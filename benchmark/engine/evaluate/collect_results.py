import os
import json
import argparse
import ast
import csv
from datetime import datetime
from itertools import count
import math
import os
import json
import pickle
import traceback
import pandas as pd
import shutil

import numpy as np
from pathlib import Path
from agent.qlib_contrib.qlib_valid import test_qlib_operator
from agent.generator_qlib import call_gen_qlib_factors, batch_call_gen_qlib_factors

from agent.qlib_contrib.qlib_expr_parsing import FactorParser

from benchmark.engine.utils import (
    get_factor_performance_batch,
    similarity_factor_output,
)
from benchmark.engine.generate.eval_fitness import start_eval_fitness
from benchmark.engine.generate.eval_diversity import (
    eval_factor_ast_distance,
    eval_pairwise_factor_similarity,
)
from benchmark.engine.evaluate.factor_eval import (
    evaluate_performance_scoring,
    evaluate_performance_ranking,
)
import re

from agent.llm_client import batch_call_llm, call_llm
from factors.lib.alpha158 import load_factors_alpha158


import os
from typing import Dict, Tuple, Any


def safe_model_name(model: str) -> str:
    return model.replace("/", "_")


# Assume this is already defined somewhere in your codebase:
# def collect_result_from_dir(output_dir) -> Tuple[Any, Any]: ...
def summarize_ranking_table(results_dict: dict) -> pd.DataFrame:
    # scenarios to keep and their short names
    scenario_map = {
        "2021-2025 Overall": "Overall",
        "CSI300 Bear (2023-2024)": "Bear",
        "CSI300 Bull (2024.10-2025.06)": "Bull",
    }

    model_order = [
        "gpt-5",
        "gemini-2.5-pro",
        "deepseek-ai_DeepSeek-R1",
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "deepseek-ai_DeepSeek-V3",
        "llama-3.1-70b-instruct",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    ]

    target_scenarios = set(scenario_map.keys())

    # group results by model base and CoT flag
    by_model = {}
    for rec in results_dict.values():
        base = safe_model_name(rec["model"])
        by_model.setdefault(base, {})
        if rec["cot"]:
            by_model[base]["cot_true"] = rec
        else:
            by_model[base]["cot_false"] = rec

    # discover all Settings present (to define columns)
    all_settings = set()
    for rec in results_dict.values():
        df = rec["merged"]
        df2 = df[df["Scenario"].isin(target_scenarios)]
        all_settings.update(df2["Setting"].unique().tolist())
    all_settings = sorted(all_settings)

    # columns: (Setting, Metric, Variant)
    metrics = ["Top-K Precision", "NDCG@k"]
    col_tuples = []
    for setting in all_settings:
        for metric in metrics:
            col_tuples.append((setting, metric, "Origin"))
            col_tuples.append((setting, metric, "CoT"))
    columns = pd.MultiIndex.from_tuples(
        col_tuples, names=["Setting", "Metric", "Variant"]
    )

    # create one table per scenario short name; rows are (Scenario, Model)
    out_tables = {}
    for long_scn, short_scn in scenario_map.items():
        row_models = [m for m in model_order if m in by_model]
        # build row index (Scenario, Model) for all models
        row_index = pd.MultiIndex.from_tuples(
            [(short_scn, m.replace("deepseek-ai_", "")) for m in row_models],
            names=["Scenario", "Model"],
        )
        out = pd.DataFrame(index=row_index, columns=columns, dtype=float)

        # fill values from each model's origin/cot entries
        for model, slots in by_model.items():
            for variant_name, flag in [("Origin", "cot_false"), ("CoT", "cot_true")]:
                rec = slots.get(flag)
                if rec is None:
                    continue
                dfm = rec["merged"].copy()
                # keep only rows for this scenario
                dfm = dfm[dfm["Scenario"] == long_scn]
                if dfm.empty:
                    continue
                dfm = dfm.set_index("Setting")
                for setting in all_settings:
                    if setting in dfm.index:
                        for metric in metrics:
                            if metric in dfm.columns:
                                out.loc[
                                    (short_scn, model), (setting, metric, variant_name)
                                ] = dfm.loc[setting, metric]

        # round to 2 decimals and '-' for missing
        out = out.applymap(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        # import pdb;pdb.set_trace()
        # rename metric if needed
        # out = out.rename(columns=lambda tup: (tup[0], "Precision@k" if tup[1]=="Top-K Precision" else tup[1], tup[2]))
        out_tables[short_scn] = out

    return out_tables


def safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _display_model_name(s: str) -> str:
    # mirror your ranking table's row label behavior
    return s.replace("deepseek-ai_", "")


def summarize_scoring_tables_split(results_dict: dict) -> dict:
    metrics = [
        "Acc_Signal",
        "MAE_Performance",
        "MAE_Stability",
        "MAE_WinRate",
        "MAE_Skewness",
    ]

    model_order = [
        "gpt-5",
        "gemini-2.5-pro",
        "deepseek-ai_DeepSeek-R1",
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "deepseek-ai_DeepSeek-V3",
        "llama-3.1-70b-instruct",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    ]

    # group results by model base and CoT flag
    by_model = {}
    envs = set()
    for rec in results_dict.values():
        base = safe_model_name(rec["model"])
        by_model.setdefault(base, {})
        if rec["cot"]:
            by_model[base]["cot_true"] = rec
        else:
            by_model[base]["cot_false"] = rec
        envs.update(list(rec["env_mae_summary"].index))

    # environments (scenarios) to split on, ordered as Bull, Bear, Overall
    env_order_key = {"Bull": 0, "Bear": 1, "Overall": 2}
    env_list = sorted(envs, key=lambda x: env_order_key.get(x, 99))

    # columns: (Metric, Variant)
    columns = pd.MultiIndex.from_tuples(
        [(m, v) for m in metrics for v in ("Origin", "CoT")],
        names=["Metric", "Variant"],
    )

    tables = {}
    # build row model order filtered to those present
    row_models = [m for m in model_order if m in by_model]
    row_index = pd.Index([_display_model_name(m) for m in row_models], name="Model")

    for env in env_list:
        out = pd.DataFrame(index=row_index, columns=columns, dtype=float)

        for base in row_models:
            disp = _display_model_name(base)
            slots = by_model.get(base, {})
            for variant_name, flag in [("Origin", "cot_false"), ("CoT", "cot_true")]:
                rec = slots.get(flag)
                if rec is None:
                    continue
                df = rec["env_mae_summary"]
                if env not in df.index:
                    continue
                for m in metrics:
                    if m in df.columns:
                        out.loc[disp, (m, variant_name)] = df.loc[env, m]

        # round 2dp and fill missing with '-'
        out = out.applymap(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        tables[env] = out

    return tables


def _safe_model_name(model: str) -> str:
    """Replace '/' with '_' to match your run folder naming."""
    return model.replace("/", "_")


def _build_expected_depth_dir(base_dir: str, model: str, cot: bool) -> str:
    """
    Expected structure:
      {base_dir}/{safe_model}_cot_{true|false}/{safe_model}_{True|False}
    """
    safe = _safe_model_name(model)
    outer = f"{safe}_cot_{str(cot).lower()}"
    inner = f"{safe}_{str(cot).capitalize()}"
    return os.path.join(base_dir, outer, inner)


def _find_single_subdir(path: str) -> str:
    """
    Descend into path until it has 0 or more than 1 subdirs.
    If at some level there's exactly one subdir, follow it.
    Stops at the deepest unique path.
    """
    cur = path
    # If the path itself does not exist, go one level up
    if not os.path.isdir(cur):
        cur = os.path.dirname(cur)

    while True:
        subs = [d for d in os.listdir(cur) if os.path.isdir(os.path.join(cur, d))]
        if len(subs) == 1:
            # go deeper
            cur = os.path.join(cur, subs[0])
        else:
            # stop if 0 or >1
            break
    return cur


def collect_all_runs(
    base_dir: str,
    model_with_cot_config,
    model_without_cot_config,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "<safe_model>_cot_true":  {"merged": ..., "env_mae_summary": ...},
        "<safe_model>_cot_false": {"merged": ..., "env_mae_summary": ...},
        ...
      }
    """
    results: Dict[str, Dict[str, Any]] = {}

    # Models that run with CoT=True
    for model in model_with_cot_config:
        cot = True
        key = f"{_safe_model_name(model)}_cot_{str(cot).lower()}"
        depth_dir = _build_expected_depth_dir(base_dir, model, cot)
        depth_dir = _find_single_subdir(depth_dir)

        if not os.path.isdir(depth_dir):
            print(f"[WARN] Missing directory: {depth_dir}")
            continue

        try:
            merged, env_mae_summary = collect_result_from_dir(depth_dir)
            results[key] = {
                "model": model,
                "cot": cot,
                "path": depth_dir,
                "merged": merged,
                "env_mae_summary": env_mae_summary,
            }
        except Exception as e:
            print(f"[ERROR] Failed on {key} ({depth_dir}): {e}")
            traceback.print_exc()  # 打印完整的堆栈信息

    # Models that run with CoT=False
    for model in model_without_cot_config:
        cot = False
        key = f"{_safe_model_name(model)}_cot_{str(cot).lower()}"
        depth_dir = _build_expected_depth_dir(base_dir, model, cot)
        depth_dir = _find_single_subdir(depth_dir)

        if not os.path.isdir(depth_dir):
            print(f"[WARN] Missing directory: {depth_dir}")
            continue

        try:
            merged, env_mae_summary = collect_result_from_dir(depth_dir)
            results[key] = {
                "model": model,
                "cot": cot,
                "path": depth_dir,
                "merged": merged,
                "env_mae_summary": env_mae_summary,
            }
        except Exception as e:
            print(f"[ERROR] Failed on {key} ({depth_dir}): {e}")

    return results


def collect_result_from_dir(outdir):

    rank_case_path = os.path.join("./benchmark/data/evaluation/all_env_scenarios.json")
    with open(rank_case_path, "r") as f:
        test_cases_ranking = json.load(f)

    rank_output_file = os.path.join(outdir, "ranking_results.json")
    with open(rank_output_file, "r") as f:
        results_ranking = json.load(f)

    precision_avg, ndcg_avg = evaluate_performance_ranking(
        test_cases_ranking, results_ranking
    )
    # print("Ranking Evaluation Results:")
    # print(precision_avg)
    # print(ndcg_avg)

    # Convert to DataFrames
    df_p = (
        pd.DataFrame(precision_avg).T.reset_index().rename(columns={"index": "Setting"})
    )
    df_n = pd.DataFrame(ndcg_avg).T.reset_index().rename(columns={"index": "Setting"})

    # Melt (tidy) both and merge on Setting + Scenario
    p_tidy = df_p.melt(
        id_vars="Setting", var_name="Scenario", value_name="Top-K Precision"
    )
    n_tidy = df_n.melt(id_vars="Setting", var_name="Scenario", value_name="NDCG@k")
    merged = p_tidy.merge(n_tidy, on=["Setting", "Scenario"])

    # Order rows
    order_settings = ["10_pick_3", "20_pick_5", "40_pick_10"]
    order_scenarios = [
        "2021-2025 Overall",
        "CSI300 Bear (2023-2024)",
        "CSI300 Bull (2024.10-2025.06)",
    ]
    merged["Setting"] = pd.Categorical(
        merged["Setting"], categories=order_settings, ordered=True
    )
    merged["Scenario"] = pd.Categorical(
        merged["Scenario"], categories=order_scenarios, ordered=True
    )
    merged = merged.sort_values(["Setting", "Scenario"]).reset_index(drop=True)

    # print(merged)

    score_output_file = os.path.join(outdir, "scoring_results.json")
    with open(score_output_file, "r") as f:
        results_scoring = json.load(f)

    score_case_path = os.path.join(
        "./benchmark/data/evaluation/alphabench_testset.json"
    )
    with open(score_case_path, "r") as f:
        test_cases_scoring = json.load(f)

    env_mae_summary, env_classification_report, per_case = evaluate_performance_scoring(
        test_cases_scoring["items"], results_scoring
    )
    # print("Scoring Evaluation Results:")
    # print(env_mae_summary)
    # print(env_classification_report)

    final_result_pkg = {
        "ranking_precision": precision_avg,
        "ranking_ndcg": ndcg_avg,
        "scoring_mae": env_mae_summary,
        "scoring_classification": env_classification_report,
        "detailed_case_results": per_case,
    }

    with open(os.path.join(outdir, "final_evaluation_results.pkl"), "wb") as f:
        pickle.dump(final_result_pkg, f)

    return merged, env_mae_summary


def export_latex_combined_table(out_tables: dict) -> str:
    """
    Build a single LaTeX table (table*) combining Overall/Bear/Bull,
    showing Origin / CoT in each cell, sorted by the custom model order,
    removing 'deepseek-ai_' in display names, inserting split lines between
    scenario groups, fitting width via adjustbox, and bolding per-scenario
    column maxima for each (Setting, Metric, Variant).
    """

    SETTINGS = ["10_pick_3", "20_pick_5", "40_pick_10"]
    METRICS = ["Top-K Precision", "NDCG@k"]
    VARIANTS = ["Origin", "CoT"]

    order_raw = [
        "gpt-5",
        "gemini-2.5-pro",
        "deepseek-ai_DeepSeek-R1",
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "deepseek-ai_DeepSeek-V3",
        "llama-3.1-70b-instruct",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    ]
    norm = lambda s: s.replace("deepseek-ai_", "")
    model_order = [norm(x) for x in order_raw]
    order_index = {m: i for i, m in enumerate(model_order)}

    def latex_escape(s: str) -> str:
        return s.replace("_", r"\_")

    def parse_num(x):
        try:
            return float(x)
        except Exception:
            return None

    # Build LaTeX
    lines = []
    header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{ll|cc|cc|cc}
\toprule
\multirow{2}{*}{Scenario} & \multirow{2}{*}{Model} 
& \multicolumn{2}{c|}{10\_pick\_3} 
& \multicolumn{2}{c|}{20\_pick\_5} 
& \multicolumn{2}{c}{40\_pick\_10} \\
\cmidrule(r){3-4} \cmidrule(r){5-6} \cmidrule(r){7-8}
 &  & Top-K Prec. & NDCG@k & Top-K Prec. & NDCG@k & Top-K Prec. & NDCG@k \\
\midrule
"""
    lines.append(header)

    for s_idx, scenario_key in enumerate(["Overall", "Bear", "Bull"]):
        df = out_tables[scenario_key].copy()
        # Drop Scenario level if present
        if "Scenario" in df.index.names:
            try:
                df = df.droplevel("Scenario")
            except Exception:
                pass

        # Group by normalized display name to dedup prefixed variants
        model_keys = list(df.index.astype(str))
        grouped = {}
        for mk in model_keys:
            grouped.setdefault(norm(mk), []).append(mk)

        # Choose representative key (prefer row with most numeric entries)
        def score_row(mk):
            cnt = 0
            for setting in SETTINGS:
                for metric in METRICS:
                    for var in VARIANTS:
                        val = df.get((setting, metric, var))
                        if val is not None:
                            v = df.loc[mk, (setting, metric, var)]
                            if parse_num(v) is not None:
                                cnt += 1
            return cnt

        chosen_key = {}
        for disp, cands in grouped.items():
            if len(cands) == 1:
                chosen_key[disp] = cands[0]
            else:
                best = max(cands, key=score_row)
                chosen_key[disp] = best

        # Scenario-wise maxima per (setting, metric, variant)
        max_map = {}  # (setting, metric, variant) -> max_value
        for setting in SETTINGS:
            for metric in METRICS:
                for var in VARIANTS:
                    vals = []
                    for disp, mk in chosen_key.items():
                        try:
                            v = parse_num(df.loc[mk, (setting, metric, var)])
                            if v is not None:
                                vals.append(v)
                        except KeyError:
                            pass
                    max_map[(setting, metric, var)] = max(vals) if vals else None

        # Order display models by custom order
        disp_models = list(chosen_key.keys())
        disp_models.sort(key=lambda m: order_index.get(m, 10**9))

        # Begin scenario block with multirow
        nrows = len(disp_models)
        if nrows == 0:
            continue
        lines.append(rf"\multirow{{{nrows}}}{{*}}{{{scenario_key}}} ")

        for i, disp in enumerate(disp_models):
            mk = chosen_key[disp]
            # Build cell texts with bold on maxima
            cell_chunks = []
            for setting in SETTINGS:
                # Top-K Precision (Origin / CoT)
                vals = []
                for var in VARIANTS:
                    try:
                        raw = df.loc[mk, (setting, "Top-K Precision", var)]
                    except KeyError:
                        raw = "-"
                    vnum = parse_num(raw)
                    maxv = max_map.get((setting, "Top-K Precision", var))
                    if (
                        vnum is not None
                        and maxv is not None
                        and abs(vnum - maxv) < 1e-9
                    ):
                        vals.append(rf"\textbf{{{raw}}}")
                    else:
                        vals.append(raw if raw != "-" else "-")
                cell_chunks.append(f"{vals[0]} / {vals[1]}")

                # NDCG@k (Origin / CoT)
                vals = []
                for var in VARIANTS:
                    try:
                        raw = df.loc[mk, (setting, "NDCG@k", var)]
                    except KeyError:
                        raw = "-"
                    vnum = parse_num(raw)
                    maxv = max_map.get((setting, "NDCG@k", var))
                    if (
                        vnum is not None
                        and maxv is not None
                        and abs(vnum - maxv) < 1e-9
                    ):
                        vals.append(rf"\textbf{{{raw}}}")
                    else:
                        vals.append(raw if raw != "-" else "-")
                cell_chunks.append(f"{vals[0]} / {vals[1]}")

            model_disp = latex_escape(norm(disp))
            # (1) model column with trailing &
            row = (
                f"& {model_disp:<25} & "
                + " & ".join(f"{c:>10}" for c in cell_chunks)
                + r" \\"
            )
            lines.append(row)

        # (2) split line after each scenario group
        lines.append(r"\midrule")

    footer = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Performance comparison under three market environments (Overall, Bear, Bull). Each cell reports \textit{Origin / CoT}. Bold indicates the best per-scenario value in each column (Origin and CoT considered separately).}
\label{tab:performance_all}
\end{table*}
"""
    lines.append(footer)
    return "\n".join(lines)


def export_latex_combined_scoring_table(out_tables: dict) -> str:
    """
    Build a single LaTeX table (table*) combining Overall/Bear/Bull
    for scoring metrics. Each cell shows 'Origin / CoT'.
    Bold rule: Acc_Signal -> higher is better; MAE_* -> lower is better.
    `out_tables` is a dict like {'Overall': df, 'Bear': df, 'Bull': df}
    produced by summarize_scoring_tables_split(...).
    """

    METRICS = [
        "Acc_Signal",
        "MAE_Performance",
        "MAE_Stability",
        "MAE_WinRate",
        "MAE_Skewness",
    ]
    VARIANTS = ["Origin", "CoT"]

    order_raw = [
        "gpt-5",
        "gemini-2.5-pro",
        "deepseek-ai_DeepSeek-R1",
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "deepseek-ai_DeepSeek-V3",
        "llama-3.1-70b-instruct",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    ]
    norm = lambda s: s.replace("deepseek-ai_", "")
    model_order = [norm(x) for x in order_raw]
    order_index = {m: i for i, m in enumerate(model_order)}

    def latex_escape(s: str) -> str:
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
        )

    def parse_num(x):
        try:
            return float(x)
        except Exception:
            return None

    def is_higher_better(metric: str) -> bool:
        return metric == "Acc_Signal"

    # ---------------- Header ----------------
    lines = []
    header = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{ll|cc|cc|cc|cc|cc}
\toprule
\multirow{2}{*}{Scenario} & \multirow{2}{*}{Model}
& \multicolumn{2}{c|}{Acc\_Signal}
& \multicolumn{2}{c|}{MAE\_Performance}
& \multicolumn{2}{c|}{MAE\_Stability}
& \multicolumn{2}{c|}{MAE\_WinRate}
& \multicolumn{2}{c}{MAE\_Skewness} \\
\cmidrule(r){3-4} \cmidrule(r){5-6} \cmidrule(r){7-8} \cmidrule(r){9-10} \cmidrule(r){11-12}
 &  & Origin & CoT & Origin & CoT & Origin & CoT & Origin & CoT & Origin & CoT \\
\midrule
"""
    lines.append(header)

    # Scenario order
    for scenario_key in ["Overall", "Bear", "Bull"]:
        df = out_tables[scenario_key].copy()

        # 行索引是 Model（已是 display 名），但谨慎处理可能的层级
        if "Scenario" in df.index.names:
            try:
                df = df.droplevel("Scenario")
            except Exception:
                pass

        # 可能存在同名（前缀差异）行，做归并并挑“信息更全”的那一行
        model_keys = list(map(str, df.index.tolist()))
        grouped = {}
        for mk in model_keys:
            grouped.setdefault(norm(mk), []).append(mk)

        def score_row(mk):
            cnt = 0
            for m in METRICS:
                for v in VARIANTS:
                    try:
                        val = df.loc[mk, (m, v)]
                    except KeyError:
                        val = None
                    if parse_num(val) is not None:
                        cnt += 1
            return cnt

        chosen_key = {}
        for disp, cands in grouped.items():
            if len(cands) == 1:
                chosen_key[disp] = cands[0]
            else:
                best = max(cands, key=score_row)
                chosen_key[disp] = best

        # 计算每列（按 metric, variant）在该场景下的最佳值（Acc 最大；MAE 最小）
        best_map = {}  # (metric, variant) -> best_value
        for metric in METRICS:
            for var in VARIANTS:
                vals = []
                for disp, mk in chosen_key.items():
                    try:
                        v = parse_num(df.loc[mk, (metric, var)])
                        if v is not None:
                            vals.append(v)
                    except KeyError:
                        pass
                if not vals:
                    best_map[(metric, var)] = None
                else:
                    best_map[(metric, var)] = (
                        max(vals) if is_higher_better(metric) else min(vals)
                    )

        # 排序行顺序
        disp_models = list(chosen_key.keys())
        disp_models.sort(key=lambda m: order_index.get(m, 10**9))

        # 写入行
        nrows = len(disp_models)
        if nrows == 0:
            continue
        lines.append(rf"\multirow{{{nrows}}}{{*}}{{{scenario_key}}} ")
        for i, disp in enumerate(disp_models):
            mk = chosen_key[disp]
            cell_vals = []
            for metric in METRICS:
                for var in VARIANTS:
                    try:
                        raw = df.loc[mk, (metric, var)]
                    except KeyError:
                        raw = "-"
                    vnum = parse_num(raw)
                    best = best_map.get((metric, var))
                    if (
                        vnum is not None
                        and best is not None
                        and abs(vnum - best) < 1e-12
                    ):
                        cell_vals.append(rf"\textbf{{{raw}}}")
                    else:
                        cell_vals.append(raw if raw != "-" else "-")

            model_disp = latex_escape(norm(disp))
            row = "& {:<25} & {} \\\\".format(
                model_disp, " & ".join(f"{c:>6}" for c in cell_vals)
            )
            lines.append(row)

        lines.append(r"\midrule")

    footer = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Scoring metrics under three market environments (Overall, Bear, Bull). Each cell reports \textit{Origin} and \textit{CoT}. Bold indicates the per-scenario best value for each metric (higher is better for Acc\_Signal; lower is better for MAE metrics).}
\label{tab:scoring_all}
\end{table*}
"""
    lines.append(footer)
    return "\n".join(lines)


if __name__ == "__main__":
    model_withou_cot_config = ["gemini-2.5-pro", "gpt-5", "deepseek-ai/DeepSeek-R1"]
    model_with_cot_config = [
        "gemini-2.5-flash",
        "gpt-4.1-mini",
        "zd",
        "llama-3.1-70b-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]

    base_dir = "/home/hluo/workdir/AlphaBench/runs/T2_official"

    model_without_cot_config = [
        "gemini-2.5-pro",
        "gpt-5",
        "gemini-2.5-flash",
        "gpt-4.1-mini",
        "deepseek-ai/DeepSeek-V3",
        # "deepseek-ai/DeepSeek-R1",
        "llama-3.1-70b-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]
    model_with_cot_config = [
        "gemini-2.5-flash",
        "gpt-4.1-mini",
        "deepseek-ai/DeepSeek-V3",
        # "deepseek-ai/DeepSeek-R1",
        "llama-3.1-70b-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "llama-3.1-8b-instruct",
        "gemini-1.5-flash-8b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]

    all_results = collect_all_runs(
        base_dir, model_with_cot_config, model_without_cot_config
    )
    ranking_df = summarize_ranking_table(all_results)
    latex_code = export_latex_combined_table(
        ranking_df
    )  # your dict with 'Overall'/'Bear'/'Bull'
    print(latex_code)  # paste into your .tex

    scoring_df = summarize_scoring_tables_split(all_results)
    latex_scoring = export_latex_combined_scoring_table(scoring_df)
    print(latex_scoring)

    import pdb

    pdb.set_trace()

    # For all float, round (2)
    # Table (1)
    # Details about ranking
    # Col: Setting (merge) -> Scenario (only keep Overall, Bear and Bull), and columns is Model name,  (Precision#k    NDCG@k, each has two column: Origin, CoT (for model don't have CoT, use -))

    # Table (2)
    # Details about scoring
    # Col: env_mae_summary.index Model name, (Acc_Signal, MAE_Performance  MAE_Stability  MAE_WinRate  MAE_Skewness, each has two column: Origin, CoT (for model don't have CoT, use -))
