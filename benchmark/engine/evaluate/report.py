"""
Performance Report Generator for AlphaBench T2 (Ranking & Scoring).

Loads existing result files from a run directory and produces:
  1. Ranking summary table  (Precision@K, NDCG@K per regime + setting)
  2. Scoring summary table  (Acc_Signal, MAE_* per regime)
  3. Classification report  (per-class Precision/Recall/F1 per regime)
  4. Optional LaTeX export  (mirrors collect_results.py format)
  5. Markdown / text summary

Usage:
    from benchmark.engine.evaluate.report import generate_report
    generate_report(
        run_dir="./runs/T2/gpt-4.1_False",
        ranking_cases_path="./benchmark/data/evaluation/all_env_scenarios.json",
        scoring_cases_path="./benchmark/data/evaluation/alphabench_testset.json",
        output_dir="./runs/T2/gpt-4.1_False/report",
    )

CLI:
    python -m benchmark.engine.evaluate.report \\
        --run_dir ./runs/T2/gpt-4.1_False \\
        --output_dir ./runs/T2/gpt-4.1_False/report
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .factor_eval import evaluate_performance_ranking, evaluate_performance_scoring


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Ranking report
# ---------------------------------------------------------------------------

def _ranking_summary(precision_avg: dict, ndcg_avg: dict) -> pd.DataFrame:
    """Build a flat DataFrame from nested {setting: {env: value}} dicts."""
    records = []
    for setting in precision_avg:
        for env in precision_avg[setting]:
            records.append({
                "Setting": setting,
                "Environment": env,
                "Precision@K": round(precision_avg[setting][env], 4),
                "NDCG@K": round(ndcg_avg[setting][env], 4),
            })
    df = pd.DataFrame(records)
    setting_order = ["10_pick_3", "20_pick_5", "40_pick_10"]
    df["Setting"] = pd.Categorical(df["Setting"], categories=setting_order, ordered=True)
    return df.sort_values(["Setting", "Environment"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    run_dir: str,
    ranking_cases_path: str = "./benchmark/data/evaluation/all_env_scenarios.json",
    scoring_cases_path: str = "./benchmark/data/evaluation/alphabench_testset.json",
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate a full performance report for a T2 evaluation run.

    Args:
        run_dir:             Directory containing ranking_results.json and scoring_results.json.
        ranking_cases_path:  Path to the ranking test-cases JSON.
        scoring_cases_path:  Path to the scoring test-cases JSON.
        output_dir:          Where to write report files (defaults to run_dir/report/).
        verbose:             Print summaries to stdout.

    Returns:
        Dict with keys: ranking_df, scoring_df, classification_df, per_case_df, output_paths
    """
    if output_dir is None:
        output_dir = os.path.join(run_dir, "report")
    os.makedirs(output_dir, exist_ok=True)

    output_paths: Dict[str, str] = {}

    # ---- Ranking ----
    rank_results_path = os.path.join(run_dir, "ranking_results.json")
    ranking_df = scoring_df = classification_df = per_case_df = None

    if os.path.exists(rank_results_path) and os.path.exists(ranking_cases_path):
        test_cases_ranking = _load_json(ranking_cases_path)
        results_ranking = _load_json(rank_results_path)

        precision_avg, ndcg_avg = evaluate_performance_ranking(
            test_cases_ranking, results_ranking
        )
        ranking_df = _ranking_summary(precision_avg, ndcg_avg)

        ranking_csv = os.path.join(output_dir, "ranking_summary.csv")
        ranking_df.to_csv(ranking_csv, index=False)
        output_paths["ranking_csv"] = ranking_csv

        if verbose:
            print("\n=== Ranking Summary ===")
            print(ranking_df.to_string(index=False))
    else:
        if verbose:
            print(f"[Report] Ranking results not found; skipping. "
                  f"(expected: {rank_results_path})")

    # ---- Scoring ----
    score_results_path = os.path.join(run_dir, "scoring_results.json")

    if os.path.exists(score_results_path) and os.path.exists(scoring_cases_path):
        test_cases_scoring_raw = _load_json(scoring_cases_path)
        # Support both {"items": [...]} and plain list
        if isinstance(test_cases_scoring_raw, dict):
            test_cases_scoring = test_cases_scoring_raw.get("items", [])
        else:
            test_cases_scoring = test_cases_scoring_raw

        results_scoring = _load_json(score_results_path)

        env_mae_summary, env_classification_report, per_case = evaluate_performance_scoring(
            test_cases_scoring, results_scoring
        )
        scoring_df = env_mae_summary
        classification_df = env_classification_report
        per_case_df = per_case

        # Write CSVs
        scoring_csv = os.path.join(output_dir, "scoring_mae_summary.csv")
        clf_csv = os.path.join(output_dir, "scoring_classification.csv")
        per_case_csv = os.path.join(output_dir, "scoring_per_case.csv")

        scoring_df.to_csv(scoring_csv)
        classification_df.to_csv(clf_csv, index=False)
        per_case.to_csv(per_case_csv, index=False)

        output_paths.update({
            "scoring_mae_csv": scoring_csv,
            "scoring_classification_csv": clf_csv,
            "scoring_per_case_csv": per_case_csv,
        })

        if verbose:
            print("\n=== Scoring MAE Summary ===")
            print(scoring_df.round(4).to_string())
            print("\n=== Classification Report (Overall) ===")
            overall = env_classification_report[
                env_classification_report["Environment"] == "ALL"
            ]
            print(overall[["Class", "Precision", "Recall", "F1", "Support"]].round(4).to_string(index=False))
    else:
        if verbose:
            print(f"[Report] Scoring results not found; skipping. "
                  f"(expected: {score_results_path})")

    # ---- Markdown summary ----
    md_lines = ["# AlphaBench T2 Performance Report\n"]
    md_lines.append(f"**Run directory:** `{run_dir}`\n")

    if ranking_df is not None:
        md_lines.append("## Ranking (Precision@K / NDCG@K)\n")
        md_lines.append(ranking_df.to_markdown(index=False))
        md_lines.append("")

    if scoring_df is not None:
        md_lines.append("## Scoring MAE Summary\n")
        md_lines.append(scoring_df.round(4).to_markdown())
        md_lines.append("")

    if classification_df is not None:
        overall = classification_df[classification_df["Environment"] == "ALL"]
        md_lines.append("## Signal Classification (ALL environments)\n")
        md_lines.append(overall[["Class", "Precision", "Recall", "F1", "Support"]].round(4).to_markdown(index=False))
        md_lines.append("")

    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    output_paths["report_md"] = md_path

    # ---- Pickle full result package (mirrors collect_results.py) ----
    result_pkg = {
        "ranking_precision": None,
        "ranking_ndcg": None,
        "scoring_mae": scoring_df,
        "scoring_classification": classification_df,
        "ranking_summary": ranking_df,
        "per_case": per_case_df,
    }
    if ranking_df is not None:
        # Reconstruct original nested dicts from flat DataFrame
        precision_dict: dict = {}
        ndcg_dict: dict = {}
        for _, row in ranking_df.iterrows():
            s, e = row["Setting"], row["Environment"]
            precision_dict.setdefault(s, {})[e] = row["Precision@K"]
            ndcg_dict.setdefault(s, {})[e] = row["NDCG@K"]
        result_pkg["ranking_precision"] = precision_dict
        result_pkg["ranking_ndcg"] = ndcg_dict

    pkl_path = os.path.join(output_dir, "final_evaluation_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(result_pkg, f)
    output_paths["pickle"] = pkl_path

    if verbose:
        print(f"\n[Report] Files written to: {output_dir}")

    return {
        "ranking_df": ranking_df,
        "scoring_df": scoring_df,
        "classification_df": classification_df,
        "per_case_df": per_case_df,
        "output_paths": output_paths,
    }


# ---------------------------------------------------------------------------
# Multi-run aggregation (compare models)
# ---------------------------------------------------------------------------

def compare_runs(
    run_dirs: Dict[str, str],   # {label: run_dir}
    ranking_cases_path: str = "./benchmark/data/evaluation/all_env_scenarios.json",
    scoring_cases_path: str = "./benchmark/data/evaluation/alphabench_testset.json",
    output_dir: str = "./runs/comparison",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Generate a side-by-side comparison of multiple evaluation runs.

    Args:
        run_dirs:   {"GPT-4.1 (no CoT)": "./runs/T2/gpt-4.1_False", ...}
        output_dir: Where to write comparison tables.

    Returns:
        {"ranking": comparison_df, "scoring": comparison_df}
    """
    os.makedirs(output_dir, exist_ok=True)

    ranking_rows: List[dict] = []
    scoring_rows: List[dict] = []

    for label, run_dir in run_dirs.items():
        result = generate_report(
            run_dir=run_dir,
            ranking_cases_path=ranking_cases_path,
            scoring_cases_path=scoring_cases_path,
            output_dir=os.path.join(output_dir, label.replace(" ", "_")),
            verbose=False,
        )

        if result["ranking_df"] is not None:
            rdf = result["ranking_df"].copy()
            rdf.insert(0, "Model", label)
            ranking_rows.append(rdf)

        if result["scoring_df"] is not None:
            sdf = result["scoring_df"].reset_index()
            sdf.insert(0, "Model", label)
            scoring_rows.append(sdf)

    output_paths: Dict[str, str] = {}

    ranking_cmp = pd.DataFrame()
    if ranking_rows:
        ranking_cmp = pd.concat(ranking_rows, ignore_index=True)
        csv_path = os.path.join(output_dir, "ranking_comparison.csv")
        ranking_cmp.to_csv(csv_path, index=False)
        output_paths["ranking_comparison"] = csv_path
        if verbose:
            print("\n=== Ranking Comparison ===")
            print(ranking_cmp.round(4).to_string(index=False))

    scoring_cmp = pd.DataFrame()
    if scoring_rows:
        scoring_cmp = pd.concat(scoring_rows, ignore_index=True)
        csv_path = os.path.join(output_dir, "scoring_comparison.csv")
        scoring_cmp.to_csv(csv_path, index=False)
        output_paths["scoring_comparison"] = csv_path
        if verbose:
            print("\n=== Scoring Comparison ===")
            print(scoring_cmp.round(4).to_string(index=False))

    return {"ranking": ranking_cmp, "scoring": scoring_cmp, "output_paths": output_paths}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate AlphaBench T2 performance report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run_dir", required=True,
                   help="Directory containing ranking_results.json and scoring_results.json.")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for report files (default: <run_dir>/report/).")
    p.add_argument("--ranking_cases",
                   default="./benchmark/data/evaluation/all_env_scenarios.json",
                   help="Path to ranking test-cases JSON.")
    p.add_argument("--scoring_cases",
                   default="./benchmark/data/evaluation/alphabench_testset.json",
                   help="Path to scoring test-cases JSON.")
    return p.parse_args()


def main():
    args = _parse_args()
    generate_report(
        run_dir=args.run_dir,
        ranking_cases_path=args.ranking_cases,
        scoring_cases_path=args.scoring_cases,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
