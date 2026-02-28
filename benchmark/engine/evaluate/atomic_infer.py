"""
Core inference runner for atomic evaluation tasks.

Loads a JSONL dataset, builds prompts, calls batch_call_llm,
computes metrics, and writes a structured report.

Supports:
  task       "binary_noise" | "pairwise_select"
  cot        chain-of-thought mode
  market_prompt  "general" | "us" | "cn" | "auto"
  num_workers    parallel workers for batch LLM calls
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .atomic_metrics import (
    compute_binary_metrics,
    compute_pairwise_metrics,
    format_report,
    normalize_ab_label,
    normalize_noise_label,
    parse_llm_output,
)
from .atomic_prompts import (
    build_noise_system_prompt,
    build_noise_user_prompt,
    build_pairwise_system_prompt,
    build_pairwise_user_prompt,
)


# ---------------------------------------------------------------------------
# LLM client import with heuristic fallback
# ---------------------------------------------------------------------------

_HAVE_LLM = False
_LLM_ERR = None
try:
    from agent.llm_client import batch_call_llm, call_llm
    _HAVE_LLM = True
except Exception as _e:
    _LLM_ERR = _e


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ---------------------------------------------------------------------------
# Heuristic stubs (offline / testing)
# ---------------------------------------------------------------------------

def _heuristic_noise(row: dict, cot: bool) -> dict:
    """Simple heuristic: expressions with only basic ops are likely noise."""
    expr = str(row.get("expression", "")).lower()
    basic_ops = ["add(", "sub(", "mul(", "div(", "mean(", "std("]
    ratio = sum(1 for op in basic_ops if op in expr) / max(len(expr), 1)
    pred = "noise" if ratio > 0.005 else "signal"
    if cot:
        return {"analysis": f"Heuristic: basic-op density={ratio:.4f}", "prediction": pred}
    return {"prediction": pred}


def _heuristic_pairwise(row: dict, cot: bool) -> dict:
    """Pick A/B by expression length as a naive proxy for complexity."""
    len_a = len(str(row.get("A", "")))
    len_b = len(str(row.get("B", "")))
    pred = "A" if len_a >= len_b else "B"
    if cot:
        return {"analysis": f"Heuristic: len(A)={len_a}, len(B)={len_b}", "prediction": pred}
    return {"prediction": pred}


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping invalid line: {line[:100]}... ({e})")
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_atomic_infer(
    data_path: str,
    output_dir: str,
    task: str,               # "binary_noise" | "pairwise_select"
    model: str = "gpt-4.1",
    cot: bool = False,
    market_prompt: str = "auto",
    num_workers: int = 8,
    temperature: float = 0.3,
    latency: float = 0.25,
    use_batch: bool = True,
    save_prompts: bool = False,
    local: bool = False,
    local_port: int = 8000,
) -> Dict[str, Any]:
    """
    Run LLM inference on an atomic evaluation JSONL dataset.

    Args:
        data_path:    Path to the input JSONL file (train.jsonl / test.jsonl).
        output_dir:   Directory for results, metrics, logs, and report.
        task:         "binary_noise" or "pairwise_select".
        model:        LLM model name.
        cot:          Enable chain-of-thought.
        market_prompt: Market style for prompting ("general","us","cn","auto").
        num_workers:  Worker threads for batch LLM.
        temperature:  Sampling temperature.
        latency:      Inter-request latency (seconds) for rate limiting.
        use_batch:    Use batch_call_llm (True) or sequential call_llm (False).
        save_prompts: Persist system/user prompts in results.
        local:        Use local LLM server.
        local_port:   Port of local LLM server.

    Returns:
        Dict with paths to results, metrics, report, and log files.
    """
    if task not in ("binary_noise", "pairwise_select"):
        raise ValueError(f"Unknown task: {task!r}. Choose 'binary_noise' or 'pairwise_select'.")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "infer.log")
    results_path = os.path.join(output_dir, "results.jsonl")
    llm_output_path = os.path.join(output_dir, "llm_output.json")
    metrics_path = os.path.join(output_dir, "metrics.json")
    report_path = os.path.join(output_dir, "report.txt")

    rows = _read_jsonl(data_path)
    if not rows:
        raise RuntimeError(f"No valid rows loaded from: {data_path}")

    # ---- Build prompts ----
    if task == "binary_noise":
        system_prompt = build_noise_system_prompt(cot=cot, market_prompt=market_prompt)
        user_prompts = [
            build_noise_user_prompt(r, cot=cot, market_prompt=market_prompt)
            for r in rows
        ]
        heuristic_fn = _heuristic_noise
        gt_key = "ground_truth"
        normalize_gt = normalize_noise_label
        default_pred = "noise"
    else:
        system_prompt = build_pairwise_system_prompt(cot=cot, market_prompt=market_prompt)
        user_prompts = [
            build_pairwise_user_prompt(r, cot=cot, market_prompt=market_prompt)
            for r in rows
        ]
        heuristic_fn = _heuristic_pairwise
        gt_key = "ground_truth"
        normalize_gt = normalize_ab_label
        default_pred = "A"

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"[{_now()}] Starting inference\n")
        lf.write(f"  task={task}  data={data_path}  model={model}\n")
        lf.write(f"  cot={cot}  market_prompt={market_prompt}  workers={num_workers}\n")
        lf.write(f"  use_batch={use_batch}  local={local}  temperature={temperature}\n")
        if not _HAVE_LLM:
            lf.write(f"[WARN] LLM client not found; using heuristic stub. ({_LLM_ERR})\n")

    # ---- Call LLM ----
    llm_outputs: List[Any]

    if _HAVE_LLM and use_batch:
        try:
            llm_outputs = batch_call_llm(
                user_prompts,
                model=model,
                num_workers=num_workers,
                system_prompt=system_prompt,
                json_output=True,
                verbose=True,
                latency=latency,
                temperature=temperature,
                local=local,
                local_port=local_port,
            )
        except Exception as e:
            with open(log_path, "a") as lf:
                lf.write(f"[{_now()}] ERROR in batch_call_llm: {e}\nFalling back to heuristic.\n")
            llm_outputs = [heuristic_fn(rows[i], cot=cot) for i in range(len(rows))]
    elif _HAVE_LLM and not use_batch:
        llm_outputs = []
        for i, prompt in enumerate(user_prompts):
            try:
                out = call_llm(
                    prompt,
                    model=model,
                    system_prompt=system_prompt,
                    json_output=True,
                    temperature=temperature,
                    local=local,
                    local_port=local_port,
                )
            except Exception as e:
                out = heuristic_fn(rows[i], cot=cot)
            llm_outputs.append(out)
            with open(log_path, "a") as lf:
                lf.write(f"[{_now()}] {i+1}/{len(rows)} done\n")
    else:
        llm_outputs = [heuristic_fn(rows[i], cot=cot) for i in range(len(rows))]

    # Persist raw outputs
    try:
        with open(llm_output_path, "w", encoding="utf-8") as f:
            json.dump(llm_outputs, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # ---- Collate results ----
    normalized_results: List[Dict[str, Any]] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for i, (row, raw) in enumerate(zip(rows, llm_outputs)):
        pred_norm, analysis = parse_llm_output(raw, task=task, cot=cot, default=default_pred)
        if pred_norm is None:
            pred_norm = default_pred

        gt_raw = row.get(gt_key)
        gt_norm = normalize_gt(gt_raw)

        if gt_norm is not None:
            y_true.append(gt_norm)
            y_pred.append(pred_norm)

        rec: Dict[str, Any] = {
            "id": row.get("id", i),
            "ground_truth": gt_norm,
            "prediction": pred_norm,
            "model": model,
            "cot": cot,
            "task": task,
        }
        # Task-specific fields for traceability
        if task == "binary_noise":
            rec["factor_name"] = row.get("factor_name")
            rec["market"] = row.get("market")
        else:
            rec["factor_name_A"] = row.get("factor_name_A")
            rec["factor_name_B"] = row.get("factor_name_B")
            rec["market"] = row.get("market")

        if cot and analysis is not None:
            rec["analysis"] = analysis

        if save_prompts:
            rec["system_prompt"] = system_prompt
            rec["user_prompt"] = user_prompts[i]

        normalized_results.append(rec)

    _write_jsonl(results_path, normalized_results)

    # ---- Metrics ----
    if task == "binary_noise":
        metrics = compute_binary_metrics(y_true, y_pred)
    else:
        metrics = compute_pairwise_metrics(y_true, y_pred)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- Report ----
    report_str = format_report(task=task, metrics=metrics, model=model, data_path=data_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(report_str)

    with open(log_path, "a") as lf:
        lf.write(f"[{_now()}] Finished. Results → {output_dir}\n")

    return {
        "task": task,
        "results_path": results_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "log_path": log_path,
        "llm_output_path": llm_output_path,
        "metrics": metrics,
    }
