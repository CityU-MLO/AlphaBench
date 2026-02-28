"""
Metrics computation for atomic evaluation tasks.

Task 1 – Pairwise Select:  accuracy (correct A/B choice).
Task 2 – Binary Noise:     accuracy, precision, recall, F1 (positive = noise).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

_NOISE_SYNS = {
    "noise", "noisy", "yes", "y", "1", "true", "random", "uninformative",
    "spurious", "useless", "white noise", "pure noise",
}
_SIGNAL_SYNS = {
    "signal", "alpha", "no", "n", "0", "false", "non-noise", "not noise",
    "non noise", "clean", "informative", "predictive", "useful", "meaningful",
}


def normalize_noise_label(value: Any) -> Optional[str]:
    """Normalize to 'noise' or 'signal' (or None if unrecognized)."""
    if isinstance(value, bool):
        return "noise" if value else "signal"
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if ("non" in s and "noise" in s) or "not noise" in s:
        return "signal"
    if s in _NOISE_SYNS or "noise" in s:
        return "noise"
    if s in _SIGNAL_SYNS:
        return "signal"
    return None


def normalize_ab_label(value: Any) -> Optional[str]:
    """Normalize to 'A' or 'B' (or None if unrecognized)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in {"a", "choice a", "option a", "select a", "1", "first", "left"}:
        return "A"
    if s in {"b", "choice b", "option b", "select b", "2", "second", "right"}:
        return "B"
    m = re.match(r"^\s*([abAB])\b", str(value).strip())
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", str(value).strip())
    if m:
        return m.group(1).upper()
    return None


def parse_llm_output(
    raw: Any,
    task: str,  # "binary_noise" | "pairwise_select"
    cot: bool,
    default: str = "noise",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse raw LLM output dict/string into (prediction_normalized, analysis).

    Returns:
        (pred_norm, analysis)  where pred_norm may be None if parsing fails.
    """
    out_dict = None
    if isinstance(raw, dict):
        out_dict = raw
    else:
        try:
            import json
            out_dict = json.loads(str(raw))
        except Exception:
            pass

    pred_raw = None
    analysis = None

    if out_dict is not None:
        analysis = out_dict.get("analysis") if cot else None
        # Try both casing conventions
        pred_raw = out_dict.get("prediction", out_dict.get("Prediction"))

    if task == "binary_noise":
        pred_norm = normalize_noise_label(pred_raw)
    else:
        pred_norm = normalize_ab_label(pred_raw)

    return pred_norm, analysis


# ---------------------------------------------------------------------------
# Task 2: Binary classification metrics (positive = "noise")
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1 for binary noise classification.
    Positive class = 'noise'.
    """
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t not in {"noise", "signal"} or p not in {"noise", "signal"}:
            continue
        if t == "noise" and p == "noise":
            tp += 1
        elif t == "signal" and p == "signal":
            tn += 1
        elif t == "signal" and p == "noise":
            fp += 1
        elif t == "noise" and p == "signal":
            fn += 1

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Task 1: Pairwise accuracy metrics
# ---------------------------------------------------------------------------

def compute_pairwise_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Compute accuracy for pairwise selection (A vs B).
    """
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred)
                  if t in {"A", "B"} and p in {"A", "B"} and t == p)
    wrong = sum(1 for t, p in zip(y_true, y_pred)
                if t in {"A", "B"} and p in {"A", "B"} and t != p)
    unknown = n - correct - wrong
    acc = correct / n if n > 0 else 0.0

    gt_a = sum(1 for t in y_true if t == "A")
    gt_b = sum(1 for t in y_true if t == "B")
    pred_a = sum(1 for p in y_pred if p == "A")
    pred_b = sum(1 for p in y_pred if p == "B")

    confusion = {
        "A→A": sum(1 for t, p in zip(y_true, y_pred) if t == "A" and p == "A"),
        "A→B": sum(1 for t, p in zip(y_true, y_pred) if t == "A" and p == "B"),
        "B→B": sum(1 for t, p in zip(y_true, y_pred) if t == "B" and p == "B"),
        "B→A": sum(1 for t, p in zip(y_true, y_pred) if t == "B" and p == "A"),
    }

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "gt_counts": {"A": gt_a, "B": gt_b},
        "pred_counts": {"A": pred_a, "B": pred_b},
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    task: str,
    metrics: Dict[str, Any],
    model: str,
    data_path: str,
) -> str:
    lines = [
        "=" * 60,
        f"AlphaBench Atomic Eval Report",
        f"  Task  : {task}",
        f"  Model : {model}",
        f"  Data  : {data_path}",
        "=" * 60,
    ]

    if task == "binary_noise":
        lines += [
            f"  N          : {metrics['n']}",
            f"  Accuracy   : {metrics['accuracy']:.4f}",
            f"  Precision  : {metrics['precision']:.4f}  (positive = noise)",
            f"  Recall     : {metrics['recall']:.4f}",
            f"  F1         : {metrics['f1']:.4f}",
            f"  TP/TN/FP/FN: {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}",
        ]
    elif task == "pairwise_select":
        lines += [
            f"  N          : {metrics['n']}",
            f"  Accuracy   : {metrics['accuracy']:.4f}",
            f"  Correct    : {metrics['correct']}",
            f"  Wrong      : {metrics['wrong']}",
            f"  Unknown    : {metrics['unknown']}",
            f"  GT  A/B    : {metrics['gt_counts']['A']}/{metrics['gt_counts']['B']}",
            f"  Pred A/B   : {metrics['pred_counts']['A']}/{metrics['pred_counts']['B']}",
            f"  Confusion  : {metrics['confusion']}",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)
