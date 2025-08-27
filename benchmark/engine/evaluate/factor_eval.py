import ast
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import math
import re

def _dcg(relevances):
    """DCG with log2 discount and (2^rel - 1) gains."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)  # i starts at 0
    return dcg


def _build_rel_map(ground_truth):
    """
    ground_truth: list of factors sorted best->worst.
    Assign graded relevance: top item has highest rel.
    rel = len(gt) - rank (1-based) + 1  ==> {len(gt), ..., 1}
    """
    n = len(ground_truth)
    return {f: (n - i) for i, f in enumerate(ground_truth, start=1)}


def ndcg_at_k(pred_list, ground_truth, k):
    """
    pred_list: predicted ordered list of factors (strings)
    ground_truth: full sorted list best->worst (strings)
    k: cutoff
    """
    if k <= 0 or not ground_truth:
        return 0.0

    rel_map = _build_rel_map(ground_truth)
    k = min(k, len(pred_list))  # in case fewer than k predictions

    # DCG for the predicted top-k
    pred_rels = [rel_map.get(f, 0) for f in pred_list[:k]]
    dcg = _dcg(pred_rels)

    # IDCG for the ideal top-k (i.e., the first k in ground_truth)
    ideal_rels = [rel_map[gt] for gt in ground_truth[:k]]
    idcg = _dcg(ideal_rels)

    return 0.0 if idcg == 0.0 else dcg / idcg


def precision(predicted, groundtruth):
    """
    Compute precision given two lists of strings.
    
    Args:
        predicted (list[str]): predicted items
        groundtruth (list[str]): ground truth items
    
    Returns:
        float: precision score
    """
    predicted_set = set(predicted)
    groundtruth_set = set(groundtruth)

    true_positive = len(predicted_set & groundtruth_set)
    total_predicted = len(predicted_set)

    if total_predicted == 0:
        return 0.0
    return true_positive / total_predicted


def average_nested_dict(collected_results):
    averaged_results = {}
    for task_type, env_dict in collected_results.items():
        averaged_results[task_type] = {}
        for environment, values in env_dict.items():
            if len(values) > 0:
                avg_val = sum(values) / len(values)
            else:
                avg_val = 0.0
            averaged_results[task_type][environment] = avg_val
    return averaged_results


def evaluate_performance_ranking(test_cases, results):
    """
    Returns:
        precision_avg: {scenario: {environment: avg_precision}}
        ndcg_avg:      {scenario: {environment: avg_ndcg}}
    """
    precision_results = {}
    ndcg_results = {}

    for case, result in zip(test_cases, results):
        scenario = case["scenario"]
        env = case["environment"]

        # init nested structures
        precision_results.setdefault(scenario, {}).setdefault(env, [])
        ndcg_results.setdefault(scenario, {}).setdefault(env, [])

        factors = case["factors"]  # full list of factor names (indexable)
        ground_truth = case["ground_truth"]  # sorted best->worst factor names
        top_k = int(scenario.split("_")[-1])

        ground_truth = list(ground_truth.keys())
        # import pdb;pdb.set_trace()
        try:
            # Parse model output to get indices
            if isinstance(result, str):
                if result.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", result, flags=re.IGNORECASE)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                    parsed = cleaned.strip()
                    # import pdb;pdb.set_trace()
                    parsed = json.loads(parsed)
                else: 
                    parsed = ast.literal_eval(result)
            elif isinstance(result, dict):
                parsed = result
            else:
                parsed = {"results": result}

            selected_idx = parsed.get("results", parsed)
            if not isinstance(selected_idx, (list, tuple)):
                raise ValueError("results must be a list/tuple of indices")

            # Convert 1-based indices -> factor names; drop invalid & duplicates (keep order)
            seen = set()
            selected_factors = []
            for idx in selected_idx:
                try:
                    i0 = int(idx) - 1
                    if 0 <= i0 < len(factors):
                        f = factors[i0]
                        if f not in seen:
                            seen.add(f)
                            selected_factors.append(f)
                except Exception:
                    continue

            # Precision@k against top_k of ground-truth
            prec = precision(selected_factors[:top_k], ground_truth[:top_k])

            # NDCG@k using full graded ground-truth ranking
            ndcg = ndcg_at_k(selected_factors, ground_truth, top_k)

            precision_results[scenario][env].append(prec)
            ndcg_results[scenario][env].append(ndcg)

        except Exception as e:
            print(f"Error processing case {scenario} - {env}: {e}")
            precision_results[scenario][env].append(0.0)
            ndcg_results[scenario][env].append(0.0)

    # Average each nested dict separately
    precision_avg = average_nested_dict(precision_results)
    ndcg_avg = average_nested_dict(ndcg_results)
    return precision_avg, ndcg_avg


# =========================================================
# Parsing helpers
# =========================================================


def _parse_pred_result(x):
    """
    Parse a prediction that may be:
      - dict
      - JSON string
      - Python literal string (e.g., produced by str(dict))
    Returns a dict with at least keys: 'signal' and 'scores', or None if parsing fails.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            if x.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", x, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                parsed = cleaned.strip()
                # import pdb;pdb.set_trace()
                parsed = json.loads(parsed)
            else: 
                parsed = ast.literal_eval(x)
            return parsed
        except Exception:
            return None
    # else:
    #     s = x.strip()
    #     # Try JSON first
    #     try:
    #         return json.loads(s)
    #     except Exception:
    #         pass
    #     # Fallback to Python literal
    #     try:
    #         return ast.literal_eval(s)
    #     except Exception:
    #         return None
    return None


def _clamp_score(v):
    """
    Clamp a numeric value into integer 1..5.
    Invalid or missing values -> np.nan (so they are excluded from averages).
    """
    try:
        iv = int(round(float(v)))
        return int(min(5, max(1, iv)))
    except Exception:
        return np.nan


# =========================================================
# Classification metrics (no sklearn)
# =========================================================


def _confusion_counts(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> np.ndarray:
    """
    Build confusion matrix counts C (len(labels) x len(labels)),
    where rows = true, cols = pred.
    """
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t not in label_to_idx or p not in label_to_idx:
            # Skip invalid labels; you may also choose to bucket them into a special class
            continue
        C[label_to_idx[t], label_to_idx[p]] += 1
    return C


def _precision_recall_f1_acc(
    C: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Given confusion matrix C (rows=true, cols=pred),
    returns:
      precision (per class),
      recall    (per class),
      f1        (per class),
      accuracy  (overall),
      support   (per class, i.e., row sums)
    """
    eps = 1e-12
    tp = np.diag(C).astype(float)
    pred_sum = C.sum(axis=0).astype(float)  # per predicted class
    true_sum = C.sum(axis=1).astype(float)  # per true class
    total = C.sum().astype(float)

    precision = tp / np.maximum(pred_sum, eps)
    recall = tp / np.maximum(true_sum, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    accuracy = float(tp.sum() / np.maximum(total, eps))
    support = true_sum
    return precision, recall, f1, accuracy, support


def _classification_report_df(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> pd.DataFrame:
    """
    Return a DataFrame with per-class Precision/Recall/F1/Support and an overall row including Accuracy.
    """
    C = _confusion_counts(y_true, y_pred, labels)
    precision, recall, f1, accuracy, support = _precision_recall_f1_acc(C)

    df = pd.DataFrame(
        {
            "Class": labels,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )
    overall = pd.DataFrame(
        [
            {
                "Class": "Overall",
                "Precision": np.mean(precision),
                "Recall": np.mean(recall),
                "F1": np.mean(f1),
                "Support": int(np.sum(support)),
                "Accuracy": accuracy,
            }
        ]
    )
    # Align columns (Accuracy only for Overall row)
    df["Accuracy"] = np.nan
    df_overall = pd.concat([df, overall], ignore_index=True)
    return df_overall


# =========================================================
# Core evaluation: MAE on 1..5 scores + 3-class classification metrics
# =========================================================


def evaluate_performance_scoring(
    test_cases: List[Dict],
    results: List[Dict],
    dims: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    env_parser: Optional[str] = "split1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate zero-shot factor scoring predictions.

    Parameters
    ----------
    test_cases : list of dict
        Each case must include:
          - 'env'    : environment string, e.g., "CSI300 Bear (2023-2024)"
          - 'signal' : ground-truth signal in {"Positive","Negative","Noise"}
          - 'scores' : dict with 1..5 per dimension
    results : list of (dict|str)
        Model predictions aligned with test_cases. Each must contain:
          - 'signal' : predicted class
          - 'scores' : dict with predicted 1..5 per dimension
    dims : list[str]
        Scoring dimensions to evaluate. Default: ["Performance","Stability","WinRate","Skewness"]
    labels : list[str]
        Classification labels / order. Default: ["Positive","Negative","Noise"]
    env_parser : str
        How to derive market_environment from 'env':
          - "split1": env.split(" ")[1] (your earlier logic)
          - "full": use env as-is
          - "first": take first token env.split(" ")[0]
        Adjust as needed to match your env naming.

    Returns
    -------
    env_mae_summary : pd.DataFrame
        Aggregated MAE by environment with columns:
            [N, Acc_Signal, MAE_All, MAE_<dim> ...]
    env_classification_report : pd.DataFrame
        Per-environment classification report (per class + Overall rows stacked).
    per_case : pd.DataFrame
        Row-wise details for debugging.
    """
    if dims is None:
        dims = ["Performance", "Stability", "WinRate", "Skewness"]
    if labels is None:
        labels = ["Positive", "Negative", "Noise"]

    # Collect row-wise records
    row_records = []

    for case, pred_raw in zip(test_cases, results):
        env_full = str(case.get("env", ""))
        if env_parser == "split1":
            try:
                market_environment = env_full.split(" ")[1]
            except Exception:
                market_environment = env_full
        elif env_parser == "first":
            try:
                market_environment = env_full.split(" ")[0]
            except Exception:
                market_environment = env_full
        elif env_parser == "full":
            market_environment = env_full
        else:
            # default fallback
            market_environment = env_full

        gt_signal = case.get("signal", None)
        gt_scores = case.get("scores", {})

        pred = _parse_pred_result(pred_raw)
        try:
            if isinstance(pred, List):
                pred = pred[0]
            try:
                if pred is None:
                    pred_signal = None
                    pred_scores = {}
                else:
                    pred_signal = pred.get("signal", None)
                    pred_scores = pred.get("scores", {})
            except Exception:
                pred_signal = None
                pred_scores = {}
        except Exception:
            pred_signal = None
            pred_scores = {}
        # Signal correctness (binary 0/1)
        signal_correct = float(pred_signal == gt_signal)

        # Dimension-wise MAE and overall MAE
        per_dim_abs = []
        mae_dims = {}
        for d in dims:
            gt = _clamp_score(gt_scores.get(d))
            pv = _clamp_score(pred_scores.get(d))
            if np.isnan(gt) or np.isnan(pv):
                mae_dims[d] = np.nan
            else:
                e = abs(pv - gt)
                mae_dims[d] = float(e)
                per_dim_abs.append(e)
        mae_all = float(np.mean(per_dim_abs)) if len(per_dim_abs) > 0 else np.nan

        row = {
            "env_full": env_full,
            "market_environment": market_environment,
            "gt_signal": gt_signal,
            "pred_signal": pred_signal,
            "signal_correct": signal_correct,
            "MAE_All": mae_all,
        }
        for d in dims:
            row[f"MAE_{d}"] = mae_dims.get(d, np.nan)
        row_records.append(row)

    per_case = pd.DataFrame(row_records)

    # =========================
    # MAE summary by environment
    # =========================
    agg = {"signal_correct": "mean", "MAE_All": "mean"}
    for d in dims:
        agg[f"MAE_{d}"] = "mean"

    env_mae_summary = (
        per_case.groupby("market_environment", dropna=False)
        .agg(agg)
        .rename(columns={"signal_correct": "Acc_Signal"})
    )

    env_counts = per_case.groupby("market_environment", dropna=False).size().rename("N")
    env_mae_summary = env_mae_summary.join(env_counts)

    # Reorder columns
    mae_cols = ["N", "Acc_Signal", "MAE_All"] + [f"MAE_{d}" for d in dims]
    env_mae_summary = env_mae_summary[mae_cols].sort_values(
        by=["Acc_Signal", "MAE_All"], ascending=[False, True]
    )

    # ==============================================
    # Classification report (per environment + overall)
    # ==============================================
    reports = []
    # Per environment
    for env_name, group in per_case.groupby("market_environment", dropna=False):
        y_true = group["gt_signal"].tolist()
        y_pred = group["pred_signal"].tolist()
        rep = _classification_report_df(y_true, y_pred, labels)
        rep.insert(0, "Environment", env_name)
        reports.append(rep)

    # Overall (all environments combined)
    y_true_all = per_case["gt_signal"].tolist()
    y_pred_all = per_case["pred_signal"].tolist()
    overall_rep = _classification_report_df(y_true_all, y_pred_all, labels)
    overall_rep.insert(0, "Environment", "ALL")
    reports.append(overall_rep)

    env_classification_report = pd.concat(reports, ignore_index=True)

    return env_mae_summary, env_classification_report, per_case
