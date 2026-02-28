"""
Scoring Dataset Builder  (paper §B.2.3)

For each regime, samples:
  - n_positive Positive factors
  - n_negative Negative factors
  - n_noise    Noise factors

Computes ground-truth 1–5 dimension scores within each category via percentile ranking.
Noise factors receive score=1 on all dimensions.

Output per regime → one JSONL file (each line = one test case):
  {
    "id":         "Overall__Positive_0012",
    "env":        "2021-2025 Overall",
    "factor":     "factor_name",
    "expression": "Div(...)",
    "signal":     "Positive" | "Negative" | "Noise",
    "scores": {
      "Performance": 4,
      "Stability":   3,
      "WinRate":     5,
      "Skewness":    2
    },
    "meta": { "mean_ic": ..., "mean_rankic": ..., ... }
  }

The combined file  `alphabench_testset.json` mirrors the format expected by
  benchmark/engine/evaluate/collect_results.py → collect_result_from_dir()
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd

from .config import BaseEvalConfig, RegimeConfig, ScoringConfig
from .stats import (
    assign_scores,
    classify_signal,
    compute_full_stats,
    factor_stats_meta,
    load_table,
)


# ---------------------------------------------------------------------------
# Balanced sampling within a regime
# ---------------------------------------------------------------------------

def _sample_balanced(
    scored: pd.DataFrame,         # has columns: signal, Performance, Stability, WinRate, Skewness
    n_positive: int,
    n_negative: int,
    n_noise: int,
    rng: random.Random,
) -> Dict[str, List[str]]:
    """
    Stratified sampling from each label bucket.
    Returns {"Positive": [...], "Negative": [...], "Noise": [...]}.
    """
    sampled: Dict[str, List[str]] = {}

    for label, target_n in [
        ("Positive", n_positive),
        ("Negative", n_negative),
        ("Noise", n_noise),
    ]:
        pool = scored[scored["signal"] == label].index.tolist()
        rng.shuffle(pool)
        sampled[label] = pool[:min(target_n, len(pool))]
        if len(sampled[label]) < target_n:
            print(f"[ScoringBuilder] WARNING: only {len(sampled[label])} {label} factors "
                  f"(requested {target_n}).")

    return sampled


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _make_record(
    factor_name: str,
    expression: str,
    env: str,
    signal: str,
    scored_row: dict,    # {Performance, Stability, WinRate, Skewness}
    meta: dict,
    idx: int,
) -> dict:
    return {
        "id": f"{env.replace(' ', '_')}_{signal}_{idx:04d}",
        "env": env,
        "factor": factor_name,
        "expression": expression,
        "signal": signal,
        "scores": {
            "Performance": int(scored_row.get("Performance", 1)),
            "Stability": int(scored_row.get("Stability", 1)),
            "WinRate": int(scored_row.get("WinRate", 1)),
            "Skewness": int(scored_row.get("Skewness", 1)),
        },
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_scoring_dataset(
    config: BaseEvalConfig,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Build scoring evaluation dataset for all regimes.

    Returns: {regime_name: path_to_jsonl}
    """
    os.makedirs(output_dir, exist_ok=True)

    ic_table = load_table(config.ic_table_path)
    rankic_table = (
        load_table(config.rankic_table_path)
        if os.path.exists(config.rankic_table_path)
        else None
    )

    with open(config.factor_pool_path, "r", encoding="utf-8") as f:
        factor_pool: Dict[str, str] = json.load(f)

    sc: ScoringConfig = config.scoring
    output_paths: Dict[str, str] = {}

    all_items: List[dict] = []   # for combined testset

    for regime in config.regimes:
        if verbose:
            print(f"[ScoringBuilder] Computing stats for regime: {regime.name} "
                  f"({regime.start} → {regime.end})")

        stats = compute_full_stats(ic_table, rankic_table, regime.start, regime.end)

        # Keep only factors present in factor pool
        available = [f for f in stats.index if f in factor_pool]
        stats = stats.loc[available]

        # Classify signals
        signal_labels = classify_signal(stats, noise_ic_threshold=sc.noise_ic_threshold)

        # Assign 1–5 scores (within each label category, percentile-based)
        scored = assign_scores(stats, signal_labels)

        # Balanced sample
        rng = random.Random(sc.seed + hash(regime.name))
        sampled = _sample_balanced(
            scored,
            n_positive=sc.n_positive,
            n_negative=sc.n_negative,
            n_noise=sc.n_noise,
            rng=rng,
        )

        # Build records
        regime_records: List[dict] = []
        for label, factors in sampled.items():
            for idx, fname in enumerate(factors):
                expr = factor_pool[fname]
                scored_row = scored.loc[fname].to_dict()
                meta = factor_stats_meta(fname, stats)
                rec = _make_record(
                    factor_name=fname,
                    expression=expr,
                    env=regime.name,
                    signal=label,
                    scored_row=scored_row,
                    meta=meta,
                    idx=idx,
                )
                regime_records.append(rec)

        # Shuffle within regime (don't leak label order to LLM)
        rng.shuffle(regime_records)

        # Write per-regime JSONL
        safe_name = (regime.name
                     .replace(" ", "_")
                     .replace("(", "").replace(")", "")
                     .replace(".", ""))
        out_path = os.path.join(output_dir, f"scoring_{safe_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in regime_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        output_paths[regime.name] = out_path
        all_items.extend(regime_records)

        if verbose:
            counts = {lbl: len(facs) for lbl, facs in sampled.items()}
            print(f"  Labels: {counts}  → {out_path}")

    # Combined file (mirrors alphabench_testset.json format)
    combined_path = os.path.join(output_dir, "alphabench_testset.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({"items": all_items}, f, indent=2, ensure_ascii=False)
    output_paths["__combined__"] = combined_path

    if verbose:
        print(f"[ScoringBuilder] Combined: {len(all_items)} items → {combined_path}")

    return output_paths
