"""
Ranking Dataset Builder  (paper §B.2.2)

For each (regime, N-pick-K setting), generates `instances_per_setting` test cases.
Each case:
  - K "good" factors sampled from the good pool (|IC|>threshold OR |RankIC|>threshold)
  - N-K "fill" factors sampled to match the empirical IC distribution of the full pool

Output per regime per setting → one JSON file:
  [
    {
      "id": "...",
      "scenario": "10_pick_3",
      "environment": "2021-2025 Overall",
      "factors": ["f1", "f2", ...],   # N factor names (ordered for LLM prompt)
      "factor_expressions": {"f1": "expr", ...},
      "ground_truth": {"f1": 5, "f2": 3, ...},  # factor_name -> rank score (best first)
      "meta": {"f1": {...}, ...},
    },
    ...
  ]

The `ground_truth` key mirrors the existing code in factor_eval.py which calls
  evaluate_performance_ranking(test_cases, results)
where ground_truth is a dict {factor_name: rank_score} sorted best->worst.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BaseEvalConfig, RankingConfig, RankingSettingConfig, RegimeConfig
from .stats import (
    classify_signal,
    compute_full_stats,
    factor_stats_meta,
    load_table,
)


# ---------------------------------------------------------------------------
# Good / fill pool selection
# ---------------------------------------------------------------------------

def _is_good(
    mean_ic: Optional[float],
    mean_rankic: Optional[float],
    ic_th: float,
    rankic_th: float,
) -> bool:
    """Paper: |IC| > ic_threshold OR |RankIC| > rankic_threshold."""
    ic_ok = (mean_ic is not None) and (abs(mean_ic) > ic_th)
    rk_ok = (mean_rankic is not None) and (abs(mean_rankic) > rankic_th)
    return ic_ok or rk_ok


def _partition_pools(
    stats: pd.DataFrame,
    factor_pool: Dict[str, str],
    ic_th: float,
    rankic_th: float,
) -> Tuple[List[str], List[str]]:
    """
    Split factors into (good_pool, fill_pool).
    Only factors that appear in both stats and factor_pool are considered.
    """
    available = [f for f in stats.index if f in factor_pool]
    good, fill = [], []
    for f in available:
        row = stats.loc[f]
        mic = row.get("mean_ic")
        mrk = row.get("mean_rankic")
        try:
            mic = float(mic) if mic is not None and not np.isnan(mic) else None
        except Exception:
            mic = None
        try:
            mrk = float(mrk) if mrk is not None and not np.isnan(mrk) else None
        except Exception:
            mrk = None
        if _is_good(mic, mrk, ic_th, rankic_th):
            good.append(f)
        else:
            fill.append(f)
    return good, fill


def _ground_truth_ranking(good_factors: List[str], stats: pd.DataFrame) -> Dict[str, int]:
    """
    Build {factor_name: rank_score} for the K good factors, ranked by |ICIR|.
    Best factor gets score K, worst gets score 1.
    Mirrors the existing factor_eval.py convention (best->worst dict).
    """
    sub = stats.loc[[f for f in good_factors if f in stats.index], "icir"].abs()
    ranked = sub.sort_values(ascending=False)
    k = len(ranked)
    return {f: k - i for i, f in enumerate(ranked.index)}


# ---------------------------------------------------------------------------
# Test case generation for one (regime, setting)
# ---------------------------------------------------------------------------

def _build_cases_for_setting(
    regime: RegimeConfig,
    setting: RankingSettingConfig,
    stats: pd.DataFrame,
    factor_pool: Dict[str, str],
    rc: RankingConfig,
    n_instances: int,
    rng: random.Random,
) -> List[dict]:
    n, k = setting.n, setting.k
    fill_count = n - k

    good_pool, fill_pool = _partition_pools(
        stats, factor_pool,
        rc.good_ic_threshold,
        rc.good_rankic_threshold,
    )

    if len(good_pool) < k:
        print(f"[RankingBuilder] WARNING: {regime.name} setting {n}_pick_{k}: "
              f"only {len(good_pool)} good factors (need {k}). Skipping.")
        return []

    if len(fill_pool) < fill_count:
        print(f"[RankingBuilder] WARNING: {regime.name} setting {n}_pick_{k}: "
              f"only {len(fill_pool)} fill factors (need {fill_count}). Capping instances.")
        n_instances = min(n_instances, len(fill_pool) // max(fill_count, 1))

    scenario_tag = f"{n}_pick_{k}"
    cases = []

    for inst_idx in range(n_instances):
        # Sample K good factors
        k_good = rng.sample(good_pool, k)
        # Sample N-K fill factors (without replacement within instance)
        k_fill = rng.sample(fill_pool, min(fill_count, len(fill_pool)))
        if len(k_fill) < fill_count:
            # Not enough fill — skip
            continue

        all_factors = k_good + k_fill
        rng.shuffle(all_factors)

        gt = _ground_truth_ranking(k_good, stats)

        case = {
            "id": f"{regime.name}__{scenario_tag}_{inst_idx:04d}",
            "scenario": scenario_tag,
            "environment": regime.name,
            "factors": all_factors,
            "factor_expressions": {f: factor_pool[f] for f in all_factors},
            "ground_truth": gt,
            "meta": {f: factor_stats_meta(f, stats) for f in all_factors},
        }
        cases.append(case)

    return cases


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_ranking_dataset(
    config: BaseEvalConfig,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Build ranking evaluation dataset for all regimes and settings.

    Returns: {regime_setting_key: path_to_json}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load tables
    ic_table = load_table(config.ic_table_path)
    rankic_table = (
        load_table(config.rankic_table_path)
        if os.path.exists(config.rankic_table_path)
        else None
    )

    # Load factor pool
    with open(config.factor_pool_path, "r", encoding="utf-8") as f:
        factor_pool: Dict[str, str] = json.load(f)

    rc = config.ranking
    output_paths: Dict[str, str] = {}

    # Collect all test cases across regimes (also write combined file)
    all_cases: List[dict] = []

    for regime in config.regimes:
        if verbose:
            print(f"[RankingBuilder] Computing stats for regime: {regime.name} "
                  f"({regime.start} → {regime.end})")

        stats = compute_full_stats(ic_table, rankic_table, regime.start, regime.end)

        regime_cases: List[dict] = []
        for setting in rc.settings:
            rng = random.Random(rc.seed + hash(regime.name + str(setting.n)))
            cases = _build_cases_for_setting(
                regime, setting, stats, factor_pool, rc,
                n_instances=rc.instances_per_setting,
                rng=rng,
            )
            if verbose:
                print(f"  Setting {setting.n}_pick_{setting.k}: {len(cases)} instances")
            regime_cases.extend(cases)

        # Write per-regime file
        safe_name = regime.name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        out_path = os.path.join(output_dir, f"ranking_{safe_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(regime_cases, f, indent=2, ensure_ascii=False)
        output_paths[regime.name] = out_path
        all_cases.extend(regime_cases)

        if verbose:
            print(f"  → Wrote {len(regime_cases)} cases → {out_path}")

    # Write combined file (compatible with existing benchmark_main.py)
    combined_path = os.path.join(output_dir, "all_env_scenarios.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    output_paths["__combined__"] = combined_path

    if verbose:
        print(f"[RankingBuilder] Combined: {len(all_cases)} cases → {combined_path}")

    return output_paths
