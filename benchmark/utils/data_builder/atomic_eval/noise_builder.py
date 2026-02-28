"""
Task 2 Dataset Builder: Binary Noise Classification

Given a backtesting IC/RankIC table and a factor pool, produces JSONL datasets
where each sample is:
    {"expression": "<expr>", "ground_truth": "noise" | "signal", "meta": {...}, ...}

Two build modes:
  A) Fresh split  — sample from the full factor pool and split into train/val/test.
  B) Refer split  — read factor names from an existing split dir (cross-market transfer).
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import AtomicEvalConfig, NoiseThreshold
from .stats import (
    compute_factor_stats,
    factor_meta,
    is_noise,
    load_ic_table,
    rank_by_noise_score,
    rank_by_signal_score,
)


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _build_record(
    factor_name: str,
    expression: str,
    label: str,   # "noise" | "signal"
    market: str,
    start_date: str,
    end_date: str,
    stats: pd.DataFrame,
) -> dict:
    return {
        "expression": expression,
        "ground_truth": label,
        "factor_name": factor_name,
        "market": market.upper(),
        "window": {"start": start_date, "end": end_date},
        "meta": factor_meta(factor_name, stats),
    }


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _stratified_sample(
    noise_pool: List[str],
    signal_pool: List[str],
    n: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """
    Pick up to n samples balanced 50/50 noise/signal.
    Returns list of (factor_name, label) tuples.
    """
    half = n // 2
    n_noise = min(half, len(noise_pool))
    n_signal = min(half, len(signal_pool))

    # Fill shortfall from the other class
    if n_noise < half:
        n_signal = min(n - n_noise, len(signal_pool))
    elif n_signal < half:
        n_noise = min(n - n_signal, len(noise_pool))

    noise_sel = noise_pool[:n_noise]
    signal_sel = signal_pool[:n_signal]

    pairs = [(f, "noise") for f in noise_sel] + [(f, "signal") for f in signal_sel]
    rng.shuffle(pairs)
    return pairs


def _split_indices(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    n_train = round(total * train_ratio)
    n_val = round(total * val_ratio)
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


# ---------------------------------------------------------------------------
# Refer-mode: load existing split factor names
# ---------------------------------------------------------------------------

def _load_refer_split(
    refer_dir: str,
) -> Dict[str, List[str]]:
    """
    Load factor names from an existing split directory produced by this builder.
    Returns {"train": [...], "val": [...], "test": [...]} where each list
    contains factor names from the corresponding JSONL file.
    """
    splits: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        fpath = os.path.join(refer_dir, f"{split}.jsonl")
        if not os.path.exists(fpath):
            continue
        names = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    name = obj.get("factor_name")
                    if name:
                        names.append(name)
                except Exception:
                    pass
        splits[split] = names
    if not splits:
        raise FileNotFoundError(f"No valid JSONL split files found in: {refer_dir}")
    return splits


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_noise_dataset(
    config: AtomicEvalConfig,
    factor_pool: Dict[str, str],          # {factor_name: expression}
    ic_table: pd.DataFrame,               # datetime-indexed IC table
    rankic_table: Optional[pd.DataFrame], # datetime-indexed RankIC table (may be None)
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Build binary noise classification dataset (Task 2).

    Args:
        config:       AtomicEvalConfig with meta, noise_threshold, split/refer.
        factor_pool:  Dict mapping factor_name -> expression string.
        ic_table:     DataFrame (datetime-index, factor columns) of daily IC values.
        rankic_table: DataFrame (datetime-index, factor columns) of daily RankIC values.
        output_dir:   Directory where train.jsonl / val.jsonl / test.jsonl are written.
        verbose:      Print progress info.

    Returns:
        Dict {"train": path, "val": path, "test": path} (val path is empty string if no val set).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(config.seed)

    meta = config.meta
    nt: NoiseThreshold = config.noise_threshold or NoiseThreshold()

    # ---- Compute stats over the configured window ----
    if verbose:
        print(f"[NoiseBuilder] Computing stats for {meta.market} "
              f"from {meta.start_date} to {meta.end_date} ...")

    stats = compute_factor_stats(
        df_ic=ic_table,
        df_rankic=rankic_table,
        start_date=meta.start_date,
        end_date=meta.end_date,
    )

    # Keep only factors present in both factor_pool and stats
    available = [f for f in stats.index if f in factor_pool]
    stats = stats.loc[available]

    if verbose:
        print(f"[NoiseBuilder] {len(available)} factors available after intersection with pool.")

    # Label each factor
    noise_labels: Dict[str, bool] = {}
    for f in available:
        label = is_noise(
            f, stats,
            ic_threshold=nt.ic,
            rankic_threshold=nt.rankic,
            use_abs=nt.abs,
            condition=nt.condition,
        )
        if label is not None:
            noise_labels[f] = label

    # Rank by noise / signal strength for deterministic ordered selection
    noise_ranked = [f for f in rank_by_noise_score(stats) if noise_labels.get(f) is True]
    signal_ranked = [f for f in rank_by_signal_score(stats) if noise_labels.get(f) is False]

    if verbose:
        print(f"[NoiseBuilder] Labelled: {len(noise_ranked)} noise, {len(signal_ranked)} signal factors.")

    # ----------------------------------------------------------------
    # Mode A: fresh split
    # ----------------------------------------------------------------
    if config.split is not None:
        sc = config.split
        pool_size = min(len(noise_ranked), len(signal_ranked)) * 2
        sample_n = min(sc.sample_n, pool_size)

        if verbose:
            print(f"[NoiseBuilder] Sampling {sample_n} factors (cap: {pool_size})...")

        sampled = _stratified_sample(noise_ranked, signal_ranked, sample_n, rng)

        n_train, n_val, n_test = _split_indices(
            len(sampled), sc.train_ratio, sc.val_ratio, sc.test_ratio
        )

        splits_data = {
            "train": sampled[:n_train],
            "val": sampled[n_train:n_train + n_val] if n_val > 0 else [],
            "test": sampled[n_train + n_val:],
        }

    # ----------------------------------------------------------------
    # Mode B: refer split (cross-market transfer)
    # ----------------------------------------------------------------
    else:
        refer_splits = _load_refer_split(config.refer)
        splits_data = {}

        for split_name, refer_names in refer_splits.items():
            matched = []
            for name in refer_names:
                if name in factor_pool and name in stats.index:
                    label = is_noise(
                        name, stats,
                        ic_threshold=nt.ic,
                        rankic_threshold=nt.rankic,
                        use_abs=nt.abs,
                        condition=nt.condition,
                    )
                    if label is not None:
                        matched.append((name, "noise" if label else "signal"))
                    else:
                        # No stats — skip with a warning
                        if verbose:
                            print(f"[NoiseBuilder] Skipping {name}: no stats available.")
            splits_data[split_name] = matched
            if verbose:
                print(f"[NoiseBuilder] Refer {split_name}: {len(matched)}/{len(refer_names)} matched.")

    # ----------------------------------------------------------------
    # Write JSONL files
    # ----------------------------------------------------------------
    output_paths: Dict[str, str] = {}

    for split_name, pairs in splits_data.items():
        if not pairs:
            output_paths[split_name] = ""
            continue

        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        records = []
        for factor_name, label in pairs:
            expr = factor_pool[factor_name]
            records.append(
                _build_record(
                    factor_name=factor_name,
                    expression=expr,
                    label=label,
                    market=meta.market,
                    start_date=meta.start_date,
                    end_date=meta.end_date,
                    stats=stats,
                )
            )

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        output_paths[split_name] = out_path
        if verbose:
            label_counts = {}
            for _, lbl in pairs:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            print(f"[NoiseBuilder] Wrote {len(records)} {split_name} records → {out_path} "
                  f"({label_counts})")

    # Write a manifest for traceability
    manifest = {
        "task": "binary_noise",
        "market": meta.market,
        "window": {"start": meta.start_date, "end": meta.end_date},
        "noise_threshold": {
            "ic": nt.ic, "rankic": nt.rankic, "abs": nt.abs, "condition": nt.condition
        },
        "mode": "split" if config.split else "refer",
        "splits": {k: os.path.basename(v) for k, v in output_paths.items() if v},
        "counts": {k: sum(1 for _ in open(v) if _.strip()) for k, v in output_paths.items() if v},
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return output_paths
