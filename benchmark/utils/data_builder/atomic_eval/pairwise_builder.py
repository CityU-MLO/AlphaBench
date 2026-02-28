"""
Task 1 Dataset Builder: Select the Better Factor (Pairwise Comparison)

Given a backtesting IC/RankIC table and factor pool, produces JSONL datasets
where each sample is:
    {
      "A": "<expr_A>",
      "B": "<expr_B>",
      "ground_truth": "A" | "B",   # which factor is the better (signal) one
      "meta_A": {...},
      "meta_B": {...},
      ...
    }

Pairs are always (signal_factor vs noise_factor) with A/B assignment randomized
so that ground_truth is uniformly distributed between A and B.

Two build modes:
  A) Fresh split  — sample noise/signal pairs from the full factor pool.
  B) Refer split  — copy the pairing structure from an existing split dir
                    (factor identity transferred; only re-label with new IC stats).
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

def _build_pair_record(
    signal_name: str,
    noise_name: str,
    signal_expr: str,
    noise_expr: str,
    market: str,
    start_date: str,
    end_date: str,
    stats: pd.DataFrame,
    rng: random.Random,
    pair_id: str,
) -> dict:
    """
    Randomly assign signal/noise to slots A and B; record ground_truth accordingly.
    """
    if rng.random() < 0.5:
        expr_a, expr_b = signal_expr, noise_expr
        name_a, name_b = signal_name, noise_name
        ground_truth = "A"
    else:
        expr_a, expr_b = noise_expr, signal_expr
        name_a, name_b = noise_name, signal_name
        ground_truth = "B"

    return {
        "id": pair_id,
        "A": expr_a,
        "B": expr_b,
        "ground_truth": ground_truth,
        "factor_name_A": name_a,
        "factor_name_B": name_b,
        "market": market.upper(),
        "window": {"start": start_date, "end": end_date},
        "meta_A": factor_meta(name_a, stats),
        "meta_B": factor_meta(name_b, stats),
    }


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _pair_pools(
    noise_ranked: List[str],
    signal_ranked: List[str],
    n: int,
) -> List[Tuple[str, str]]:
    """
    Create up to n (signal, noise) pairs deterministically from ranked lists.
    Each factor appears at most once per pair list.
    """
    usable = min(n, len(noise_ranked), len(signal_ranked))
    return list(zip(signal_ranked[:usable], noise_ranked[:usable]))


def _split_pairs(
    pairs: List[Tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Tuple[str, str]]]:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val] if n_val > 0 else [],
        "test": shuffled[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Refer-mode: load existing split pairs (factor names)
# ---------------------------------------------------------------------------

def _load_refer_pairs(refer_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Read factor name pairs from existing split JSONL files.
    Returns {"train": [(signal_name, noise_name), ...], ...}.

    We reconstruct signal/noise identity from ground_truth + factor_name_A/B.
    """
    splits: Dict[str, List[Tuple[str, str]]] = {}
    for split in ("train", "val", "test"):
        fpath = os.path.join(refer_dir, f"{split}.jsonl")
        if not os.path.exists(fpath):
            continue
        pairs = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    gt = obj.get("ground_truth", "A")
                    name_a = obj.get("factor_name_A", "")
                    name_b = obj.get("factor_name_B", "")
                    # Recover which was the signal
                    if gt == "A":
                        signal_name, noise_name = name_a, name_b
                    else:
                        signal_name, noise_name = name_b, name_a
                    if signal_name and noise_name:
                        pairs.append((signal_name, noise_name))
                except Exception:
                    pass
        if pairs:
            splits[split] = pairs
    if not splits:
        raise FileNotFoundError(f"No valid JSONL split files found in: {refer_dir}")
    return splits


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_pairwise_dataset(
    config: AtomicEvalConfig,
    factor_pool: Dict[str, str],
    ic_table: pd.DataFrame,
    rankic_table: Optional[pd.DataFrame],
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Build pairwise "select the better" dataset (Task 1).

    Args:
        config:       AtomicEvalConfig with meta, noise_threshold, split/refer.
        factor_pool:  Dict mapping factor_name -> expression string.
        ic_table:     DataFrame (datetime-index, factor columns) of daily IC values.
        rankic_table: DataFrame (datetime-index, factor columns) of daily RankIC values.
        output_dir:   Directory where train.jsonl / val.jsonl / test.jsonl are written.
        verbose:      Print progress info.

    Returns:
        Dict {"train": path, "val": path, "test": path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(config.seed)

    meta = config.meta
    nt: NoiseThreshold = config.noise_threshold or NoiseThreshold()

    # ---- Compute stats ----
    if verbose:
        print(f"[PairwiseBuilder] Computing stats for {meta.market} "
              f"from {meta.start_date} to {meta.end_date} ...")

    stats = compute_factor_stats(
        df_ic=ic_table,
        df_rankic=rankic_table,
        start_date=meta.start_date,
        end_date=meta.end_date,
    )

    available = [f for f in stats.index if f in factor_pool]
    stats = stats.loc[available]

    if verbose:
        print(f"[PairwiseBuilder] {len(available)} factors available.")

    # ---- Label each factor ----
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

    noise_ranked = [f for f in rank_by_noise_score(stats) if noise_labels.get(f) is True]
    signal_ranked = [f for f in rank_by_signal_score(stats) if noise_labels.get(f) is False]

    if verbose:
        print(f"[PairwiseBuilder] {len(noise_ranked)} noise, {len(signal_ranked)} signal factors.")

    # ----------------------------------------------------------------
    # Mode A: fresh split
    # ----------------------------------------------------------------
    if config.split is not None:
        sc = config.split
        max_pairs = min(len(noise_ranked), len(signal_ranked))
        n_pairs = min(sc.sample_n, max_pairs)
        if verbose:
            print(f"[PairwiseBuilder] Building {n_pairs} pairs (cap: {max_pairs})...")

        all_pairs = _pair_pools(noise_ranked, signal_ranked, n_pairs)
        splits_data = _split_pairs(
            all_pairs, sc.train_ratio, sc.val_ratio, sc.test_ratio, rng
        )

    # ----------------------------------------------------------------
    # Mode B: refer split
    # ----------------------------------------------------------------
    else:
        refer_pairs = _load_refer_pairs(config.refer)
        splits_data = {}

        for split_name, pairs in refer_pairs.items():
            matched = []
            for signal_name, noise_name in pairs:
                # Both factors must be in pool and stats
                if (signal_name in factor_pool and noise_name in factor_pool
                        and signal_name in stats.index and noise_name in stats.index):
                    matched.append((signal_name, noise_name))
                else:
                    if verbose:
                        missing = [n for n in (signal_name, noise_name)
                                   if n not in factor_pool or n not in stats.index]
                        print(f"[PairwiseBuilder] Skipping pair: missing {missing}")
            splits_data[split_name] = matched
            if verbose:
                print(f"[PairwiseBuilder] Refer {split_name}: "
                      f"{len(matched)}/{len(pairs)} pairs matched.")

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
        for i, (signal_name, noise_name) in enumerate(pairs):
            pair_id = f"{meta.market.upper()}_{split_name.upper()}_{i:04d}"
            rec = _build_pair_record(
                signal_name=signal_name,
                noise_name=noise_name,
                signal_expr=factor_pool[signal_name],
                noise_expr=factor_pool[noise_name],
                market=meta.market,
                start_date=meta.start_date,
                end_date=meta.end_date,
                stats=stats,
                rng=rng,
                pair_id=pair_id,
            )
            records.append(rec)

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        output_paths[split_name] = out_path

        if verbose:
            gt_counts = {"A": sum(1 for r in records if r["ground_truth"] == "A"),
                         "B": sum(1 for r in records if r["ground_truth"] == "B")}
            print(f"[PairwiseBuilder] Wrote {len(records)} {split_name} pairs → {out_path} "
                  f"(GT: {gt_counts})")

    # Write manifest
    manifest = {
        "task": "pairwise_select",
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
