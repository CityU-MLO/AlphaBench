"""
Build Binary Noise Classification Dataset (T4 - Task 2)

Each record asks the LLM to classify whether a factor expression is "signal" or "noise".

Reads:
  benchmark/data/evaluate/raw/ic_table_csi300.csv
  benchmark/data/evaluate/raw/rankic_table_csi300.csv
  benchmark/data/evaluate/pool/factor_pool.json

Writes to output_dir/noise/:
  train.jsonl       ← training split
  val.jsonl         ← validation split
  test.jsonl        ← test split
  manifest.json     ← split sizes & config summary

Record format:
  {
    "expression":   "Div($close, Add(Mean($close, 15), 1e-12))",
    "ground_truth": "noise" | "signal",
    "factor_name":  "close_to_rolling_mean_...",
    "market":       "CSI300",
    "window":       {"start": "2021-01-01", "end": "2025-06-30"},
    "meta":         {"mean_ic": 0.031, "mean_rankic": 0.028, ...}
  }

Usage (default config):
  python example/01_build_datasets/build_atomic_noise_dataset.py

Usage (custom config):
  python example/01_build_datasets/build_atomic_noise_dataset.py \
      --config example/01_build_datasets/configs/atomic_noise_csi300.yaml \
      --output_dir benchmark/data/evaluate/atomic/noise_csi300

Usage (cross-market transfer from existing CSI300 split):
  python example/01_build_datasets/build_atomic_noise_dataset.py \
      --ic_table     benchmark/data/evaluate/raw/ic_table_sp500.csv \
      --rankic_table benchmark/data/evaluate/raw/rankic_table_sp500.csv \
      --market       SP500 \
      --refer        benchmark/data/evaluate/atomic/noise_csi300 \
      --output_dir   benchmark/data/evaluate/atomic/noise_sp500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.utils.data_builder.atomic_eval import (
    AtomicEvalConfig,
    build_noise_dataset,
    load_ic_table,
)


_DEFAULT_IC_TABLE     = "benchmark/data/evaluate/raw/ic_table_csi300.csv"
_DEFAULT_RANKIC_TABLE = "benchmark/data/evaluate/raw/rankic_table_csi300.csv"
_DEFAULT_POOL         = "benchmark/data/evaluate/pool/factor_pool.json"
_DEFAULT_OUTPUT       = "benchmark/data/evaluate/atomic/noise_csi300"
_DEFAULT_CONFIG       = "example/01_build_datasets/configs/atomic_noise_csi300.yaml"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Build binary noise classification dataset for T4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None,
                   help=f"Path to YAML config. Default: {_DEFAULT_CONFIG}")
    p.add_argument("--ic_table", default=None,
                   help=f"IC table CSV path. Default: {_DEFAULT_IC_TABLE}")
    p.add_argument("--rankic_table", default=None,
                   help=f"RankIC table CSV path. Default: {_DEFAULT_RANKIC_TABLE}")
    p.add_argument("--factor_pool", default=None,
                   help=f"factor_pool.json path. Default: {_DEFAULT_POOL}")
    p.add_argument("--market", default=None,
                   help="Market name override (e.g. CSI300, SP500).")
    p.add_argument("--refer", default=None,
                   help="Path to existing split dir (cross-market transfer mode). "
                        "If given, copies the factor index from that split.")
    p.add_argument("--sample_n", type=int, default=None,
                   help="Override: number of factors to sample (fresh split mode only).")
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT,
                   help=f"Output directory. Default: {_DEFAULT_OUTPUT}")
    return p.parse_args()


def main():
    args = _parse_args()

    # ---- Load config ----
    if args.config and os.path.exists(args.config):
        print(f"[Config] Loading from YAML: {args.config}")
        cfg = AtomicEvalConfig.from_yaml(args.config)
    elif args.refer:
        # Cross-market mode: require --ic_table / --rankic_table
        cfg = AtomicEvalConfig.from_dict({
            "meta": {
                "market": args.market or "SP500",
                "start_date": "2021-01-01",
                "end_date":   "2025-06-30",
            },
            "noise_threshold": {
                "ic": 0.005, "rankic": 0.025,
                "abs": True, "condition": "and",
            },
            "refer": args.refer,
        })
    else:
        print("[Config] Using default paper settings (CSI300, fresh split)")
        cfg = AtomicEvalConfig.from_dict({
            "meta": {
                "market": "CSI300",
                "start_date": "2021-01-01",
                "end_date":   "2025-06-30",
            },
            "noise_threshold": {
                "ic": 0.005, "rankic": 0.025,
                "abs": True, "condition": "and",
            },
            "split": {
                "sample_n": 300,
                "train_ratio": 0.6,
                "val_ratio":   0.1,
                "test_ratio":  0.3,
            },
        })

    # ---- CLI overrides ----
    ic_table_path     = args.ic_table     or _DEFAULT_IC_TABLE
    rankic_table_path = args.rankic_table or _DEFAULT_RANKIC_TABLE
    pool_path         = args.factor_pool  or _DEFAULT_POOL
    if args.market:
        cfg.meta.market = args.market
    if args.refer:
        cfg.split = None
        cfg.refer = args.refer
    if args.sample_n and cfg.split:
        cfg.split.sample_n = args.sample_n

    # ---- Verify inputs ----
    for label, path in [
        ("IC table",     ic_table_path),
        ("RankIC table", rankic_table_path),
        ("Factor pool",  pool_path),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)
        print(f"[Input] {label}: {path}")

    output_dir = args.output_dir
    print(f"[Output] Writing to: {output_dir}")

    mode = "refer" if cfg.refer else "fresh split"
    print(f"[Mode] {mode}")
    if cfg.split:
        print(f"[Split] sample_n={cfg.split.sample_n}, "
              f"train={cfg.split.train_ratio}, val={cfg.split.val_ratio}, "
              f"test={cfg.split.test_ratio}")
    if cfg.noise_threshold:
        nt = cfg.noise_threshold
        print(f"[Noise threshold] ic={nt.ic}, rankic={nt.rankic}, "
              f"abs={nt.abs}, condition={nt.condition!r}")

    # ---- Load data ----
    print("\n[Loading] IC table...")
    ic_table = load_ic_table(ic_table_path)

    print("[Loading] RankIC table...")
    rankic_table = load_ic_table(rankic_table_path)

    print("[Loading] Factor pool...")
    with open(pool_path, "r", encoding="utf-8") as f:
        factor_pool = json.load(f)
    print(f"[Loading] {len(factor_pool)} factors in pool.")

    # ---- Build ----
    print("\n" + "=" * 60)
    print("Building BINARY NOISE dataset (T4 Task-2)")
    print("=" * 60)

    paths = build_noise_dataset(
        config=cfg,
        factor_pool=factor_pool,
        ic_table=ic_table,
        rankic_table=rankic_table,
        output_dir=output_dir,
        verbose=True,
    )

    print("\n[Done] Output files:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
