"""
Build Base Evaluation Datasets (T2: Ranking & Scoring)

Reads:
  benchmark/data/evaluate/raw/ic_table_csi300.csv
  benchmark/data/evaluate/raw/rankic_table_csi300.csv
  benchmark/data/evaluate/pool/factor_pool.json

Writes to output_dir:
  all_env_scenarios.json      ← ranking test cases (all regimes combined)
  ranking_<regime>.json       ← per-regime ranking test cases
  alphabench_testset.json     ← scoring test cases (all regimes combined)
  scoring_<regime>.json       ← per-regime scoring test cases

Usage (default config, from repo root):
  python example/01_build_datasets/build_base_eval_datasets.py

Usage (custom config YAML):
  python example/01_build_datasets/build_base_eval_datasets.py \
      --config example/01_build_datasets/configs/base_eval_csi300.yaml \
      --output_dir benchmark/data/evaluate/built

Usage (SP500 with inline overrides):
  python example/01_build_datasets/build_base_eval_datasets.py \
      --ic_table     benchmark/data/evaluate/raw/ic_table_sp500.csv \
      --rankic_table benchmark/data/evaluate/raw/rankic_table_sp500.csv \
      --market       SP500 \
      --output_dir   benchmark/data/evaluate/built_sp500
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root or from this file's directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.utils.data_builder.base_eval import (
    BaseEvalConfig,
    RegimeConfig,
    build_ranking_dataset,
    build_scoring_dataset,
)


# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------

_DEFAULT_IC_TABLE     = "benchmark/data/evaluate/raw/ic_table_csi300.csv"
_DEFAULT_RANKIC_TABLE = "benchmark/data/evaluate/raw/rankic_table_csi300.csv"
_DEFAULT_POOL         = "benchmark/data/evaluate/pool/factor_pool.json"
_DEFAULT_OUTPUT       = "benchmark/data/evaluate/built"
_DEFAULT_CONFIG       = "example/01_build_datasets/configs/base_eval_csi300.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build ranking and scoring datasets for T2 evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None,
                   help="Path to YAML config (overrides all defaults). "
                        f"Default: {_DEFAULT_CONFIG}")
    p.add_argument("--ic_table", default=None,
                   help=f"Path to IC table CSV. Default: {_DEFAULT_IC_TABLE}")
    p.add_argument("--rankic_table", default=None,
                   help=f"Path to RankIC table CSV. Default: {_DEFAULT_RANKIC_TABLE}")
    p.add_argument("--factor_pool", default=None,
                   help=f"Path to factor_pool.json. Default: {_DEFAULT_POOL}")
    p.add_argument("--market", default=None,
                   help="Market name (e.g. CSI300, SP500). Overrides config.")
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT,
                   help=f"Output directory. Default: {_DEFAULT_OUTPUT}")
    p.add_argument("--tasks", nargs="+", choices=["ranking", "scoring", "all"],
                   default=["all"],
                   help="Which datasets to build. Default: all")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = _parse_args()

    # ---- Load or build config ----
    if args.config and os.path.exists(args.config):
        print(f"[Config] Loading from YAML: {args.config}")
        cfg = BaseEvalConfig.from_yaml(args.config)
    else:
        print(f"[Config] Using default CSI300 paper settings")
        cfg = BaseEvalConfig()

    # ---- Apply CLI overrides ----
    if args.ic_table:
        cfg.ic_table_path = args.ic_table
    if args.rankic_table:
        cfg.rankic_table_path = args.rankic_table
    if args.factor_pool:
        cfg.factor_pool_path = args.factor_pool
    if args.market:
        cfg.market = args.market

    # ---- Verify inputs ----
    for label, path in [
        ("IC table",       cfg.ic_table_path),
        ("RankIC table",   cfg.rankic_table_path),
        ("Factor pool",    cfg.factor_pool_path),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)
        print(f"[Input] {label}: {path}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Output] Writing datasets to: {output_dir}")

    # ---- Determine which tasks to run ----
    tasks = args.tasks
    if "all" in tasks:
        tasks = ["ranking", "scoring"]

    # ---- Build datasets ----
    if "ranking" in tasks:
        print("\n" + "=" * 60)
        print("Building RANKING dataset (T2 §B.2.2)")
        print("=" * 60)
        ranking_dir = os.path.join(output_dir, "ranking")
        paths = build_ranking_dataset(cfg, output_dir=ranking_dir, verbose=args.verbose)
        print(f"\n[Ranking] Output files:")
        for k, v in paths.items():
            print(f"  {k}: {v}")

    if "scoring" in tasks:
        print("\n" + "=" * 60)
        print("Building SCORING dataset (T2 §B.2.3)")
        print("=" * 60)
        scoring_dir = os.path.join(output_dir, "scoring")
        paths = build_scoring_dataset(cfg, output_dir=scoring_dir, verbose=args.verbose)
        print(f"\n[Scoring] Output files:")
        for k, v in paths.items():
            print(f"  {k}: {v}")

    print("\n[Done] All requested datasets built.")
    print(f"       Evaluation datasets at: {output_dir}")


if __name__ == "__main__":
    main()
