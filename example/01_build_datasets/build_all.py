"""
Build ALL evaluation datasets in one shot.

Builds:
  1. T2 Ranking dataset   → benchmark/data/evaluate/built/ranking/
  2. T2 Scoring dataset   → benchmark/data/evaluate/built/scoring/
  3. T4 Noise dataset     → benchmark/data/evaluate/atomic/noise_csi300/
  4. T4 Pairwise dataset  → benchmark/data/evaluate/atomic/pairwise_csi300/

Assumes input files at:
  benchmark/data/evaluate/raw/ic_table_csi300.csv
  benchmark/data/evaluate/raw/rankic_table_csi300.csv
  benchmark/data/evaluate/pool/factor_pool.json

Usage (from repo root):
  python example/01_build_datasets/build_all.py

Usage (SP500 in addition to CSI300):
  python example/01_build_datasets/build_all.py --also_sp500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.utils.data_builder.base_eval import (
    BaseEvalConfig,
    build_ranking_dataset,
    build_scoring_dataset,
)
from benchmark.utils.data_builder.atomic_eval import (
    AtomicEvalConfig,
    build_noise_dataset,
    build_pairwise_dataset,
    load_ic_table,
)


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

RAW_DIR  = "benchmark/data/evaluate/raw"
POOL_DIR = "benchmark/data/evaluate/pool"

INPUTS = {
    "csi300": {
        "ic_table":     os.path.join(RAW_DIR, "ic_table_csi300.csv"),
        "rankic_table": os.path.join(RAW_DIR, "rankic_table_csi300.csv"),
        "factor_pool":  os.path.join(POOL_DIR, "factor_pool.json"),
        "market":       "CSI300",
        "start_date":   "2021-01-01",
        "end_date":     "2025-06-30",
    },
    "sp500": {
        "ic_table":     os.path.join(RAW_DIR, "ic_table_sp500.csv"),
        "rankic_table": os.path.join(RAW_DIR, "rankic_table_sp500.csv"),
        "factor_pool":  os.path.join(POOL_DIR, "factor_pool.json"),
        "market":       "SP500",
        "start_date":   "2021-01-01",
        "end_date":     "2025-06-30",
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_inputs(info: dict) -> bool:
    ok = True
    for key in ("ic_table", "rankic_table", "factor_pool"):
        if not os.path.exists(info[key]):
            print(f"  [MISSING] {info[key]}")
            ok = False
    return ok


def _build_base(info: dict, out_root: str) -> None:
    """Build T2 ranking + scoring for one market."""
    market = info["market"]
    cfg = BaseEvalConfig(
        market=market,
        start_date=info["start_date"],
        end_date=info["end_date"],
        ic_table_path=info["ic_table"],
        rankic_table_path=info["rankic_table"],
        factor_pool_path=info["factor_pool"],
    )
    ranking_dir = os.path.join(out_root, f"ranking_{market.lower()}")
    scoring_dir = os.path.join(out_root, f"scoring_{market.lower()}")

    print(f"\n[T2 Ranking] {market}")
    build_ranking_dataset(cfg, output_dir=ranking_dir, verbose=True)

    print(f"\n[T2 Scoring] {market}")
    build_scoring_dataset(cfg, output_dir=scoring_dir, verbose=True)


def _build_atomic(info: dict, out_root: str, refer_noise: str | None = None,
                  refer_pairwise: str | None = None) -> tuple[str, str]:
    """Build T4 noise + pairwise for one market.  Returns (noise_dir, pairwise_dir)."""
    market = info["market"].lower()

    def _fresh_cfg(mode: str) -> dict:
        return {
            "meta": {
                "market":     info["market"],
                "start_date": info["start_date"],
                "end_date":   info["end_date"],
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
        }

    def _refer_cfg(refer_path: str) -> dict:
        return {
            "meta": {
                "market":     info["market"],
                "start_date": info["start_date"],
                "end_date":   info["end_date"],
            },
            "noise_threshold": {
                "ic": 0.005, "rankic": 0.025,
                "abs": True, "condition": "and",
            },
            "refer": refer_path,
        }

    # ---- Load data (shared for both tasks) ----
    print(f"[Loading] IC / RankIC tables for {info['market']}...")
    ic_table     = load_ic_table(info["ic_table"])
    rankic_table = load_ic_table(info["rankic_table"])
    with open(info["factor_pool"], "r", encoding="utf-8") as _f:
        factor_pool = json.load(_f)
    print(f"[Loading] {len(factor_pool)} factors in pool.")

    # Noise dataset
    noise_dir = os.path.join(out_root, f"noise_{market}")
    noise_raw = _refer_cfg(refer_noise) if refer_noise else _fresh_cfg("noise")
    noise_cfg = AtomicEvalConfig.from_dict(noise_raw)
    print(f"\n[T4 Noise] {info['market']}" +
          (f" (refer: {refer_noise})" if refer_noise else " (fresh split)"))
    build_noise_dataset(
        config=noise_cfg,
        factor_pool=factor_pool,
        ic_table=ic_table,
        rankic_table=rankic_table,
        output_dir=noise_dir,
        verbose=True,
    )

    # Pairwise dataset
    pairwise_dir = os.path.join(out_root, f"pairwise_{market}")
    pairwise_raw = _refer_cfg(refer_pairwise) if refer_pairwise else _fresh_cfg("pairwise")
    pairwise_cfg = AtomicEvalConfig.from_dict(pairwise_raw)
    print(f"\n[T4 Pairwise] {info['market']}" +
          (f" (refer: {refer_pairwise})" if refer_pairwise else " (fresh split)"))
    build_pairwise_dataset(
        config=pairwise_cfg,
        factor_pool=factor_pool,
        ic_table=ic_table,
        rankic_table=rankic_table,
        output_dir=pairwise_dir,
        verbose=True,
    )

    return noise_dir, pairwise_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Build all AlphaBench evaluation datasets.")
    p.add_argument("--base_out",   default="benchmark/data/evaluate/built",
                   help="Root output dir for T2 (ranking + scoring) datasets.")
    p.add_argument("--atomic_out", default="benchmark/data/evaluate/atomic",
                   help="Root output dir for T4 (noise + pairwise) datasets.")
    p.add_argument("--also_sp500", action="store_true",
                   help="Also build SP500 datasets (using CSI300 split as reference).")
    return p.parse_args()


def main():
    args = _parse_args()
    t0 = time.time()

    print("=" * 70)
    print("AlphaBench — Build All Evaluation Datasets")
    print("=" * 70)

    # ---- Check CSI300 inputs ----
    print("\n[Checking CSI300 inputs]")
    if not _check_inputs(INPUTS["csi300"]):
        print("[ERROR] Missing CSI300 input files. Aborting.")
        sys.exit(1)

    # ---- T2: CSI300 ----
    print("\n" + "=" * 40)
    print("T2 Base Evaluation — CSI300")
    print("=" * 40)
    _build_base(INPUTS["csi300"], args.base_out)

    # ---- T4: CSI300 (fresh split) ----
    print("\n" + "=" * 40)
    print("T4 Atomic Evaluation — CSI300 (fresh split)")
    print("=" * 40)
    noise_csi, pairwise_csi = _build_atomic(INPUTS["csi300"], args.atomic_out)

    # ---- SP500 (optional, uses CSI300 split as reference) ----
    if args.also_sp500:
        print("\n[Checking SP500 inputs]")
        if not _check_inputs(INPUTS["sp500"]):
            print("[WARN] Missing SP500 files; skipping SP500.")
        else:
            print("\n" + "=" * 40)
            print("T2 Base Evaluation — SP500")
            print("=" * 40)
            _build_base(INPUTS["sp500"], args.base_out)

            print("\n" + "=" * 40)
            print("T4 Atomic Evaluation — SP500 (cross-market refer)")
            print("=" * 40)
            _build_atomic(
                INPUTS["sp500"], args.atomic_out,
                refer_noise=noise_csi,
                refer_pairwise=pairwise_csi,
            )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All datasets built in {elapsed:.1f}s")
    print(f"  T2 datasets: {args.base_out}")
    print(f"  T4 datasets: {args.atomic_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
