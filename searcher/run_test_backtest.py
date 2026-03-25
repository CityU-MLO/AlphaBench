#!/usr/bin/env python3
"""
AlphaBench — Test Period Portfolio Backtesting

After a search run completes, use this script to evaluate the discovered
factors on the held-out test period with a full portfolio backtest.

This script:
  1. Loads the final_pool.jsonl from a completed search run.
  2. Ranks factors by search-period RankIC (or val RankIC if available).
  3. Selects the top-N factors for portfolio construction.
  4. Evaluates each factor on the test period via the FFO server
     (full backtest mode: fast=False).
  5. Saves test-period results to <results_dir>/test_results/.

Usage
─────
  # Basic usage (reads config from the search run's saved config.yaml):
  python run_test_backtest.py --results-dir ./results/search_001

  # Override test parameters:
  python run_test_backtest.py --results-dir ./results/search_001 \\
      --top-n 50 --drop-k 5 --n-jobs 4

  # Use a different config file:
  python run_test_backtest.py --config config.yaml --pool final_pool.jsonl \\
      --top-n 50 --drop-k 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure AlphaBench root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_ALPHABENCH_ROOT = _SCRIPT_DIR.parent
if str(_ALPHABENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALPHABENCH_ROOT))

from searcher.backtester import Backtester
from searcher.config.config import load_config_from_yaml, FullConfig
from searcher.utils.logger import SearchLogger


def load_final_pool(pool_path: str) -> list:
    """Load factors from a JSONL file."""
    factors = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                factors.append(json.loads(line))
    return factors


def rank_factors(factors: list, metric_key: str = "rank_ic") -> list:
    """Rank factors by a metric, preferring val_metrics if available."""
    def sort_key(f):
        # Prefer val_metrics for ranking (out-of-sample)
        val_m = f.get("val_metrics", {})
        search_m = f.get("search_metrics", f.get("metrics", {}))
        if val_m and metric_key in val_m:
            return float(val_m.get(metric_key, -1e9))
        return float(search_m.get(metric_key, -1e9))

    return sorted(factors, key=sort_key, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AlphaBench — Test Period Portfolio Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to a completed search run directory (contains config.yaml and final_pool.jsonl)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file (if not using --results-dir)"
    )
    parser.add_argument(
        "--pool", type=str, default=None,
        help="Path to final_pool.jsonl (if not using --results-dir)"
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of top factors to select for portfolio (default: 50)"
    )
    parser.add_argument(
        "--drop-k", type=int, default=5,
        help="Positions dropped per rebalance (default: 5)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4,
        help="Parallel evaluation workers (default: 4)"
    )
    parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Use fast mode (IC only) instead of full portfolio backtest"
    )
    parser.add_argument(
        "--rank-by", type=str, default="rank_ic",
        choices=["rank_ic", "ic", "icir", "rank_icir"],
        help="Metric to rank factors by for top-N selection (default: rank_ic)"
    )

    args = parser.parse_args()

    logger = SearchLogger("TestBacktest")

    # ── Resolve config and pool paths ────────────────────────────────────
    if args.results_dir:
        results_dir = Path(args.results_dir)
        config_path = str(results_dir / "config.yaml")
        pool_path = str(results_dir / "final_pool.jsonl")
    else:
        if not args.config or not args.pool:
            parser.error("Either --results-dir or both --config and --pool are required")
        config_path = args.config
        pool_path = args.pool
        results_dir = Path(pool_path).parent

    # Resolve config path
    if not Path(config_path).is_absolute():
        candidate = _SCRIPT_DIR / config_path
        if candidate.exists():
            config_path = str(candidate)

    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    if not Path(pool_path).exists():
        logger.error(f"Pool file not found: {pool_path}")
        return 1

    try:
        logger.info("=" * 70)
        logger.info("AlphaBench — Test Period Portfolio Backtesting")
        logger.info("=" * 70)

        # ── Load config and pool ─────────────────────────────────────────
        config = load_config_from_yaml(config_path)
        logger.info(f"Config: {config_path}")

        factors = load_final_pool(pool_path)
        logger.info(f"Loaded {len(factors)} factors from {pool_path}")

        if not factors:
            logger.error("No factors found in pool file")
            return 1

        # ── Rank and select top-N factors ────────────────────────────────
        ranked = rank_factors(factors, metric_key=args.rank_by)
        selected = ranked[:args.top_n]
        logger.info(f"Selected top {len(selected)} factors (ranked by {args.rank_by})")

        # ── Create test-period backtester ────────────────────────────────
        test_backtester = Backtester.for_test(
            backtest_config=config.backtesting,
            verification_config=config.verification,
            top_k=args.top_n,
            n_drop=args.drop_k,
            n_jobs=args.n_jobs,
            fast=args.fast,
            logger=logger,
        )

        logger.info(f"  Test period   : {config.verification.test_start} ~ {config.verification.test_end}")
        logger.info(f"  Market        : {config.backtesting.market}")
        logger.info(f"  Top N         : {args.top_n}")
        logger.info(f"  Drop K        : {args.drop_k}")
        logger.info(f"  N jobs        : {args.n_jobs}")
        logger.info(f"  Fast mode     : {args.fast}")
        logger.info(f"  FFO server    : {config.backtesting.ffo_server}")
        logger.info("")

        # ── Health check ─────────────────────────────────────────────────
        if not test_backtester.check_health():
            logger.warning("FFO backend may be unavailable")

        # ── Evaluate on test period ──────────────────────────────────────
        logger.info(f"Evaluating {len(selected)} factors on test period …")
        t0 = time.time()

        eval_input = [
            {"name": f.get("name", f"factor_{i}"), "expression": f.get("expression", "")}
            for i, f in enumerate(selected)
            if f.get("expression")
        ]
        test_results = test_backtester.evaluate_batch(eval_input)
        elapsed = time.time() - t0

        n_success = sum(1 for r in test_results if r.get("success"))
        logger.info(f"Evaluation done: {n_success}/{len(eval_input)} successful in {elapsed:.1f}s")

        # ── Merge test metrics into factors ──────────────────────────────
        test_factors = []
        for f, result in zip(selected, test_results):
            merged = dict(f)
            merged["test_metrics"] = result.get("metrics", {})
            merged["test_success"] = result.get("success", False)
            test_factors.append(merged)

        # Sort by test-period RankIC
        test_factors.sort(
            key=lambda x: float(x.get("test_metrics", {}).get("rank_ic", -1e9)),
            reverse=True,
        )

        # ── Save results ─────────────────────────────────────────────────
        test_output_dir = results_dir / "test_results"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Save full test results
        test_results_path = test_output_dir / "test_results.jsonl"
        with open(test_results_path, "w", encoding="utf-8") as f:
            for tf in test_factors:
                f.write(json.dumps(tf, ensure_ascii=False) + "\n")

        # Save summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": config_path,
            "pool_path": pool_path,
            "test_period": f"{config.verification.test_start} ~ {config.verification.test_end}",
            "market": config.backtesting.market,
            "top_n": args.top_n,
            "drop_k": args.drop_k,
            "fast": args.fast,
            "n_factors_evaluated": len(eval_input),
            "n_success": n_success,
            "elapsed_sec": round(elapsed, 1),
            "rank_by": args.rank_by,
        }

        # Add aggregate test metrics
        test_rics = [
            float(tf.get("test_metrics", {}).get("rank_ic", 0))
            for tf in test_factors if tf.get("test_success")
        ]
        test_ics = [
            float(tf.get("test_metrics", {}).get("ic", 0))
            for tf in test_factors if tf.get("test_success")
        ]
        if test_rics:
            summary["mean_test_rank_ic"] = round(sum(test_rics) / len(test_rics), 6)
            summary["max_test_rank_ic"] = round(max(test_rics), 6)
            summary["min_test_rank_ic"] = round(min(test_rics), 6)
        if test_ics:
            summary["mean_test_ic"] = round(sum(test_ics) / len(test_ics), 6)

        summary_path = test_output_dir / "test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Test Backtesting Complete!")
        logger.info("=" * 70)
        logger.info(f"  Factors tested : {n_success}/{len(eval_input)}")
        if test_rics:
            logger.info(f"  Mean RankIC    : {summary['mean_test_rank_ic']:.6f}")
            logger.info(f"  Max  RankIC    : {summary['max_test_rank_ic']:.6f}")
        if test_ics:
            logger.info(f"  Mean IC        : {summary['mean_test_ic']:.6f}")
        logger.info(f"  Results saved  : {test_output_dir}")

        # Print top 10 factors
        logger.info("")
        logger.info("Top 10 factors by test-period RankIC:")
        for i, tf in enumerate(test_factors[:10], 1):
            tm = tf.get("test_metrics", {})
            sm = tf.get("search_metrics", tf.get("metrics", {}))
            vm = tf.get("val_metrics", {})
            name = (tf.get("name") or "?")[:30]
            test_ric = tm.get("rank_ic", float("nan"))
            test_ic = tm.get("ic", float("nan"))
            search_ric = sm.get("rank_ic", float("nan"))
            val_ric = vm.get("rank_ic", float("nan"))
            logger.info(
                f"  {i:2d}. {name:<30} "
                f"TestRIC={test_ric:.4f}  TestIC={test_ic:.4f}  "
                f"SearchRIC={search_ric:.4f}  ValRIC={val_ric:.4f}"
            )

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
