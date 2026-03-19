#!/usr/bin/env python
"""
Demo 05 -- No-Train Portfolio Combine
======================================
Combines multiple alpha factors into a single trading signal using
z-score normalization + equal-weight averaging (no model training).

Demonstrates:

  - combine_factors() API function
  - Per-factor vs combined IC comparison
  - Optional portfolio backtest on combined signal

Start the backend first:

    ppo start backend

Then run this demo:

    python demos/05_portfolio_combine.py
"""

import time
from ffo.api import combine_factors, server_health


# A diverse set of alpha factors to combine
FACTORS_TO_COMBINE = [
    # Momentum
    "Rank($close, 20)",
    # Volume ratio
    "Mean($volume, 5) / Mean($volume, 20)",
    # Price-volume correlation
    "Corr($close, $volume, 10)",
    # Volatility
    "Std($close / Delay($close, 1) - 1, 20)",
    # VWAP deviation
    "($close - $vwap) / $vwap",
]


def main():
    print("=" * 70)
    print("Demo 05: No-Train Portfolio Combine")
    print("=" * 70)

    # ── Health check ─────────────────────────────────────────────────
    health = server_health()
    if not health.is_healthy:
        print(f"\nERROR: Server not running ({health.error})")
        print("  Start with:  ppo start backend")
        return

    print(f"\nServer healthy  (latency={health.latency_ms:.0f}ms)")
    print(f"Combining {len(FACTORS_TO_COMBINE)} factors...")
    print(f"Market: csi300  |  Period: 2023-01-01 -> 2024-01-01\n")

    # ── Fast combine (IC only, no backtest) ──────────────────────────
    print("Phase 1: Fast combine (IC metrics only)")
    print("-" * 70)
    t0 = time.time()
    result = combine_factors(
        expressions=FACTORS_TO_COMBINE,
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
    )
    elapsed = time.time() - t0

    if not result.success:
        print(f"\nERROR: {result.error}")
        return

    print(f"Completed in {elapsed:.1f}s\n")

    # ── Per-factor metrics ───────────────────────────────────────────
    print(f"{'Factor':<50} {'IC':>7} {'Rank IC':>9} {'ICIR':>7}")
    print("-" * 75)
    for r in result.per_factor_results:
        expr = r.expression if len(r.expression) <= 48 else r.expression[:45] + "..."
        m = r.metrics
        print(f"{expr:<50} {m.ic:>+7.4f} {m.rank_ic:>+9.4f} {m.icir:>+7.3f}")

    # ── Combined signal metrics ──────────────────────────────────────
    cm = result.combined_metrics
    print("-" * 75)
    print(f"{'** COMBINED (equal-weight z-score avg) **':<50} {cm.ic:>+7.4f} {cm.rank_ic:>+9.4f} {cm.icir:>+7.3f}")
    print(f"\nCombined signal: IC={cm.ic:+.4f}  Rank_IC={cm.rank_ic:+.4f}  ICIR={cm.icir:+.3f}  Rank_ICIR={cm.rank_icir:+.3f}  ({cm.n_dates} dates)")

    # ── Compare with best individual factor ──────────────────────────
    if result.per_factor_results:
        best_individual = max(result.per_factor_results, key=lambda r: abs(r.metrics.rank_ic))
        improvement = abs(cm.rank_ic) - abs(best_individual.metrics.rank_ic)
        print(f"\nBest individual: {best_individual.expression[:50]}  Rank_IC={best_individual.metrics.rank_ic:+.4f}")
        print(f"Improvement over best individual: {improvement:+.4f} ({improvement / max(abs(best_individual.metrics.rank_ic), 1e-6) * 100:+.1f}%)")

    # ── Full combine with portfolio backtest ─────────────────────────
    print(f"\n{'=' * 70}")
    print("Phase 2: Full combine with portfolio backtest")
    print("-" * 70)
    t0 = time.time()
    result_full = combine_factors(
        expressions=FACTORS_TO_COMBINE,
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=False,
        topk=50,
        n_drop=5,
    )
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s\n")

    if result_full.portfolio_metrics:
        pm = result_full.portfolio_metrics
        for return_type in ["excess_return_without_cost", "excess_return_with_cost"]:
            if return_type in pm:
                m = pm[return_type]
                print(f"{return_type}:")
                print(f"  Annualized Return : {m.get('annualized_return', 0):.2f}%")
                print(f"  Information Ratio : {m.get('information_ratio', 0):.4f}")
                print(f"  Max Drawdown      : {m.get('max_drawdown', 0):.2f}%")
                print()
    else:
        print("No portfolio metrics available (backtest may have been skipped).")

    print("Done.")


if __name__ == "__main__":
    main()
