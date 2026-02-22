#!/usr/bin/env python
"""
Demo 02 — Batch Factor Evaluation with Ranking
===============================================
Evaluates a basket of alpha factors in parallel and ranks them by
Rank IC. Demonstrates:

  - batch_evaluate_factors() with parallel execution
  - Progress reporting
  - Sorting and filtering results

Start the backend first:

    ppo start backend

Then run this demo:

    python demos/02_batch_eval.py
"""

import time
from ffo.api import batch_evaluate_factors, server_health


# A diverse set of alpha factor candidates
CANDIDATE_FACTORS = [
    # ── Price momentum / reversal
    "Rank($close, 5)",
    "Rank($close, 10)",
    "Rank($close, 20)",
    "Rank($close, 60)",
    # ── Return-based
    "$close / Delay($close, 1) - 1",
    "$close / Delay($close, 5) - 1",
    "$close / Delay($close, 20) - 1",
    # ── Volume
    "Mean($volume, 5) / Mean($volume, 20)",
    "Rank($volume, 10)",
    "Log(Mean($volume, 5))",
    # ── Volatility
    "Std($close / Delay($close, 1) - 1, 20)",
    "Std($close, 20) / Mean($close, 20)",
    # ── Price-volume interaction
    "Corr($close, $volume, 10)",
    "Corr($close, $volume, 20)",
    "Rank(Corr($vwap, $volume, 10), 20)",
    # ── High-low range
    "($high - $low) / $close",
    "Mean(($high - $low) / $close, 5)",
    # ── VWAP deviation
    "($close - $vwap) / $vwap",
    "Rank(($close - $vwap) / $vwap, 20)",
    # ── Combined
    "Rank($close, 20) - Rank($volume, 20)",
]


def main():
    print("=" * 70)
    print("Demo 02: Batch Factor Evaluation with Ranking")
    print("=" * 70)

    # Check server
    health = server_health()
    if not health.is_healthy:
        print(f"\nERROR: Server not running ({health.error})")
        print("  Start with:  ppo start backend")
        return

    print(f"\nServer healthy  (latency={health.latency_ms:.0f}ms)")
    print(f"Evaluating {len(CANDIDATE_FACTORS)} factor candidates...")
    print(f"Market: csi300  |  Period: 2023-01-01 → 2024-01-01  |  Mode: fast\n")

    # ── Parallel batch evaluation ──────────────────────────────────
    t0 = time.time()
    results = batch_evaluate_factors(
        expressions=CANDIDATE_FACTORS,
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
        parallel=True,
        max_workers=8,
        progress=True,
    )
    elapsed = time.time() - t0

    # ── Separate successful and failed ─────────────────────────────
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Successful : {len(successful)}/{len(results)}")
    print(f"  Failed     : {len(failed)}")

    # ── Rank by Rank IC (descending absolute value) ────────────────
    ranked = sorted(successful, key=lambda r: abs(r.metrics.rank_ic), reverse=True)

    print("\n" + "─" * 70)
    print(f"{'Rank':<5} {'Expression':<45} {'IC':>7} {'Rank IC':>9} {'ICIR':>7}")
    print("─" * 70)

    for i, r in enumerate(ranked, 1):
        m = r.metrics
        expr = r.expression
        if len(expr) > 43:
            expr = expr[:40] + "..."
        star = " ★" if abs(m.rank_ic) > 0.03 else ""
        print(f"{i:<5} {expr:<45} {m.ic:>+7.4f} {m.rank_ic:>+9.4f} {m.icir:>+7.3f}{star}")

    # ── Summary statistics ─────────────────────────────────────────
    if successful:
        ics = [r.metrics.ic for r in successful]
        rics = [r.metrics.rank_ic for r in successful]
        best = ranked[0]

        print("\n" + "─" * 70)
        print("Summary:")
        print(f"  Best factor     : {best.expression}")
        print(f"  Best Rank IC    : {best.metrics.rank_ic:+.4f}")
        print(f"  Mean |IC|       : {sum(abs(x) for x in ics) / len(ics):.4f}")
        print(f"  Mean |Rank IC|  : {sum(abs(x) for x in rics) / len(rics):.4f}")
        print(f"  Factors w/ |IC| > 0.02  : {sum(1 for x in ics if abs(x) > 0.02)}")

    if failed:
        print(f"\nFailed evaluations:")
        for r in failed:
            print(f"  {r.expression[:50]:50s} → {r.error}")

    print("\nDone.")


if __name__ == "__main__":
    main()
