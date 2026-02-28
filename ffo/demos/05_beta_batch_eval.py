#!/usr/bin/env python
"""
Demo 05 — Beta Batch Evaluation: Speed Benchmark
=================================================
Compares two evaluation strategies for the **same** 10 factors over a
2-year window on CSI 300:

  Strategy A — batch_evaluate_factors()
      Evaluates factors one-by-one inside parallel HTTP threads.
      Each factor triggers its own Qlib data-load + pandas IC computation.

  Strategy B — beta_evaluate_factors()
      Packages all 10 factors into a SINGLE compute_factor_data() call,
      then computes IC / RankIC for all factors simultaneously using
      PyTorch GPU (falls back to CPU if no GPU is available).

Expected speedup:
  - Data loading: ~10× fewer Qlib calls (1 vs 10).
  - IC computation: matrix operations on GPU vs pandas groupby loops.
  - Total wall-clock: typically 3–8× faster on GPU, 1.5–3× on CPU.

Start the backend first:

    ppo start backend

Then run:

    python demos/05_beta_batch_eval.py
"""

import time
import statistics
from typing import List

from ffo.api import batch_evaluate_factors, beta_evaluate_factors, server_health
from ffo.api.functions import FactorResult

# ── Factor basket (10 diverse alpha factors, 2-year CSI 300 window) ──────────
FACTORS: List[str] = [
    "Rank($close, 5)",
    "Rank($close, 20)",
    "$close / Min($close, 5) - 1",
    "$close / Min($close, 20) - 1",
    "Mean($volume, 5) / Mean($volume, 20)",
    "Std($close / Min($close, 1) - 1, 20)",
    "Corr($close, $volume, 10)",
    "($high - $low) / $close",
    "($close - $vwap) / $vwap",
    "Rank($close, 20) - Rank($volume, 20)",
]

MARKET = "csi300"
START  = "2022-01-01"
END    = "2024-01-01"
LABEL  = "close_return"

SEP    = "─" * 72


def _print_results(results: List[FactorResult]) -> None:
    """Print a ranked metrics table."""
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]

    ranked = sorted(successful, key=lambda r: abs(r.metrics.rank_ic), reverse=True)

    print(f"\n{'#':<4} {'Expression':<44} {'IC':>7} {'RankIC':>8} {'ICIR':>7}")
    print(SEP)
    for i, r in enumerate(ranked, 1):
        expr = r.expression if len(r.expression) <= 42 else r.expression[:39] + "..."
        m = r.metrics
        flag = " ★" if abs(m.rank_ic) > 0.03 else ""
        print(f"{i:<4} {expr:<44} {m.ic:>+7.4f} {m.rank_ic:>+8.4f} {m.icir:>+7.3f}{flag}")

    if failed:
        print(f"\n  [!] {len(failed)} factor(s) failed:")
        for r in failed:
            print(f"      {r.expression[:55]:55s} → {r.error}")


def run_strategy_a(use_cache: bool = False) -> tuple[List[FactorResult], float]:
    """Strategy A: one-by-one parallel HTTP batch."""
    t0 = time.perf_counter()
    results = batch_evaluate_factors(
        expressions=FACTORS,
        market=MARKET,
        start=START,
        end=END,
        label=LABEL,
        fast=True,
        use_cache=use_cache,
        parallel=True,
        max_workers=8,
        progress=False,
    )
    return results, time.perf_counter() - t0


def run_strategy_b(use_cache: bool = False) -> tuple[List[FactorResult], float]:
    """Strategy B: single batch load + GPU IC/RankIC."""
    t0 = time.perf_counter()
    results = beta_evaluate_factors(
        expressions=FACTORS,
        market=MARKET,
        start=START,
        end=END,
        label=LABEL,
        device="auto",
        timeout=600,
    )
    return results, time.perf_counter() - t0


def main() -> None:
    print("=" * 72)
    print("Demo 05: Beta Batch Evaluation — Speed Benchmark")
    print("=" * 72)
    print(f"  Factors : {len(FACTORS)}")
    print(f"  Market  : {MARKET}")
    print(f"  Period  : {START} → {END}  (2 years)")
    print(f"  Label   : {LABEL}")
    print()

    # ── Server health ─────────────────────────────────────────────────────────
    health = server_health()
    if not health.is_healthy:
        print(f"ERROR: Backend not reachable — {health.error}")
        print("  Start with:  ppo start backend")
        return
    print(f"Backend healthy  (latency={health.latency_ms:.0f} ms)\n")

    # ── Warm-up run (fills Qlib internal caches, not FFO cache) ──────────────
    print("Warming up (filling Qlib internal caches)...")
    _ = run_strategy_b(use_cache=False)   # beta warms Qlib in one shot
    print("Warm-up done.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Strategy A: batch_evaluate_factors (parallel threads, per-factor loads)
    # ══════════════════════════════════════════════════════════════════════════
    print(SEP)
    print("Strategy A — batch_evaluate_factors()  (parallel, per-factor)")
    print(f"  10 separate Qlib data-load calls + pandas IC loops")
    print(SEP)

    results_a, elapsed_a = run_strategy_a(use_cache=False)

    successful_a = [r for r in results_a if r.success]
    print(f"  Completed : {len(successful_a)}/{len(FACTORS)} factors succeeded")
    print(f"  Wall time : {elapsed_a:.2f} s")
    _print_results(results_a)

    # ══════════════════════════════════════════════════════════════════════════
    # Strategy B: beta_evaluate_factors (single load + GPU)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("Strategy B — beta_evaluate_factors()   (single load + GPU/CPU)")
    print(f"  1 Qlib data-load call, PyTorch matrix IC for all factors")
    print(SEP)

    results_b, elapsed_b = run_strategy_b(use_cache=False)

    successful_b = [r for r in results_b if r.success]
    device_used  = next(
        (getattr(r, "__dict__", {}).get("device", "?") for r in results_b if r.success),
        "?"
    )
    print(f"  Completed : {len(successful_b)}/{len(FACTORS)} factors succeeded")
    print(f"  Wall time : {elapsed_b:.2f} s")
    _print_results(results_b)

    # ══════════════════════════════════════════════════════════════════════════
    # Speed comparison
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("Speed comparison")
    print(SEP)

    speedup = elapsed_a / elapsed_b if elapsed_b > 0 else float("inf")

    print(f"  Strategy A  (batch, per-factor)  : {elapsed_a:>7.2f} s")
    print(f"  Strategy B  (beta, single+GPU)   : {elapsed_b:>7.2f} s")
    print(f"  Speedup     B vs A               : {speedup:>7.2f}×")

    if speedup >= 2:
        print(f"\n  beta_evaluate_factors is {speedup:.1f}× faster!")
    elif speedup >= 1:
        print(f"\n  beta_evaluate_factors is {speedup:.1f}× faster.")
    else:
        print(f"\n  Note: batch was faster this run (may be due to caching / cold GPU).")

    # ── Metric agreement check ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Metric agreement (Strategy A vs Strategy B)")
    print(SEP)

    expr_to_a = {r.expression: r for r in results_a if r.success}
    expr_to_b = {r.expression: r for r in results_b if r.success}

    common = set(expr_to_a) & set(expr_to_b)
    if common:
        ic_diffs   = [abs(expr_to_a[e].metrics.ic       - expr_to_b[e].metrics.ic)       for e in common]
        ric_diffs  = [abs(expr_to_a[e].metrics.rank_ic  - expr_to_b[e].metrics.rank_ic)  for e in common]

        print(f"  Compared  : {len(common)} factors")
        print(f"  Max |ΔIC|   : {max(ic_diffs):.6f}")
        print(f"  Mean |ΔIC|  : {statistics.mean(ic_diffs):.6f}")
        print(f"  Max |ΔRankIC| : {max(ric_diffs):.6f}")
        print(f"  Mean |ΔRankIC|: {statistics.mean(ric_diffs):.6f}")

        tol = 1e-4
        if max(ic_diffs) < tol and max(ric_diffs) < tol:
            print(f"\n  [✓] Results are numerically consistent (tol={tol}).")
        else:
            print(f"\n  [!] Some differences exceed {tol}; verify data alignment.")
    else:
        print("  No common successful factors to compare.")

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()
