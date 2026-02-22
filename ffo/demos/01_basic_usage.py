#!/usr/bin/env python
"""
Demo 01 — Basic Factor Evaluation
==================================
Shows the simplest way to evaluate a single alpha factor using the
FFO API. Start the backend first:

    ppo start backend

Then run this demo:

    python demos/01_basic_usage.py
"""

from ffo.api import evaluate_factor, check_factor, server_health


def main():
    print("=" * 60)
    print("Demo 01: Basic Factor Evaluation")
    print("=" * 60)

    # ── Step 1: Verify the server is running ──────────────────────
    print("\n[1] Checking server health...")
    health = server_health()

    if not health.is_healthy:
        print(f"  ERROR: Server is not running ({health.error})")
        print("  Start it with:  ppo start backend")
        return

    print(f"  Status   : {health.status}")
    print(f"  Latency  : {health.latency_ms:.0f} ms")
    print(f"  Cache    : {health.cache}")

    # ── Step 2: Validate syntax before evaluation ──────────────────
    expression = "Rank($close, 20)"
    print(f"\n[2] Checking syntax: {expression!r}")

    check = check_factor(expression)
    if not check.is_valid:
        print(f"  Invalid expression: {check.error}")
        return
    print(f"  Syntax OK")

    # ── Step 3: Evaluate the factor ────────────────────────────────
    print(f"\n[3] Evaluating factor...")
    print(f"    Expression : {expression}")
    print(f"    Market     : csi300")
    print(f"    Period     : 2023-01-01 → 2024-01-01")
    print(f"    Mode       : fast (IC only)")

    result = evaluate_factor(
        expression=expression,
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
    )

    # ── Step 4: Print results ──────────────────────────────────────
    print("\n[4] Results:")
    if result.success:
        m = result.metrics
        print(f"  IC          : {m.ic:+.4f}   (|IC| > 0.02 is useful)")
        print(f"  Rank IC     : {m.rank_ic:+.4f}")
        print(f"  ICIR        : {m.icir:+.4f}   (> 0.5 is good)")
        print(f"  Rank ICIR   : {m.rank_icir:+.4f}")
        print(f"  Turnover    : {m.turnover:.4f}   (lower = cheaper)")
        print(f"  N Dates     : {m.n_dates}")
        print(f"  Cached      : {result.cached}")
    else:
        print(f"  ERROR: {result.error}")

    # ── Step 5: Try an invalid expression ──────────────────────────
    print("\n[5] Testing invalid expression...")
    bad_result = evaluate_factor("NotAnOp($close, 20)")
    print(f"  Success: {bad_result.success}")
    if not bad_result.success:
        print(f"  Error  : {bad_result.error}")

    print("\nDone.")


if __name__ == "__main__":
    main()
