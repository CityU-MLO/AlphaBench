#!/usr/bin/env python
"""
FFO Client Usage Examples

This file demonstrates the unified FFO client interface with:
- Simple function-style API calls
- Parallel batch processing
- Progress tracking
- Context managers for resource management
"""

import sys
import os
import time

# Add parent directory to path to import client module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from client import evaluate, evaluate_batch, FactorEvalClient


# ==================== Example 1: Simple Function-Style Usage ====================

def example_simple_usage():
    """
    Simplest way to evaluate factors - just call functions.
    No need to manage client instances.
    """
    print("=" * 60)
    print("Example 1: Simple Function-Style Usage")
    print("=" * 60)

    # Single factor evaluation
    result = evaluate("Rank($close, 20)", fast=True)

    print(f"Factor: {result.get('expression', 'N/A')}")
    print(f"Success: {result.get('success', False)}")
    if result.get('success'):
        metrics = result.get('metrics', {})
        print(f"IC: {metrics.get('ic', 0):.4f}")
        print(f"Rank IC: {metrics.get('rank_ic', 0):.4f}")
        print(f"ICIR: {metrics.get('icir', 0):.4f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print()


# ==================== Example 2: Batch Evaluation with Parallelization ====================

def example_batch_parallel():
    """
    Evaluate multiple factors in parallel for better performance.
    """
    print("=" * 60)
    print("Example 2: Batch Evaluation with Parallelization")
    print("=" * 60)

    factors = [
        "Rank($close, 20)",
        "Mean($volume, 5)",
        "StdDev($close, 10)",
        "Corr($close, $volume, 20)",
        "Rank($close / Delay($close, 1) - 1, 10)",
    ]

    # Sequential evaluation
    print("Sequential evaluation...")
    start_time = time.time()
    results_seq = evaluate_batch(factors, parallel=False, fast=True)
    seq_time = time.time() - start_time
    print(f"Time: {seq_time:.2f}s")

    # Parallel evaluation
    print("\nParallel evaluation (8 workers)...")
    start_time = time.time()
    results_par = evaluate_batch(factors, parallel=True, max_workers=8, fast=True, progress=True)
    par_time = time.time() - start_time
    print(f"Time: {par_time:.2f}s")

    if par_time > 0 and seq_time > 0:
        print(f"Speedup: {seq_time/par_time:.2f}x")

    # Display results
    print("\nResults:")
    for result in results_par:
        if result.get('success'):
            expr = result.get('expression', '')[:30]
            ic = result.get('metrics', {}).get('ic', 0)
            print(f"{expr}: IC={ic:.4f}")
        else:
            expr = result.get('expression', '')[:30]
            error = result.get('error', 'Unknown')
            print(f"{expr}: ERROR - {error}")

    print()


# ==================== Example 3: Context Manager Usage ====================

def example_context_manager():
    """
    Use context manager for automatic resource cleanup.
    """
    print("=" * 60)
    print("Example 3: Context Manager Usage")
    print("=" * 60)

    factors = [
        "Rank($close, 20)",
        "Mean($volume, 5)",
        "StdDev($close, 10)",
    ]

    # Client is automatically closed when exiting the context
    with FactorEvalClient() as client:
        print("Client initialized")

        # Check health
        if not client.health_check():
            print("WARNING: API server not healthy!")
            return

        # Evaluate with progress tracking
        def progress(completed, total):
            print(f"\rProgress: {completed}/{total}", end="")
            if completed == total:
                print()

        results = client.evaluate_batch_parallel(
            factors,
            max_workers=4,
            fast=True,
            progress_callback=progress,
        )

        print(f"\nEvaluated {len(results)} factors")

    print("Client closed (automatic cleanup)")
    print()


# ==================== Example 4: Progress Tracking ====================

def example_progress_tracking():
    """
    Track progress for long-running batch evaluations.
    """
    print("=" * 60)
    print("Example 4: Progress Tracking")
    print("=" * 60)

    factors = [f"Rank($close, {i})" for i in range(10, 40, 2)]

    with FactorEvalClient() as client:
        print(f"Evaluating {len(factors)} factors with progress tracking...\n")

        def progress(completed, total):
            percent = 100 * completed // total
            bar_length = 50
            filled = bar_length * completed // total
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f"\r[{bar}] {completed}/{total} ({percent}%)", end="")
            if completed == total:
                print()

        results = client.evaluate_batch_parallel(
            factors,
            max_workers=8,
            fast=True,
            progress_callback=progress,
        )

        successful = sum(1 for r in results if r.get('success'))
        print(f"\nCompleted: {successful}/{len(results)} successful")

    print()


# ==================== Example 5: Error Handling ====================

def example_error_handling():
    """
    Demonstrate robust error handling.
    """
    print("=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    # Include some invalid expressions
    factors = [
        "Rank($close, 20)",  # Valid
        "InvalidOperator($close, 20)",  # Invalid
        "Mean($volume, 5)",  # Valid
        "Rank($close, -10)",  # Invalid parameter
    ]

    print("Evaluating with some invalid expressions...\n")

    results = evaluate_batch(factors, parallel=True, fast=True)

    for result in results:
        expr = result.get('expression', 'N/A')[:40]
        success = result.get('success', False)

        print(f"Expression: {expr}")
        print(f"Success: {success}")

        if success:
            ic = result.get('metrics', {}).get('ic', 0)
            print(f"IC: {ic:.4f}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")

        print()


# ==================== Example 6: Cache Management ====================

def example_cache_management():
    """
    Demonstrate cache statistics and management.
    """
    print("=" * 60)
    print("Example 6: Cache Management")
    print("=" * 60)

    with FactorEvalClient() as client:
        # Get cache stats
        stats = client.get_cache_stats()
        print(f"Cache stats: {stats}")

        # Evaluate with cache
        print("\nFirst evaluation (cache miss)...")
        result1 = evaluate("Rank($close, 20)", fast=True)
        cached1 = result1.get('cached', False)
        print(f"Cached: {cached1}")

        print("\nSecond evaluation (cache hit)...")
        result2 = evaluate("Rank($close, 20)", fast=True)
        cached2 = result2.get('cached', False)
        print(f"Cached: {cached2}")

        # Get updated stats
        stats_after = client.get_cache_stats()
        print(f"\nCache stats after: {stats_after}")

    print()


# ==================== Main ====================

def main():
    """Run all examples."""
    print("\n")
    print("=" * 60)
    print("FFO Client Usage Examples")
    print("=" * 60)
    print()

    try:
        # Run examples
        example_simple_usage()
        example_batch_parallel()
        example_context_manager()
        example_progress_tracking()
        example_error_handling()
        example_cache_management()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the FFO API server is running on http://127.0.0.1:19330")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
