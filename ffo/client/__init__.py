"""
FFO Client Module - Unified Factor Evaluation Client

This module provides a unified interface for interacting with the FFO API:

1. **FactorEvalClient** - Full-featured client class with:
   - Basic evaluation methods
   - Parallel batch processing
   - Context manager support
   - Progress callbacks

2. **Convenience Functions** - Simple function-style API:
   - evaluate() - Evaluate a single factor
   - evaluate_batch() - Evaluate multiple factors with optional parallelization

Usage Examples:

    # Simple function-style usage (recommended for quick tasks)
    >>> from client import evaluate, evaluate_batch
    >>>
    >>> result = evaluate("Rank($close, 20)", fast=True)
    >>> print(f"IC: {result['metrics']['ic']:.4f}")
    >>>
    >>> results = evaluate_batch(
    ...     ["Rank($close, 20)", "Mean($volume, 5)"],
    ...     parallel=True,
    ...     max_workers=8
    ... )

    # Client class usage (recommended for production)
    >>> from client import FactorEvalClient
    >>>
    >>> with FactorEvalClient() as client:
    ...     results = client.evaluate_batch_parallel(
    ...         ["Rank($close, 20)", "Mean($volume, 5)"],
    ...         max_workers=8,
    ...         fast=True
    ...     )
"""

from .factor_eval_client import (
    FactorEvalClient,
    ExpressionInput,
    get_client,
    evaluate,
    evaluate_batch,
    check_factor_via_api,
    evaluate_factor_via_api,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT,
)

__all__ = [
    # Main client class
    "FactorEvalClient",

    # Types
    "ExpressionInput",

    # Client factory
    "get_client",

    # Simple convenience functions (recommended)
    "evaluate",
    "evaluate_batch",

    # Legacy/advanced functions
    "check_factor_via_api",
    "evaluate_factor_via_api",

    # Constants
    "DEFAULT_API_URL",
    "DEFAULT_TIMEOUT",
]
