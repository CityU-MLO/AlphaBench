"""
FFO Agent-Friendly API

Clean, typed, well-documented functions designed for direct use by
LLM agents, scripts, and the MCP server.

All functions communicate with the running FFO backend via HTTP.
Start the backend with ``ppo start backend`` before calling these.

Quick start::

    from ffo.api import evaluate_factor, batch_evaluate_factors, check_factor

    result = evaluate_factor("Rank($close, 20)")
    print(result.metrics.ic)

    results = batch_evaluate_factors(
        ["Rank($close, 20)", "Mean($volume, 5)"],
        parallel=True,
    )
"""

from .functions import (
    # Data classes (returned by functions)
    FactorMetrics,
    FactorResult,
    SyntaxCheckResult,
    CacheStats,
    ServerHealth,
    # Core functions
    evaluate_factor,
    batch_evaluate_factors,
    check_factor,
    batch_check_factors,
    # Cache management
    get_cache_stats,
    clear_cache,
    # Server health
    server_health,
)

__all__ = [
    # Data classes
    "FactorMetrics",
    "FactorResult",
    "SyntaxCheckResult",
    "CacheStats",
    "ServerHealth",
    # Functions
    "evaluate_factor",
    "batch_evaluate_factors",
    "check_factor",
    "batch_check_factors",
    "get_cache_stats",
    "clear_cache",
    "server_health",
]
