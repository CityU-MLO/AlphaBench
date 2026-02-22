"""
FFO — Factor Feature Oracle
===========================
High-performance factor evaluation service for quantitative trading.

Quick start::

    # Install and run the CLI
    pip install -e .
    ppo start backend      # Start the API server
    ppo status             # Check status
    ppo eval "Rank($close, 20)"   # Quick evaluation

    # Python API
    from ffo.api import evaluate_factor, batch_evaluate_factors

    result = evaluate_factor("Rank($close, 20)")
    print(result.metrics.ic)

    # MCP server (for LLM agents)
    ppo start mcp

Sub-packages
------------
ffo.api      Agent-friendly typed API functions
ffo.config   Unified configuration management
ffo.cli      CLI entry points (ppo command)
ffo.mcp      MCP server for LLM agent integration
ffo.client   Low-level HTTP client (FactorEvalClient)
"""

from ffo.api import (
    evaluate_factor,
    batch_evaluate_factors,
    check_factor,
    get_cache_stats,
    clear_cache,
    server_health,
    FactorResult,
    FactorMetrics,
    SyntaxCheckResult,
    CacheStats,
    ServerHealth,
)
from ffo.config import get_config

__version__ = "0.1.0"

__all__ = [
    # High-level API (recommended entry points)
    "evaluate_factor",
    "batch_evaluate_factors",
    "check_factor",
    "get_cache_stats",
    "clear_cache",
    "server_health",
    # Return types
    "FactorResult",
    "FactorMetrics",
    "SyntaxCheckResult",
    "CacheStats",
    "ServerHealth",
    # Config
    "get_config",
    # Version
    "__version__",
]
