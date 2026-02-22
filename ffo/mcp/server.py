#!/usr/bin/env python
"""
FFO MCP Server

Exposes the FFO factor evaluation API as MCP (Model Context Protocol) tools,
making it directly usable by Claude, GPT-4, and other MCP-compatible agents.

Tools exposed:
  evaluate_factor         - Evaluate a single alpha factor
  batch_evaluate_factors  - Evaluate many factors (with parallelism)
  check_factor_syntax     - Validate factor expression syntax
  get_server_health       - Check if the FFO backend is running
  get_cache_stats         - Show evaluation cache statistics
  clear_cache             - Clear the evaluation cache

Usage (stdio transport, for Claude Desktop / agent SDK)::

    ppo start mcp
    # or directly:
    python -m ffo.mcp.server

Usage (SSE transport, for web agents)::

    ppo start mcp --transport sse --port 8765

Claude Desktop config (~/.claude/claude_desktop_config.json)::

    {
      "mcpServers": {
        "ffo": {
          "command": "ppo",
          "args": ["start", "mcp"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# ── Create MCP server ─────────────────────────────────────────────────────────

mcp = FastMCP(
    name="FFO Factor Evaluation",
    instructions=(
        "FFO (Factor Feature Oracle) is a quantitative finance tool for evaluating "
        "alpha factor expressions on Chinese stock markets (A-shares). "
        "Factors are Qlib-style expressions that combine price/volume fields: "
        "$close (close price), $open, $high, $low, $volume, $vwap, $amount. "
        "Common operators: Rank(), Mean(), Std(), Corr(), Delay(), Delta(), "
        "Max(), Min(), Sum(), If(), Sign(). "
        "Evaluate factors to get IC (Information Coefficient) metrics that "
        "measure predictive power for future stock returns. "
        "Higher absolute IC / Rank IC = stronger factor signal."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def evaluate_factor(
    expression: str,
    market: str = "csi300",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    fast: bool = True,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a quantitative alpha factor expression and return IC metrics.

    Computes the Information Coefficient (IC) between the factor signal and
    next-period stock returns. IC measures the factor's predictive power.

    Args:
        expression: Qlib-style factor formula. Use stock price/volume fields
                    prefixed with $. Available fields: $close, $open, $high,
                    $low, $volume, $vwap, $amount.
                    Available operators: Rank(x, n), Mean(x, n), Std(x, n),
                    Corr(x, y, n), Delay(x, n), Delta(x, n), Max(x, n),
                    Min(x, n), Sum(x, n), Abs(x), Log(x), Sign(x), If(c,a,b).
                    Examples:
                      "Rank($close, 20)"
                      "Mean($volume, 5) / Std($volume, 20)"
                      "$close / Delay($close, 1) - 1"
                      "Corr($close, $volume, 10)"
        market:     Stock universe. One of: "csi300" (300 large caps),
                    "csi500" (500 mid caps), "csi1000" (1000 small caps).
        start:      Evaluation period start date (YYYY-MM-DD).
        end:        Evaluation period end date (YYYY-MM-DD).
        fast:       If True (default), compute IC metrics only (fast, ~5s).
                    If False, also run portfolio backtest (slow, ~60s).
        use_cache:  Use cached result if available. Recommended: True.

    Returns:
        dict with keys:
          success (bool): Whether evaluation succeeded.
          expression (str): The evaluated expression.
          metrics (dict): IC metrics:
            - ic (float): Mean IC, range ~[-0.1, 0.1]. |IC| > 0.02 is useful.
            - rank_ic (float): Mean Rank IC, more robust than IC.
            - icir (float): IC / std(IC). Value > 0.5 is good.
            - rank_icir (float): Rank IC / std(Rank IC).
            - turnover (float): Daily portfolio turnover [0,1].
            - n_dates (int): Number of trading days evaluated.
          error (str | None): Error message if success=False.
          cached (bool): Whether result came from cache.
    """
    from ffo.api import evaluate_factor as _eval

    result = _eval(
        expression=expression,
        market=market,
        start=start,
        end=end,
        fast=fast,
        use_cache=use_cache,
    )
    return result.to_dict()


@mcp.tool()
def batch_evaluate_factors(
    expressions: List[str],
    market: str = "csi300",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    fast: bool = True,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple alpha factor expressions in parallel.

    More efficient than calling evaluate_factor() repeatedly.
    Results are returned in the same order as the input expressions.

    Args:
        expressions: List of factor expressions to evaluate. Each element is
                     a Qlib-style expression string (see evaluate_factor docs).
                     Example: ["Rank($close, 20)", "Mean($volume, 5)"]
        market:      Stock universe: "csi300", "csi500", or "csi1000".
        start:       Evaluation period start (YYYY-MM-DD).
        end:         Evaluation period end (YYYY-MM-DD).
        fast:        Fast mode (IC only). Recommended for bulk evaluation.
        max_workers: Parallel threads. Increase for larger batches.

    Returns:
        List of result dicts (same schema as evaluate_factor). Results are
        sorted to match the input order.

    Example:
        Evaluate a basket of momentum factors and rank by IC:
        expressions = [
            "Rank($close, 5)",
            "Rank($close, 10)",
            "Rank($close, 20)",
            "Mean($volume, 5) / Mean($volume, 20)",
        ]
    """
    from ffo.api import batch_evaluate_factors as _batch

    results = _batch(
        expressions=expressions,
        market=market,
        start=start,
        end=end,
        fast=fast,
        parallel=True,
        max_workers=max_workers,
    )
    return [r.to_dict() for r in results]


@mcp.tool()
def check_factor_syntax(
    expression: str,
    market: str = "csi300",
) -> Dict[str, Any]:
    """
    Validate a factor expression without running a full evaluation.

    Fast check that verifies the expression is parseable and executable
    by the Qlib engine. Use this before expensive evaluations.

    Args:
        expression: Factor expression to validate.
        market:     Market to use for validation data.

    Returns:
        dict with keys:
          is_valid (bool): True if expression is valid and executable.
          expression (str): The checked expression.
          error (str | None): Error description if invalid.
          name (str): Parsed factor name.

    Example:
        check_factor_syntax("Rank($close, 20)")
        # → {"is_valid": True, "expression": "Rank($close, 20)", ...}

        check_factor_syntax("InvalidOp($close)")
        # → {"is_valid": False, "error": "Unknown operator: InvalidOp", ...}
    """
    from ffo.api import check_factor as _check

    result = _check(expression=expression, market=market)
    return result.to_dict()


@mcp.tool()
def get_server_health() -> Dict[str, Any]:
    """
    Check if the FFO backend server is running and healthy.

    Use this before calling evaluation tools to verify the backend is up.
    If unhealthy, start it with: ppo start backend

    Returns:
        dict with keys:
          is_healthy (bool): True if server is reachable and healthy.
          status (str): Server status string (e.g., "healthy", "unreachable").
          latency_ms (float): Response latency in milliseconds.
          cache (dict): Cache statistics from the server.
          error (str | None): Error message if unhealthy.
    """
    from ffo.api import server_health as _health

    result = _health()
    return result.to_dict()


@mcp.tool()
def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the factor evaluation cache.

    The cache stores previously computed factor evaluations to avoid
    redundant computation. Repeated calls with the same parameters
    return instantly from cache.

    Returns:
        dict with keys:
          cache_size (int): Number of cached entries.
          max_cache_size (int): Maximum cache capacity.
          hit_rate (float): Cache hit rate (0.0–1.0).
    """
    from ffo.api import get_cache_stats as _stats

    result = _stats()
    return result.to_dict()


@mcp.tool()
def clear_cache() -> Dict[str, Any]:
    """
    Clear all cached factor evaluations.

    Use this when you want to force fresh evaluation (e.g., after data update).

    Returns:
        dict with keys:
          success (bool): True if cache was cleared.
          message (str): Human-readable result message.
    """
    from ffo.api import clear_cache as _clear

    ok = _clear()
    return {
        "success": ok,
        "message": "Cache cleared successfully." if ok else "Failed to clear cache.",
    }


# ── Resources ─────────────────────────────────────────────────────────────────


@mcp.resource("ffo://operators")
def list_operators() -> str:
    """List all supported Qlib operators for factor expression building."""
    return """# Supported Qlib Operators for Factor Expressions

## Price/Volume Fields (prefix with $)
- $close   - Daily closing price
- $open    - Daily opening price
- $high    - Daily high price
- $low     - Daily low price
- $volume  - Daily trading volume
- $vwap    - Volume-weighted average price
- $amount  - Daily trading amount (price × volume)

## Time-Series Operators (x=signal, n=lookback window)
- Rank(x, n)       - Cross-sectional rank at each date (normalized 0-1)
- Mean(x, n)       - Rolling n-day mean
- Std(x, n)        - Rolling n-day standard deviation
- Sum(x, n)        - Rolling n-day sum
- Max(x, n)        - Rolling n-day maximum
- Min(x, n)        - Rolling n-day minimum
- Delay(x, n)      - Value n days ago (lag)
- Delta(x, n)      - x - Delay(x, n)  (n-day change)
- Corr(x, y, n)    - Rolling n-day Pearson correlation

## Element-wise Operators
- Abs(x)           - Absolute value
- Log(x)           - Natural logarithm
- Sign(x)          - Sign (-1, 0, 1)
- Power(x, n)      - x raised to power n
- If(cond, a, b)   - Conditional: a if cond else b

## Arithmetic
- +, -, *, /       - Standard arithmetic

## Example Expressions
- Rank($close, 20)                          # 20-day price momentum rank
- Mean($volume, 5) / Mean($volume, 20)     # Short/long volume ratio
- Corr($close, $volume, 10)                # Price-volume correlation
- Delta($close, 1) / Delay($close, 1)      # 1-day return
- Rank(Corr($vwap, $volume, 10), 20)       # Rank of price-volume correlation
"""


@mcp.resource("ffo://markets")
def list_markets() -> str:
    """List supported market universes."""
    return """# Supported Market Universes

## csi300 (default)
- CSI 300 Index constituent stocks
- ~300 large-cap A-share stocks
- Most liquid, lowest transaction cost

## csi500
- CSI 500 Index constituent stocks
- ~500 mid-cap A-share stocks
- Higher alpha potential than csi300

## csi1000
- CSI 1000 Index constituent stocks
- ~1000 small-cap A-share stocks
- Highest alpha potential, highest transaction cost

## Tips
- Start with csi300 for baseline evaluation
- Factors often have different IC profiles across universes
- Transaction costs matter more for smaller-cap universes
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FFO MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port (only used with --transport sse or streamable-http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="streamable-http", port=args.port)


if __name__ == "__main__":
    main()
