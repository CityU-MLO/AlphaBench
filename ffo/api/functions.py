"""
FFO Agent-Friendly API Functions

This module provides clean, typed functions for interacting with the FFO
backend. It is designed to be:

- **LLM agent friendly**: every function has a rich docstring, clear type
  annotations, and structured return types so agents can call them correctly.
- **MCP-ready**: return types are dataclasses that serialise to plain dicts.
- **Fail-safe**: all functions return structured results even on error,
  never raise unless explicitly requested.

Typical usage::

    from ffo.api import evaluate_factor, batch_evaluate_factors

    # Evaluate a single factor
    result = evaluate_factor("Rank($close, 20)")
    if result.success:
        print(f"IC = {result.metrics.ic:.4f}")
    else:
        print(f"Error: {result.error}")

    # Evaluate many factors in parallel
    results = batch_evaluate_factors(
        ["Rank($close, 20)", "Mean($volume, 5)"],
        parallel=True,
        fast=True,
    )
    for r in results:
        print(r.expression, r.metrics.ic if r.success else r.error)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import requests

from ffo.config import get_config


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class FactorMetrics:
    """
    Evaluation metrics for a single factor expression.

    Attributes:
        ic:         Mean Information Coefficient (Pearson correlation with
                    next-period returns). Range typically [-1, 1].
                    Absolute value > 0.02 is generally meaningful.
        rank_ic:    Mean Rank IC (Spearman correlation). More robust to outliers.
        icir:       IC Information Ratio = ic / std(daily_ic). Measures consistency.
                    Value > 0.5 is considered good.
        rank_icir:  Rank IC Information Ratio.
        turnover:   Average daily portfolio turnover [0, 1]. Lower is cheaper.
        n_dates:    Number of trading days evaluated.
    """
    ic: float = 0.0
    rank_ic: float = 0.0
    icir: float = 0.0
    rank_icir: float = 0.0
    turnover: float = 1.0
    n_dates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorResult:
    """
    Result of evaluating a single factor expression.

    Attributes:
        success:    True if evaluation succeeded.
        expression: The factor expression that was evaluated.
        metrics:    Evaluation metrics (populated when success=True).
        error:      Error message (populated when success=False).
        cached:     True if this result was served from cache.
        market:     Market universe used.
        start:      Evaluation start date.
        end:        Evaluation end date.
    """
    success: bool
    expression: str
    metrics: FactorMetrics = field(default_factory=FactorMetrics)
    error: Optional[str] = None
    cached: bool = False
    market: str = "csi300"
    start: str = ""
    end: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_api_response(cls, raw: Dict[str, Any]) -> "FactorResult":
        """Build a FactorResult from a raw API response dict."""
        metrics_raw = raw.get("metrics") or {}
        metrics = FactorMetrics(
            ic=float(metrics_raw.get("ic", 0.0)),
            rank_ic=float(metrics_raw.get("rank_ic", 0.0)),
            icir=float(metrics_raw.get("icir") or metrics_raw.get("ir", 0.0)),
            rank_icir=float(metrics_raw.get("rank_icir", 0.0)),
            turnover=float(metrics_raw.get("turnover", 1.0)),
            n_dates=int(metrics_raw.get("n_dates", 0)),
        )
        return cls(
            success=bool(raw.get("success", False)),
            expression=str(raw.get("expression", "")),
            metrics=metrics,
            error=raw.get("error") or raw.get("message"),
            cached=bool(raw.get("cached", False)),
            market=str(raw.get("market", "")),
            start=str(raw.get("start_date", "")),
            end=str(raw.get("end_date", "")),
        )


@dataclass
class SyntaxCheckResult:
    """
    Result of checking factor expression syntax.

    Attributes:
        is_valid:   True if the expression is syntactically valid and can be
                    computed by the Qlib engine.
        expression: The expression that was checked.
        error:      Error message if invalid.
        name:       Parsed factor name (if provided).
    """
    is_valid: bool
    expression: str
    error: Optional[str] = None
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CacheStats:
    """
    Statistics about the evaluation cache.

    Attributes:
        cache_size:     Number of entries currently in cache.
        max_cache_size: Maximum allowed cache entries.
        hit_rate:       Cache hit rate (0.0–1.0) if tracked.
    """
    cache_size: int = 0
    max_cache_size: int = 0
    hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServerHealth:
    """
    Health status of the FFO backend server.

    Attributes:
        is_healthy: True if the server is reachable and healthy.
        status:     Status string from the server.
        latency_ms: Response latency in milliseconds.
        cache:      Cache statistics embedded in the health response.
        error:      Error message if unhealthy.
    """
    is_healthy: bool
    status: str = "unknown"
    latency_ms: float = 0.0
    cache: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Internal HTTP helper ──────────────────────────────────────────────────────


def _post(endpoint: str, payload: Dict[str, Any], timeout: int = 120) -> Optional[Any]:
    """POST to the FFO backend. Returns parsed JSON or None on failure."""
    cfg = get_config()
    url = f"{cfg.backend_url}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if not r.ok:
            # Try to surface the actual JSON error body before raising
            try:
                return r.json()
            except Exception:
                pass
            r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


def _get(endpoint: str, timeout: int = 10) -> Optional[Any]:
    cfg = get_config()
    url = f"{cfg.backend_url}{endpoint}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


# ── Public API Functions ──────────────────────────────────────────────────────


def evaluate_factor(
    expression: str,
    market: str = "csi300",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    label: str = "close_return",
    fast: bool = True,
    use_cache: bool = True,
    topk: int = 50,
    n_drop: int = 5,
    timeout: int = 180,
    forward_n: int = 1,
) -> FactorResult:
    """
    Evaluate a quantitative alpha factor expression.

    Computes IC (Information Coefficient) between the factor signal and
    future stock returns over the given date range and market universe.

    Args:
        expression: Qlib-style factor expression.
                    Examples:
                      - "Rank($close, 20)"
                      - "Mean($volume, 5) / Std($volume, 20)"
                      - "Corr($close, $volume, 10)"
                      - "$close / Delay($close, 1) - 1"
        market:     Market universe. Choices: "csi300", "csi500", "csi1000".
                    Default: "csi300" (CSI 300 index constituent stocks).
        start:      Evaluation start date (inclusive), format "YYYY-MM-DD".
        end:        Evaluation end date (inclusive), format "YYYY-MM-DD".
        label:      Return label for IC computation. Default: "close_return".
        fast:       If True, skip portfolio backtest and return IC metrics only
                    (5-10x faster). Default: True.
        use_cache:  Use persistent cache for repeated evaluations. Default: True.
        topk:       Top-K stocks selected for the long portfolio. Default: 50.
        n_drop:     Stocks dropped (rebalanced) per period. Default: 5.
        timeout:    Per-factor timeout in seconds. Default: 180.

    Returns:
        FactorResult with success flag, metrics (IC, Rank IC, ICIR, …),
        and optional error message.

    Example::

        result = evaluate_factor(
            "Rank($close, 20)",
            market="csi300",
            start="2023-01-01",
            end="2024-01-01",
            fast=True,
        )
        if result.success:
            print(f"IC={result.metrics.ic:.4f}  Rank_IC={result.metrics.rank_ic:.4f}")
    """
    cfg = get_config()
    payload = {
        "expression": expression,
        "market": market,
        "start": start,
        "end": end,
        "label": label,
        "fast": fast,
        "use_cache": use_cache,
        "topk": topk,
        "n_drop": n_drop,
        "timeout": timeout,
        "forward_n": forward_n,
    }
    raw = _post("/factors/eval", payload, timeout=timeout + 30)

    if raw is None:
        return FactorResult(
            success=False,
            expression=expression,
            error="No response from FFO backend. Is it running? Try: ppo start backend",
        )

    # API returns a list even for single expressions
    items = raw if isinstance(raw, list) else [raw]
    item = items[0] if items else {}

    return FactorResult.from_api_response({**item, "expression": expression})


def batch_evaluate_factors(
    expressions: List[str],
    market: str = "csi300",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    label: str = "close_return",
    fast: bool = True,
    use_cache: bool = True,
    topk: int = 50,
    n_drop: int = 5,
    timeout: int = 180,
    parallel: bool = True,
    max_workers: int = 8,
    forward_n: int = 1,
    progress: bool = False,
) -> List[FactorResult]:
    """
    Evaluate multiple factor expressions efficiently.

    When ``fast=True`` (default), all expressions are sent in a single
    HTTP request. The server loads Qlib data once and computes IC/RankIC
    for all factors in one pass — typically 5-8x faster than per-factor
    evaluation.

    When ``fast=False``, factors are evaluated individually (with optional
    parallel threads) so that portfolio backtests can run per factor.

    Args:
        expressions:  List of Qlib-style factor expressions to evaluate.
        market:       Market universe. Default: "csi300".
        start:        Evaluation start date. Default: "2023-01-01".
        end:          Evaluation end date. Default: "2024-01-01".
        label:        Return label. Default: "close_return".
        fast:         Fast mode (IC only). Default: True.
        use_cache:    Use cache. Default: True.
        topk:         Long portfolio size. Default: 50.
        n_drop:       Rebalance drop count. Default: 5.
        timeout:      Timeout in seconds. For fast=True this is the total
                      batch timeout; for fast=False it is per-factor.
                      Default: 180.
        parallel:     Evaluate in parallel using thread pool (only used
                      when fast=False). Default: True.
        max_workers:  Number of parallel threads (fast=False only).
                      Default: 8.
        progress:     Print progress to stdout. Default: False.

    Returns:
        List of FactorResult, one per expression (same order as input).

    Example::

        results = batch_evaluate_factors(
            ["Rank($close, 20)", "Mean($volume, 5)", "Corr($close, $volume, 10)"],
            fast=True,
            progress=True,
        )
        for r in sorted(results, key=lambda x: -x.metrics.rank_ic):
            print(f"{r.expression[:40]:40s}  IC={r.metrics.ic:.4f}")
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── Fast batch path: single HTTP request, server-side batch eval ──────
    if fast and len(expressions) > 1:
        payload = {
            "expression": expressions,
            "market": market,
            "start": start,
            "end": end,
            "label": label,
            "fast": True,
            "use_cache": use_cache,
            "topk": topk,
            "n_drop": n_drop,
            "timeout": timeout,
            "forward_n": forward_n,
        }
        raw = _post("/factors/eval", payload, timeout=timeout + 60)

        if raw is None:
            return [
                FactorResult(
                    success=False,
                    expression=expr,
                    error="No response from FFO backend. Is it running? Try: ppo start backend",
                )
                for expr in expressions
            ]

        items = raw if isinstance(raw, list) else [raw]
        results: List[FactorResult] = []
        for i, expr in enumerate(expressions):
            item = items[i] if i < len(items) else {}
            results.append(
                FactorResult.from_api_response({**item, "expression": expr})
            )
        return results

    # ── Per-factor path: sequential or parallel HTTP requests ─────────────
    kwargs = dict(
        market=market,
        start=start,
        end=end,
        label=label,
        fast=fast,
        use_cache=use_cache,
        topk=topk,
        n_drop=n_drop,
        timeout=timeout,
        forward_n=forward_n,
    )

    if not parallel or len(expressions) == 1:
        results = []
        for i, expr in enumerate(expressions):
            results.append(evaluate_factor(expression=expr, **kwargs))
            if progress:
                print(f"\r[{i+1}/{len(expressions)}] {expr[:40]}", end="", flush=True)
        if progress:
            print()
        return results

    # Parallel path (fast=False with backtest)
    futures = {}
    results: List[Optional[FactorResult]] = [None] * len(expressions)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, expr in enumerate(expressions):
            fut = pool.submit(evaluate_factor, expression=expr, **kwargs)
            futures[fut] = i

        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = FactorResult(
                    success=False,
                    expression=expressions[idx],
                    error=str(e),
                )
            completed += 1
            if progress:
                pct = 100 * completed // len(expressions)
                print(f"\rProgress: {completed}/{len(expressions)} ({pct}%)", end="", flush=True)

    if progress:
        print()

    return [r for r in results if r is not None]


def check_factor(
    expression: str,
    market: str = "csi300",
    start: str = "2020-01-01",
    end: str = "2020-01-06",
) -> SyntaxCheckResult:
    """
    Validate a factor expression without running a full evaluation.

    This is a fast syntax and executability check. Use it to validate
    user-provided expressions before running expensive evaluations.

    Args:
        expression: Factor expression to validate.
        market:     Market to use for validation data. Default: "csi300".
        start:      Short date window start (for faster checking).
        end:        Short date window end.

    Returns:
        SyntaxCheckResult with is_valid flag and optional error message.

    Example::

        result = check_factor("Rank($close, 20)")
        # result.is_valid → True

        result = check_factor("InvalidOp($close, 20)")
        # result.is_valid → False
        # result.error    → "Unknown operator: InvalidOp"
    """
    payload = {
        "expression": expression,
        "instruments": market.upper(),
        "start": start,
        "end": end,
    }
    raw = _post("/factors/check", payload, timeout=60)

    if raw is None:
        return SyntaxCheckResult(
            is_valid=False,
            expression=expression,
            error="No response from FFO backend. Is it running? Try: ppo start backend",
        )

    items = raw if isinstance(raw, list) else [raw]
    item = items[0] if items else {}

    return SyntaxCheckResult(
        is_valid=bool(item.get("is_valid", item.get("success", False))),
        expression=expression,
        error=item.get("error") or item.get("message"),
        name=item.get("name", ""),
    )


def batch_check_factors(
    expressions: List[str],
    market: str = "csi300",
    start: str = "2020-01-01",
    end: str = "2020-01-15",
    timeout: int = 120,
) -> List[SyntaxCheckResult]:
    """
    Validate multiple factor expressions in a single request.

    Sends all expressions to the server in one HTTP call. The server
    checks each expression against Qlib in a short date window.

    Args:
        expressions: List of factor expressions to validate.
        market:      Market to use for validation data. Default: "csi300".
        start:       Short date window start. Default: "2020-01-01".
        end:         Short date window end. Default: "2020-01-15".
        timeout:     Request timeout in seconds. Default: 120.

    Returns:
        List of SyntaxCheckResult, one per expression (same order as input).

    Example::

        results = batch_check_factors([
            "Rank($close, 20)",
            "InvalidOp($close)",
            "Mean($volume, 5)",
        ])
        for r in results:
            status = "VALID" if r.is_valid else f"INVALID: {r.error}"
            print(f"{r.expression:40s}  {status}")
    """
    payload = {
        "expression": expressions,
        "instruments": market.upper(),
        "start": start,
        "end": end,
    }
    raw = _post("/factors/check", payload, timeout=timeout)

    if raw is None:
        return [
            SyntaxCheckResult(
                is_valid=False,
                expression=expr,
                error="No response from FFO backend. Is it running? Try: ppo start backend",
            )
            for expr in expressions
        ]

    items = raw if isinstance(raw, list) else [raw]

    results: List[SyntaxCheckResult] = []
    for i, expr in enumerate(expressions):
        item = items[i] if i < len(items) else {}
        results.append(
            SyntaxCheckResult(
                is_valid=bool(item.get("is_valid", item.get("success", False))),
                expression=expr,
                error=item.get("error") or item.get("error_message"),
                name=item.get("name", ""),
            )
        )

    return results


def get_cache_stats() -> CacheStats:
    """
    Retrieve cache statistics from the FFO backend.

    Returns:
        CacheStats with current cache size and capacity.

    Example::

        stats = get_cache_stats()
        print(f"Cache: {stats.cache_size}/{stats.max_cache_size}")
    """
    raw = _get("/cache_stats")

    if not isinstance(raw, dict):
        return CacheStats()

    # Handle both top-level and nested formats
    data = raw.get("cache") or raw
    return CacheStats(
        cache_size=int(data.get("cache_size", data.get("size", 0))),
        max_cache_size=int(data.get("max_cache_size", data.get("max_size", 0))),
        hit_rate=float(data.get("hit_rate", 0.0)),
    )


def clear_cache() -> bool:
    """
    Clear all cached factor evaluations from the FFO backend.

    Returns:
        True if cache was cleared successfully, False otherwise.

    Example::

        ok = clear_cache()
        # ok → True
    """
    raw = _post("/clear_cache", {}, timeout=30)
    if not isinstance(raw, dict):
        return False
    return bool(raw.get("success", False))


def server_health() -> ServerHealth:
    """
    Check the health of the FFO backend server.

    Returns a structured health status including latency and cache info.
    Use this to verify the server is running before batch operations.

    Returns:
        ServerHealth with is_healthy flag, latency, and cache stats.

    Example::

        health = server_health()
        if not health.is_healthy:
            print("Backend is down! Run: ppo start backend")
        else:
            print(f"Backend OK (latency: {health.latency_ms:.0f}ms)")
    """
    cfg = get_config()
    url = f"{cfg.backend_url}/health"
    t0 = time.time()
    try:
        r = requests.get(url, timeout=5)
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            return ServerHealth(
                is_healthy=data.get("status") == "healthy",
                status=data.get("status", "unknown"),
                latency_ms=latency_ms,
                cache=data.get("cache", {}),
            )
        return ServerHealth(
            is_healthy=False,
            status=f"HTTP {r.status_code}",
            latency_ms=latency_ms,
            error=r.text[:200],
        )
    except requests.RequestException as e:
        return ServerHealth(
            is_healthy=False,
            status="unreachable",
            error=str(e),
        )
