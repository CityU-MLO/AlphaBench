#!/usr/bin/env python
"""
Factor Evaluation REST API Client - Unified Version

This unified client combines the original client with enhanced features:
- Simple function-style API
- Context manager support for automatic cleanup
- Parallel batch processing with thread pools
- Progress callbacks
- Better error handling and retries

Endpoints:
- GET  /health
- POST /factors/check
- POST /factors/eval
- POST /clear_cache
- GET  /cache_stats

Usage:
    # Simple single evaluation
    >>> from client import FactorEvalClient, evaluate
    >>> result = evaluate("Rank($close, 20)")

    # Parallel batch evaluation
    >>> from client import evaluate_batch
    >>> results = evaluate_batch(
    ...     ["Rank($close, 20)", "Mean($volume, 5)"],
    ...     parallel=True,
    ...     max_workers=8
    ... )

    # Context manager for resource cleanup
    >>> with FactorEvalClient() as client:
    ...     results = client.evaluate_batch_parallel(expressions)
"""

import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:19777"
DEFAULT_TIMEOUT = 120  # seconds
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds


ExpressionInput = Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]]


class FactorEvalClient:
    """
    Unified Factor Evaluation API Client.

    Features:
    - Simple API for single/batch factor evaluation
    - Context manager support for automatic cleanup
    - Parallel batch processing
    - Progress tracking
    - Automatic retries and error handling

    Args:
        base_url: API server URL (default: http://127.0.0.1:19777)
        timeout: Request timeout in seconds (default: 120)

    Example:
        >>> client = FactorEvalClient()
        >>> result = client.evaluate_factor("Rank($close, 20)")
        >>> print(result[0]['metrics']['ic'])

        # Or with context manager
        >>> with FactorEvalClient() as client:
        ...     results = client.evaluate_batch_parallel(expressions)
    """

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        logger.info("FactorEvalClient initialized with base URL: %s", self.base_url)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False

    def close(self):
        """Close connection pools and cleanup resources."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        if self.session:
            self.session.close()

    # ---------------------------
    # low-level request wrapper
    # ---------------------------
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Any]:
        """
        Make an HTTP request with retry logic.

        Returns:
            Parsed JSON (dict/list) or None if failed.
        """
        url = f"{self.base_url}{endpoint}"

        for retry in range(MAX_RETRIES):
            try:
                resp = self.session.request(
                    method=method, url=url, timeout=self.timeout, **kwargs
                )

                # accept 200 only; you can also accept 400 as json if you want
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except Exception:
                        logger.warning("Non-JSON response: %s", resp.text[:300])
                        return {
                            "success": False,
                            "error": "Non-JSON response",
                            "_raw": resp.text,
                        }

                # for debugging: still try parse json
                try:
                    j = resp.json()
                except Exception:
                    j = {"_raw": resp.text}

                logger.warning(
                    "API request failed: %s %s -> %s, body=%s",
                    method,
                    endpoint,
                    resp.status_code,
                    str(j)[:300],
                )

            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Connection error (attempt %d/%d)", retry + 1, MAX_RETRIES
                )
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (retry + 1))

            except requests.exceptions.Timeout:
                logger.warning("Timeout (attempt %d/%d)", retry + 1, MAX_RETRIES)
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

            except Exception as e:
                logger.exception("Unexpected error: %s", e)
                break

        return None

    # ---------------------------
    # helpers
    # ---------------------------
    @staticmethod
    def _ensure_list_result(result: Any) -> List[Dict]:
        """
        Server now returns list for both single/multi,
        but just in case, normalize:
        - dict -> [dict]
        - list -> list
        - None -> []
        """
        if result is None:
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return [
            {"success": False, "error": "Unexpected response type", "result": result}
        ]

    # ---------------------------
    # public API
    # ---------------------------
    def health_check(self) -> bool:
        result = self._make_request("GET", "/health")
        return isinstance(result, dict) and result.get("status") == "healthy"

    def check_factor(
        self,
        expression: ExpressionInput,
        instruments: str = "CSI300",
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict]:
        """
        POST /factors/check
        Returns: always list of results
        """
        payload: Dict[str, Any] = {"expression": expression, "instruments": instruments}
        if start is not None:
            payload["start"] = start
        if end is not None:
            payload["end"] = end
        if timeout is not None:
            payload["timeout"] = int(timeout)

        result = self._make_request("POST", "/factors/check", json=payload)
        if result is None:
            return [
                {
                    "success": False,
                    "name": "",
                    "expression": "" if not isinstance(expression, str) else expression,
                    "error": "Server check failed",
                }
            ]
        return self._ensure_list_result(result)

    def evaluate_factor(
        self,
        expression: ExpressionInput,
        market: str = "csi300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        label: str = "close_return",
        use_cache: bool = True,
        topk: int = 50,
        n_drop: int = 5,
        timeout: int = 120,
        fast: bool = False,
        n_jobs_backtest: int = 4,
        forward_n: int = 1,
    ) -> List[Dict]:
        """
        POST /factors/eval
        Returns: always list
        """
        payload = {
            "expression": expression,
            "start": start_date,
            "end": end_date,
            "market": market,
            "label": label,
            "use_cache": use_cache,
            "topk": int(topk),
            "n_drop": int(n_drop),
            "timeout": int(timeout),
            "fast": bool(fast),
            "n_jobs_backtest": int(n_jobs_backtest),
            "forward_n": int(forward_n),
        }

        result = self._make_request("POST", "/factors/eval", json=payload)
        if result is None:
            # default failure list
            return [
                {
                    "success": False,
                    "error": "API request failed",
                    "metrics": {
                        "ic": 0.0,
                        "rank_ic": 0.0,
                        "ir": 0.0,
                        "icir": 0.0,
                        "rank_icir": 0.0,
                        "turnover": 1.0,
                    },
                }
            ]

        return self._ensure_list_result(result)

    def batch_evaluate_factors(
        self,
        factors: List[Dict[str, str]],
        market: str = "csi300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        label: str = "close_return",
        timeout: int = 300,
        n_jobs: int = 1,
    ) -> Dict:
        """
        If you still keep /factors/batch_eval:
        Body:
        {
          "factors": [{"name":"F1","expression":"..."}, ...],
          ...
        }
        """
        payload = {
            "factors": factors,
            "start": start_date,
            "end": end_date,
            "market": market,
            "label": label,
            "timeout": int(timeout),
            "n_jobs": int(n_jobs),
        }
        result = self._make_request("POST", "/factors/batch_eval", json=payload)
        if result is None:
            return {
                "success": False,
                "error": "Batch API request failed",
                "results": [],
            }
        return result

    def clear_cache(self) -> bool:
        result = self._make_request("POST", "/clear_cache", json={})
        return isinstance(result, dict) and result.get("success", False)

    def get_cache_stats(self) -> Dict:
        result = self._make_request("GET", "/cache_stats")
        return (
            result
            if isinstance(result, dict)
            else {"cache_size": 0, "max_cache_size": 0}
        )

    # ---------------------------
    # Enhanced parallel batch processing
    # ---------------------------

    def evaluate_batch_parallel(
        self,
        expressions: List[Union[str, Dict[str, str]]],
        max_workers: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Evaluate multiple factors in parallel using thread pool.

        Args:
            expressions: List of factor expressions or dicts
            max_workers: Number of parallel workers (default: 8)
            progress_callback: Optional callback(completed, total) for progress
            **kwargs: Additional arguments passed to evaluate_factor

        Returns:
            List of evaluation results

        Example:
            >>> with FactorEvalClient() as client:
            ...     results = client.evaluate_batch_parallel(
            ...         ["Rank($close, 20)", "Mean($volume, 5)"],
            ...         max_workers=8,
            ...         fast=True
            ...     )
        """
        if not self._thread_pool:
            self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Normalize expressions
        factor_list = []
        for expr in expressions:
            if isinstance(expr, str):
                factor_list.append(expr)
            elif isinstance(expr, dict):
                factor_list.append(expr)

        # Submit all tasks
        futures = {}
        for expr in factor_list:
            future = self._thread_pool.submit(
                self.evaluate_factor,
                expression=expr,
                **kwargs
            )
            futures[future] = expr

        # Collect results
        results = []
        completed = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result[0] if isinstance(result, list) and len(result) > 0 else result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(factor_list))

            except Exception as e:
                logger.error(f"Failed to evaluate {futures[future]}: {e}")
                results.append({
                    "success": False,
                    "expression": str(futures[future]),
                    "error": str(e)
                })
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(factor_list))

        return results


# ---------------------------
# global convenience wrappers
# ---------------------------
_global_client: Optional[FactorEvalClient] = None


def get_client(
    base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT
) -> FactorEvalClient:
    global _global_client
    base_url = base_url.rstrip("/")
    if (
        _global_client is None
        or _global_client.base_url != base_url
        or _global_client.timeout != timeout
    ):
        _global_client = FactorEvalClient(base_url=base_url, timeout=timeout)
    return _global_client


def check_factor_via_api(
    expression: ExpressionInput, api_url: str = DEFAULT_API_URL
) -> List[Dict]:
    client = get_client(api_url)
    return client.check_factor(expression)


def evaluate_factor_via_api(
    expression: ExpressionInput,
    market: str = "csi300",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    label: str = "close_return",
    use_cache: bool = True,
    topk: int = 50,
    n_drop: int = 5,
    timeout: int = 120,
    fast: bool = False,
    n_jobs_backtest: int = 4,
    api_url: str = DEFAULT_API_URL,
) -> List[Dict]:
    client = get_client(api_url)
    return client.evaluate_factor(
        expression=expression,
        market=market,
        start_date=start_date,
        end_date=end_date,
        label=label,
        use_cache=use_cache,
        topk=topk,
        n_drop=n_drop,
        timeout=timeout,
        fast=fast,
        n_jobs_backtest=n_jobs_backtest,
    )


def batch_evaluate_factors_via_api(
    expressions: List[ExpressionInput],
    market: str = "csi300",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    label: str = "close_return",
    use_cache: bool = True,
    topk: int = 50,
    n_drop: int = 5,
    timeout: int = 120,
    fast: bool = False,
    n_jobs_backtest: int = 4,
    api_url: str = DEFAULT_API_URL,
    max_workers: int = 8,
) -> List[Dict]:
    """Evaluate multiple factor expressions via the FFO API (parallel).

    This is a convenience wrapper matching the old ``api.factor_eval_client``
    interface so that existing searchers (EA / CoT / ToT) keep working.
    """
    client = get_client(api_url)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                client.evaluate_factor,
                expression=expr,
                market=market,
                start_date=start_date,
                end_date=end_date,
                label=label,
                use_cache=use_cache,
                topk=topk,
                n_drop=n_drop,
                timeout=timeout,
                fast=fast,
                n_jobs_backtest=n_jobs_backtest,
            ): i
            for i, expr in enumerate(expressions)
        }
        results = [None] * len(expressions)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logger.error("batch eval [%d] failed: %s", idx, e)
                results[idx] = {"error": str(e), "expression": expressions[idx]}
    return results


# ---------------------------
# New simple convenience functions
# ---------------------------

def evaluate(expression: str, **kwargs) -> Dict[str, Any]:
    """
    Evaluate a single factor using the global client (simple interface).

    This is the simplest way to evaluate a factor - just call the function
    without managing client instances.

    Args:
        expression: Factor expression (e.g., "Rank($close, 20)")
        **kwargs: Additional evaluation parameters

    Returns:
        Single evaluation result dict (not a list)

    Example:
        >>> from client.factor_eval_client import evaluate
        >>> result = evaluate("Rank($close, 20)", fast=True)
        >>> print(f"IC: {result['metrics']['ic']:.4f}")
    """
    client = get_client()
    results = client.evaluate_factor(expression=expression, **kwargs)
    return results[0] if isinstance(results, list) and len(results) > 0 else results


def evaluate_batch(
    expressions: List[str],
    parallel: bool = False,
    max_workers: int = 8,
    progress: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple factors (sequential or parallel).

    Args:
        expressions: List of factor expressions
        parallel: Use parallel execution (default: False)
        max_workers: Number of parallel workers if parallel=True (default: 8)
        progress: Show progress (default: False)
        **kwargs: Additional evaluation parameters

    Returns:
        List of evaluation results

    Example:
        >>> from client.factor_eval_client import evaluate_batch
        >>> results = evaluate_batch(
        ...     ["Rank($close, 20)", "Mean($volume, 5)"],
        ...     parallel=True,
        ...     fast=True,
        ...     progress=True
        ... )
        >>> for r in results:
        ...     print(f"{r['expression']}: IC={r['metrics']['ic']:.4f}")
    """
    client = get_client()

    if progress:
        def progress_callback(completed, total):
            print(f"\rProgress: {completed}/{total} ({100*completed//total}%)", end="")
            if completed == total:
                print()  # New line when done
    else:
        progress_callback = None

    if parallel:
        return client.evaluate_batch_parallel(
            expressions,
            max_workers=max_workers,
            progress_callback=progress_callback,
            **kwargs
        )
    else:
        results = []
        for i, expr in enumerate(expressions):
            result = evaluate(expr, **kwargs)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(expressions))
        return results
