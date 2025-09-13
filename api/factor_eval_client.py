#!/usr/bin/env python
"""
Client for Factor Evaluation REST API
This module provides functions to interact with the factor evaluation API.
"""

import requests
import json
import logging
from typing import Dict, List, Optional
from urllib.parse import quote
import time

logger = logging.getLogger(__name__)

# Default API configuration
DEFAULT_API_URL = "http://localhost:9889"
DEFAULT_TIMEOUT = 120  # seconds
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds


class FactorEvalClient:
    """Client for Factor Evaluation API."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response JSON or None if failed
        """
        url = f"{self.base_url}{endpoint}"

        for retry in range(MAX_RETRIES):
            try:
                response = self.session.request(
                    method=method, url=url, timeout=self.timeout, **kwargs
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {retry + 1}/{MAX_RETRIES})")
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (retry + 1))

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {retry + 1}/{MAX_RETRIES})")
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        return None

    def health_check(self) -> bool:
        """
        Check if the API server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        result = self._make_request("GET", "/health")
        return result is not None and result.get("status") == "healthy"

    def check_factor(self, expr) -> Dict:
        """
        Check if a factor expression is valid.
        
        Args:
            expr: Factor expression to check
            
        Returns:
            Dictionary with success status and error message if any
        """
        result = self._make_request("POST", "/check", json={"expression": expr})
        return result if result else {"success": False, "error": "Server Check failed"}

    def evaluate_factor(
        self,
        expr: str,
        market: str = "csi300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        label: str = "close_return",
        use_cache: bool = True,
    ) -> Dict:
        """
        Evaluate a single factor expression.
        
        Args:
            expr: Factor expression
            market: Market identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            label: Label type for evaluation
            use_cache: Whether to use cache
            
        Returns:
            Evaluation results dictionary
        """
        # Use POST method for complex expressions
        if len(expr) > 1000:
            # Use POST for long expressions
            result = self._make_request(
                "POST",
                "/eval",
                json={
                    "expression": expr,
                    "start": start_date,
                    "end": end_date,
                    "market": market,
                    "label": label,
                    "use_cache": use_cache,
                },
            )
        else:
            # Use GET for short expressions
            params = {
                "expression": expr,
                "start": start_date,
                "end": end_date,
                "market": market,
                "label": label,
                "use_cache": str(use_cache).lower(),
            }
            result = self._make_request("GET", "/eval", params=params)

        if result is None:
            # Return default failure response
            return {
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

        return result

    def batch_evaluate_factors(
        self,
        factors: List[Dict],
        market: str = "csi300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        label: str = "close_return",
        use_cache: bool = True,
    ) -> Dict:
        """
        Evaluate multiple factors in batch.
        
        Args:
            factors: List of factor dictionaries with 'name' and 'expression' keys
            market: Market identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            label: Label type for evaluation
            use_cache: Whether to use cache
            
        Returns:
            Batch evaluation results
        """
        result = self._make_request(
            "POST",
            "/batch_eval",
            json={
                "factors": factors,
                "start": start_date,
                "end": end_date,
                "market": market,
                "label": label,
                "use_cache": use_cache,
            },
        )

        if result is None:
            # Return default failure response
            return {
                "success": False,
                "error": "Batch API request failed",
                "results": [],
            }

        return result

    def clear_cache(self) -> bool:
        """
        Clear the server's evaluation cache.
        
        Returns:
            True if successful, False otherwise
        """
        result = self._make_request("POST", "/clear_cache")
        return result is not None and result.get("success", False)

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics from the server.
        
        Returns:
            Cache statistics dictionary
        """
        result = self._make_request("GET", "/cache_stats")
        return result if result else {"cache_size": 0, "max_cache_size": 0}


# Global client instance
_global_client = None


def get_client(base_url: str = DEFAULT_API_URL) -> FactorEvalClient:
    """
    Get or create a global API client instance.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        FactorEvalClient instance
    """
    global _global_client
    if _global_client is None or _global_client.base_url != base_url.rstrip("/"):
        _global_client = FactorEvalClient(base_url)
    return _global_client


def check_factor_via_api(expr: str, api_url: str = DEFAULT_API_URL) -> Dict:
    """
    Convenience function to check a factor expression via API.
    
    Args:
        expr: Factor expression to check
        api_url: API server URL
        
    Returns:
        Check result dictionary with success status and error message if any
    """
    client = get_client(api_url)
    return client.check_factor(expr)


def evaluate_factor_via_api(
    expr: str,
    market: str = "csi300",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    label: str = "close_return",
    use_cache: bool = True,
    api_url: str = DEFAULT_API_URL,
) -> Dict:
    """
    Convenience function to evaluate a factor via API.
    
    Args:
        expr: Factor expression
        market: Market identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        label: Label type for evaluation
        use_cache: Whether to use cache
        api_url: API server URL
        
    Returns:
        Evaluation results dictionary with metrics
    """
    client = get_client(api_url)
    return client.evaluate_factor(
        expr=expr,
        market=market,
        start_date=start_date,
        end_date=end_date,
        label=label,
        use_cache=use_cache,
    )


def batch_evaluate_factors_via_api(
    factors: List[Dict],
    market: str = "csi300",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    label: str = "close_return",
    use_cache: bool = True,
    api_url: str = DEFAULT_API_URL,
) -> List[Dict]:
    """
    Convenience function to evaluate multiple factors via API.
    
    Args:
        factors: List of factor dictionaries with 'name' and 'expression' keys
        market: Market identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        label: Label type for evaluation
        use_cache: Whether to use cache
        api_url: API server URL
        
    Returns:
        List of evaluation results
    """
    client = get_client(api_url)
    result = client.batch_evaluate_factors(
        factors=factors,
        market=market,
        start_date=start_date,
        end_date=end_date,
        label=label,
        use_cache=use_cache,
    )

    if result.get("success", False):
        return result.get("results", [])
    else:
        # Return empty results for each factor
        return [{"success": False, "metrics": {"ic": 0, "rank_ic": 0}} for _ in factors]


if __name__ == "__main__":
    # Test the client
    import sys

    # Check if API server is running
    client = FactorEvalClient()
    if not client.health_check():
        print("API server is not running. Please start it first:")
        print("python api/factor_eval_api.py")
        sys.exit(1)

    print("API server is healthy!")

    from agent.qlib_contrib.qlib_expr_parsing import FactorParser, print_tree
    
    fs = FactorParser()

    # Test single factor evaluation
    test_expr = "Less($volume, Quantile($volume, 21, 0.8))"
    ast = fs.parse(test_expr) 
    copx = fs.get_complexity(ast)
    print_tree(ast)
    print("Complex:", copx)
    
    # Check factor if wrong
    print(check_factor_via_api("Les($volume, Quantile($volume, 21, 0.8))"))
    print(check_factor_via_api("Less($volume, Quantile($volume, 21))"))
    result = check_factor_via_api(test_expr)
    print("Check:", result)
    print(f"\nEvaluating test factor: {test_expr}")

    result = evaluate_factor_via_api(test_expr)
    if result["success"]:
        print(f"Evaluation successful!")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
    else:
        print(f"Evaluation failed: {result.get('error', 'Unknown error')}")

    # Test batch evaluation
    test_factors = [
        {"name": "factor1", "expression": "Rank($close, 20)"},
        {"name": "factor2", "expression": "Mean($volume, 10)"},
        {"name": "factor3", "expression": "Corr($close, $volume, 30)"},
    ]

    print(f"\nBatch evaluating {len(test_factors)} factors...")
    results = batch_evaluate_factors_via_api(test_factors)

    for i, result in enumerate(results):
        if result.get("success", False):
            print(
                f"{test_factors[i]['name']}: Rank IC = {result['metrics']['rank_ic']:.4f}"
            )
        else:
            print(
                f"{test_factors[i]['name']}: Failed - {result.get('error', 'Unknown error')}"
            )
