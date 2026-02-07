"""
Evaluator agent for evaluating factors via FFO API.

Follows EvoAlpha's FactorEvaluator pattern:
- Calls FFO server /factors/eval endpoint
- Supports fast (IC only) and full (IC + portfolio) modes
- Parallel evaluation with thread pool
- Automatic retry on failure
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from ..core.schemas import EvaluatedFactor, FactorCandidate


@dataclass
class EvaluatorConfig:
    """
    Configuration for evaluator agent.

    Attributes:
        ffo_url: FFO API base URL (e.g., "http://127.0.0.1:19350")
        market: Market identifier (e.g., "csi300")
        start_date: Backtest start date
        end_date: Backtest end date
        top_k: Number of top stocks to select (for full backtest)
        n_drop: Number of stocks to drop (for full backtest)
        fast: Use fast mode (IC only) or full mode
        n_jobs: Number of parallel evaluation workers
        timeout: Request timeout in seconds
        max_retries: Maximum retries per evaluation
    """
    ffo_url: str = "http://127.0.0.1:19350"
    market: str = "csi300"
    start_date: str = "2022-01-01"
    end_date: str = "2023-01-01"
    top_k: int = 30
    n_drop: int = 5
    fast: bool = True
    n_jobs: int = 4
    timeout: int = 120
    max_retries: int = 3


class FactorEvaluator:
    """
    Agent for evaluating factors via FFO API.

    Supports:
    - Fast mode: IC metrics only (~1-2 sec/factor)
    - Full mode: IC + Portfolio metrics (~10-30 sec/factor)
    - Parallel evaluation with thread pool
    - Automatic retry on failure
    """

    def __init__(self, config: EvaluatorConfig, logger=None):
        self.config = config
        if logger is None:
            from ..utils.logger import SearchLogger
            logger = SearchLogger("Evaluator")
        self.logger = logger

    def evaluate_single(
        self, candidate: FactorCandidate, retry_count: int = 0
    ) -> EvaluatedFactor:
        """
        Evaluate a single factor via FFO API.

        Args:
            candidate: Factor to evaluate
            retry_count: Current retry count

        Returns:
            Evaluated factor with metrics
        """
        try:
            payload = {
                "expression": candidate.expression,
                "start": self.config.start_date,
                "end": self.config.end_date,
                "market": self.config.market,
                "fast": self.config.fast,
                "use_cache": True,
            }

            if not self.config.fast:
                payload["topk"] = self.config.top_k
                payload["n_drop"] = self.config.n_drop

            url = f"{self.config.ffo_url}/factors/eval"
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            result = response.json()

            # Handle list response
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if not result.get("success", False):
                raise ValueError(
                    f"Evaluation failed: {result.get('error', 'Unknown error')}"
                )

            metrics = result.get("metrics", {})
            portfolio_metrics = result.get("portfolio_metrics")

            return EvaluatedFactor(
                candidate=candidate,
                metrics=metrics,
                portfolio_metrics=portfolio_metrics,
                backtest_mode="full" if not self.config.fast else "fast",
                success=True,
                error="",
            )

        except Exception as e:
            error_msg = str(e)

            if retry_count < self.config.max_retries:
                self.logger.warning(
                    f"Evaluation failed for {candidate.name}, "
                    f"retry {retry_count + 1}/{self.config.max_retries}: {error_msg}"
                )
                time.sleep(1)
                return self.evaluate_single(candidate, retry_count + 1)

            self.logger.error(f"Evaluation failed for {candidate.name}: {error_msg}")
            return EvaluatedFactor(
                candidate=candidate,
                metrics={},
                portfolio_metrics=None,
                backtest_mode="failed",
                success=False,
                error=error_msg,
            )

    def evaluate_batch(
        self, candidates: List[FactorCandidate], parallel: bool = True
    ) -> List[EvaluatedFactor]:
        """
        Evaluate a batch of factors.

        Args:
            candidates: Factors to evaluate
            parallel: Whether to evaluate in parallel

        Returns:
            List of evaluated factors
        """
        if not candidates:
            return []

        self.logger.info(f"Evaluating {len(candidates)} factors (parallel={parallel})...")
        start_time = time.time()

        if parallel and len(candidates) > 1:
            results = self._evaluate_parallel(candidates)
        else:
            results = [self.evaluate_single(c) for c in candidates]

        elapsed = time.time() - start_time
        num_success = sum(1 for r in results if r.success)
        self.logger.info(
            f"Evaluated {len(candidates)} factors in {elapsed:.1f}s "
            f"({num_success} successful)"
        )

        return results

    def _evaluate_parallel(
        self, candidates: List[FactorCandidate]
    ) -> List[EvaluatedFactor]:
        """Evaluate factors in parallel using thread pool."""
        results = [None] * len(candidates)

        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            future_to_idx = {
                executor.submit(self.evaluate_single, candidate): idx
                for idx, candidate in enumerate(candidates)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"Unexpected error in parallel evaluation: {e}")
                    results[idx] = EvaluatedFactor(
                        candidate=candidates[idx],
                        metrics={},
                        success=False,
                        error=str(e),
                    )

        return results

    def check_api_health(self) -> bool:
        """
        Check if FFO API is healthy and reachable.

        Returns:
            True if API is healthy
        """
        try:
            url = f"{self.config.ffo_url}/health"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            result = response.json()
            is_healthy = result.get("status") == "healthy"

            if is_healthy:
                self.logger.info(f"FFO API is healthy: {self.config.ffo_url}")
            else:
                self.logger.warning(f"FFO API unhealthy: {result}")

            return is_healthy

        except Exception as e:
            self.logger.error(f"FFO API unreachable: {e}")
            return False


def create_evaluator(config_dict: Dict[str, Any], logger=None) -> FactorEvaluator:
    """
    Factory function to create a FactorEvaluator from config dict.

    Args:
        config_dict: Configuration dictionary with keys matching EvaluatorConfig fields
        logger: Optional logger

    Returns:
        Configured FactorEvaluator instance
    """
    config = EvaluatorConfig(
        ffo_url=config_dict.get("ffo_url", "http://127.0.0.1:19350"),
        market=config_dict.get("market", "csi300"),
        start_date=config_dict.get("period_start", "2022-01-01"),
        end_date=config_dict.get("period_end", "2023-01-01"),
        top_k=config_dict.get("top_k", 30),
        n_drop=config_dict.get("n_drop", 5),
        fast=config_dict.get("fast", True),
        n_jobs=config_dict.get("n_jobs", 4),
        timeout=config_dict.get("timeout", 120),
        max_retries=config_dict.get("max_retries", 3),
    )
    return FactorEvaluator(config, logger=logger)
