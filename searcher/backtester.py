"""
Backtester — FFO-backed factor evaluation for the searcher platform.

All factor evaluation in the search pipeline goes through this class.
It wraps the FFO client (ffo.client.factor_eval_client) so that:
  - Market / date / portfolio params are set once from BacktestConfig
  - Algos receive clean callables with no URL or config concerns
  - The FFO server is the single source of truth for backtest results

Usage:
    bt = Backtester.from_config(backtest_config)
    result = bt.evaluate_single("Rank($close, 20)")
    results = bt.evaluate_batch([{"name": "f1", "expression": "Rank($close, 20)"}])

    # Pass callables to algos
    algo = create_algo("ea", ..., evaluate_fn=bt.as_evaluate_fn(), ...)
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from ffo.client.factor_eval_client import FactorEvalClient


class Backtester:
    """
    FFO-backed backtester for the searcher platform.

    Wraps FactorEvalClient with:
      - Bound market / date / portfolio params from config
      - Normalised return dicts: {"success", "expression", "metrics", "name"}
      - Two batch interfaces: list (for EA) and dict-by-name (for ToT)
      - Parallel evaluation support
    """

    def __init__(
        self,
        ffo_url: str = "http://127.0.0.1:19777",
        market: str = "csi300",
        start_date: str = "2016-01-01",
        end_date: str = "2021-01-01",
        top_k: int = 30,
        n_drop: int = 1,
        fast: bool = True,
        n_jobs: int = 4,
        timeout: int = 120,
        logger=None,
        label: str = "search",
    ):
        self.ffo_url = ffo_url
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.top_k = top_k
        self.n_drop = n_drop
        self.fast = fast
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.logger = logger
        self.label = label
        self._client = FactorEvalClient(base_url=ffo_url, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config, logger=None) -> "Backtester":
        """
        Create a Backtester from a BacktestConfig dataclass.

        Args:
            config: BacktestConfig instance (searcher.config.config)
            logger: Optional logger
        """
        return cls(
            ffo_url=config.get_api_url(),
            market=config.market,
            start_date=config.search_start,
            end_date=config.search_end,
            top_k=config.top_k,
            n_drop=config.n_drop,
            fast=config.fast,
            n_jobs=config.n_jobs,
            timeout=getattr(config, "timeout", 120),
            logger=logger,
            label="search",
        )

    @classmethod
    def for_validation(cls, backtest_config, verification_config, logger=None) -> "Backtester":
        """
        Create a Backtester for the validation period.

        Uses the FFO server and market from backtest_config,
        but dates from verification_config.val_start/val_end.
        Always uses fast=True for validation (IC metrics only).
        """
        return cls(
            ffo_url=backtest_config.get_api_url(),
            market=backtest_config.market,
            start_date=verification_config.val_start,
            end_date=verification_config.val_end,
            top_k=backtest_config.top_k,
            n_drop=backtest_config.n_drop,
            fast=True,
            n_jobs=backtest_config.n_jobs,
            timeout=getattr(backtest_config, "timeout", 120),
            logger=logger,
            label="val",
        )

    @classmethod
    def for_test(
        cls,
        backtest_config,
        verification_config,
        *,
        top_k: int = 50,
        n_drop: int = 5,
        n_jobs: int = 4,
        fast: bool = False,
        logger=None,
    ) -> "Backtester":
        """
        Create a Backtester for the test period (full portfolio backtest).

        Uses dates from verification_config.test_start/test_end.
        Defaults to full backtest (fast=False) with test-specific portfolio params.
        """
        return cls(
            ffo_url=backtest_config.get_api_url(),
            market=backtest_config.market,
            start_date=verification_config.test_start,
            end_date=verification_config.test_end,
            top_k=top_k,
            n_drop=n_drop,
            fast=fast,
            n_jobs=n_jobs,
            timeout=getattr(backtest_config, "timeout", 120),
            logger=logger,
            label="test",
        )

    # ------------------------------------------------------------------ #
    # Core evaluation
    # ------------------------------------------------------------------ #

    def evaluate_single(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate one factor via FFO.

        Returns:
            {"success": bool, "expression": str, "metrics": {...}, "error": str|None}
        """
        try:
            results = self._client.evaluate_factor(
                expression=expression,
                market=self.market,
                start_date=self.start_date,
                end_date=self.end_date,
                fast=self.fast,
                topk=self.top_k,
                n_drop=self.n_drop,
                use_cache=True,
            )
            if isinstance(results, list) and results:
                raw = results[0]
            elif isinstance(results, dict):
                raw = results
            else:
                raw = {}

            return {
                "success": bool(raw.get("success", False)),
                "expression": expression,
                "metrics": raw.get("metrics", {}),
                "error": raw.get("error"),
                "cached": raw.get("cached", False),
            }
        except Exception as e:
            self._log_warning(f"evaluate_single failed for '{expression[:60]}': {e}")
            return {"success": False, "expression": expression, "metrics": {}, "error": str(e)}

    def evaluate_batch(self, factors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of factors in parallel via FFO.

        Args:
            factors: [{"name": str, "expression": str}, ...]

        Returns:
            List of result dicts (same order as input), each with
            {"success", "expression", "name", "metrics", "error"}.
        """
        if not factors:
            return []

        results: List[Optional[Dict]] = [None] * len(factors)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            future_to_idx = {
                pool.submit(self.evaluate_single, f.get("expression", "")): i
                for i, f in enumerate(factors)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "success": False,
                        "expression": factors[idx].get("expression", ""),
                        "metrics": {},
                        "error": str(e),
                    }
                r["name"] = factors[idx].get("name", "")
                results[idx] = r

        return [r for r in results if r is not None]

    def evaluate_batch_by_name(self, factors: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a batch; return results keyed by factor name.

        Args:
            factors: [{"name": str, "expression": str}, ...]

        Returns:
            {name: {"success", "expression", "metrics", ...}}
        """
        results = self.evaluate_batch(factors)
        return {r.get("name", r.get("expression", "")): r for r in results}

    # ------------------------------------------------------------------ #
    # Callable factories (for passing to algos)
    # ------------------------------------------------------------------ #

    def as_evaluate_fn(self) -> Callable:
        """Return evaluate_single as a plain callable (for CoT, ToT single eval)."""
        return self.evaluate_single

    def as_batch_evaluate_fn(self) -> Callable:
        """Return evaluate_batch as a callable returning List[Dict] (for EA)."""
        return self.evaluate_batch

    def as_batch_evaluate_fn_dict(self) -> Callable:
        """Return evaluate_batch_by_name as a callable returning Dict[str, Dict] (for ToT)."""
        return self.evaluate_batch_by_name

    # ------------------------------------------------------------------ #
    # Health + utilities
    # ------------------------------------------------------------------ #

    def check_health(self) -> bool:
        """Check if the FFO backend is reachable and healthy."""
        try:
            healthy = self._client.health_check()
            if healthy:
                self._log_info(f"FFO backend healthy: {self.ffo_url}")
            else:
                self._log_warning(f"FFO backend unhealthy: {self.ffo_url}")
            return healthy
        except Exception as e:
            self._log_warning(f"FFO backend unreachable ({self.ffo_url}): {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"Backtester(url={self.ffo_url}, market={self.market}, "
            f"{self.start_date}~{self.end_date}, fast={self.fast}, label={self.label})"
        )

    # ------------------------------------------------------------------ #
    # Internal logging helpers
    # ------------------------------------------------------------------ #

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _log_warning(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
