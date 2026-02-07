"""
Unified Optimization Pipeline

This module provides a comprehensive pipeline that integrates three optimization approaches:
1. Baseline (Equal-Weight) - Reference portfolio
2. LASSO (Linear Regression) - Predictive optimization
3. IC Optimization (Information Coefficient) - Correlation-based optimization

All three approaches support both fixed (single period) and dynamic (multi-period) training.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Structured result for a single optimization method."""

    method: str
    status: str
    period: Dict  # {'start': str, 'end': str}
    weights: List[float]
    factor_names: List[str]
    metrics: Dict
    training_period: Optional[Dict] = None
    message: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PeriodResults:
    """Structured results for a single period across all methods."""

    period_index: int
    period: Dict  # {'start': str, 'end': str}
    baseline: Optional[OptimizationResult] = None
    lasso: Optional[OptimizationResult] = None
    ic_optimization: Optional[OptimizationResult] = None
    training_period: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "period_index": self.period_index,
            "period": self.period,
        }
        if self.training_period:
            result["training_period"] = self.training_period
        if self.baseline:
            result["baseline"] = self.baseline.to_dict()
        if self.lasso:
            result["lasso"] = self.lasso.to_dict()
        if self.ic_optimization:
            result["ic_optimization"] = self.ic_optimization.to_dict()
        return result


class UnifiedOptimizationPipeline:
    """
    Comprehensive pipeline for factor combination optimization.

    Integrates three optimization approaches:
    - Baseline: Equal-weight reference portfolio
    - LASSO: L1-regularized regression
    - IC Optimization: Correlation-based weighting

    Supports both fixed and dynamic training approaches.
    """

    def __init__(
        self,
        instruments: str = "csi300",
        data_path: str = "~/.qlib/qlib_data/cn_data",
        region: str = "cn",
        model_save_path: str = "./cache_data/factor_models",
    ):
        """
        Initialize the optimization pipeline.

        Args:
            instruments: Market/instrument universe (e.g., "csi300", "csi500")
            data_path: Path to Qlib data
            region: Market region (e.g., "cn" for China)
            model_save_path: Path to save model results
        """
        self.instruments = instruments
        self.data_path = data_path
        self.region = region
        self.model_save_path = model_save_path

        # ============ Phase 2: Period-Level Caching ============
        # Cache to store period data and avoid reloading within same period
        self.period_data_cache = (
            {}
        )  # Key: (period_start, period_end), Value: loaded data

        os.makedirs(model_save_path, exist_ok=True)
        logger.info(
            f"Pipeline initialized for {instruments} market with caching enabled"
        )

    def run_fixed_period_optimization(
        self,
        factor_expressions: List[str],
        start_date: str,
        end_date: str,
        methods: List[str] = None,
        method_configs: Optional[Dict] = None,
        include_baseline: bool = True,
    ) -> Dict:
        """
        Run optimization for a single fixed period.

        Args:
            factor_expressions: List of Qlib factor expressions
            start_date: Period start date (YYYY-MM-DD)
            end_date: Period end date (YYYY-MM-DD)
            methods: List of methods to run
            method_configs: Configuration for each method
            include_baseline: Whether to include baseline

        Returns:
            Dict containing optimization results
        """
        if methods is None:
            methods = ["lasso"]

        logger.info("=" * 70)
        logger.info("FIXED PERIOD OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Methods: {', '.join(methods)}")

        # Prepare configurations
        default_configs = self._get_default_configs()
        if method_configs:
            for method, config in method_configs.items():
                if method in default_configs:
                    default_configs[method].update(config)

        # Results container
        period_result = PeriodResults(
            period_index=0,
            period={"start": start_date, "end": end_date},
            training_period={"start": start_date, "end": end_date},
        )

        # Compute baseline
        if include_baseline:
            logger.info("Computing Baseline...")
            baseline_result = self._compute_baseline_fixed(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(
                f"Baseline result status: {baseline_result.get('status', 'UNKNOWN')}"
            )
            if baseline_result.get("status") == "success":
                logger.info("Setting period_result.baseline...")
                period_result.baseline = OptimizationResult(
                    method="baseline",
                    status="success",
                    period={"start": start_date, "end": end_date},
                    weights=(
                        baseline_result["weights"].tolist()
                        if hasattr(baseline_result["weights"], "tolist")
                        else list(baseline_result["weights"])
                    ),
                    factor_names=baseline_result["factor_names"],
                    metrics=baseline_result["metrics"],
                    message="Baseline computed",
                )
                logger.info(f"Baseline set: {period_result.baseline is not None}")
            else:
                logger.error(
                    f"Baseline failed: {baseline_result.get('message', 'Unknown error')}"
                )

        # Compute LASSO
        if "lasso" in methods:
            logger.info("Computing LASSO...")
            lasso_config = default_configs["lasso"]
            lasso_result = self._compute_lasso_fixed(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
                **lasso_config,
            )
            logger.info(f"LASSO result status: {lasso_result.get('status', 'UNKNOWN')}")
            if lasso_result.get("status") == "success":
                logger.info("Setting period_result.lasso...")
                period_result.lasso = OptimizationResult(
                    method="lasso",
                    status="success",
                    period={"start": start_date, "end": end_date},
                    weights=lasso_result["weights"],
                    factor_names=lasso_result["factor_names"],
                    metrics={
                        "r2": lasso_result.get("r2", 0),
                        "mse": lasso_result.get("mse", 0),
                        "rolling_r2_mean": lasso_result.get("rolling_r2_mean", 0),
                        "n_samples": lasso_result.get("n_samples", 0),
                    },
                    message="LASSO computed",
                )
                logger.info(f"LASSO set: {period_result.lasso is not None}")
            else:
                logger.error(
                    f"LASSO failed: {lasso_result.get('message', 'Unknown error')}"
                )

        # Compute IC Optimization
        if "ic_optimization" in methods:
            logger.info("Computing IC Optimization...")
            ic_config = default_configs["ic_optimization"]
            ic_result = self._compute_ic_fixed(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
                **ic_config,
            )
            logger.info(
                f"IC Optimization result status: {ic_result.get('status', 'UNKNOWN')}"
            )
            if ic_result.get("status") == "success":
                logger.info("Setting period_result.ic_optimization...")
                period_result.ic_optimization = OptimizationResult(
                    method="ic_optimization",
                    status="success",
                    period={"start": start_date, "end": end_date},
                    weights=ic_result["weights"],
                    factor_names=ic_result["factor_names"],
                    metrics={
                        "combined_ic_mean": ic_result.get("expected_return", 0),
                        "combined_ic_ir": ic_result.get("sharpe_ratio", 0),
                        "num_factors_used": int(
                            np.sum(np.array(ic_result["weights"]) > 0)
                        ),
                    },
                    message="IC optimization computed",
                )
                logger.info(
                    f"IC Optimization set: {period_result.ic_optimization is not None}"
                )
            else:
                logger.error(
                    f"IC Optimization failed: {ic_result.get('message', 'Unknown error')}"
                )

        # Final logging before return
        logger.info(f"Final period_result before to_dict():")
        logger.info(f"  - baseline: {period_result.baseline is not None}")
        logger.info(f"  - lasso: {period_result.lasso is not None}")
        logger.info(f"  - ic_optimization: {period_result.ic_optimization is not None}")

        period_dict = period_result.to_dict()
        logger.info(f"After to_dict(), keys in period: {list(period_dict.keys())}")

        return {
            "status": "success",
            "total_periods": 1,
            "period_type": "fixed",
            "periods": [period_dict],
            "summary": self._create_summary([period_result]),
        }

    def run_dynamic_period_optimization(
        self,
        periods_config: List[Dict],
        methods: List[str] = None,
        method_configs: Optional[Dict] = None,
        include_baseline: bool = True,
        lookback_window: Optional[int] = None,
    ) -> Dict:
        """
        Run optimization for multiple dynamic periods.

        Args:
            periods_config: List of period configurations
            methods: List of methods to run
            method_configs: Configuration for each method
            include_baseline: Whether to include baseline
            lookback_window: Global lookback window (T) in days. Overrides method configs.

        Returns:
            Dict containing optimization results for all periods
        """
        if methods is None:
            methods = ["lasso"]

        # ============ Phase 2: Clear cache at start of new optimization run ============
        self.period_data_cache.clear()
        logger.info("Cache cleared - starting fresh optimization run")

        # Prepare configurations
        default_configs = self._get_default_configs()
        if method_configs:
            for method, config in method_configs.items():
                if method in default_configs:
                    default_configs[method].update(config)

        # Extract global lookback window (T) preference
        # Priority:
        # 1. Explicit argument 'lookback_window'
        # 2. Explicit user config in 'method_configs' (for any active method)
        # 3. Default config of active methods (LASSO > IC)

        global_lookback = 60  # Ultimate fallback

        if lookback_window is not None:
            global_lookback = lookback_window
        else:
            # Check if user explicitly provided lookback in any config
            user_provided_T = None
            if method_configs:
                for method in methods:
                    if (
                        method in method_configs
                        and "lookback_window" in method_configs[method]
                    ):
                        user_provided_T = method_configs[method]["lookback_window"]
                        break  # Use the first explicit one found

            if user_provided_T is not None:
                global_lookback = user_provided_T
            else:
                # Fallback to defaults logic
                if "lasso" in methods and "lasso" in default_configs:
                    global_lookback = default_configs["lasso"].get(
                        "lookback_window", 60
                    )
                elif (
                    "ic_optimization" in methods
                    and "ic_optimization" in default_configs
                ):
                    global_lookback = default_configs["ic_optimization"].get(
                        "lookback_window", 60
                    )

        # Validate periods - check for both old format (start/end) and new format (test_start/test_end)
        # Convert new format to old format for validation
        periods_for_validation = []
        for period in periods_config:
            if "test_start" in period and "test_end" in period:
                # New format from dynamic optimization
                periods_for_validation.append(
                    {
                        "start": period["test_start"],
                        "end": period["test_end"],
                        "factors": period.get("factors", []),
                    }
                )
            elif "start" in period and "end" in period:
                # Old format
                periods_for_validation.append(period)

        validation = self.validate_periods_continuity(periods_for_validation)
        if not validation["is_valid"]:
            logger.error(f"Period validation failed: {validation['issues']}")
            return {
                "status": "failed",
                "message": "Period validation failed",
                "validation": validation,
            }

        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(warning)

        logger.info("=" * 70)
        logger.info("DYNAMIC PERIOD OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Total periods: {len(periods_config)}")
        logger.info(f"Methods: {', '.join(methods)}")
        logger.info(f"Global Lookback Window (T): {global_lookback} days")

        all_period_results = []

        # Process each period
        for period_idx, period_config in enumerate(periods_config):

            # Support both old format (start/end) and new format (test_start/test_end, train_start/train_end)
            if "test_start" in period_config:
                # New format from dynamic optimization UI
                period_start = period_config.get("train_start")
                period_end = period_config.get("train_end")
                test_start = period_config.get("test_start")
                test_end = period_config.get("test_end")
                factors = period_config.get("factors", [])
            else:
                # Old format - auto-calculate training period
                test_start = period_config.get("start")
                test_end = period_config.get("end")
                factors = period_config.get("factors", [])

                # If no explicit training period provided, calculate it automatically
                if (
                    "train_start" not in period_config
                    or "train_end" not in period_config
                ):
                    from datetime import datetime, timedelta

                    test_start_dt = datetime.strptime(test_start, "%Y-%m-%d")

                    # Training End = Test Start - 1 Day (Prevent Leakage)
                    training_end_dt = test_start_dt - timedelta(days=1)

                    # Training Start = Training End - Lookback Window (T)
                    # We multiply by 1.5 to approximate calendar days from trading days
                    est_calendar_days = int(global_lookback * 1.5)
                    training_start_dt = training_end_dt - timedelta(
                        days=est_calendar_days
                    )

                    period_start = training_start_dt.strftime("%Y-%m-%d")
                    period_end = training_end_dt.strftime("%Y-%m-%d")
                else:
                    period_start = period_config.get("train_start")
                    period_end = period_config.get("train_end")

            logger.info(
                f"Period {period_idx + 1}: Train {period_start} to {period_end}, Test {test_start} to {test_end}"
            )

            period_result = PeriodResults(
                period_index=period_idx,
                period={"start": test_start, "end": test_end},
                training_period={"start": period_start, "end": period_end},
            )

            # Compute baseline
            if include_baseline:
                # Baseline should be evaluated on the TEST period to show 1/N performance
                baseline_result = self._compute_baseline_fixed(
                    factor_expressions=factors, start_date=test_start, end_date=test_end
                )

                if baseline_result["status"] == "success":
                    period_result.baseline = OptimizationResult(
                        method="baseline",
                        status="success",
                        period={"start": test_start, "end": test_end},
                        weights=(
                            baseline_result["weights"].tolist()
                            if hasattr(baseline_result["weights"], "tolist")
                            else list(baseline_result["weights"])
                        ),
                        factor_names=baseline_result["factor_names"],
                        metrics=baseline_result["metrics"],
                    )

            # Compute LASSO
            if "lasso" in methods:
                lasso_config = default_configs["lasso"]
                lasso_result = self._compute_lasso_dynamic(
                    factor_expressions=factors,
                    period_start=period_start,
                    period_end=period_end,
                    **lasso_config,
                )

                if lasso_result["status"] == "success":
                    period_result.lasso = OptimizationResult(
                        method="lasso",
                        status="success",
                        period={"start": period_start, "end": period_end},
                        training_period=lasso_result.get("training_period"),
                        weights=lasso_result["weights"],
                        factor_names=lasso_result["factor_names"],
                        metrics=lasso_result.get("train_metrics", {}),
                    )
                    period_result.training_period = lasso_result.get("training_period")

            # Compute IC Optimization
            if "ic_optimization" in methods:
                ic_config = default_configs["ic_optimization"]
                ic_result = self._compute_ic_dynamic(
                    factor_expressions=factors,
                    period_start=period_start,
                    period_end=period_end,
                    **ic_config,
                )

                if ic_result["status"] == "success":
                    period_result.ic_optimization = OptimizationResult(
                        method="ic_optimization",
                        status="success",
                        period={"start": period_start, "end": period_end},
                        training_period=ic_result.get("training_period"),
                        weights=ic_result["weights"],
                        factor_names=ic_result["factor_names"],
                        metrics={
                            "combined_ic_mean": ic_result.get("combined_ic_mean", 0),
                            "combined_ic_ir": ic_result.get("combined_ic_ir", 0),
                        },
                    )

            all_period_results.append(period_result)

        return {
            "status": "success",
            "total_periods": len(periods_config),
            "period_type": "dynamic",
            "periods": [p.to_dict() for p in all_period_results],
            "summary": self._create_summary(all_period_results),
            "validation": validation,
        }

    # ==================== EXPORT/IMPORT ====================

    def export_results(
        self,
        results: Dict,
        configuration: Dict,
        experiment_name: str = "optimization_result",
    ) -> str:
        """
        Export optimization results in standardized format.

        Args:
            results: Results dict from run_*_optimization
            configuration: Configuration used (periods, factors, methods)
            experiment_name: Name for export file

        Returns:
            Path to saved file
        """
        export_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "pipeline_version": "1.0",
                "type": "factor_combination",
                "mode": results.get("period_type", "unknown"),
                "description": experiment_name,
            },
            "configuration": configuration,
            "results": results,
            "summary": results.get("summary", {}),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.model_save_path, filename)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Results exported to: {filepath}")
        return filepath

    @staticmethod
    def import_results(filepath: str) -> Tuple[Dict, Dict]:
        """
        Import previously exported results.

        Args:
            filepath: Path to exported JSON file

        Returns:
            Tuple of (results, configuration)
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        results = data.get("results", {})
        configuration = data.get("configuration", {})

        logger.info(f"Imported from: {filepath}")
        return results, configuration

    @staticmethod
    def validate_periods_continuity(periods_config: List[Dict]) -> Dict:
        """
        Validate period continuity for dynamic optimization.

        Args:
            periods_config: List of period configurations

        Returns:
            Validation result dict
        """
        result = {
            "is_valid": True,
            "total_periods": len(periods_config),
            "issues": [],
            "warnings": [],
        }

        if not periods_config:
            result["is_valid"] = False
            result["issues"].append("No periods provided")
            return result

        sorted_periods = sorted(periods_config, key=lambda p: p.get("start", ""))

        for i, period in enumerate(sorted_periods):
            if "start" not in period or "end" not in period:
                result["is_valid"] = False
                result["issues"].append(f"Period {i+1}: Missing start or end date")

            if "factors" not in period or not period["factors"]:
                result["is_valid"] = False
                result["issues"].append(f"Period {i+1}: No factors specified")

            try:
                start = datetime.strptime(period["start"], "%Y-%m-%d")
                end = datetime.strptime(period["end"], "%Y-%m-%d")

                if start >= end:
                    result["is_valid"] = False
                    result["issues"].append(f"Period {i+1}: Start date >= end date")
            except ValueError as e:
                result["is_valid"] = False
                result["issues"].append(f"Period {i+1}: Invalid date format")

        for i in range(len(sorted_periods) - 1):
            try:
                current_end = datetime.strptime(sorted_periods[i]["end"], "%Y-%m-%d")
                next_start = datetime.strptime(
                    sorted_periods[i + 1]["start"], "%Y-%m-%d"
                )

                gap_days = (next_start - current_end).days

                if gap_days < 1:
                    result["is_valid"] = False
                    result["issues"].append(f"Periods {i+1}-{i+2}: Overlap")
                elif gap_days > 3:
                    result["warnings"].append(f"Periods {i+1}-{i+2}: {gap_days}d gap")
            except ValueError:
                pass

        return result

    # ==================== HELPER METHODS ====================

    def _get_default_configs(self) -> Dict:
        """Get default configurations."""
        return {
            "lasso": {
                "alpha": 0.01,
                "rolling_window": 60,
                "lookback_window": 60,
                "max_iter": 1000,
            },
            "ic_optimization": {
                "method": "equal_top",
                "top_k": 5,
                "lookback_window": 60,
            },
        }

    def _compute_baseline_fixed(
        self, factor_expressions: List[str], start_date: str, end_date: str
    ) -> Dict:
        """Compute baseline for fixed period."""
        from data.pipeline.optim.baseline import compute_equal_weight_baseline

        try:
            return compute_equal_weight_baseline(
                factor_expressions=factor_expressions,
                market=self.instruments,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Baseline error: {e}")
            return {"status": "failed", "message": str(e)}

    def _compute_lasso_fixed(
        self,
        factor_expressions: List[str],
        start_date: str,
        end_date: str,
        alpha: float = 0.01,
        rolling_window: int = 60,
        max_iter: int = 1000,
        **kwargs,
    ) -> Dict:
        """Compute LASSO for fixed period."""
        from data.pipeline.optim.ml_training import train_lasso_on_period

        try:
            # Use the stateless training function for consistency
            result = train_lasso_on_period(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
                instruments=self.instruments,
                alpha=alpha,
                max_iter=max_iter,
            )

            if result["status"] == "success":
                # Map the result to match expected fixed period output format
                return {
                    "status": "success",
                    "weights": result.get("weights", []),
                    "factor_names": result.get("factor_names", []),
                    "r2": result.get("metrics", {}).get("r2", 0),
                    "mse": result.get("metrics", {}).get("mse", 0),
                    "rolling_r2_mean": 0,  # Not applicable for fixed period
                    "n_samples": result.get("n_samples", 0),
                }
            else:
                return result

        except Exception as e:
            logger.error(f"LASSO error: {e}")
            return {"status": "failed", "message": str(e)}

    def _compute_lasso_dynamic(
        self,
        factor_expressions: List[str],
        period_start: str,
        period_end: str,
        alpha: float = 0.01,
        rolling_window: int = 60,
        lookback_window: int = 60,
        max_iter: int = 1000,
        **kwargs,
    ) -> Dict:
        """Compute LASSO for dynamic period (Train on specific period)."""
        from data.pipeline.optim.ml_training import train_lasso_on_period

        try:
            # Use the explicit training function
            result = train_lasso_on_period(
                factor_expressions=factor_expressions,
                start_date=period_start,
                end_date=period_end,
                instruments=self.instruments,
                alpha=alpha,
                max_iter=max_iter,
            )

            if result["status"] == "success":
                return {
                    "status": "success",
                    "weights": result.get("weights", []),
                    "factor_names": result.get("factor_names", []),
                    "training_period": {"start": period_start, "end": period_end},
                    "train_metrics": result.get("metrics", {}),
                }
            else:
                return result

        except Exception as e:
            logger.error(f"Dynamic LASSO error: {e}")
            return {"status": "failed", "message": f"LASSO failed: {str(e)}"}

    def _compute_ic_fixed(
        self,
        factor_expressions: List[str],
        start_date: str,
        end_date: str,
        method: str = "equal_top",
        top_k: int = 5,
        **kwargs,
    ) -> Dict:
        """Compute IC for fixed period."""
        from data.pipeline.optim.ic_optimization import optimize_factor_weights_ic

        try:
            return optimize_factor_weights_ic(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
                method=method,
                top_k=top_k,
                instruments=self.instruments,
            )
        except Exception as e:
            logger.error(f"IC error: {e}")
            return {"status": "failed", "message": str(e)}

    def _compute_ic_dynamic(
        self,
        factor_expressions: List[str],
        period_start: str,
        period_end: str,
        method: str = "equal_top",
        top_k: int = 5,
        lookback_window: int = 60,
        **kwargs,
    ) -> Dict:
        """Compute IC for dynamic period (Train on specific period)."""
        from data.pipeline.optim.ic_optimization import optimize_factor_weights_ic

        try:
            # Use the fixed optimization function on the training period
            # This avoids double-lookback logic issues
            return optimize_factor_weights_ic(
                factor_expressions=factor_expressions,
                start_date=period_start,
                end_date=period_end,
                method=method,
                top_k=top_k,
                instruments=self.instruments,
            )
        except Exception as e:
            logger.error(f"Dynamic IC error: {e}")
            return {"status": "failed", "message": f"IC failed: {str(e)}"}

    def _create_summary(self, period_results: List[PeriodResults]) -> Dict:
        """Create summary statistics."""
        return {
            "total_periods": len(period_results),
            "successful_periods": len(
                [
                    p
                    for p in period_results
                    if p.baseline or p.lasso or p.ic_optimization
                ]
            ),
        }
