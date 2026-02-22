import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
import qlib
from qlib.data.dataset.loader import QlibDataLoader
import warnings

warnings.filterwarnings("ignore")


def compute_equal_weight_baseline(
    factor_expressions: List[str],
    market: str,
    start_date: str,
    end_date: str,
    rolling_window: int = 60,
) -> Dict:
    """
    Compute equal-weight baseline portfolio and performance metrics.

    All factors are given equal weight (1/n), providing a baseline to compare
    against optimized weights from LASSO or IC optimization.

    Args:
        factor_expressions: List of Qlib factor expressions
        market: Market/instrument universe (e.g., "csi300", "csi500")
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        rolling_window: Window size for metrics calculation

    Returns:
        Dict containing:
            - status: 'success' or 'failed'
            - weights: Equal weights for each factor (1/n for each)
            - factor_names: Names of factors
            - factor_expressions: Original factor expressions
            - metrics: Performance metrics matching optimized results
                - r2: R² score
                - mse: Mean squared error
                - n_samples: Total samples
                - rolling_r2_mean: Mean R² across windows
                - rolling_r2_std: Std dev of R² across windows
            - predictions: Portfolio predictions for backtest period
            - actuals: Actual next-day returns
            - message: Error message if failed
    """
    try:
        # Initialize Qlib
        try:
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        except:
            pass  # Already initialized

        n_factors = len(factor_expressions)
        if n_factors == 0:
            return {"status": "failed", "message": "No factors provided"}

        # Equal weights for all factors
        equal_weights = np.array([1.0 / n_factors] * n_factors)

        # Load factor data
        factor_data_config = {
            "feature": (factor_expressions, [f"factor_{i}" for i in range(n_factors)])
        }
        loader = QlibDataLoader(config=factor_data_config)

        factor_data = loader.load(
            instruments=market, start_time=start_date, end_time=end_date
        )

        if "feature" not in factor_data:
            return {"status": "failed", "message": "Failed to load factor data"}

        feature_df = factor_data["feature"]

        # Load labels (next-day returns)
        label_config = {"label": (["Ref($close, -1)/$close - 1"], ["LABEL"])}
        label_loader = QlibDataLoader(config=label_config)

        label_data = label_loader.load(
            instruments=market, start_time=start_date, end_time=end_date
        )

        if "label" not in label_data:
            return {"status": "failed", "message": "Failed to load label data"}

        label_df = label_data["label"]

        # Align data
        common_index = feature_df.index.intersection(label_df.index)
        X = feature_df.loc[common_index].fillna(0).values
        y = label_df.loc[common_index].fillna(0).values.flatten()

        if len(X) == 0:
            return {
                "status": "failed",
                "message": "No data overlap between factors and labels",
            }

        # Remove any remaining NaN values
        mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            return {
                "status": "failed",
                "message": "No valid data after removing NaN values",
            }

        # Compute portfolio predictions using equal weights
        predictions = np.dot(X, equal_weights)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score

        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Rolling window metrics
        n_windows = max(1, len(y) - rolling_window + 1)
        rolling_r2_scores = []

        for i in range(n_windows):
            y_window = y[i : i + rolling_window]
            pred_window = predictions[i : i + rolling_window]

            if len(y_window) == rolling_window:
                r2_window = r2_score(y_window, pred_window)
                rolling_r2_scores.append(r2_window)

        rolling_r2_mean = np.mean(rolling_r2_scores) if rolling_r2_scores else r2
        rolling_r2_std = np.std(rolling_r2_scores) if rolling_r2_scores else 0.0

        # Factor names - use the actual factor expressions
        factor_names = factor_expressions

        # Sparsity (not applicable for equal weights - always fully dense)
        non_zero_weights = n_factors
        sparsity_ratio = non_zero_weights / n_factors

        return {
            "status": "success",
            "weights": equal_weights,
            "factor_names": factor_names,
            "factor_expressions": factor_expressions,
            "metrics": {
                "r2": r2,
                "mse": mse,
                "n_samples": len(y),
                "rolling_r2_mean": rolling_r2_mean,
                "rolling_r2_std": rolling_r2_std,
                "n_windows": n_windows,
                "window_size": rolling_window,
                "sparsity_ratio": sparsity_ratio,
                "non_zero_weights": non_zero_weights,
            },
            "predictions": (
                predictions.tolist()
                if hasattr(predictions, "tolist")
                else list(predictions)
            ),
            "actuals": y.tolist() if hasattr(y, "tolist") else list(y),
            "message": "Equal-weight baseline computed successfully",
        }

    except Exception as e:
        import traceback

        return {
            "status": "failed",
            "message": f"Baseline computation failed: {str(e)}",
            "error_details": traceback.format_exc(),
        }
