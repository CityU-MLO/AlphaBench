import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import qlib
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.loader import QlibDataLoader
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Fixed Period LASSO (for Category A testing)
# ============================================================================
def train_lasso_factor_combination(
    factor_expressions: List[str],
    start_date: str,
    end_date: str,
    instruments: str = "csi300",
    market: str = None,
    alpha: float = 0.01,
    rolling_window: int = 60,
    max_iter: int = 1000,
    n_jobs: int = 1,
    **kwargs,
) -> Dict:
    """
    Train LASSO for a single fixed period (Category A testing).

    Args:
        factor_expressions: List of factor expressions
        start_date: Test period start (YYYY-MM-DD)
        end_date: Test period end (YYYY-MM-DD)
        instruments: Market universe (default: csi300)
        market: Alternative parameter for market universe
        alpha: LASSO regularization parameter
        rolling_window: Rolling window size for training
        max_iter: Maximum iterations
        n_jobs: Number of parallel jobs for data loading (default: 1)

    Returns:
        Dict with weights and metrics
    """
    print(f"[LASSO Fixed] Training with {len(factor_expressions)} factors")
    print(f"[LASSO Fixed] Period: {start_date} to {end_date}")
    print(f"[LASSO Fixed] Alpha: {alpha}, Rolling window: {rolling_window}")

    try:
        from datetime import datetime, timedelta
        import time
        import os

        # Use market parameter if instruments not provided
        if market is not None:
            instruments = market

        # Initialize Qlib
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

        # Calculate training period (lookback before start_date)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        lookback_start = start_dt - timedelta(days=int(rolling_window * 1.5))
        lookback_start_str = lookback_start.strftime("%Y-%m-%d")

        print(f"[LASSO Fixed] Training data: {lookback_start_str} to {start_date}")

        # Prepare data loader
        fields = factor_expressions
        names = [f"factor_{i}" for i in range(len(factor_expressions))]
        labels = ["Ref($close, -1)/$close - 1"]
        label_names = ["LABEL"]

        data_loader_config = {
            "feature": (fields, names),
            "label": (labels, label_names),
        }
        data_loader = QlibDataLoader(config=data_loader_config)

        # Load data
        load_start = time.time()

        # Configure parallel loading
        if n_jobs > 1:
            os.environ["QLIB_ENABLE_PARALLEL"] = str(n_jobs)
            print(f"[LASSO Fixed] Parallel loading enabled with {n_jobs} jobs")
        else:
            os.environ["QLIB_ENABLE_PARALLEL"] = "0"

        df = data_loader.load(
            instruments=instruments,
            start_time=lookback_start_str,
            end_time=end_date,
        )
        load_time = time.time() - load_start

        feature_df = df["feature"].fillna(0)
        label_df = df["label"].fillna(0)

        if feature_df.empty or label_df.empty:
            return {
                "status": "failed",
                "message": "No data loaded. Check Qlib data availability.",
            }

        print(f"[LASSO Fixed] Loaded {feature_df.shape[0]} rows in {load_time:.2f}s")

        # Split into training (before start_date) and test (start_date to end_date)
        start_dt_ts = pd.Timestamp(start_date)
        all_dates = feature_df.index.get_level_values(0).unique()
        train_dates = sorted([d for d in all_dates if d < start_dt_ts])

        if len(train_dates) < max(len(factor_expressions) * 2, 10):
            return {
                "status": "failed",
                "message": f"Insufficient training data: {len(train_dates)} days",
            }

        # Prepare training data
        X_train = feature_df.loc[train_dates].values
        y_train = label_df.loc[train_dates].values.flatten()

        # Remove NaN/Inf values (same as notebook)
        valid_mask = ~(
            np.isnan(X_train).any(axis=1)
            | np.isnan(y_train)
            | np.isinf(X_train).any(axis=1)
            | np.isinf(y_train)
        )
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        print(f"[LASSO Fixed] Training samples: {X_train.shape[0]} (after filtering)")

        if len(X_train) < max(len(factor_expressions) * 2, 10):
            return {
                "status": "failed",
                "message": f"Insufficient valid training data after filtering: {len(X_train)} samples",
            }

        # Scale and train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        print(f"[LASSO Fixed] Using user's alpha: {alpha}")

        # Train LASSO with user's alpha
        model = Lasso(alpha=alpha, max_iter=max_iter, tol=1e-4)
        model.fit(X_scaled, y_train)

        weights = model.coef_.tolist()
        non_zero = np.sum(np.abs(weights) > 1e-6)
        training_method = f"LASSO (alpha={alpha})"

        # Fallback to Ridge if all weights are zero
        if non_zero == 0:
            print(
                "[LASSO Fixed] ⚠️  LASSO produced all zero weights - switching to Ridge Regression..."
            )
            model = Ridge(alpha=0.01, solver="auto", max_iter=10000)
            model.fit(X_scaled, y_train)
            weights = model.coef_.tolist()
            non_zero = np.sum(np.abs(weights) > 1e-6)
            training_method = "Ridge Regression (LASSO fallback)"

        # Calculate metrics
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)

        print(f"[LASSO Fixed] Method: {training_method}")
        print(
            f"[LASSO Fixed] R2: {r2:.4f}, MSE: {mse:.6f}, Non-zero: {non_zero}/{len(weights)}"
        )

        return {
            "status": "success",
            "weights": weights,
            "factor_names": factor_expressions,
            "training_method": training_method,
            "metrics": {
                "r2": float(r2),
                "mse": float(mse),
                "non_zero_weights": int(non_zero),
                "total_factors": len(factor_expressions),
            },
            "training_period": {"start": lookback_start_str, "end": start_date},
            "test_period": {"start": start_date, "end": end_date},
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "failed", "message": str(e)}


# ============================================================================
# Pure Training Function (No Split, No Lookback)
# ============================================================================
def train_lasso_on_period(
    factor_expressions: List[str],
    start_date: str,
    end_date: str,
    instruments: str = "csi300",
    alpha: float = 0.01,
    max_iter: int = 1000,
    n_jobs: int = 1,
    **kwargs,
) -> Dict:
    """
    Train LASSO on a specific period (start_date to end_date) and return weights.
    The ENTIRE period is used for training. No train/test split.

    Args:
        factor_expressions: List of factor expressions
        start_date: Training start date
        end_date: Training end date
        instruments: Market universe
        alpha: LASSO regularization
        max_iter: Max iterations
        n_jobs: Number of parallel jobs for data loading (default: 1)

    Returns:
        Dict with weights and training metrics
    """
    print(f"[LASSO Train] Training on {start_date} to {end_date}")

    try:
        import time
        import os
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.metrics import mean_squared_error, r2_score

        # Initialize Qlib
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

        # Prepare data loader
        fields = factor_expressions
        names = [f"factor_{i}" for i in range(len(factor_expressions))]
        labels = ["Ref($close, -1)/$close - 1"]
        label_names = ["LABEL"]

        data_loader_config = {
            "feature": (fields, names),
            "label": (labels, label_names),
        }
        data_loader = QlibDataLoader(config=data_loader_config)

        # Load data
        # Configure parallel loading
        if n_jobs > 1:
            os.environ["QLIB_ENABLE_PARALLEL"] = str(n_jobs)
        else:
            os.environ["QLIB_ENABLE_PARALLEL"] = "0"

        df = data_loader.load(
            instruments=instruments,
            start_time=start_date,
            end_time=end_date,
        )

        feature_df = df["feature"].fillna(0)
        label_df = df["label"].fillna(0)

        if feature_df.empty or label_df.empty:
            return {
                "status": "failed",
                "message": "No data loaded for training period.",
            }

        X_train = feature_df.values
        y_train = label_df.values.flatten()

        # Remove NaN/Inf
        valid_mask = ~(
            np.isnan(X_train).any(axis=1)
            | np.isnan(y_train)
            | np.isinf(X_train).any(axis=1)
            | np.isinf(y_train)
        )
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        if len(X_train) < max(len(factor_expressions) * 2, 10):
            return {
                "status": "failed",
                "message": f"Insufficient training data: {len(X_train)} samples",
            }

        # Scale and train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = Lasso(alpha=alpha, max_iter=max_iter, tol=1e-4)
        model.fit(X_scaled, y_train)

        weights = model.coef_.tolist()
        non_zero = np.sum(np.abs(weights) > 1e-6)
        training_method = f"LASSO (alpha={alpha})"

        # Fallback
        if non_zero == 0:
            model = Ridge(alpha=0.01)
            model.fit(X_scaled, y_train)
            weights = model.coef_.tolist()
            training_method = "Ridge (Fallback)"

        # Metrics (In-Sample)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)

        return {
            "status": "success",
            "weights": weights,
            "factor_names": factor_expressions,
            "training_method": training_method,
            "metrics": {
                "r2": float(r2),
                "mse": float(mse),
                "non_zero_weights": int(non_zero),
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "failed", "message": str(e)}
