import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_ic_weights(
    ic_df: pd.DataFrame, lambda_risk=1.0, alpha_l1=0.0, non_negative=True
):
    """
    Optimize linear combination weights for IC mean-variance with optional L1 sparsity.

    Args:
        ic_df (pd.DataFrame): rows = dates, columns = factors
        lambda_risk (float): risk aversion term (higher = smoother IC)
        alpha_l1 (float): L1 regularization strength (higher = sparser weights)
        non_negative (bool): enforce w >= 0 constraint
    """
    mu = ic_df.mean().values
    Sigma = ic_df.cov().values
    n = len(mu)

    def objective(w):
        # mean-variance + L1
        return -(mu @ w - 0.5 * lambda_risk * (w.T @ Sigma @ w)) + alpha_l1 * np.sum(
            np.abs(w)
        )

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) if non_negative else (None, None)] * n
    w0 = np.ones(n) / n

    res = minimize(objective, w0, bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=ic_df.columns, name="weight")


if __name__ == "__main__":
    # ---- Step 1: generate dummy IC data ----
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200)
    factors = [f"factor_{i}" for i in range(5)]

    # random scores in [-1, 1]
    ic_data = np.random.uniform(-1, 1, size=(len(dates), len(factors)))
    ic_df = pd.DataFrame(ic_data, index=dates, columns=factors)

    print("Sample IC data:")
    print(ic_df.head(), "\n")

    weights = optimize_ic_weights(ic_df, lambda_risk=2.0)
    print("Optimized Weights:")
    print(weights, "\n")

    # ---- Step 4: compute combined IC ----
    ic_comb = ic_df @ weights
    print("Combined IC mean:", ic_comb.mean())
    print("Combined IC std:", ic_comb.std())
