import pandas as pd

import pandas as pd


def top_n_linear_combination(
    df: pd.DataFrame,
    top_n: int = 5,
    weights=None,
    window_size: int = 22,
    start_date: str = None,
    end_date: str = None,
):
    """
    Compute top-N factor linear combinations over a rolling window.

    Args:
        df (pd.DataFrame): columns=factors, index=datetime (daily)
        top_n (int): number of top factors to select each day
        weights (list[float] or None): optional custom weights for selected factors
        window_size (int): lookback window size (days)
        start_date (str): backtest start date (e.g. "2020-01-01")
        end_date (str): backtest end date (e.g. "2021-01-01")

    Returns:
        dict[date_str, dict[factor_name, weight]]
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    df = df.sort_index()

    # --- check date coverage ---
    df_start, df_end = df.index.min(), df.index.max()
    if start_date and (pd.Timestamp(start_date) < df_start):
        print(f"⚠️ start_date earlier than data start; adjusted to {df_start.date()}")
        start_date = df_start
    if end_date and (pd.Timestamp(end_date) > df_end):
        print(f"⚠️ end_date later than data end; adjusted to {df_end.date()}")
        end_date = df_end

    if start_date is None:
        start_date = df_start
    if end_date is None:
        end_date = df_end

    # --- prepare parameters ---
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    trading_days = df.loc[start_date:end_date].index

    results = {}

    # --- rolling computation ---
    for t in trading_days:
        current_date = pd.Timestamp(t)
        past_start = current_date - pd.Timedelta(
            days=window_size * 2
        )  # generous offset
        history = df.loc[df.index < current_date].iloc[
            -window_size:
        ]  # last window_size days before t

        if len(history) == 0:
            continue

        factor_mean = history.mean().sort_values(ascending=False)
        top_factors = factor_mean.head(top_n).index.tolist()

        # assign weights
        if weights is None:
            w = [1 / len(top_factors)] * len(top_factors)
        else:
            if len(weights) != len(top_factors):
                raise ValueError("weights length must match number of selected factors")
            w = weights

        results[str(current_date.date())] = dict(zip(top_factors, w))

    return results
