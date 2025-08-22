import pandas as pd


def dropna_pair(x, y):
    """Drop rows where either x or y is NaN."""
    df = pd.concat([x, y], axis=1)
    df = df.dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) data points after dropping NaNs.")
    return df.iloc[:, 0], df.iloc[:, 1]


def group_by_time(factor, label):
    """
    Group factor and label by time (outer index level if MultiIndex, else by index).
    :return: iterator of (time, (factor_series, label_series))
    """
    if isinstance(factor.index, pd.MultiIndex):
        level = 0
    else:
        level = None
    grouped_factor = factor.groupby(level=level)
    grouped_label = label.groupby(level=level)
    for t in grouped_factor.groups.keys():
        yield t, (grouped_factor.get_group(t), grouped_label.get_group(t))
