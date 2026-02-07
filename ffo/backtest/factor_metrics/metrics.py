import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from .utils import dropna_pair, group_by_time


def FL_Ic(factor, label, **kwargs):
    """
    Compute the Information Coefficient (IC) as Pearson correlation between factor and label.
    """
    x, y = dropna_pair(factor, label)
    if len(x) < 2:
        return np.nan
    return pearsonr(x, y)[0]


def FL_RankIc(factor, label, **kwargs):
    """
    Compute the rank-based Information Coefficient (Rank IC) as Spearman correlation.
    """
    x, y = dropna_pair(factor.rank(), label.rank())
    if len(x) < 2:
        return np.nan
    return spearmanr(x, y)[0]


def FL_Ir(factor, label, **kwargs):
    """
    Compute the Information Ratio (IR) as the mean of ICs over time
    divided by their standard deviation.
    """
    ics = []
    for _, (xf, yf) in group_by_time(factor, label):
        try:
            ics.append(FL_Ic(xf, yf))
        except:
            continue
    ics = pd.Series(ics).dropna()
    return ics.mean() / (ics.std() + 1e-12) if not ics.empty else np.nan


def FL_Icir(factor, label, **kwargs):
    """
    Compute IC IR (mean IC multiplied by sqrt(n) divided by standard deviation of ICs).
    """
    ics = []
    for _, (xf, yf) in group_by_time(factor, label):
        try:
            ics.append(FL_Ic(xf, yf))
        except:
            continue
    ics = pd.Series(ics).dropna()
    n = len(ics)
    return ics.mean() * np.sqrt(n) / (ics.std() + 1e-12) if n > 0 else np.nan


def FL_RankIcir(factor, label, **kwargs):
    """
    Compute Rank IC IR (mean rank IC multiplied by sqrt(n) divided by standard deviation).
    """
    ics = []
    for _, (xf, yf) in group_by_time(factor, label):
        try:
            ics.append(FL_RankIc(xf, yf))
        except:
            continue
    ics = pd.Series(ics).dropna()
    n = len(ics)
    return ics.mean() * np.sqrt(n) / (ics.std() + 1e-12) if n > 0 else np.nan


def FL_QuantileReturn(factor, label, quantiles=5, **kwargs):
    """
    Compute the average return for each factor quantile and the spread between
    top and bottom quantiles across time.
    """
    returns = {}
    try:
        for t, (xf, yf) in group_by_time(factor, label):
            df = pd.concat([xf.rename("factor"), yf.rename("label")], axis=1).dropna()
            if df.empty:
                continue
            df["group"] = pd.qcut(df["factor"], quantiles, labels=False) + 1
            grp_ret = df.groupby("group")["label"].mean()
            for q, val in grp_ret.items():
                returns.setdefault(f"Q{q}", []).append(val)
            spread = grp_ret.iloc[-1] - grp_ret.iloc[0]
            returns.setdefault("spread", []).append(spread)
        return {k: np.nanmean(v) if v else np.nan for k, v in returns.items()}
    except Exception as e:
        print(f"[Error] Quantile return computation failed: {e}")
        return None


def FL_Turnover(factor, label=None, quantile=5, **kwargs):
    """
    Compute turnover rate of the top quantile selection across time.
    Turnover is defined as 1 - (overlap / previous top assets).
    """
    prev_top = None
    turnovers = []
    try:
        for t, (xf, _) in group_by_time(factor, factor):
            df = xf.dropna()
            if df.empty:
                continue
            cutoff = np.nanpercentile(df, 100 * (1 - 1 / quantile))
            top_assets = df[df >= cutoff].index
            if prev_top is not None:
                overlap = len(set(prev_top) & set(top_assets))
                turnovers.append(1 - overlap / len(prev_top))
            prev_top = top_assets
        return np.nanmean(turnovers) if turnovers else np.nan
    except Exception as e:
        print(f"[Error] Turnover computation failed: {e}")
        return None
