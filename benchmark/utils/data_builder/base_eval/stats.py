"""
Factor statistics for the base evaluation benchmark (T2).

Computes for a given regime window:
  - mean_ic         : mean daily IC
  - mean_rankic     : mean daily RankIC
  - icir            : ICIR = mean(IC) / std(IC)
  - rankicir        : RankICIR = mean(RankIC) / std(RankIC)
  - turnover_rate   : sign-flip ratio (proxy for strategy turnover)
  - win_rate        : fraction of days where IC has the correct sign relative to mean
  - skewness        : skewness of daily IC series

These are used to:
  (a) classify factors as Positive / Negative / Noise
  (b) assign per-dimension 1–5 scores via within-category percentile ranking
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

def load_table(path: str) -> pd.DataFrame:
    """
    Load a daily IC/RankIC CSV. Expects either a 'datetime' column or datetime index.
    Returns a datetime-indexed DataFrame sorted ascending, all-NaN columns dropped.
    """
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    df = df.sort_index()
    return df.dropna(axis=1, how="all")


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


# ---------------------------------------------------------------------------
# Per-series statistics
# ---------------------------------------------------------------------------

def _sign_flip_ratio(series: pd.Series) -> float:
    """Fraction of consecutive days where IC sign flips (proxy for turnover)."""
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    signs = np.sign(s.values)
    flips, prev = 0, signs[0]
    for cur in signs[1:]:
        if prev == 0:
            prev = cur
            continue
        if cur == 0:
            continue
        if prev * cur < 0:
            flips += 1
        prev = cur
    return flips / max(len(signs) - 1, 1)


def _win_rate(series: pd.Series) -> float:
    """
    Fraction of days where IC has the expected sign (sign of mean IC).
    If mean IC ≈ 0, returns 0.5.
    """
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    mean_ic = s.mean()
    if abs(mean_ic) < 1e-12:
        return 0.5
    if mean_ic > 0:
        return float((s > 0).mean())
    else:
        return float((s < 0).mean())


def _skewness(series: pd.Series) -> float:
    """Fisher–Pearson skewness (pandas default)."""
    s = series.dropna()
    if len(s) < 3:
        return float("nan")
    return float(s.skew())


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full factor stats for a window
# ---------------------------------------------------------------------------

def compute_full_stats(
    ic_table: pd.DataFrame,
    rankic_table: Optional[pd.DataFrame],
    start: str,
    end: str,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Compute per-factor stats over [start, end].

    Columns returned:
        mean_ic, std_ic, icir,
        mean_rankic, std_rankic, rankicir,
        turnover_rate, win_rate, skewness

    Factors with < min_obs observations are given NaN across all columns.
    """
    sub_ic = slice_window(ic_table, start, end)
    sub_rk = slice_window(rankic_table, start, end) if rankic_table is not None else None

    records = {}
    for factor in sub_ic.columns:
        ic = sub_ic[factor].dropna()
        if len(ic) < min_obs:
            records[factor] = {k: np.nan for k in (
                "mean_ic", "std_ic", "icir",
                "mean_rankic", "std_rankic", "rankicir",
                "turnover_rate", "win_rate", "skewness",
            )}
            continue

        mean_ic = float(ic.mean())
        std_ic = float(ic.std()) if len(ic) > 1 else np.nan
        icir = float(mean_ic / std_ic) if (std_ic and std_ic > 1e-12) else 0.0

        mean_rankic = std_rankic = rankicir = np.nan
        if sub_rk is not None and factor in sub_rk.columns:
            rk = sub_rk[factor].dropna()
            if len(rk) >= min_obs:
                mean_rankic = float(rk.mean())
                std_rankic = float(rk.std()) if len(rk) > 1 else np.nan
                rankicir = float(mean_rankic / std_rankic) if (std_rankic and std_rankic > 1e-12) else 0.0

        records[factor] = {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "mean_rankic": mean_rankic,
            "std_rankic": std_rankic,
            "rankicir": rankicir,
            "turnover_rate": _sign_flip_ratio(ic),
            "win_rate": _win_rate(ic),
            "skewness": _skewness(ic),
        }

    return pd.DataFrame.from_dict(records, orient="index")


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def classify_signal(
    stats: pd.DataFrame,
    noise_ic_threshold: float = 0.01,
) -> pd.Series:
    """
    Classify each factor as 'Positive', 'Negative', or 'Noise'.

    Rules (paper §B.2.3):
      Noise    : avg |IC| < noise_ic_threshold
      Positive : avg IC > 0 and |IC| >= threshold
      Negative : avg IC < 0 and |IC| >= threshold
    """
    labels = {}
    for f, row in stats.iterrows():
        mic = _safe_float(row.get("mean_ic"))
        if mic is None:
            labels[f] = "Noise"
            continue
        if abs(mic) < noise_ic_threshold:
            labels[f] = "Noise"
        elif mic > 0:
            labels[f] = "Positive"
        else:
            labels[f] = "Negative"
    return pd.Series(labels, name="signal")


# ---------------------------------------------------------------------------
# 1–5 score assignment (within-category percentile ranking)
# ---------------------------------------------------------------------------

def _percentile_to_score(values: pd.Series) -> pd.Series:
    """
    Map a numeric series to integer scores 1–5 by within-series quintile rank.
    Rank is ascending (rank 1 = lowest value).
    NaN values get score NaN.
    """
    ranks = values.rank(pct=True, method="average", na_option="keep")
    scores = np.ceil(ranks * 5).clip(1, 5)
    return scores.round().astype("Int64")


def assign_scores(
    stats: pd.DataFrame,
    signal_labels: pd.Series,
) -> pd.DataFrame:
    """
    Assign integer 1–5 scores for each dimension (Performance, Stability,
    WinRate, Skewness) within each (category × regime) group.

    Noise factors receive fixed score 1 on all dimensions (paper §B.2.3).

    Dimension → metric mapping:
      Performance : |icir|      (higher ICIR = stronger signal)
      Stability   : 1 – turnover_rate  (lower turnover = more stable)
      WinRate     : win_rate    (fraction of days with correct sign)
      Skewness    : skewness    (higher right-skew preferred)

    Returns DataFrame with columns: signal, Performance, Stability, WinRate, Skewness
    """
    df = stats.copy()
    df["signal"] = signal_labels

    result_records = {}

    for label in ("Positive", "Negative"):
        subset = df[df["signal"] == label]
        if subset.empty:
            continue

        perf_vals = subset["icir"].abs()
        stab_vals = 1.0 - subset["turnover_rate"].fillna(1.0)
        wr_vals = subset["win_rate"].fillna(0.5)
        sk_vals = subset["skewness"].fillna(0.0)

        scores_perf = _percentile_to_score(perf_vals)
        scores_stab = _percentile_to_score(stab_vals)
        scores_wr = _percentile_to_score(wr_vals)
        scores_sk = _percentile_to_score(sk_vals)

        for f in subset.index:
            result_records[f] = {
                "signal": label,
                "Performance": int(scores_perf.get(f, 3)),
                "Stability": int(scores_stab.get(f, 3)),
                "WinRate": int(scores_wr.get(f, 3)),
                "Skewness": int(scores_sk.get(f, 3)),
            }

    # Noise: all scores = 1 (paper §B.2.3)
    noise_factors = df[df["signal"] == "Noise"].index
    for f in noise_factors:
        result_records[f] = {
            "signal": "Noise",
            "Performance": 1,
            "Stability": 1,
            "WinRate": 1,
            "Skewness": 1,
        }

    return pd.DataFrame.from_dict(result_records, orient="index")


def factor_stats_meta(factor_name: str, stats: pd.DataFrame) -> dict:
    """Return a metadata dict for a single factor (serialization-friendly)."""
    if factor_name not in stats.index:
        return {}
    row = stats.loc[factor_name]
    return {
        "mean_ic": _safe_float(row.get("mean_ic")),
        "mean_rankic": _safe_float(row.get("mean_rankic")),
        "icir": _safe_float(row.get("icir")),
        "rankicir": _safe_float(row.get("rankicir")),
        "turnover_rate": _safe_float(row.get("turnover_rate")),
        "win_rate": _safe_float(row.get("win_rate")),
        "skewness": _safe_float(row.get("skewness")),
    }
