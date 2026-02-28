"""
IC / RankIC statistics helpers for atomic evaluation dataset builders.

Extracted and generalized from rebuttal_eval_design build scripts.
All functions are stateless and operate on pandas DataFrames.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def sign_flip_ratio(series: pd.Series) -> float:
    """Fraction of consecutive-day sign flips in an IC series."""
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


def _to_float_or_none(x) -> Optional[float]:
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def select_window(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Slice a datetime-indexed DataFrame to [start_date, end_date].
    The index must already be a DatetimeIndex (sorted).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df.loc[(df.index >= start) & (df.index <= end)]


def load_ic_table(path: str) -> pd.DataFrame:
    """
    Load an IC/RankIC CSV table.
    Expected: a 'datetime' column (or datetime index) + one column per factor.
    Returns a DataFrame with a DatetimeIndex, sorted ascending, all-NaN columns dropped.
    """
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    df = df.sort_index()
    df = df.dropna(axis=1, how="all")
    return df


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_factor_stats(
    df_ic: pd.DataFrame,
    df_rankic: Optional[pd.DataFrame],
    start_date: str,
    end_date: str,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Compute per-factor statistics over the given date window.

    Returns a DataFrame indexed by factor name with columns:
        mean_ic, std_ic, icir, mean_rankic, sign_flip_ratio

    Factors with fewer than `min_obs` non-NaN IC observations are given NaN stats.
    """
    sub_ic = select_window(df_ic, start_date, end_date)
    sub_rk = select_window(df_rankic, start_date, end_date) if df_rankic is not None else None

    records = {}
    for factor in sub_ic.columns:
        ic_series = sub_ic[factor].dropna()
        if len(ic_series) < min_obs:
            records[factor] = {
                "mean_ic": np.nan,
                "std_ic": np.nan,
                "icir": np.nan,
                "mean_rankic": np.nan,
                "sign_flip_ratio": np.nan,
            }
            continue

        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std()) if len(ic_series) > 1 else np.nan
        icir = float(mean_ic / std_ic) if (std_ic and std_ic > 0) else 0.0
        sfr = float(sign_flip_ratio(ic_series))

        mean_rankic = np.nan
        if sub_rk is not None and factor in sub_rk.columns:
            rk_series = sub_rk[factor].dropna()
            if len(rk_series) >= min_obs:
                mean_rankic = float(rk_series.mean())

        records[factor] = {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "mean_rankic": mean_rankic,
            "sign_flip_ratio": sfr,
        }

    return pd.DataFrame.from_dict(records, orient="index")


def factor_meta(
    factor_name: str,
    stats: pd.DataFrame,
) -> dict:
    """
    Extract per-factor meta dict from the stats DataFrame.
    Returns {"ic", "rankic", "icir", "sign_flip_ratio"} with None for missing values.
    """
    if factor_name not in stats.index:
        return {"ic": None, "rankic": None, "icir": None, "sign_flip_ratio": None}
    row = stats.loc[factor_name]
    return {
        "ic": _to_float_or_none(row.get("mean_ic")),
        "rankic": _to_float_or_none(row.get("mean_rankic")),
        "icir": _to_float_or_none(row.get("icir")),
        "sign_flip_ratio": _to_float_or_none(row.get("sign_flip_ratio")),
    }


# ---------------------------------------------------------------------------
# Noise / clean labelling helpers
# ---------------------------------------------------------------------------

def is_noise(
    factor_name: str,
    stats: pd.DataFrame,
    ic_threshold: float,
    rankic_threshold: float,
    use_abs: bool,
    condition: str,
) -> Optional[bool]:
    """
    Return True if the factor is labelled as noise, False if clean, None if stats unavailable.

    Noise gate logic:
        ic_gate   = (|mean_ic|   if use_abs else mean_ic)   < ic_threshold
        rk_gate   = (|mean_rankic| if use_abs else mean_rankic) < rankic_threshold
        noise = ic_gate AND rk_gate  (condition='and')
              or ic_gate OR  rk_gate  (condition='or')
    """
    if factor_name not in stats.index:
        return None
    row = stats.loc[factor_name]

    mean_ic = _to_float_or_none(row.get("mean_ic"))
    mean_rankic = _to_float_or_none(row.get("mean_rankic"))

    if mean_ic is None:
        return None

    ic_val = abs(mean_ic) if use_abs else mean_ic
    ic_gate = ic_val < ic_threshold

    # RankIC gate — skip if not available, default to matching ic_gate
    if mean_rankic is not None:
        rk_val = abs(mean_rankic) if use_abs else mean_rankic
        rk_gate = rk_val < rankic_threshold
    else:
        rk_gate = ic_gate  # fall back to IC gate only

    if condition == "and":
        return ic_gate and rk_gate
    else:  # "or"
        return ic_gate or rk_gate


def rank_by_noise_score(stats: pd.DataFrame, use_abs: bool = True) -> pd.Index:
    """
    Rank factors from most-noise-like to least-noise-like.
    Used to select balanced noise/signal samples.

    Noise score = abs(mean_ic) (lower is more noise-like),
                  tiebreak by sign_flip_ratio (higher is more noise-like).
    Returns factor names in that order.
    """
    df = stats.copy()
    df["_abs_ic"] = df["mean_ic"].abs() if use_abs else df["mean_ic"]
    df["_sfr"] = df["sign_flip_ratio"].fillna(1.0)
    df["_abs_icir"] = df["icir"].abs().fillna(0.0)
    # sort: noisy factors first
    df_sorted = df.sort_values(
        by=["_abs_ic", "_sfr", "_abs_icir"],
        ascending=[True, False, True],
        na_position="last",
    )
    return df_sorted.index


def rank_by_signal_score(stats: pd.DataFrame, use_abs: bool = True) -> pd.Index:
    """
    Rank factors from most-signal-like to least-signal-like.
    Returns factor names in that order (strongest signal first).
    """
    df = stats.copy()
    df["_abs_ic"] = df["mean_ic"].abs() if use_abs else df["mean_ic"]
    df["_abs_rankic"] = df["mean_rankic"].abs().fillna(0.0)
    df["_abs_icir"] = df["icir"].abs().fillna(0.0)
    df["_sfr"] = df["sign_flip_ratio"].fillna(1.0)
    df["_combined"] = df[["_abs_ic", "_abs_rankic"]].max(axis=1)
    df_sorted = df.sort_values(
        by=["_combined", "_abs_icir", "_sfr"],
        ascending=[False, False, True],
        na_position="last",
    )
    return df_sorted.index
