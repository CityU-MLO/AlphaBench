from typing import Any, Dict, Tuple
import qlib
from multiprocessing import Pool, cpu_count

# import agent.qlib_contrib.qlib_extend_ops
from qlib.constant import REG_CN
from qlib.data.dataset.loader import QlibDataLoader
import pandas as pd
import os
import numpy as np
import time

DEFAULT_PROVIDER_URI = os.environ.get(
    "QLIB_PROVIDER_URI", os.path.expanduser("~/.qlib/qlib_data/cn_data")
)
DEFAULT_REGION = os.environ.get("QLIB_REGION", "cn")
DEFAULT_INSTRUMENTS = os.environ.get("QLIB_INSTRUMENTS", "CSI300")

CACHE_PATH = os.environ.get("FACTOR_API_CACHE_PATH", "factor_cache.sqlite")
CACHE_MAX_ENTRIES = int(
    os.environ.get("FACTOR_API_CACHE_MAX_ENTRIES", "50000")
)  # LRU target
CACHE_PRUNE_BATCH = int(
    os.environ.get("FACTOR_API_CACHE_PRUNE_BATCH", "5000")
)  # delete this many when over

CPU_JOBS = max(1, int(os.environ.get("FACTOR_API_CPU_JOBS", str(os.cpu_count() or 4))))


try:
    provider_uri = os.environ.get("QLIB_DATA_PATH", "~/.qlib/qlib_data/cn_data")
    print(f"Using QLIB_DATA_PATH: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    print(f"Qlib initialized with data provider at {provider_uri}")
    QLIB_VALID_FLAG = True
except Exception as e:
    print(f"Error initializing Qlib: {e}")
    QLIB_VALID_FLAG = False
    raise


def _ir(mean_val: float, std_val: float) -> float:
    if std_val is None or not np.isfinite(std_val) or std_val <= 0:
        return float(0)
    return float(mean_val / std_val)


def summarize_ic_rankic(
    ic_daily: pd.Series, rankic_daily: pd.Series
) -> Dict[str, float]:
    ic_mean = float(ic_daily.mean()) if len(ic_daily) else float(0)
    ic_std = float(ic_daily.std(ddof=1)) if len(ic_daily) > 1 else float(0)
    rankic_mean = float(rankic_daily.mean()) if len(rankic_daily) else float(0)
    rankic_std = float(rankic_daily.std(ddof=1)) if len(rankic_daily) > 1 else float(0)
    return {
        "ic": ic_mean,
        "icir": _ir(ic_mean, ic_std),
        "rank_ic": rankic_mean,
        "rank_icir": _ir(rankic_mean, rankic_std),
        "n_dates": int(len(ic_daily.index.unique())),
    }


def _daily_turnover(feature_s: pd.Series) -> pd.Series:
    """
    Compute rank-based turnover between consecutive dates.

    Turnover_t = 1 - spearman_corr(f_t, f_{t-1})

    Operates directly on the factor score series (MultiIndex: datetime, instrument).
    Uses aligned asset universes at t and t-1, handles missing values.
    """
    groups = feature_s.dropna().groupby(level="datetime")
    dates = sorted(groups.groups.keys())

    turnovers = {}
    prev_scores = None

    for date in dates:
        curr_scores = groups.get_group(date).droplevel("datetime")

        if prev_scores is not None:
            common_idx = curr_scores.index.intersection(prev_scores.index)
            if len(common_idx) >= 5:
                c = curr_scores.loc[common_idx]
                p = prev_scores.loc[common_idx]
                rho = c.rank().corr(p.rank())
                if np.isfinite(rho):
                    turnovers[date] = 1.0 - rho

        prev_scores = curr_scores

    return pd.Series(turnovers, dtype=float)


def _calc_ic_rankic_for_group(group):
    f, y = group["f"], group["y"]
    ic = f.corr(y)
    rankic = f.rank().corr(y.rank())
    return group.index.get_level_values("datetime")[0], ic, rankic


def _daily_ic_rankic(
    feature_s: pd.Series, label_s: pd.Series, numworker: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute daily IC (Pearson) and RankIC (Spearman via ranks) by date.
    Both inputs are aligned Series with a MultiIndex containing 'datetime'.

    Parameters
    ----------
    feature_s : pd.Series
        Feature (signal) series with MultiIndex containing 'datetime'.
    label_s : pd.Series
        Label (return) series with MultiIndex containing 'datetime'.
    numworker : int, default=1
        Number of worker processes. If >1, use multiprocessing for parallel computation.

    Returns
    -------
    ic_daily : pd.Series
        Daily IC values (Pearson correlation by date).
    rankic_daily : pd.Series
        Daily RankIC values (Spearman correlation by date).
    """

    df = pd.concat({"f": feature_s, "y": label_s}, axis=1).dropna()
    if df.empty:
        print("Empty dataframe")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if numworker <= 1:
        g = df.groupby(level="datetime", sort=False)
        ic_daily = g.apply(lambda x: x["f"].corr(x["y"]))
        rankic_daily = g.apply(lambda x: x["f"].rank().corr(x["y"].rank()))
        return ic_daily, rankic_daily

    groups = [g for _, g in df.groupby(level="datetime", sort=False)]

    numworker = min(numworker, cpu_count())
    with Pool(numworker) as pool:
        results = pool.map(_calc_ic_rankic_for_group, groups)

    dates, ic, rankic = zip(*results)
    ic_daily = pd.Series(ic, index=dates)
    rankic_daily = pd.Series(rankic, index=dates)

    return ic_daily, rankic_daily


def compute_factor_data(
    factor_definitions,
    label="close_return",
    instruments="csi300",
    start_time="2021-01-01",
    end_time="2025-01-01",
):
    """
    Compute factor data and label data using QlibDataLoader.

    :param factor_definitions: List of dicts, each with 'name' and 'expression'
                               e.g. [{'name': 'factor1', 'expression': 'Rank(Div($high, $low), 10)'}]
    :param label: String, label expression, e.g. 'Ref($close, -2)/Ref($close, -1) - 1'
    :param instruments: String, instruments universe, default 'csi300'
    :param start_time: String, start date
    :param end_time: String, end date
    :return: pandas DataFrame containing factor data and label data
    """
    # Extract fields and names from factor_definitions
    fields = [f["expression"] for f in factor_definitions]
    names = [f["name"] for f in factor_definitions]

    # Prepare label
    if label == "close_return_lag":
        label = "Ref($close, -2)/Ref($close, -1) - 1"
    elif label == "close_return":
        label = "Ref($close, -1)/$close - 1"
    elif label == "close":
        label = "Ref($close, -1)"
    else:
        raise ValueError(
            f"Unsupported label type: {label}. Supported types are 'close_return' or 'close'."
        )

    labels = [label]
    label_names = ["LABEL"]

    # Build config
    data_loader_config = {"feature": (fields, names), "label": (labels, label_names)}

    # Load data
    data_loader = QlibDataLoader(config=data_loader_config)
    try:
        df = data_loader.load(
            instruments=instruments, start_time=start_time, end_time=end_time
        )

        if df.empty:
            print("Error: Loaded DataFrame is empty.")
            return None
        return df
    except Exception as e:
        print(f"Exception during factor computation: {e}")
        return None


def performance_single_alpha(
    expr="Rank(Div($high, $low), 10)",
    market="CSI300",
    start="2020-01-01",
    end="2020-01-23",
    label="close_return",
    num_worker=1,
) -> Dict[str, Any]:
    """
    Runs inside child process. Heavy libs are imported here, so the parent can hard-kill safely.
    """
    import qlib
    from backtest.qlib.dataloader import compute_factor_data

    # Per-process init (safer across OS / forks)
    qlib.init(provider_uri=DEFAULT_PROVIDER_URI, region=DEFAULT_REGION)

    factor_list = [{"name": "api_factor", "expression": expr}]
    out = compute_factor_data(
        factor_list,
        label=label,
        instruments=market.lower(),
        start_time=start,
        end_time=end,
    )

    if (
        out is None
        or "feature" not in out
        or "label" not in out
        or "api_factor" not in out["feature"]
        or "LABEL" not in out["label"]
    ):
        raise RuntimeError("Missing feature/label output from compute_factor_data")

    f_s: pd.Series = out["feature"]["api_factor"]
    y_s: pd.Series = out["label"]["LABEL"]

    ic_d, rankic_d = _daily_ic_rankic(f_s, y_s, num_worker=num_worker)
    metrics = summarize_ic_rankic(ic_d, rankic_d)

    # Compute rank-based turnover from factor scores
    turnover_d = _daily_turnover(f_s)
    mean_turnover = float(turnover_d.mean()) if len(turnover_d) else 0.0

    return {
        "success": True,
        "expression": expr,
        "market": market,
        "start_date": start,
        "end_date": end,
        "metrics": {
            "ic": metrics["ic"],
            "rank_ic": metrics["rank_ic"],
            "ir": metrics["icir"],  # backward compatible alias
            "icir": metrics["icir"],
            "rank_icir": metrics["rank_icir"],
            "turnover": mean_turnover,
            "n_dates": metrics["n_dates"],
        },
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "raw": {"ic_daily": ic_d, "rankic_daily": rankic_d, "turnover_daily": turnover_d},
    }


if __name__ == "__main__":
    # Test query factor computed data
    factor_definitions = [
        {
            "name": "vol_momentum",
            "expression": "(Sum(Greater($volume-Ref($volume, 1), 0), 10)-Sum(Greater(Ref($volume, 1)-$volume, 0), 10))/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
        },
        {"name": "price_ratio", "expression": "Rank(Div($high, $low), 10)"},
    ]

    label = "close_return"

    # Single alpha IC
    df = compute_factor_data(
        factor_definitions, label, start_time="2020-01-01", end_time="2022-01-23"
    )
    print("Loaded factor data:")
    print(df.head())

    factor_df = df["feature"]["vol_momentum"]
    label_df = df["label"]["LABEL"]

    # === Single ===
    start_time = time.time()
    ic_1, rankic_1 = _daily_ic_rankic(factor_df, label_df, numworker=1)
    t1 = time.time() - start_time
    print(f"\nSingle-thread time: {t1:.3f}s")

    # === Parallel (8) ===
    start_time = time.time()
    ic_8, rankic_8 = _daily_ic_rankic(factor_df, label_df, numworker=8)
    t8 = time.time() - start_time
    print(f"Parallel (8 workers) time: {t8:.3f}s")

    ic_diff = (ic_1 - ic_8).abs().max()
    rankic_diff = (rankic_1 - rankic_8).abs().max()

    print("\n=== Consistency Check ===")
    print(f"IC diff max: {ic_diff:.6e}")
    print(f"RankIC diff max: {rankic_diff:.6e}")

    same_ic = np.allclose(ic_1, ic_8, atol=1e-10, equal_nan=True)
    same_rankic = np.allclose(rankic_1, rankic_8, atol=1e-10, equal_nan=True)

    if same_ic and same_rankic:
        print("✅ IC and RankIC results are identical.")
    else:
        print("⚠️ Results differ slightly (possible float or group order difference).")

    print("\n=== Summary ===")
    print(f"Speedup: {t1/t8:.2f}x")

    # Single alpha overall performance
    report = performance_single_alpha(
        expr="Rank(Div($high, $low), 10)",
        market="CSI300",
        start="2020-01-01",
        end="2020-01-23",
        label="close_return",
        num_worker=8,
    )

    print(report)
