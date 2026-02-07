from pprint import pprint

import qlib
import pandas as pd

# import agent.qlib_contrib.qlib_extend_ops
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.loader import QlibDataLoader

from data.worker.factors_portfolio_test import (
    backtest_by_single_alpha,
    backtest_daily,
    get_portfolio_analysis,
)


def backtest_linear_alpha_combination(
    alpha_factor_list: list[dict],
    topk=50,
    n_drop=5,
    data_path="~/.qlib/qlib_data/cn_data",
    instruments="csi300",
    region="cn",
    benchmark="SH000300",
    account=10000,
    exchange_kwargs: dict = None,
):
    """
    Run backtest for a linear combination of daily alpha factors.

    Supports two configuration formats:

    Format 1 (Legacy - daily configurations):
    Each dict in alpha_factor_list represents ONE trading day:
        Option 1 (legacy single factor):
        {
            "date": "YYYY-MM-DD",
            "factor_expr": "QlLib expression string",
            "ref": "YYYY-MM-DD"   # optional: copy strategy from another day
        }

        Option 2 (multiple factors with weights):
        {
            "date": "YYYY-MM-DD",
            "factors": [
                {"expr": "factor_expression_1", "weight": 0.5},
                {"expr": "factor_expression_2", "weight": 0.3},
                {"expr": "factor_expression_3", "weight": 0.2}
            ],
            "ref": "YYYY-MM-DD"   # optional: copy configuration from another day
        }

    Format 2 (New - period-based):
    Each dict represents a period:
        {
            "factor_expr": "single Qlib expression string" OR ["expr1", "expr2", "expr3"],
            "weights": [0.5, 0.3, 0.2],  # optional, same length as factor_expr list
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"
        }

    Notes:
        - Weights are automatically normalized to sum to 1.0
        - Performs only one backtest across combined results.

    Returns:
        analysis_df, report_normal, positions_normal
    """
    qlib.init(provider_uri=data_path, region=region)

    # --- Step 1: preprocess and expand configurations ---
    expanded_configs = []

    for item in alpha_factor_list:
        if "start_date" in item and "end_date" in item:
            # Format 2: Period-based configuration
            factor_expr = item.get("factor_expr")
            weights = item.get("weights", None)
            start_date = item["start_date"]
            end_date = item["end_date"]

            # Convert single expression to list format
            if isinstance(factor_expr, str):
                factor_expr = [factor_expr]
                if weights is None:
                    weights = [1.0]
            elif isinstance(factor_expr, list):
                if weights is None:
                    weights = [1.0 / len(factor_expr)] * len(factor_expr)
                elif len(weights) != len(factor_expr):
                    raise ValueError(
                        f"Weights length ({len(weights)}) must match factor_expr length ({len(factor_expr)})"
                    )

            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Create factors configuration
            factors_config = [
                {"expr": expr, "weight": weight}
                for expr, weight in zip(factor_expr, normalized_weights)
            ]

            # Generate daily configurations for the period
            current_date = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            while current_date <= end_dt:
                date_str = current_date.strftime("%Y-%m-%d")
                expanded_configs.append({"date": date_str, "factors": factors_config})
                current_date += pd.Timedelta(days=1)

        else:
            # Format 1: Legacy daily configuration
            expanded_configs.append(item)

    # --- Step 2: preprocess reference mapping ---
    # Build lookup for factor expressions, handle "ref"
    factor_expr = {}
    for item in expanded_configs:
        date = item.get("date")
        expr = item.get("factor_expr")
        ref = item.get("ref")

        if ref:
            if ref not in factor_expr:
                raise ValueError(f"Reference date {ref} not defined before {date}.")
            factor_expr[date] = factor_expr[ref]
        else:
            if "factor_expr" in item:
                factor_expr[date] = [{"expr": item["factor_expr"], "weight": 1.0}]
            elif "factors" in item:
                factors = item["factors"]
                total_weight = sum(f.get("weight", 1.0) for f in factors)
                normalized_factors = []
                for f in factors:
                    weight = f.get("weight", 1.0) / total_weight
                    normalized_factors.append({"expr": f["expr"], "weight": weight})
                factor_expr[date] = normalized_factors
            else:
                raise ValueError(
                    f"Invalid format for date {date}: must have 'factor_expr' or 'factors'"
                )

    # --- Step 2: compute score for each day individually ---
    daily_scores = []

    for date, factors in factor_expr.items():
        if not factors:
            continue

        # Build fields and names for multiple factors
        fields = [f["expr"] for f in factors]
        names = [f"factor_{i}" for i in range(len(factors))]
        weights = [f["weight"] for f in factors]

        labels = ["Ref($close, -2)/Ref($close, -1) - 1"]
        label_names = ["LABEL"]

        data_loader_config = {
            "feature": (fields, names),
            "label": (labels, label_names),
        }
        data_loader = QlibDataLoader(config=data_loader_config)

        # Load one day window
        df = data_loader.load(
            instruments=instruments,
            start_time=date,
            end_time=date,
        )

        feature_df = df["feature"]
        if feature_df.empty:
            print(f"No data available on {date}")
            continue

        # Combine multiple factor scores using weighted sum
        combined_score = sum(
            feature_df.iloc[:, i] * weights[i] for i in range(len(factors))
        )

        # Create result dataframe
        result_df = pd.DataFrame(
            {"combined_score": combined_score, "date": date}, index=feature_df.index
        )

        daily_scores.append(result_df)

    if not daily_scores:
        raise ValueError("No valid daily scores computed!")

    # --- Step 3: combine all daily scores ---
    all_scores = pd.concat(daily_scores)
    all_scores = (
        all_scores.reset_index()
        .drop_duplicates(subset=["datetime", "instrument"], keep="last")
        .set_index(["datetime", "instrument"])
        .sort_index()
    )
    all_scores = all_scores.rename(columns={"score": "combined_score"})

    # --- Step 4: one unified backtest ---
    STRATEGY_CONFIG = {
        "topk": topk,
        "n_drop": n_drop,
        "signal": all_scores["combined_score"],
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

    start_time = min(factor_expr.keys())
    end_time = max(factor_expr.keys())

    report_normal, positions_normal = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy_obj,
        benchmark=benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
    )

    analysis_df = get_portfolio_analysis(report_normal)

    return analysis_df, report_normal, positions_normal


def backtest_single_alpha_pipeline(
    alpha_factor_list: list[dict],
    topk=50,
    n_drop=5,
    data_path="~/.qlib/qlib_data/cn_data",
    instruments="csi300",
    region="cn",
    benchmark="SH000300",
    account=10000,
    exchange_kwargs: dict = None,
):
    """
    Run backtest on multiple alpha factors sequentially (each period/expr defined in list),
    combine all scores into one signal, then perform ONE final backtest.

    Each dict in alpha_factor_list should be:
        {
            "factor_expr": "some qlib expression",
            "start_time": "YYYY-MM-DD",
            "end_time": "YYYY-MM-DD"
        }

    Returns:
        analysis_df, report_normal, positions_normal
    """

    # --- init qlib ---
    qlib.init(provider_uri=data_path, region=region)

    # --- collect all periods ---
    all_scores = []

    for i, alpha_info in enumerate(alpha_factor_list):
        factor_expr = alpha_info["factor_expr"]
        start_time = alpha_info["start_time"]
        end_time = alpha_info["end_time"]

        print(
            f"🚀 [{i+1}/{len(alpha_factor_list)}] Loading factor: {factor_expr} ({start_time} → {end_time})"
        )

        fields = [factor_expr]
        names = [f"alpha_{i}"]

        labels = ["Ref($close, -2)/Ref($close, -1) - 1"]
        label_names = ["LABEL"]

        data_loader_config = {
            "feature": (fields, names),
            "label": (labels, label_names),
        }
        data_loader = QlibDataLoader(config=data_loader_config)

        df = data_loader.load(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
        )

        feature_df = df["feature"].copy()
        feature_df.columns = ["score"]
        feature_df["period"] = f"{start_time}_{end_time}"

        feature_df = feature_df.copy()
        feature_df["__order__"] = i
        all_scores.append(feature_df)

    # --- combine all scores ---
    all_scores = pd.concat(all_scores)
    all_scores = all_scores.sort_index()

    # Handle overlaps: keep the latest (highest __order__) if same (datetime, instrument)
    all_scores = (
        all_scores.reset_index()
        .sort_values(["datetime", "instrument", "__order__"])
        .drop_duplicates(subset=["datetime", "instrument"], keep="last")
        .set_index(["datetime", "instrument"])
        .sort_index()
    )

    # Rename and clean up
    all_scores = all_scores.rename(columns={"score": "combined_score"})
    all_scores = all_scores[["combined_score", "period"]]  # keep useful cols only

    # --- one unified backtest ---
    STRATEGY_CONFIG = {
        "topk": topk,
        "n_drop": n_drop,
        "signal": all_scores["combined_score"],
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

    report_normal, positions_normal = backtest_daily(
        start_time=alpha_factor_list[0]["start_time"],
        end_time=alpha_factor_list[-1]["end_time"],
        strategy=strategy_obj,
        benchmark=benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
    )

    analysis_df = get_portfolio_analysis(report_normal)

    return analysis_df, report_normal, positions_normal


if __name__ == "__main__":
    # Example 1: Legacy single factor format
    print("=== Example 1: Single Factor Backtest ===")
    alpha_factors_single = [
        {"date": "2017-03-01", "factor_expr": "Div($close, Mean($close, 5))"},
        {"date": "2017-03-02", "factor_expr": "Sub($close, Ref($close, 1))"},
        {"date": "2017-03-03", "ref": "2017-03-02"},  # copy previous day's strategy
        {
            "date": "2017-03-06",
            "factor_expr": "Mean(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), 7)",
        },
    ]

    try:
        analysis_df, reports, positions = backtest_linear_alpha_combination(
            alpha_factor_list=alpha_factors_single,
            topk=20,
            n_drop=5,
            data_path="~/.qlib/qlib_data/cn_data",
            instruments="csi300",
            region="cn",
            benchmark="SH000300",
        )
        print("Single factor backtest performance:")
        pprint(analysis_df)
    except Exception as e:
        print(f"Single factor backtest failed: {e}")

    # Example 2: Multiple factors with weights (legacy format)
    print("\n=== Example 2: Multiple Factors with Weights (Legacy) ===")
    alpha_factors_multi = [
        {
            "date": "2017-03-01",
            "factors": [
                {"expr": "Div($close, Mean($close, 5))", "weight": 0.6},
                {"expr": "Sub($close, Ref($close, 1))", "weight": 0.4},
            ],
        },
        {
            "date": "2017-03-02",
            "factors": [
                {"expr": "Div($close, Mean($close, 5))", "weight": 0.3},
                {"expr": "Sub($close, Ref($close, 1))", "weight": 0.4},
                {"expr": "Mean($volume, 10)", "weight": 0.3},
            ],
        },
        {
            "date": "2017-03-03",
            "ref": "2017-03-02",  # copy previous day's factor configuration
        },
        {
            "date": "2017-03-06",
            "factors": [
                {
                    "expr": "Mean(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), 7)",
                    "weight": 1.0,
                }
            ],
        },
    ]

    try:
        analysis_df, reports, positions = backtest_linear_alpha_combination(
            alpha_factor_list=alpha_factors_multi,
            topk=20,
            n_drop=5,
            data_path="~/.qlib/qlib_data/cn_data",
            instruments="csi300",
            region="cn",
            benchmark="SH000300",
        )
        print("Multiple factors backtest performance:")
        pprint(analysis_df)
    except Exception as e:
        print(f"Multiple factors backtest failed: {e}")

    # Example 3: New period-based format
    print("\n=== Example 3: Period-Based Format ===")
    alpha_factors_period = [
        {
            "factor_expr": [
                "Div($close, Mean($close, 5))",
                "Sub($close, Ref($close, 1))",
            ],
            "weights": [0.7, 0.3],
            "start_date": "2017-03-01",
            "end_date": "2017-03-10",
        },
        {
            "factor_expr": "Mean(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), 7)",  # Single factor
            "start_date": "2017-03-13",
            "end_date": "2017-03-15",
        },
    ]

    try:
        analysis_df, reports, positions = backtest_linear_alpha_combination(
            alpha_factor_list=alpha_factors_period,
            topk=20,
            n_drop=5,
            data_path="~/.qlib/qlib_data/cn_data",
            instruments="csi300",
            region="cn",
            benchmark="SH000300",
        )
        print("Period-based backtest performance:")
        pprint(analysis_df)
    except Exception as e:
        print(f"Period-based backtest failed: {e}")
