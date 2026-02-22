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


def get_portfolio_analysis(report_normal):
    analysis = dict()

    # default frequency will be daily (i.e. "day")
    analysis["benchmark"] = risk_analysis(report_normal["bench"])

    analysis["pure_return_without_cost"] = risk_analysis(report_normal["return"])
    analysis["pure_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["cost"]
    )

    analysis["excess_return_without_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"]
    )
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"]
    )

    analysis_df = pd.concat(analysis)
    return analysis_df


def backtest_by_scores(
    factor_scores,
    topk=50,
    n_drop=5,
    start_time="2017-01-01",
    end_time="2020-08-01",
    data_path="~/.qlib/qlib_data/cn_data",
    region="cn",
    BENCH="SH000300",
):

    qlib.init(provider_uri=data_path, region=region)
    STRATEGY_CONFIG = {"topk": topk, "n_drop": n_drop, "signal": factor_scores}

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = backtest_daily(
        start_time=start_time, end_time=end_time, strategy=strategy_obj, benchmark=BENCH
    )

    analysis_df = get_portfolio_analysis(report_normal)

    return analysis_df


def backtest_by_single_alpha(
    alpha_factor,
    topk=50,
    n_drop=5,
    start_time="2017-01-01",
    end_time="2020-08-01",
    data_path="~/.qlib/qlib_data/cn_data",
    instruments="csi300",
    region="cn",
    BENCH="SH000300",
):
    """
    Run a simple backtest using a single alpha factor.

    Args:
        alpha_factor (str): The alpha factor expression to test, e.g. "Ref($close, -1)/$close - 1".
        topk (int): Number of top-ranked stocks to long each day.
        n_drop (int): Number of worst-ranked stocks to drop each day.
        start_time (str): Start date for the backtest period.
        end_time (str): End date for the backtest period.
        data_path (str): Path to Qlib data directory.
        instruments (str): The market or index to select stocks from (e.g. "csi300").
        region (str): Data region (e.g. "cn" for China market).
        BENCH (str): Benchmark index symbol, e.g. "SH000300".

    Returns:
        tuple:
            - analysis_df (pd.DataFrame): A summary of portfolio performance metrics
              such as mean return, volatility, annualized return, information ratio,
              and max drawdown across multiple types of returns (benchmark, pure, excess).
              Example structure:

                  risk
                  benchmark                  mean               0.000007
                                             std                0.013023
                                             annualized_return  0.001777
                                             information_ratio  0.008845
                                             max_drawdown      -0.473380
                  pure_return_with_cost      mean               0.000529
                                             std                0.017519
                                             annualized_return  0.125872
                                             information_ratio  0.465738
                                             max_drawdown      -0.506976

            - report_normal (pd.DataFrame): Daily portfolio backtest records including:
                - account (float): portfolio total value
                - return (float): daily return
                - total_turnover / turnover (float): trading activity
                - total_cost / cost (float): transaction costs
                - value / cash (float): position and cash holdings
                - bench (float): benchmark daily return

              Example row:
                  datetime     account        return  total_turnover  turnover ...
                  2020-01-02  1.000000e+08  0.000000e+00    0.000000e+00  ...

            - positions_normal (dict): Dictionary of daily positions (holdings) produced by
              the strategy, where keys are dates and values contain account position details.

    """

    fields = [alpha_factor]
    names = ["test_expr"]

    labels = ["Ref($close, -2)/Ref($close, -1) - 1"]  # label
    label_names = ["LABEL"]

    data_loader_config = {"feature": (fields, names), "label": (labels, label_names)}

    qlib.init(provider_uri=data_path, region=region)
    data_loader = QlibDataLoader(config=data_loader_config)
    df = data_loader.load(
        instruments=instruments, start_time=start_time, end_time=end_time
    )

    STRATEGY_CONFIG = {"topk": topk, "n_drop": n_drop, "signal": df["feature"]}

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = backtest_daily(
        start_time=start_time, end_time=end_time, strategy=strategy_obj, benchmark=BENCH
    )

    analysis_df = get_portfolio_analysis(report_normal)

    return analysis_df, report_normal, positions_normal


def backtest_by_single_alpha_in_pool(
    factor_data,
    topk=50,
    n_drop=5,
    start_time="2017-01-01",
    end_time="2020-08-01",
    data_path="~/.qlib/qlib_data/cn_data",
    instruments="csi300",
    region="cn",
    BENCH="SH000300",
):

    factor_names = factor_data["feature"].columns.tolist()
    for name in factor_names:
        factor_scores = factor_data["feature"][name]
        STRATEGY_CONFIG = {"topk": topk, "n_drop": n_drop, "signal": factor_scores}

        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy_obj,
            benchmark=BENCH,
        )
        analysis = dict()

        # default frequency will be daily (i.e. "day")
        analysis["benchmark"] = risk_analysis(report_normal["bench"])

        analysis["pure_return_without_cost"] = risk_analysis(report_normal["return"])
        analysis["pure_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["cost"]
        )

        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"]
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame

    # import pdb;pdb.set_trace()
    return analysis_df


if __name__ == "__main__":
    data_path = "~/.qlib/qlib_data/cn_data"
    region = "cn"

    instruments = "csi300"
    start_time = "2020-01-01"
    end_time = "2022-12-31"

    qlib.init(provider_uri=data_path, region=region)
    # data_loader = Alpha158DL()

    # df = data_loader.load(
    #     instruments=instruments, start_time=start_time, end_time=end_time
    # )
    # backtest_by_single_alpha_in_pool(
    #     factor_data=df,
    #     topk=20,
    #     n_drop=5,
    #     start_time=start_time,
    #     end_time=end_time,
    #     data_path=data_path,
    #     instruments=instruments,
    #     region=region,
    #     BENCH="SH000300",
    # )

    factor_expr = "If(Gt(Corr($close, $volume, 10), 0.5), If(Gt(Delta($close, 20), 0), Mean($close, 20), Mean($close, 10)), Mean($close, 10))"
    # factor_expr = "If(Gt(Sub(EMA($close, 12), EMA($close, 26)), 0), Div($close, $volume + 1e-12), Std(EMA($close, 26), 20))"
    result = backtest_by_single_alpha(
        alpha_factor=factor_expr,
        topk=20,
        n_drop=5,
        start_time="2020-01-01",
        end_time="2022-12-31",
        data_path="~/.qlib/qlib_data/cn_data",
        instruments="csi300",
        region="cn",
        BENCH="SH000300",
    )
    print("Backtest performance (CN):")
    pprint(result)

    result = backtest_by_single_alpha(
        alpha_factor=factor_expr,
        topk=20,
        n_drop=5,
        start_time="2020-01-01",
        end_time="2022-12-31",
        data_path="~/.qlib/qlib_data/us_data_ours",
        instruments="sp500",
        region="us",
        BENCH="^gspc",
    )
    print("Backtest performance (US):")
    pprint(result)
