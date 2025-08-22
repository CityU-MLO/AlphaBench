#!/usr/bin/env python
"""
REST API for Factor Evaluation
This API provides endpoints for evaluating factor expressions during search processes.
"""


from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
import traceback
import logging

import os
import re
import sys
import qlib
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from urllib.parse import unquote
from flask import jsonify, request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # optional, for progress bar
from qlib.data.dataset.loader import QlibDataLoader
from backtest.factor_metrics import (
    FL_Ic,
    FL_RankIc,
    FL_Ir,
    FL_Icir,
    FL_RankIcir,
    FL_QuantileReturn,
    FL_Turnover,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation functions
from backtest.qlib.dataloader import compute_factor_data
from backtest.factor_metrics import get_performance
import threading

from api.utils import compute_ic_for_col, summarize_ic_tables, qlib_load_data_thread


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "market": "csi300",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "label": "close_return",
    "cache_enabled": True,
}

# Simple in-memory cache for factor results
factor_cache = {}
MAX_CACHE_SIZE = 100000
MAX_WORKERS = 8  # tune this according to your CPU cores

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")


logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 120


def _default_fail_result(
    expr, market, start_date, end_date, error_msg="Factor computation failed"
):
    return {
        "success": False,
        "error": error_msg,
        "expression": expr,
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "metrics": {
            "ic": 0.0,
            "rank_ic": 0.0,
            "ir": 0.0,
            "icir": 0.0,
            "rank_icir": 0.0,
            "turnover": 1.0,
        },
        "timestamp": datetime.now().isoformat(),
    }


def _safe_ir(mean_val: float, std_val: float) -> float:
    return (
        float(mean_val / std_val)
        if (std_val is not None and std_val > 0)
        else float("nan")
    )


def _compute_daily_ic_rankic(
    feature_s: pd.Series, label_s: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized daily IC (Pearson) and RankIC (Spearman via ranks) for one factor.
    Both feature_s and label_s must share the same MultiIndex with level 'datetime'
    (and typically 'instrument' or similar as the other level).
    """
    # Align and drop NaNs once
    df = pd.concat({"factor": feature_s, "label": label_s}, axis=1).dropna()

    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Group by date
    g = df.groupby(level="datetime", sort=False)

    # IC: Pearson per date
    ic_by_date = g.apply(lambda x: x["factor"].corr(x["label"]))

    # RankIC: Spearman via within-date ranks, then Pearson
    # Rank once per date for both columns, then correlate
    def _rankic_group(x: pd.DataFrame) -> float:
        fr = x["factor"].rank(method="average")
        lr = x["label"].rank(method="average")
        return fr.corr(lr)

    rankic_by_date = g.apply(_rankic_group)

    return ic_by_date, rankic_by_date


def evaluate_factor_expr(
    expr: str,
    market: str,
    start_date: str,
    end_date: str,
    label: str = "close_return",
    use_cache: bool = True,
) -> dict:
    """
    Evaluate a factor expression and return performance metrics.
    Optimization:
      - compute daily IC & RankIC first, then summarize (mean / IR)
      - vectorized per-date computations (groupby), no Python loops per-row
      - cache results
      - hard timeout on compute_factor_data
    """
    cache_key = f"{expr}||{market}||{start_date}||{end_date}||{label}"

    if use_cache and cache_key in factor_cache:
        logger.info("Cache hit for expression: %s...", expr[:60])
        return factor_cache[cache_key]

    try:
        fixed_expr = expr  # plug in your fixer if you have one
        factor_list = [{"name": "api_factor", "expr": fixed_expr}]

        def _compute():
            return compute_factor_data(
                factor_list,
                label=label,
                instruments=market.lower(),
                start_time=start_date,
                end_time=end_date,
            )

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_compute)
            try:
                output = future.result(timeout=TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                logger.error(
                    "Timeout (%ss) for expression: %r", TIMEOUT_SECONDS, expr[:120]
                )
                return _default_fail_result(
                    expr, market, start_date, end_date, error_msg="Timeout"
                )

        if (
            output is None
            or "feature" not in output
            or "label" not in output
            or "api_factor" not in output["feature"]
            or "LABEL" not in output["label"]
        ):
            return _default_fail_result(
                expr, market, start_date, end_date, error_msg="Missing feature/label"
            )

        # Extract series (expect MultiIndex with level 'datetime')
        factor_s: pd.Series = output["feature"]["api_factor"]
        label_s: pd.Series = output["label"]["LABEL"]

        # Compute daily IC / RankIC efficiently
        ic_daily, rankic_daily = _compute_daily_ic_rankic(factor_s, label_s)

        # Summaries
        ic_mean = float(ic_daily.mean()) if not ic_daily.empty else float("nan")
        ic_std = float(ic_daily.std(ddof=1)) if len(ic_daily) > 1 else float("nan")
        icir = _safe_ir(ic_mean, ic_std)

        rankic_mean = (
            float(rankic_daily.mean()) if not rankic_daily.empty else float("nan")
        )
        rankic_std = (
            float(rankic_daily.std(ddof=1)) if len(rankic_daily) > 1 else float("nan")
        )
        rankicir = _safe_ir(rankic_mean, rankic_std)

        result = {
            "success": True,
            "expression": expr,
            "fixed_expression": fixed_expr,
            "market": market,
            "start_date": start_date,
            "end_date": end_date,
            "metrics": {
                "ic": ic_mean,
                "rank_ic": rankic_mean,
                # keep 'ir' key if downstream expects it; here we mirror icir to stay backward-compatible
                "ir": icir,
                "icir": icir,
                "rank_icir": rankicir,
                "turnover": 0.0,  # fill if you compute it elsewhere
                "n_dates": int(len(ic_daily.index.unique())),
            },
            "timestamp": datetime.now().isoformat(),
        }

        if use_cache and len(factor_cache) < MAX_CACHE_SIZE:
            factor_cache[cache_key] = result

        return result

    except Exception as e:
        logger.error("Error evaluating factor: %s", str(e))
        logger.error(traceback.format_exc())
        return _default_fail_result(
            expr, market, start_date, end_date, error_msg=str(e)
        )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "Factor Evaluation API",
            "timestamp": datetime.now().isoformat(),
            "cache_size": len(factor_cache),
        }
    )


@app.route("/check", methods=["POST"])
def check_factor():
    data = request.get_json()
    expr = data.get("expr", "")

    if not expr:
        return (
            jsonify(
                {
                    "success": False,
                    "error_message": "Missing required parameter: expr",
                    "error_type": "EMPTY_EXPR",
                }
            ),
            400,
        )

    fields = [expr]
    names = ["test_expr"]
    data_loader_config = {"feature": (fields, names)}

    result_container = []
    timeout = 30

    thread = threading.Thread(
        target=qlib_load_data_thread, args=(result_container, data_loader_config)
    )
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        error_msg = f"Timeout: execution exceeded {timeout}s"
        return (
            jsonify(
                {"success": False, "error_message": error_msg, "error_type": "TIMEOUT"}
            ),
            200,
        )

    if not result_container:
        error_msg = "No result returned from data loader."
        return (
            jsonify(
                {
                    "success": False,
                    "error_message": error_msg,
                    "error_type": "DATA_LOADER_ERROR",
                }
            ),
            200,
        )

    success, result = result_container[0]
    if not success:
        # Read error message from Qlib
        if re.search(r"missing \d+ required positional argument", result):
            return (
                jsonify(
                    {
                        "success": False,
                        "error_message": result,
                        "error_type": "INVALID_PARA",
                    }
                ),
                200,
            )
        elif re.search(r"The operator \[.*?\] is not registered", result):
            return (
                jsonify(
                    {
                        "success": False,
                        "error_message": result,
                        "error_type": "UNREGISTERED_OPERATOR",
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error_message": result,
                        "error_type": "UNKNOWN_ERROR",
                    }
                ),
                200,
            )

    df = result
    ndf = df["feature"]
    if ndf.empty:
        error_msg = "Loaded DataFrame is empty. Which means factor doesn't contribute to any output. Test failed."
        return (
            jsonify(
                {
                    "success": False,
                    "error_message": error_msg,
                    "error_type": "EMPTY_DATA",
                }
            ),
            500,
        )

    nan_ratio = ndf.isna().mean().mean()
    if nan_ratio > 0.01:
        error_msg = f"High NaN ratio: {nan_ratio:.2%}. Test failed."
        return (
            jsonify(
                {
                    "success": False,
                    "error_message": error_msg,
                    "error_type": "HIGH_NAN_RATIO",
                }
            ),
            500,
        )
    elif nan_ratio > 0.001:
        return jsonify({"success": True}), 200
    else:
        return jsonify({"success": True}), 200


@app.route("/eval", methods=["GET", "POST"])
def evaluate_factor():
    """
    Evaluate a factor expression.
    
    GET parameters:
        expr: Factor expression (URL encoded)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        market: Market identifier (default: csi300)
        label: Label type (default: close_return)
        use_cache: Whether to use cache (default: true)
    
    POST body (JSON):
        {
            "expr": "factor expression",
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
            "market": "csi300",
            "label": "close_return",
            "use_cache": true
        }
    """
    try:
        if request.method == "GET":
            # Parse GET parameters
            expr = request.args.get("expr", "")
            # Remove quotes if present
            expr = expr.strip('"').strip("'")
            # URL decode
            expr = unquote(expr)

            start_date = request.args.get("start", DEFAULT_CONFIG["start_date"])
            start_date = start_date.strip("'").strip('"')

            end_date = request.args.get("end", DEFAULT_CONFIG["end_date"])
            end_date = end_date.strip("'").strip('"')

            market = request.args.get("market", DEFAULT_CONFIG["market"])
            market = market.strip("'").strip('"').lower()

            label = request.args.get("label", DEFAULT_CONFIG["label"])
            use_cache = request.args.get("use_cache", "true").lower() == "true"

        else:  # POST
            data = request.get_json()
            expr = data.get("expr", "")
            start_date = data.get("start", DEFAULT_CONFIG["start_date"])
            end_date = data.get("end", DEFAULT_CONFIG["end_date"])
            market = data.get("market", DEFAULT_CONFIG["market"]).lower()
            label = data.get("label", DEFAULT_CONFIG["label"])
            use_cache = data.get("use_cache", True)

        # Validate required parameters
        if not expr:
            return (
                jsonify(
                    {"success": False, "error": "Missing required parameter: expr"}
                ),
                400,
            )

        # Log request
        logger.info(
            f"Evaluating factor: {expr[:100]}... for {market} from {start_date} to {end_date}"
        )

        # Evaluate factor
        result = evaluate_factor_expr(
            expr=expr,
            market=market,
            start_date=start_date,
            end_date=end_date,
            label=label,
            use_cache=use_cache,
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/batch_eval", methods=["POST"])
def batch_evaluate_factors():
    """
    Evaluate multiple factor expressions in batch.
    
    POST body (JSON):
        {
            "factors": [
                {"name": "factor1", "expr": "expression1"},
                {"name": "factor2", "expr": "expression2"}
            ],
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
            "market": "csi300",
            "label": "close_return",
            "use_cache": true
        }
    """
    try:
        data = request.get_json()
        factors = data.get("factors", [])
        start_date = data.get("start", DEFAULT_CONFIG["start_date"])
        end_date = data.get("end", DEFAULT_CONFIG["end_date"])
        market = data.get("market", DEFAULT_CONFIG["market"]).lower()
        label = data.get("label", DEFAULT_CONFIG["label"])
        use_cache = data.get("use_cache", True)

        if not factors:
            return jsonify({"success": False, "error": "No factors provided"}), 400

        fields = [f["expr"] for f in factors]
        names = [f["name"] for f in factors]
        data_loader = QlibDataLoader(
            config={
                "feature": (fields, names),
                "label": (["Ref($close, -1) / $close -1"], ["returns"]),
            }
        )

        factor_data = data_loader.load(
            instruments="CSI300", start_time=start_date, end_time=end_date
        )
        # Parallel compute
        factor_data = factor_data.fillna(0)
        returns = factor_data[("label", "returns")]
        results = {}

        for metric_name, metric_fn in [("IC", FL_Ic), ("RankIC", FL_RankIc)]:
            out = Parallel(n_jobs=16)(
                delayed(compute_ic_for_col)(factor_data, col, metric_fn, returns)
                for col in factor_data["feature"].columns
            )

            metric_values = [v for _, v in out if v is not None]
            metric_cols = [c for c, v in out if v is not None]
            results[metric_name] = pd.concat(metric_values, axis=1)
            results[metric_name].columns = metric_cols

        ic_table = results["IC"]
        rankic_table = results["RankIC"]
        results = summarize_ic_tables(ic_table, rankic_table)

        return jsonify(
            {
                "success": True,
                "count": len(results),
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Batch API error: {str(e)}")
        logger.error(traceback.format_exc())

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear the factor evaluation cache."""
    global factor_cache
    factor_cache = {}

    return jsonify(
        {
            "success": True,
            "message": "Cache cleared",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/cache_stats", methods=["GET"])
def cache_stats():
    """Get cache statistics."""
    return jsonify(
        {
            "cache_size": len(factor_cache),
            "max_cache_size": MAX_CACHE_SIZE,
            "cache_keys": list(factor_cache.keys())[:10],  # Show first 10 keys
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get("PORT", 9888))
    debug = os.environ.get("DEBUG", "False").lower() == "true"

    print(f"Starting Factor Evaluation API on port {port}")
    print(f"Debug mode: {debug}")
    print(
        f"Example URL: http://localhost:{port}/eval?expr=\"Rank(Corr($close,$volume,10),252)\"&start='2023-01-01'&end='2024-01-01'&market='csi300'"
    )

    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
