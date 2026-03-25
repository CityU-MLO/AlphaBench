"""
Simple Web UI for Factor Cache Query and Visualization
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path BEFORE local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.utils

# Local imports
from ffo.utils.factor_cache_manager import FactorCacheManager
from ffo.utils.factor_store import FactorStore
from ffo.utils.utils import expr_hash, _normalize_expr
from ffo.config.manager import get_config
from ffo.client import FactorEvalClient

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("WebApp")


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    else:
        return obj


def build_standardized_export(
    optimization_type: str,
    optimization_results: Dict,
    configuration: Dict,
    factors: List[Dict],
    periods_config: List[Dict],
    continuity_validation: Dict,
    experiment_name: str = "optimization_result",
) -> Dict:
    """
    Build standardized JSON export format for factor optimization results.

    Supports both Category A (fixed) and Category B (dynamic) formats.
    Ready for Category C (review/backtesting) upload.
    """
    from datetime import datetime

    export_data = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "export_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pipeline_version": "1.0",
            "description": experiment_name,
            "optimization_type": optimization_type,
        },
        "configuration": {
            "type": optimization_type,
            "market": configuration.get("market", "csi300"),
            "lookback_window": configuration.get("lookback_window", 60),
            "alpha": configuration.get("alpha", 0.01),
            "methods": configuration.get(
                "methods", ["baseline", "lasso", "ic_optimization"]
            ),
            "include_baseline": configuration.get("include_baseline", True),
            "period_config": {
                "type": optimization_type,
                "continuity_validated": continuity_validation.get("is_valid", True),
                "continuity_warnings": continuity_validation.get("warnings", []),
            },
        },
        "factors": [
            {
                "index": i,
                "formula": f.get("formula") or f.get("name", f"factor_{i}"),
                "name": f.get("name", f"factor_{i}"),
            }
            for i, f in enumerate(factors)
        ],
        "results": {
            "status": optimization_results.get("status", "unknown"),
            "total_periods": optimization_results.get("total_periods", 0),
            "successful_periods": sum(
                1
                for p in optimization_results.get("periods", [])
                if (p.get("baseline") or p.get("lasso") or p.get("ic_optimization"))
            ),
            "periods": optimization_results.get("periods", []),
        },
        "computation_details": {
            "data_loading": {
                "market": configuration.get("market", "csi300"),
                "factor_count": len(factors),
                "valid_factors": len(factors),
                "invalid_factors": 0,
            },
            "weight_computation": {
                "lookback_window": configuration.get("lookback_window", 60),
                "rolling_window": configuration.get("rolling_window", 60),
                "max_iterations": configuration.get("max_iterations", 1000),
                "alpha": configuration.get("alpha", 0.01),
            },
        },
    }

    # Add period-specific config based on type
    if optimization_type == "fixed" and periods_config:
        p = periods_config[0] if isinstance(periods_config, list) else periods_config
        export_data["configuration"]["period_config"]["single_period"] = {
            "train_start": p.get("train_start") or p.get("start"),
            "train_end": p.get("train_end") or p.get("end"),
            "test_start": p.get("test_start") or p.get("start"),
            "test_end": p.get("test_end") or p.get("end"),
        }

    elif optimization_type == "dynamic" and periods_config:
        export_data["configuration"]["period_config"]["dynamic_periods"] = [
            {
                "period_index": i,
                "test_start": p.get("test_start") or p.get("start"),
                "test_end": p.get("test_end") or p.get("end"),
                "training": {
                    "type": (
                        "global_lookback" if optimization_type == "dynamic" else "fixed"
                    ),
                    "lookback_days": configuration.get("lookback_window", 60),
                    "computed_start": p.get("train_start") or p.get("start"),
                    "computed_end": p.get("train_end") or p.get("end"),
                },
            }
            for i, p in enumerate(periods_config)
        ]

    # Calculate success rate
    total = export_data["results"]["total_periods"]
    successful = export_data["results"]["successful_periods"]
    export_data["results"]["success_rate"] = (
        (successful / total * 100) if total > 0 else 0
    )

    return export_data


# Initialize cache manager (kept for query history tracking only)
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache_data"))
cache_manager = FactorCacheManager(cache_dir=cache_dir)

# Incremental daily IC cache + factor score storage (primary data source)
factor_store = FactorStore(cache_dir=cache_dir)

# Initialize API client — base URL from config
_cfg = get_config()
api_client = FactorEvalClient(base_url=_cfg.backend_url)


@app.route("/")
def index():
    """Main page with query interface."""
    return render_template("index.html")


@app.route("/api/search_cache", methods=["POST"])
def search_cache():
    """Search for cached factor results using FactorStore."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")

    eh = expr_hash(_normalize_expr(expr))
    min_d, max_d, count = factor_store.get_coverage(eh, market, label, start_date, end_date)

    if count > 0:
        daily_data = factor_store.get_daily_ic(eh, market, label, start_date, end_date)
        metrics = FactorStore.compute_summary(daily_data)
        result = {
            "success": True,
            "expression": expr,
            "market": market,
            "start_date": start_date,
            "end_date": end_date,
            "metrics": metrics,
            "daily_metrics": daily_data,
            "cached": True,
        }
        return jsonify({"success": True, "data": result, "from_cache": True})

    return jsonify({"success": False, "message": "No cache found for this query"})


@app.route("/api/evaluate_factor", methods=["POST"])
def evaluate_factor():
    """Evaluate a factor expression via backend API."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")
    use_cache = data.get("use_cache", True)
    fast = data.get("fast", True)
    forward_n = max(1, int(data.get("forward_n", 1)))
    topk = max(1, int(data.get("topk", 50)))
    n_drop = max(0, int(data.get("n_drop", 5)))

    # Optional exchange kwargs overrides from frontend
    exchange_overrides = {}
    if "open_cost" in data:
        exchange_overrides["open_cost"] = float(data["open_cost"])
    if "close_cost" in data:
        exchange_overrides["close_cost"] = float(data["close_cost"])
    if "min_cost" in data:
        exchange_overrides["min_cost"] = float(data["min_cost"])
    if "limit_threshold" in data:
        exchange_overrides["limit_threshold"] = data["limit_threshold"]

    try:
        # Client returns a list — extract first item
        results = api_client.evaluate_factor(
            expression=expr,
            market=market,
            start_date=start_date,
            end_date=end_date,
            label=label,
            use_cache=use_cache,
            fast=fast,
            forward_n=forward_n,
            topk=topk,
            n_drop=n_drop,
            exchange_kwargs=exchange_overrides if exchange_overrides else None,
        )
        result = results[0] if isinstance(results, list) and results else results

        if result.get("success"):
            # Log to query history
            metrics = result.get("metrics", {})
            cache_manager.add_to_query_history(
                expression=expr,
                market=market,
                start_date=start_date,
                end_date=end_date,
                ic=metrics.get("ic", 0.0),
                rank_ic=metrics.get("rank_ic", 0.0),
                icir=metrics.get("icir"),
                rank_icir=metrics.get("rank_icir"),
            )
            return jsonify({"success": True, "data": result})
        else:
            # Pass through backend error directly (don't double-wrap)
            error_msg = result.get("error", "Unknown API error")
            return jsonify(
                {"success": False, "message": error_msg}
            )

    except Exception as e:
        logger.exception("evaluate_factor error")
        return jsonify(
            {
                "success": False,
                "message": f"API connection error: {e}. Is the API server running?",
            }
        )


@app.route("/api/batch_evaluate_factor", methods=["POST"])
def batch_evaluate_factor():
    """
    Batch Factor Evaluation — single backend request for all factors.

    Sends all expressions in one request to the backend's /factors/eval
    endpoint with fast=True. The backend loads Qlib data once and computes
    IC/RankIC for all factors in a single pass.
    """
    data = request.json or {}
    expressions = data.get("expressions", [])
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")
    use_cache = data.get("use_cache", True)
    timeout = int(data.get("timeout", 600))
    forward_n = max(1, int(data.get("forward_n", 1)))

    if not expressions:
        return jsonify({"success": False, "message": "No expressions provided"})

    try:
        resp = requests.post(
            f"{api_client.base_url}/factors/eval",
            json={
                "expression": expressions,
                "market": market,
                "start": start_date,
                "end": end_date,
                "label": label,
                "fast": True,
                "use_cache": use_cache,
                "timeout": timeout,
                "forward_n": forward_n,
            },
            timeout=timeout + 30,
        )
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        return jsonify({"success": False, "message": f"Backend error: {e}"})

    # Convert list → dict keyed by expression for easy client lookup
    results = {}
    for item in items:
        expr_key = item.get("expression", "")
        results[expr_key] = item

        # Log successful evaluations to query history
        if item.get("success"):
            metrics = item.get("metrics", {})
            try:
                cache_manager.add_to_query_history(
                    expression=expr_key,
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                    ic=metrics.get("ic", 0.0),
                    rank_ic=metrics.get("rank_ic", 0.0),
                    icir=metrics.get("icir"),
                    rank_icir=metrics.get("rank_icir"),
                )
            except Exception:
                pass

    return jsonify({"success": True, "results": results, "total": len(expressions)})


@app.route("/api/get_cached_expressions", methods=["GET"])
def get_cached_expressions():
    """Get list of cached expressions from FactorStore."""
    limit = request.args.get("limit", 100, type=int)

    # Query expressions table in FactorStore
    try:
        with factor_store._connect() as conn:
            rows = conn.execute(
                "SELECT expr_hash, expression, created_at FROM expressions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        expressions = [
            {"expr_hash": r[0], "expression": r[1], "created_at": r[2]}
            for r in rows
        ]
    except Exception:
        expressions = []

    return jsonify({"success": True, "expressions": expressions})


@app.route("/api/run_portfolio_backtest", methods=["POST"])
def run_portfolio_backtest():
    """Run portfolio backtest for a single factor (fast=False triggers backtest on backend)."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")
    topk = max(1, int(data.get("topk", 50)))
    n_drop = max(0, int(data.get("n_drop", 5)))

    # Optional exchange kwargs overrides from frontend
    exchange_overrides = {}
    if "open_cost" in data:
        exchange_overrides["open_cost"] = float(data["open_cost"])
    if "close_cost" in data:
        exchange_overrides["close_cost"] = float(data["close_cost"])
    if "min_cost" in data:
        exchange_overrides["min_cost"] = float(data["min_cost"])
    if "limit_threshold" in data:
        exchange_overrides["limit_threshold"] = data["limit_threshold"]

    try:
        # Call backend with fast=False to trigger portfolio backtest
        results = api_client.evaluate_factor(
            expression=expr,
            market=market,
            start_date=start_date,
            end_date=end_date,
            label=label,
            use_cache=True,
            fast=False,
            topk=topk,
            n_drop=n_drop,
            exchange_kwargs=exchange_overrides if exchange_overrides else None,
        )
        result = results[0] if isinstance(results, list) and results else results

        if result.get("success") and result.get("portfolio_metrics"):
            resp = {
                "success": True,
                "portfolio_metrics": result["portfolio_metrics"],
            }
            if result.get("portfolio_details"):
                resp["portfolio_details"] = result["portfolio_details"]
            return jsonify(resp)
        elif result.get("success"):
            return jsonify({
                "success": False,
                "message": "Backtest completed but no portfolio metrics returned",
            })
        else:
            return jsonify({
                "success": False,
                "message": result.get("error", "Portfolio backtest failed"),
            })

    except Exception as e:
        logger.exception("run_portfolio_backtest error")
        return jsonify({
            "success": False,
            "message": f"Portfolio backtest error: {e}",
        })


@app.route("/api/get_daily_chart", methods=["POST"])
def get_daily_chart():
    """Generate daily IC/RankIC chart data."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")

    eh = expr_hash(_normalize_expr(expr))
    daily_data = factor_store.get_daily_ic(eh, market, label, start_date, end_date)

    if not daily_data:
        return jsonify({"success": False, "message": "No daily data available"})

    df = pd.DataFrame(daily_data)

    # Create plotly chart
    fig = go.Figure()

    # Add IC line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ic"],
            mode="lines+markers",
            name="IC",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        )
    )

    # Add Rank IC line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rank_ic"],
            mode="lines+markers",
            name="Rank IC",
            line=dict(color="red", width=2),
            marker=dict(size=4),
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Daily IC and Rank IC - {market.upper()}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify({"success": True, "chart": chart_json})


@app.route("/api/get_performance_stats", methods=["POST"])
def get_performance_stats():
    """Get performance statistics for a factor."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")

    eh = expr_hash(_normalize_expr(expr))
    daily_metrics = factor_store.get_daily_ic(eh, market, label, start_date, end_date)

    if not daily_metrics:
        return jsonify({"success": False, "message": "No data available"})

    metrics = FactorStore.compute_summary(daily_metrics)

    # Calculate additional statistics
    stats = {
        "mean_ic": metrics.get("ic", 0),
        "mean_rank_ic": metrics.get("rank_ic", 0),
        "icir": metrics.get("icir", 0),
        "rank_icir": metrics.get("rank_icir", 0),
        "n_dates": metrics.get("n_dates", 0),
    }

    if daily_metrics:
        df = pd.DataFrame(daily_metrics)
        ic_values = df["ic"].dropna()
        rank_ic_values = df["rank_ic"].dropna()

        if len(ic_values) > 0:
            stats["ic_std"] = float(ic_values.std())
            stats["ic_min"] = float(ic_values.min())
            stats["ic_max"] = float(ic_values.max())
            stats["ic_positive_ratio"] = float((ic_values > 0).mean())

        if len(rank_ic_values) > 0:
            stats["rank_ic_std"] = float(rank_ic_values.std())
            stats["rank_ic_min"] = float(rank_ic_values.min())
            stats["rank_ic_max"] = float(rank_ic_values.max())
            stats["rank_ic_positive_ratio"] = float((rank_ic_values > 0).mean())

    return jsonify({"success": True, "stats": stats})


@app.route("/api/cache_stats", methods=["GET"])
def get_cache_stats():
    """Get cache statistics from FactorStore."""
    stats = factor_store.get_store_stats()
    return jsonify({"success": True, "stats": stats})


@app.route("/api/clear_cache", methods=["POST"])
def clear_cache():
    """Clear FactorStore cache entries."""
    data = request.json or {}
    expr = data.get("expression")

    try:
        eh = expr_hash(_normalize_expr(expr)) if expr else None
        factor_store.clear(expr_hash=eh)
        return jsonify({"success": True, "message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/compare_factors", methods=["POST"])
def compare_factors():
    """Compare multiple factors via backend batch eval."""
    data = request.json
    expressions = data.get("expressions", [])
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")

    if len(expressions) < 2:
        return jsonify(
            {"success": False, "message": "Need at least 2 expressions to compare"}
        )

    try:
        # Call backend /factors/eval with all expressions (batch mode)
        resp = requests.post(
            f"{api_client.base_url}/factors/eval",
            json={
                "expression": expressions,
                "market": market,
                "start": start_date,
                "end": end_date,
                "label": label,
                "fast": True,
                "use_cache": True,
            },
            timeout=300,
        )
        resp.raise_for_status()
        items = resp.json()

        comparison = []
        for i, item in enumerate(items):
            metrics = item.get("metrics", {})
            comparison.append(
                {
                    "expression": item.get("expression", expressions[i] if i < len(expressions) else ""),
                    "ic": metrics.get("ic", 0),
                    "rank_ic": metrics.get("rank_ic", 0),
                    "icir": metrics.get("icir", 0),
                    "rank_icir": metrics.get("rank_icir", 0),
                    "turnover": metrics.get("turnover", 0),
                    "n_dates": metrics.get("n_dates", 0),
                }
            )
        return jsonify({"success": True, "comparison": comparison})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/get_factor_history", methods=["POST"])
def get_factor_history():
    """Get evaluation history for a factor from query history."""
    data = request.json
    expr = data.get("expression", "")
    market = data.get("market")

    if not expr:
        return jsonify({"success": False, "message": "Expression required"})

    # Use query history from cache_manager (tracks evaluations over time)
    history = cache_manager.get_factor_history(expr, market)
    return jsonify({"success": True, "history": history})


@app.route("/api/export_data", methods=["POST"])
def export_data():
    """Export factor data."""
    data = request.json
    expr = data.get("expression", "")
    market = data.get("market", "csi300")
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    export_format = data.get("format", "csv")

    try:
        # Get data via evaluate_factor (returns list)
        results_list = api_client.evaluate_factor(
            expression=expr, market=market, start_date=start_date, end_date=end_date,
            fast=True, use_cache=True,
        )
        result = results_list[0] if isinstance(results_list, list) and results_list else results_list

        if not result.get("success"):
            return jsonify(
                {"success": False, "message": result.get("error", "Export failed")}
            )

        # Format data
        daily_metrics = result.get("daily_metrics", [])
        if not daily_metrics:
            return jsonify({"success": False, "message": "No data to export"})

        df = pd.DataFrame(daily_metrics)

        if export_format == "csv":
            return jsonify(
                {
                    "success": True,
                    "data": df.to_csv(index=False),
                    "format": "csv",
                    "filename": "factor_export.csv",
                }
            )
        else:
            return jsonify(
                {
                    "success": True,
                    "data": df.to_dict(orient="records"),
                    "format": "json",
                    "filename": "factor_export.json",
                }
            )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/get_top_factors", methods=["GET"])
def get_top_factors():
    """Get top performing factors based on IC or Rank IC from query history."""
    limit = request.args.get("limit", 5, type=int)
    metric = request.args.get("metric", "rank_ic")  # rank_ic or ic

    # Get top factors from query history (same source as Recent Activity)
    result = cache_manager.get_top_factors_from_history(limit=limit, metric=metric)

    return jsonify(result)


@app.route("/api/get_recent_activity", methods=["GET"])
def get_recent_activity():
    """Get recently evaluated factors."""
    limit = request.args.get("limit", 20, type=int)
    search = request.args.get("search", None)

    # Get query history with optional search
    history = cache_manager.get_query_history(limit=limit, search_query=search)

    return jsonify({"success": True, "recent_activity": history})


@app.route("/api/get_distribution_chart", methods=["POST"])
def get_distribution_chart():
    """Generate IC distribution chart data."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")

    eh = expr_hash(_normalize_expr(expr))
    daily_data = factor_store.get_daily_ic(eh, market, label, start_date, end_date)

    if not daily_data:
        return jsonify({"success": False, "message": "No daily data available"})

    df = pd.DataFrame(daily_data)

    # Create IC distribution histogram
    fig = go.Figure()

    # IC histogram
    fig.add_trace(
        go.Histogram(
            x=df["ic"].dropna(),
            name="IC Distribution",
            nbinsx=30,
            opacity=0.7,
            marker_color="blue",
        )
    )

    fig.update_layout(
        title=f"IC Distribution - {market.upper()}",
        xaxis_title="IC Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
    )

    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify({"success": True, "chart": chart_json})


@app.route("/api/get_cumulative_chart", methods=["POST"])
def get_cumulative_chart():
    """Generate cumulative IC chart data."""
    data = request.json
    expr = data.get("expression", "")
    market = (data.get("market", "csi300") or "csi300").lower()
    start_date = data.get("start_date", "2023-01-01")
    end_date = data.get("end_date", "2024-01-01")
    label = data.get("label", "close_return")

    eh = expr_hash(_normalize_expr(expr))
    daily_data = factor_store.get_daily_ic(eh, market, label, start_date, end_date)

    if not daily_data:
        return jsonify({"success": False, "message": "No daily data available"})

    df = pd.DataFrame(daily_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Calculate cumulative IC
    df["cumulative_ic"] = df["ic"].cumsum()
    df["cumulative_rank_ic"] = df["rank_ic"].cumsum()

    # Create chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_ic"],
            mode="lines",
            name="Cumulative IC",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_rank_ic"],
            mode="lines",
            name="Cumulative Rank IC",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title=f"Cumulative IC - {market.upper()}",
        xaxis_title="Date",
        yaxis_title="Cumulative Value",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify({"success": True, "chart": chart_json})


@app.route("/api/factor_combination/dynamic", methods=["POST"])
def factor_combination_dynamic():
    """
    Handle dynamic factor combination training with multiple periods via pipeline.

    Request JSON format:
    {
        "periods": [
            {
                "period_id": 1,
                "train_start": "2020-01-01",
                "train_end": "2020-06-30",
                "test_start": "2020-07-01",
                "test_end": "2020-12-31"
            }
        ],
        "lookback_window": 60,
        "alpha": 0.01,
        "include_baseline": true
    }
    """
    from data.pipeline.optim.training_pipeline import UnifiedOptimizationPipeline
    from datetime import datetime, timedelta
    import logging

    logger = logging.getLogger(__name__)

    try:
        data = request.json

        # Validate input
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided"})

        # Extract parameters
        market = data.get("market", "csi300")
        lookback_window = data.get("lookback_window", 60)
        alpha = data.get("alpha", 0.01)
        include_baseline = data.get("include_baseline", True)
        methods = data.get("methods", ["baseline", "lasso", "ic_optimization"])
        periods_config = data.get("periods", [])

        # Validate periods
        if not periods_config or not isinstance(periods_config, list):
            # Debug info
            logger = logging.getLogger(__name__)
            logger.error(
                f"Invalid periods_config: type={type(periods_config)}, value={periods_config}"
            )
            logger.error(f"Full request data keys: {data.keys()}")
            return jsonify(
                {
                    "success": False,
                    "message": "Please provide periods as a list",
                    "debug": {
                        "received_type": str(type(periods_config)),
                        "received_value": str(periods_config)[:100],
                        "available_keys": list(data.keys()),
                    },
                }
            )

        if len(periods_config) == 0:
            return jsonify(
                {"success": False, "message": "At least one period is required"}
            )

        # Get optional training period from UI (overrides JSON)
        ui_train_start = data.get("training_start_date") or data.get("train_start")
        ui_train_end = data.get("training_end_date") or data.get("train_end")

        # Validate and transform periods
        transformed_periods = []
        global_train_start = None
        global_train_end = None

        for i, period in enumerate(periods_config):
            # Support two formats:
            # Format 1: {start, end, factors} - convert to testing period
            # Format 2: {period_id, train_start, train_end, test_start, test_end, factors}

            if "start" in period and "end" in period:
                # Format 1: Simple format with start/end as testing period
                test_start = period["start"]
                test_end = period["end"]

                transformed_period = {
                    "period_id": i,
                    "test_start": test_start,
                    "test_end": test_end,
                    "factors": period.get("factors", []),
                }

                # Use UI training dates if provided, otherwise calculate rolling window per period
                if ui_train_start and ui_train_end:
                    transformed_period["train_start"] = ui_train_start
                    transformed_period["train_end"] = ui_train_end
                else:
                    # Dynamic Rolling Window: Train on data immediately preceding the test period
                    # Default to 1 year of training data if not specified
                    # This ensures NO data leakage (training ends before test starts)
                    test_start_dt = datetime.strptime(test_start, "%Y-%m-%d")
                    train_end_dt = test_start_dt - timedelta(days=1)
                    train_start_dt = train_end_dt - timedelta(
                        days=365
                    )  # Default 1 year lookback

                    transformed_period["train_start"] = train_start_dt.strftime(
                        "%Y-%m-%d"
                    )
                    transformed_period["train_end"] = train_end_dt.strftime("%Y-%m-%d")

            elif (
                "train_start" in period
                and "train_end" in period
                and "test_start" in period
                and "test_end" in period
            ):
                # Format 2: Full format with all dates
                transformed_period = {
                    "period_id": period.get("period_id", i),
                    "train_start": ui_train_start or period["train_start"],
                    "train_end": ui_train_end or period["train_end"],
                    "test_start": period["test_start"],
                    "test_end": period["test_end"],
                    "factors": period.get("factors", []),
                }
            else:
                return jsonify(
                    {
                        "success": False,
                        "message": f"Period {i+1} has invalid format. Use either (start, end, factors) or (train_start, train_end, test_start, test_end, factors)",
                    }
                )

            # Validate factors
            if not transformed_period.get("factors"):
                return jsonify(
                    {
                        "success": False,
                        "message": f"Period {i+1} missing 'factors' field",
                    }
                )

            transformed_periods.append(transformed_period)

        # Validate period continuity - check test_start/test_end for continuity
        def validate_dynamic_continuity(periods):
            """Validate that testing periods are continuous."""
            if len(periods) < 2:
                return {"is_valid": True, "issues": [], "warnings": []}

            issues = []
            warnings = []

            # Sort by test_start
            sorted_periods = sorted(periods, key=lambda p: p.get("test_start", ""))

            for i in range(len(sorted_periods) - 1):
                try:
                    from datetime import datetime, timedelta

                    current_end = datetime.strptime(
                        sorted_periods[i]["test_end"], "%Y-%m-%d"
                    )
                    next_start = datetime.strptime(
                        sorted_periods[i + 1]["test_start"], "%Y-%m-%d"
                    )

                    gap_days = (next_start - current_end).days

                    # Ideally should be 1 day gap (next trading day) or account for weekends
                    if gap_days < 1:
                        issues.append(
                            f"Periods {i+1}-{i+2}: Overlap in testing periods"
                        )
                    elif gap_days > 3:
                        warnings.append(
                            f"Periods {i+1}-{i+2}: {gap_days} day gap in testing periods (may skip weekends/holidays)"
                        )
                except ValueError as e:
                    issues.append(f"Period {i+1}: Invalid date format")

            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
            }

        continuity_result = validate_dynamic_continuity(transformed_periods)

        if not continuity_result["is_valid"]:
            return jsonify(
                {
                    "success": False,
                    "message": "Testing periods are not continuous. Each period must start the day after (or next trading day after) the previous period ends.",
                    "validation": continuity_result,
                }
            )

        if continuity_result["warnings"]:
            for warning in continuity_result["warnings"]:
                logger.warning(f"Period validation warning: {warning}")

        # Call pipeline with transformed periods
        pipeline = UnifiedOptimizationPipeline(
            instruments=market, model_save_path="./cache_data/factor_models"
        )

        # Prepare method configurations with lookback_window
        method_configs = {}
        if "lasso" in methods:
            method_configs["lasso"] = {
                "alpha": alpha,
                "lookback_window": lookback_window,
                "rolling_window": 60,
                "max_iter": 1000,
            }
        if "ic_optimization" in methods:
            method_configs["ic_optimization"] = {"method": "equal_top", "top_k": 5}

        result = pipeline.run_dynamic_period_optimization(
            periods_config=transformed_periods,
            methods=methods,
            method_configs=method_configs,
            include_baseline=include_baseline,
            lookback_window=lookback_window,
        )

        # Convert result to JSON-serializable format
        result_serializable = convert_to_serializable(result)

        # Build standardized export format
        export_format = build_standardized_export(
            optimization_type="dynamic",
            optimization_results=result_serializable,
            configuration={
                "market": market,
                "lookback_window": lookback_window,
                "alpha": alpha,
                "methods": methods,
                "include_baseline": include_baseline,
                "rolling_window": 60,
                "max_iterations": 1000,
            },
            factors=[
                {"formula": f}
                for f in (
                    periods_config[0].get("factors", []) if periods_config else []
                )
            ],
            periods_config=transformed_periods,
            continuity_validation=continuity_result,
            experiment_name="dynamic_factor_optimization",
        )

        # Build backtest-ready format (simplified for direct backtesting)
        backtest_config = {
            "market": market,
            "optimization_method": "dynamic_periods",
            "periods": [],
        }

        for period_result in result_serializable.get("periods", []):
            # Create a period object that contains all methods
            backtest_period = {
                "test_start": period_result.get("test_start"),
                "test_end": period_result.get("test_end"),
                "period_id": period_result.get("period_id"),
            }

            has_valid_method = False
            for method in ["lasso", "ic_optimization", "baseline"]:
                if (
                    period_result.get(method)
                    and period_result[method].get("status") == "success"
                ):
                    backtest_period[method] = {
                        "factor_names": period_result[method].get("factor_names", []),
                        "weights": period_result[method].get("weights", []),
                        "combined_expression": " + ".join(
                            [
                                f"({w:.8f}) * ({f})"
                                for f, w in zip(
                                    period_result[method].get("factor_names", []),
                                    period_result[method].get("weights", []),
                                )
                                if w != 0
                            ]
                        ),
                    }
                    has_valid_method = True

            if has_valid_method:
                backtest_config["periods"].append(backtest_period)

        return jsonify(
            {
                "success": True,
                "message": "Dynamic factor combination training completed",
                "results": result_serializable,
                "export_format": export_format,
                "backtest_config": backtest_config,  # NEW: Simplified format for backtesting
                "summary": {
                    "total_periods": len(periods_config),
                    "market": market,
                    "lookback_window": lookback_window,
                    "methods": methods,
                    "baseline_enabled": include_baseline,
                    "continuity_validated": continuity_result["is_valid"],
                },
            }
        )

    except Exception as e:
        import traceback

        return jsonify(
            {
                "success": False,
                "message": f"Dynamic factor combination failed: {str(e)}",
                "error_details": traceback.format_exc(),
            }
        )


@app.route("/api/factor_combination/fixed", methods=["POST"])
def factor_combination_fixed():
    """
    Handle fixed factor combination training for a single period.

    Category A: Fixed Period Optimization with train/test separation

    Request JSON format (Option 1 - Direct):
    {
        "type": "fixed",
        "market": "csi300",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "factors": ["factor1", "factor2", ...],
        "methods": ["lasso", "ic_optimization"],
        "alpha": 0.01,
        "include_baseline": true
    }

    Request JSON format (Option 2 - Same as Category B):
    {
        "periods": [
            {
                "start": "2020-01-01",
                "end": "2020-12-31",
                "factors": ["factor1", "factor2", ...]
            }
        ],
        "market": "csi300",
        "methods": ["lasso", "ic_optimization"],
        "alpha": 0.01,
        "include_baseline": true
    }
    Note: Will use the first period from the array
    """
    try:
        data = request.json

        # Validate input
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided"})

        # Extract parameters - support both old and new format
        market = data.get("market", "csi300")

        # Check if using periods array format (like Category B)
        if (
            "periods" in data
            and isinstance(data["periods"], list)
            and len(data["periods"]) > 0
        ):
            # Use first period from array
            first_period = data["periods"][0]
            training_start = first_period.get("start")
            training_end = first_period.get("end")
            testing_start = first_period.get("start")
            testing_end = first_period.get("end")
            factors = first_period.get("factors", [])
            print(
                f"[CATEGORY A] Using periods array format - first period: {training_start} to {training_end}, {len(factors)} factors"
            )
        else:
            # Original format
            training_start = data.get("training_start_date") or data.get("start_date")
            training_end = data.get("training_end_date") or data.get("end_date")
            testing_start = data.get("testing_start_date") or data.get("start_date")
            testing_end = data.get("testing_end_date") or data.get("end_date")
            factors = data.get("factors", [])

        methods = data.get("methods", ["baseline", "lasso", "ic_optimization"])
        alpha = data.get("alpha", 0.01)
        include_baseline = data.get("include_baseline", True)

        # Validate required fields
        if (
            not training_start
            or not training_end
            or not testing_start
            or not testing_end
        ):
            return jsonify(
                {
                    "success": False,
                    "message": "start and end dates are required (either in periods array or as direct fields)",
                }
            )

        if not factors or not isinstance(factors, list):
            return jsonify(
                {"success": False, "message": "factors must be a non-empty list"}
            )

        # Initialize pipeline
        from data.pipeline.optim.training_pipeline import UnifiedOptimizationPipeline

        pipeline = UnifiedOptimizationPipeline(
            instruments=market, model_save_path="./cache_data/factor_models"
        )

        # Prepare method configurations
        method_configs = {}
        if "lasso" in methods:
            method_configs["lasso"] = {
                "alpha": alpha,
                "rolling_window": 60,
                "max_iter": 1000,
            }
        if "ic_optimization" in methods:
            method_configs["ic_optimization"] = {"method": "equal_top", "top_k": 5}

        # Run fixed period optimization with training period
        result = pipeline.run_fixed_period_optimization(
            factor_expressions=factors,
            start_date=training_start,
            end_date=training_end,
            methods=methods,
            method_configs=method_configs,
            include_baseline=include_baseline,
        )

        # Convert result to JSON-serializable format
        result_serializable = convert_to_serializable(result)

        # Debug logging
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Pipeline result: {result}")
        logger.info(f"Serializable result: {result_serializable}")

        # Extract the first period's results for easier client-side access
        period_results = (
            result_serializable.get("periods", [{}])[0]
            if result_serializable.get("periods")
            else {}
        )
        logger.info(f"Extracted period results: {period_results}")

        # Add testing period info
        period_results["testing_period"] = {"start": testing_start, "end": testing_end}
        period_results["training_period"] = {
            "start": training_start,
            "end": training_end,
        }

        # Add performance field to each method for easy access
        for method_key in ["baseline", "lasso", "ic_optimization"]:
            if method_key in period_results and period_results[method_key]:
                method_data = period_results[method_key]
                # Use combined_ic_mean from metrics, or 0 if not available
                metrics = method_data.get("metrics", {})
                method_data["performance"] = metrics.get("combined_ic_mean", 0)

        return jsonify(
            {
                "success": True,
                "message": "Fixed factor combination training completed",
                "results": period_results,  # Flattened results with baseline, lasso, ic_optimization at top level
                "summary": {
                    "training_period": {"start": training_start, "end": training_end},
                    "testing_period": {"start": testing_start, "end": testing_end},
                    "factors_count": len(factors),
                    "methods": methods,
                    "market": market,
                    "include_baseline": include_baseline,
                },
            }
        )

    except Exception as e:
        import traceback

        return jsonify(
            {
                "success": False,
                "message": f"Fixed factor combination failed: {str(e)}",
                "error_details": traceback.format_exc(),
            }
        )


@app.route("/api/factor_combination/review", methods=["POST"])
def factor_combination_review():
    """
    Handle review and re-upload of previously exported optimization results.

    Category C: Review Mode - Download JSON with results, re-upload for backtesting

    Request JSON format (for loading):
    {
        "action": "load",
        "filepath": "path/to/exported_results.json"
    }

    Request JSON format (for export):
    {
        "action": "export",
        "results": {...},  // from previous optimization
        "configuration": {...},
        "experiment_name": "my_experiment"
    }
    """
    try:
        data = request.json

        if not data:
            return jsonify({"success": False, "message": "No JSON data provided"})

        action = data.get("action", "").lower()

        # Action: Load previously exported results
        if action == "load":
            filepath = data.get("filepath")
            if not filepath:
                return jsonify(
                    {
                        "success": False,
                        "message": "filepath is required for load action",
                    }
                )

            from data.pipeline.optim.training_pipeline import (
                UnifiedOptimizationPipeline,
            )

            try:
                results, configuration = UnifiedOptimizationPipeline.import_results(
                    filepath
                )

                return jsonify(
                    {
                        "success": True,
                        "message": "Results loaded successfully",
                        "data": {
                            "results": convert_to_serializable(results),
                            "configuration": convert_to_serializable(configuration),
                        },
                    }
                )
            except FileNotFoundError:
                return jsonify(
                    {"success": False, "message": f"File not found: {filepath}"}
                )

        # Action: Export results for download/review
        elif action == "export":
            results = data.get("results")
            configuration = data.get("configuration", {})
            experiment_name = data.get("experiment_name", "optimization_result")

            if not results:
                return jsonify(
                    {
                        "success": False,
                        "message": "results are required for export action",
                    }
                )

            from data.pipeline.optim.training_pipeline import (
                UnifiedOptimizationPipeline,
            )

            pipeline = UnifiedOptimizationPipeline(
                model_save_path="./cache_data/factor_models"
            )

            filepath = pipeline.export_results(
                results=results,
                configuration=configuration,
                experiment_name=experiment_name,
            )

            return jsonify(
                {
                    "success": True,
                    "message": "Results exported successfully",
                    "data": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            )

        # Action: Validate periods continuity
        elif action == "validate_periods":
            periods_config = data.get("periods", [])

            if not periods_config:
                return jsonify(
                    {
                        "success": False,
                        "message": "periods are required for validate_periods action",
                    }
                )

            from data.pipeline.optim.training_pipeline import (
                UnifiedOptimizationPipeline,
            )

            validation = UnifiedOptimizationPipeline.validate_periods_continuity(
                periods_config
            )

            return jsonify(
                {
                    "success": validation["is_valid"],
                    "message": "Periods validation completed",
                    "data": {
                        "is_valid": validation["is_valid"],
                        "total_periods": validation["total_periods"],
                        "issues": validation["issues"],
                        "warnings": validation["warnings"],
                    },
                }
            )

        else:
            return jsonify(
                {
                    "success": False,
                    "message": f"Unknown action: {action}. Supported actions: load, export, validate_periods",
                }
            )

    except Exception as e:
        import traceback

        return jsonify(
            {
                "success": False,
                "message": f"Review operation failed: {str(e)}",
                "error_details": traceback.format_exc(),
            }
        )


@app.route("/api/factor_combination/backtest", methods=["POST"])
def run_backtest_from_weights():
    """
    Run backtest using pre-computed weights from uploaded JSON.

    This enables out-of-sample testing:
    1. Train on Period 1 → Get weights
    2. Upload weights JSON
    3. Specify NEW Period 2 → Backtest with those fixed weights

    Request JSON format:
    {
        "weights_data": {
            "periods": [{
                "weights": [0.3, 0.5, 0.2],
                "factor_names": ["factor1", "factor2", "factor3"],
                "method": "lasso"  // or "baseline", "ic_optimization"
            }]
        },
        "backtest_config": {
            "market": "csi300",
            "start_date": "2023-07-01",
            "end_date": "2023-12-31",
            "method": "lasso"  // which method's weights to use
        }
    }
    """
    try:
        data = request.json

        if not data:
            return jsonify({"success": False, "message": "No JSON data provided"})

        weights_data = data.get("weights_data")
        backtest_config = data.get("backtest_config", {})

        if not weights_data:
            return jsonify({"success": False, "message": "weights_data is required"})

        # Extract configuration
        market = (backtest_config.get("market", "csi300") or "csi300").lower()
        method = backtest_config.get("method", "lasso")

        # Resolve market config from centralized config
        mcfg = get_config().get_market_config(market)
        bt_data_path = mcfg["data_path"]
        bt_region = mcfg["region"]
        bt_benchmark = mcfg.get("benchmark", "SH000300")
        bt_instruments = mcfg.get("instruments", market)

        # Extract weights and factors from the uploaded data
        periods = weights_data.get("periods", [])
        if not periods:
            return jsonify(
                {"success": False, "message": "No periods found in weights_data"}
            )

        # Import backtest engine + worker pool
        from backtest.qlib.single_alpha_backtest import _ensure_qlib_init
        from qlib.data import D
        from routes.factors import _get_worker_pool

        # Initialize Qlib in main process for D.features() benchmark calls
        _ensure_qlib_init(bt_data_path, bt_region)

        # Get worker pool for heavy backtest calls
        bt_pool = _get_worker_pool()

        # ============ DETECT MODE: DYNAMIC vs FIXED ============
        # Check if periods have date ranges (dynamic) or not (fixed)
        has_period_dates = any(
            "period" in p
            or ("train_start" in p and "test_start" in p)
            or ("test_start" in p and "test_end" in p)
            for p in periods
        )

        backtest_mode = "dynamic" if has_period_dates else "fixed"
        print(f"[BACKTEST] Mode detected: {backtest_mode}")
        print(f"[BACKTEST] Method requested: {method}")
        print(f"[BACKTEST] Total periods: {len(periods)}")

        # Helper to calculate advanced metrics
        def calculate_advanced_metrics(daily_returns_list, benchmark_returns_list=None):
            if not daily_returns_list:
                return {}

            returns = np.array([r["return"] for r in daily_returns_list])

            # Basic metrics
            total_return = np.sum(returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Annualized (assuming 252 days)
            ann_factor = 252
            annualized_return = mean_return * ann_factor
            annualized_volatility = std_return * np.sqrt(ann_factor)

            # Sharpe
            sharpe_ratio = (
                (annualized_return / annualized_volatility)
                if annualized_volatility > 0
                else 0
            )

            # Max Drawdown
            cum_returns = np.cumsum(returns)
            max_dd = 0
            peak = -99999
            for r in cum_returns:
                if r > peak:
                    peak = r
                dd = peak - r
                if dd > max_dd:
                    max_dd = dd

            # Win Rate
            win_rate = np.sum(returns > 0) / len(returns)

            metrics = {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_dd),  # Positive value representing the drop
                "win_rate": float(win_rate),
                "trading_days": len(returns),
            }

            # Benchmark comparison if available
            if benchmark_returns_list:
                # Align dates
                bench_map = {r["date"]: r["return"] for r in benchmark_returns_list}
                excess_returns = []
                for r in daily_returns_list:
                    b_ret = bench_map.get(r["date"], 0.0)
                    excess_returns.append(r["return"] - b_ret)

                excess_returns = np.array(excess_returns)
                excess_mean = np.mean(excess_returns)
                excess_std = np.std(excess_returns)

                metrics["information_ratio"] = (
                    float((excess_mean / excess_std) * np.sqrt(ann_factor))
                    if excess_std > 0
                    else 0
                )

            return metrics

        if backtest_mode == "fixed":
            # ============ FIXED MODE: Use first period weights for user-specified date range ============
            # For fixed mode, still allow user to specify dates
            start_date = backtest_config.get("start_date")
            end_date = backtest_config.get("end_date")

            if not start_date or not end_date:
                return jsonify(
                    {
                        "success": False,
                        "message": "start_date and end_date are required for fixed mode",
                    }
                )

            selected_period = None
            for period in periods:
                if period.get(method):
                    selected_period = period[method]
                    print(
                        f"[DEBUG] Found period with method '{method}': {len(selected_period.get('weights', []))} weights"
                    )
                    break

            if not selected_period:
                return jsonify(
                    {
                        "success": False,
                        "message": f"No weights found for method '{method}' in uploaded data",
                    }
                )

            weights = selected_period.get("weights", [])
            factor_names = selected_period.get("factor_names", [])

            # Debug: show first few weights
            print(
                f"[DEBUG] Method '{method}' - First 5 weights: {weights[:5] if len(weights) >= 5 else weights}"
            )

            if not weights or not factor_names:
                return jsonify(
                    {
                        "success": False,
                        "message": "Invalid weights or factor_names in selected period",
                    }
                )

            if len(weights) != len(factor_names):
                return jsonify(
                    {
                        "success": False,
                        "message": f"Weights length ({len(weights)}) != factor_names length ({len(factor_names)})",
                    }
                )

            # Create combined factor expression using fixed weights
            weighted_terms = []
            for w, factor_expr in zip(weights, factor_names):
                if abs(w) > 1e-10:  # Skip near-zero weights
                    weighted_terms.append(f"({w})*({factor_expr})")

            # Fetch Benchmark Data
            try:
                bench_df = D.features(
                    [bt_benchmark],
                    ["$close/$close(1)-1"],
                    start_time=start_date,
                    end_time=end_date,
                )
                benchmark_returns = []
                if bench_df is not None and not bench_df.empty:
                    bench_df = bench_df.reset_index()
                    for _, row in bench_df.iterrows():
                        benchmark_returns.append(
                            {
                                "date": row["datetime"].strftime("%Y-%m-%d"),
                                "return": float(row["$close/$close(1)-1"]),
                            }
                        )
            except Exception as e:
                logger.warning("Failed to fetch benchmark data: %s", e)
                benchmark_returns = []

            if not weighted_terms:
                return jsonify(
                    {
                        "success": False,
                        "message": "All weights are near zero, cannot create combined factor",
                    }
                )

            combined_expression = " + ".join(weighted_terms)
            print(
                f"[BACKTEST] Fixed mode - Combined expression (first 200 chars): {combined_expression[:200]}..."
            )
            print(
                f"[BACKTEST] Fixed mode - Number of weighted terms: {len(weighted_terms)}"
            )
            period_info = {
                "mode": "fixed",
                "weights": weights,
                "factor_names": factor_names,
            }

        else:
            # ============ DYNAMIC MODE: Use period dates from JSON ============
            # NO user-specified dates needed - use what's in the JSON!

            # Extract all periods with their date ranges
            period_weights = []
            for idx, period in enumerate(periods):
                print(f"[DEBUG] Period {idx} keys: {list(period.keys())}")

                # Parse period dates (try multiple formats)
                period_start = None
                period_end = None

                # Format 1: "period" field with "YYYY-MM-DD to YYYY-MM-DD"
                if "period" in period and isinstance(period["period"], str):
                    parts = period["period"].split(" to ")
                    if len(parts) == 2:
                        try:
                            period_start = datetime.strptime(
                                parts[0].strip(), "%Y-%m-%d"
                            )
                            period_end = datetime.strptime(parts[1].strip(), "%Y-%m-%d")
                        except ValueError as e:
                            print(f"[WARNING] Period {idx} date parse error: {e}")

                # Format 2: "period" field as dict with start/end
                elif "period" in period and isinstance(period["period"], dict):
                    p_dict = period["period"]
                    if "start" in p_dict and "end" in p_dict:
                        try:
                            period_start = datetime.strptime(
                                p_dict["start"], "%Y-%m-%d"
                            )
                            period_end = datetime.strptime(p_dict["end"], "%Y-%m-%d")
                        except ValueError as e:
                            print(f"[WARNING] Period {idx} date parse error: {e}")

                # Format 3: train/test split - use test period
                elif "test_start" in period and "test_end" in period:
                    try:
                        period_start = datetime.strptime(
                            period["test_start"], "%Y-%m-%d"
                        )
                        period_end = datetime.strptime(period["test_end"], "%Y-%m-%d")
                    except ValueError as e:
                        print(f"[WARNING] Period {idx} date parse error: {e}")

                if not period_start or not period_end:
                    print(f"[WARNING] Period {idx} has no valid date range, skipping")
                    continue

                # Extract method weights
                if method not in period or not period[method]:
                    print(f"[WARNING] Period {idx} missing method '{method}', skipping")
                    continue

                method_data = period[method]
                weights = method_data.get("weights", [])
                factor_names = method_data.get("factor_names", [])

                if not weights or not factor_names or len(weights) != len(factor_names):
                    print(
                        f"[WARNING] Period {idx} has invalid weights/factors, skipping"
                    )
                    continue

                period_weights.append(
                    {
                        "period_index": idx,
                        "start": period_start,
                        "end": period_end,
                        "weights": weights,
                        "factor_names": factor_names,
                    }
                )
                print(
                    f"[INFO] ✓ Period {idx}: {period_start.date()} to {period_end.date()} - {len(weights)} factors"
                )

            if not period_weights:
                return jsonify(
                    {
                        "success": False,
                        "message": f"No valid periods found in JSON with method '{method}'",
                    }
                )

            # Sort periods by start date
            period_weights.sort(key=lambda x: x["start"])

            # Calculate overall date range
            start_date = period_weights[0]["start"].strftime("%Y-%m-%d")
            end_date = period_weights[-1]["end"].strftime("%Y-%m-%d")

            # Fetch Benchmark Data for Dynamic Mode
            try:
                bench_df = D.features(
                    [bt_benchmark],
                    ["$close/$close(1)-1"],
                    start_time=start_date,
                    end_time=end_date,
                )
                benchmark_returns = []
                if bench_df is not None and not bench_df.empty:
                    bench_df = bench_df.reset_index()
                    for _, row in bench_df.iterrows():
                        benchmark_returns.append(
                            {
                                "date": row["datetime"].strftime("%Y-%m-%d"),
                                "return": float(row["$close/$close(1)-1"]),
                            }
                        )
            except Exception as e:
                logger.warning("Failed to fetch benchmark data: %s", e)
                benchmark_returns = []

            print(
                f"[BACKTEST] Dynamic mode: {len(period_weights)} periods covering {start_date} to {end_date}"
            )

            combined_expression = f"DYNAMIC_WEIGHTS_{len(period_weights)}_PERIODS"
            period_info = {
                "mode": "dynamic",
                "total_periods": len(period_weights),
                "periods": [
                    {
                        "index": p["period_index"],
                        "start": p["start"].strftime("%Y-%m-%d"),
                        "end": p["end"].strftime("%Y-%m-%d"),
                        "num_factors": len(p["weights"]),
                    }
                    for p in period_weights
                ],
            }

        # Run backtest based on mode
        try:
            if backtest_mode == "fixed":
                # ============ FIXED MODE: Single backtest with fixed weights ============
                print(
                    f"[BACKTEST] Running backtest: {start_date} to {end_date}, market={market}"
                )
                print(
                    f"[BACKTEST] Expression hash: {hash(combined_expression)}"
                )  # To detect if expression changes

                analysis_df, report_normal, positions_normal = bt_pool.submit_backtest(
                    region=bt_region,
                    job_type="backtest_by_single_alpha",
                    kwargs=dict(
                        alpha_factor=combined_expression,
                        topk=50, n_drop=5,
                        start_time=start_date, end_time=end_date,
                        data_path=bt_data_path, instruments=bt_instruments,
                        region=bt_region, BENCH=bt_benchmark,
                    ),
                )

                # Extract daily returns
                daily_returns = []
                cumulative_returns = []
                if report_normal is not None and len(report_normal) > 0:
                    cumulative = 0.0
                    for idx, row in report_normal.iterrows():
                        date_str = (
                            idx.strftime("%Y-%m-%d")
                            if hasattr(idx, "strftime")
                            else str(idx)
                        )
                        daily_ret = float(row.get("return", 0))
                        cumulative += daily_ret

                        daily_returns.append({"date": date_str, "return": daily_ret})
                        cumulative_returns.append(
                            {"date": date_str, "cumulative_return": cumulative}
                        )

                # Calculate advanced metrics
                metrics = calculate_advanced_metrics(daily_returns, benchmark_returns)

                result_data = {
                    "combined_expression": combined_expression,
                    "weights": period_info["weights"],
                    "factor_names": period_info["factor_names"],
                    "method": method,
                    "mode": "fixed",
                    "period": {"start": start_date, "end": end_date},
                    "market": market,
                    "metrics": metrics,
                    "daily_returns": daily_returns,
                    "cumulative_returns": cumulative_returns,
                    "benchmark_returns": benchmark_returns,
                }

            else:
                # ============ DYNAMIC MODE: Multiple backtests per period + aggregate ============
                print(
                    f"[BACKTEST] Running {len(period_weights)} period-specific backtests..."
                )

                all_period_results = []
                all_daily_returns = []

                for p_idx, period in enumerate(period_weights):
                    period_start_str = period["start"].strftime("%Y-%m-%d")
                    period_end_str = period["end"].strftime("%Y-%m-%d")

                    # Create combined expression for this period
                    weighted_terms = []
                    for w, factor_expr in zip(
                        period["weights"], period["factor_names"]
                    ):
                        if abs(w) > 1e-10:
                            weighted_terms.append(f"({w})*({factor_expr})")

                    if not weighted_terms:
                        print(
                            f"[WARNING] Period {p_idx} has all zero weights, skipping"
                        )
                        continue

                    period_expression = " + ".join(weighted_terms)

                    print(
                        f"[BACKTEST] Period {p_idx+1}/{len(period_weights)}: {period_start_str} to {period_end_str}"
                    )

                    # Run backtest for this period
                    try:
                        analysis_df, report_normal, positions_normal = bt_pool.submit_backtest(
                            region=bt_region,
                            job_type="backtest_by_single_alpha",
                            kwargs=dict(
                                alpha_factor=period_expression,
                                topk=50, n_drop=5,
                                start_time=period_start_str, end_time=period_end_str,
                                data_path=bt_data_path, instruments=bt_instruments,
                                region=bt_region, BENCH=bt_benchmark,
                            ),
                        )

                        # Extract period metrics
                        period_metrics = {}
                        if analysis_df is not None and len(analysis_df) > 0:
                            try:
                                level0_values = analysis_df.index.get_level_values(
                                    0
                                ).unique()
                                pure_metrics = analysis_df.loc[level0_values[0]]

                                period_metrics["annualized_return"] = (
                                    float(pure_metrics.loc["annualized_return"])
                                    if "annualized_return" in pure_metrics.index
                                    else 0
                                )
                                period_metrics["information_ratio"] = (
                                    float(pure_metrics.loc["information_ratio"])
                                    if "information_ratio" in pure_metrics.index
                                    else 0
                                )
                                period_metrics["max_drawdown"] = (
                                    float(pure_metrics.loc["max_drawdown"])
                                    if "max_drawdown" in pure_metrics.index
                                    else 0
                                )
                            except:
                                pass

                        # Collect daily returns
                        if report_normal is not None and len(report_normal) > 0:
                            for idx, row in report_normal.iterrows():
                                all_daily_returns.append(
                                    {
                                        "date": (
                                            idx.strftime("%Y-%m-%d")
                                            if hasattr(idx, "strftime")
                                            else str(idx)
                                        ),
                                        "return": float(row.get("return", 0)),
                                        "period_index": p_idx,
                                    }
                                )

                        all_period_results.append(
                            {
                                "period_index": p_idx,
                                "start": period_start_str,
                                "end": period_end_str,
                                "metrics": period_metrics,
                                "num_factors": len(period["weights"]),
                                "expression": (
                                    period_expression[:100] + "..."
                                    if len(period_expression) > 100
                                    else period_expression
                                ),
                            }
                        )

                    except Exception as period_error:
                        print(f"[ERROR] Period {p_idx} backtest failed: {period_error}")
                        continue

                # Aggregate results across all periods
                if not all_daily_returns:
                    return jsonify(
                        {"success": False, "message": "All period backtests failed"}
                    )

                # Sort by date and calculate cumulative returns
                all_daily_returns.sort(key=lambda x: x["date"])
                cumulative = 0.0
                cumulative_returns = []

                for ret in all_daily_returns:
                    cumulative += ret["return"]
                    cumulative_returns.append(
                        {"date": ret["date"], "cumulative_return": cumulative}
                    )

                # Calculate aggregate metrics using advanced helper
                aggregate_metrics = calculate_advanced_metrics(
                    all_daily_returns, benchmark_returns
                )

                print(
                    f"[BACKTEST] Dynamic mode complete: {len(all_period_results)} periods, {len(all_daily_returns)} trading days"
                )

                result_data = {
                    "combined_expression": f"DYNAMIC ({len(all_period_results)} periods)",
                    "method": method,
                    "mode": "dynamic",
                    "period": {"start": start_date, "end": end_date},
                    "market": market,
                    "metrics": aggregate_metrics,
                    "daily_returns": all_daily_returns,
                    "cumulative_returns": cumulative_returns,
                    "benchmark_returns": benchmark_returns,
                    "period_results": all_period_results,
                    "total_periods": len(all_period_results),
                }

        except Exception as backtest_error:
            import traceback

            traceback.print_exc()
            return jsonify(
                {
                    "success": False,
                    "message": f"Backtest execution failed: {str(backtest_error)}",
                }
            )

        # Format response
        result = {
            "success": True,
            "message": f"Backtest completed successfully ({backtest_mode} mode)",
            "backtest_results": result_data,
        }

        return jsonify(result)

    except Exception as e:
        import traceback

        return jsonify(
            {
                "success": False,
                "message": f"Backtest failed: {str(e)}",
                "error_details": traceback.format_exc(),
            }
        )


def _get_backtest_pool():
    """Lazily import and return the qlib worker pool (only when backtest is needed)."""
    from routes.factors import _get_worker_pool
    return _get_worker_pool()


if __name__ == "__main__":
    web_cfg = get_config()
    port = web_cfg.get("server.web.port", 19787)
    host = web_cfg.get("server.web.host", "0.0.0.0")

    # Check if something is already listening on localhost:port
    # (catches VSCode port forwarding, stale processes, etc.)
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(1)
        probe.connect(("127.0.0.1", port))
        probe.close()
        # Connection succeeded → something is already listening
        print(f"\nERROR: Port {port} is already in use on localhost.")
        print(f"  To find the process:  lsof -i :{port}")
        print(f"  To use another port:  FFO_WEB_PORT=<port> python web_app.py")
        print(f"  If VSCode is forwarding this port, stop it in the Ports panel.\n")
        sys.exit(1)
    except (ConnectionRefusedError, OSError):
        pass  # Port is free

    app.run(
        debug=web_cfg.get("server.web.debug", False),
        host=host,
        port=port,
    )
