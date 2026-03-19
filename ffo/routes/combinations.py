#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factor Evaluation API (clean rewrite)

Improvements:
- Hard timeouts with kill-on-timeout using subprocesses
- Persistent SQLite cache keyed by expr hash + params
- Vectorized IC/RankIC, faster batch eval
- Smaller, clearer endpoints with robust error handling
"""

import os
import logging
from datetime import datetime, timezone
from urllib.parse import unquote

from flask_cors import CORS
from utils.utils import PersistentCache
from flask import Blueprint, request, jsonify

bp = Blueprint("combinations", __name__, url_prefix="/combination")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("FactorAPI")

# -----------------------------
# Defaults
# -----------------------------
DEFAULTS = {
    "market": os.environ.get("DEFAULT_MARKET", "csi300"),
    "start": os.environ.get("DEFAULT_START", "2023-01-01"),
    "end": os.environ.get("DEFAULT_END", "2024-01-01"),
    "label": os.environ.get("DEFAULT_LABEL", "close_return"),
    "check_start": os.environ.get("CHECK_START", "2020-01-01"),
    "check_end": os.environ.get("CHECK_END", "2020-01-15"),
    "use_cache": True,
    "timeout_eval": int(os.environ.get("TIMEOUT_EVAL_SEC", "180")),
    "timeout_check": int(os.environ.get("TIMEOUT_CHECK_SEC", "120")),
    "timeout_batch": int(os.environ.get("TIMEOUT_BATCH_SEC", "600")),
}

# Persistent cache
CACHE = PersistentCache()


@bp.route("/train", methods=["POST"])
def train_factor_combination():
    """
    Train factor combination models using LASSO or IC optimization.

    Body JSON:
    {
      "factor_expressions": ["expr1", "expr2", "expr3"],
      "method": "lasso" or "ic_optimization",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "market": "csi300",
      "method_config": {
        "alpha": 0.01,           // for LASSO
        "lambda_risk": 1.0,      // for IC optimization
        "alpha_l1": 0.0,         // for IC optimization
        "non_negative": true     // for IC optimization
      }
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        factor_expressions = data.get("factor_expressions", [])
        method = data.get("method", "lasso")
        start_date = data.get("start_date", "2018-01-01")
        end_date = data.get("end_date", "2020-12-31")
        market = data.get("market", "csi300")
        method_config = data.get("method_config", {})
        n_jobs = int(data.get("n_jobs", 1))

        if not factor_expressions:
            return (
                jsonify({"success": False, "error": "Missing factor_expressions"}),
                400,
            )

        if method not in ["lasso", "ic_optimization"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Method must be 'lasso' or 'ic_optimization'",
                    }
                ),
                400,
            )

        # Import training functions
        if method == "lasso":
            from data.pipeline.optim.ml_training import train_lasso_factor_combination

            result = train_lasso_factor_combination(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
                instruments=market,
                n_jobs=n_jobs,
                **method_config,
            )
        else:  # ic_optimization
            from data.pipeline.optim.ic_optimization import optimize_factor_weights_ic

            result = optimize_factor_weights_ic(
                factor_expressions=factor_expressions,
                start_date=start_date,
                end_date=end_date,
                instruments=market,
                n_jobs=n_jobs,
                **method_config,
            )

        return jsonify(
            {
                "success": True,
                "method": method,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Factor combination training error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            500,
        )


@bp.route("/backtest", methods=["POST"])
def backtest_factor_combination():
    """
    Run backtest using trained factor combination weights.

    Body JSON:
    {
      "trained_model": {...},  // Result from /factor_combination/train
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "market": "csi300",
      "topk": 50,
      "n_drop": 5
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        trained_model = data.get("trained_model")
        start_date = data.get("start_date", "2021-01-01")
        end_date = data.get("end_date", "2021-12-31")
        market = data.get("market", "csi300")
        topk = int(data.get("topk", 50))
        n_drop = int(data.get("n_drop", 5))

        if not trained_model:
            return jsonify({"success": False, "error": "Missing trained_model"}), 400

        method = (
            trained_model.get("factor_expressions", []) and "lasso" or "ic_optimization"
        )

        # Run backtest
        if method == "lasso":
            from data.pipeline.optim.ml_training import (
                evaluate_lasso_combination_backtest,
            )

            analysis_df, report_normal, positions_normal = (
                evaluate_lasso_combination_backtest(
                    trained_model=trained_model,
                    start_date=start_date,
                    end_date=end_date,
                    topk=topk,
                    n_drop=n_drop,
                    instruments=market,
                )
            )
        else:
            from data.pipeline.optim.ic_optimization import (
                evaluate_ic_optimized_combination_backtest,
            )

            analysis_df, report_normal, positions_normal = (
                evaluate_ic_optimized_combination_backtest(
                    optimized_model=trained_model,
                    start_date=start_date,
                    end_date=end_date,
                    topk=topk,
                    n_drop=n_drop,
                    instruments=market,
                )
            )

        return jsonify(
            {
                "success": True,
                "backtest_result": {
                    "analysis": (
                        analysis_df.to_dict() if analysis_df is not None else None
                    ),
                    "report_normal": report_normal,
                    "positions_normal": positions_normal,
                },
                "method": method,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Factor combination backtest error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            500,
        )


@bp.route("/automated_training", methods=["POST"])
def automated_factor_training():
    """
    Run automated rolling window training pipeline.

    Body JSON:
    {
      "factor_expressions": ["expr1", "expr2", "expr3"],
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "training_window_days": 252,
      "update_frequency_days": 21,
      "methods": ["lasso", "ic_optimization"],
      "method_configs": {
        "lasso": {"alpha": 0.01},
        "ic_optimization": {"lambda_risk": 1.0}
      },
      "market": "csi300"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        factor_expressions = data.get("factor_expressions", [])
        start_date = data.get("start_date", "2018-01-01")
        end_date = data.get("end_date", "2022-12-31")
        training_window_days = int(data.get("training_window_days", 252))
        update_frequency_days = int(data.get("update_frequency_days", 21))
        methods = data.get("methods", ["lasso"])
        method_configs = data.get("method_configs", {})
        market = data.get("market", "csi300")

        if not factor_expressions:
            return (
                jsonify({"success": False, "error": "Missing factor_expressions"}),
                400,
            )

        # Run automated training
        from data.pipeline.optim.training_pipeline import (
            AutomatedFactorTrainingPipeline,
        )

        pipeline = AutomatedFactorTrainingPipeline(
            factor_expressions=factor_expressions, instruments=market
        )

        results = pipeline.run_rolling_training(
            start_date=start_date,
            end_date=end_date,
            training_window_days=training_window_days,
            update_frequency_days=update_frequency_days,
            methods=methods,
            method_configs=method_configs,
        )

        return jsonify(
            {
                "success": True,
                "automated_training_result": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Automated training error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            500,
        )


@bp.route("/compare_methods", methods=["POST"])
def compare_factor_methods():
    """
    Compare LASSO vs IC optimization approaches.

    Body JSON:
    {
      "factor_expressions": ["expr1", "expr2", "expr3"],
      "train_start": "YYYY-MM-DD",
      "train_end": "YYYY-MM-DD",
      "test_start": "YYYY-MM-DD",
      "test_end": "YYYY-MM-DD",
      "market": "csi300",
      "lasso_config": {"alpha": 0.01},
      "ic_config": {"lambda_risk": 1.0, "alpha_l1": 0.0}
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        factor_expressions = data.get("factor_expressions", [])
        train_start = data.get("train_start", "2018-01-01")
        train_end = data.get("train_end", "2020-12-31")
        test_start = data.get("test_start", "2021-01-01")
        test_end = data.get("test_end", "2021-12-31")
        market = data.get("market", "csi300")
        lasso_config = data.get("lasso_config", {"alpha": 0.01})
        ic_config = data.get("ic_config", {"lambda_risk": 1.0, "alpha_l1": 0.0})

        if not factor_expressions:
            return (
                jsonify({"success": False, "error": "Missing factor_expressions"}),
                400,
            )

        # Run comparison
        from data.pipeline.optim.ic_optimization import compare_ml_vs_ic_optimization

        comparison_result = compare_ml_vs_ic_optimization(
            factor_expressions=factor_expressions,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            instruments=market,
            lasso_alpha=lasso_config.get("alpha", 0.01),
            ic_lambda_risk=ic_config.get("lambda_risk", 1.0),
            ic_alpha_l1=ic_config.get("alpha_l1", 0.0),
        )

        return jsonify(
            {
                "success": True,
                "comparison": comparison_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Method comparison error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            500,
        )


@bp.route("/generate_report", methods=["POST"])
def generate_factor_report():
    """
    Generate comprehensive report for factor combination results.

    Body JSON:
    {
      "training_results": {...},  // From automated training
      "backtest_results": {...},  // Optional backtest results
      "report_title": "My Factor Report"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        training_results = data.get("training_results")
        backtest_results = data.get("backtest_results")
        report_title = data.get("report_title", "Factor Combination Report")

        if not training_results:
            return jsonify({"success": False, "error": "Missing training_results"}), 400

        # Generate report
        from data.pipeline.optim.reporting import FactorCombinationReporter

        reporter = FactorCombinationReporter()
        report_result = reporter.generate_comprehensive_report(
            training_results=training_results,
            backtest_results=backtest_results,
            report_title=report_title,
        )

        return jsonify(
            {
                "success": True,
                "report_result": report_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Report generation error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            500,
        )


@bp.route("/list_models", methods=["GET"])
def list_saved_models():
    """
    List saved trained models.

    Query params:
      method: "lasso" or "ic_optimization" (optional)
    """
    try:
        method_filter = request.args.get("method")

        from data.pipeline.optim.training_pipeline import (
            AutomatedFactorTrainingPipeline,
        )

        # Create pipeline instance to access model directory
        pipeline = AutomatedFactorTrainingPipeline(factor_expressions=[])
        model_dir = pipeline.model_save_path

        if not os.path.exists(model_dir):
            return jsonify(
                {"success": True, "models": [], "message": "No models directory found"}
            )

        models = []
        for filename in os.listdir(model_dir):
            if filename.endswith(".json"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    method = parts[0]
                    date = parts[1].split(".")[0]

                    if method_filter and method != method_filter:
                        continue

                    models.append(
                        {
                            "filename": filename,
                            "method": method,
                            "training_date": date,
                            "path": os.path.join(model_dir, filename),
                        }
                    )

        # Sort by date descending
        models.sort(key=lambda x: x["training_date"], reverse=True)

        return jsonify({"success": True, "models": models, "total_count": len(models)})

    except Exception as e:
        logger.exception("List models error")
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/load_model", methods=["GET"])
def load_saved_model():
    """
    Load a saved trained model.

    Query params:
      method: "lasso" or "ic_optimization"
      training_date: "YYYY-MM-DD"
    """
    try:
        method = request.args.get("method")
        training_date = request.args.get("training_date")

        if not method or not training_date:
            return (
                jsonify({"success": False, "error": "Missing method or training_date"}),
                400,
            )

        from data.pipeline.optim.training_pipeline import (
            AutomatedFactorTrainingPipeline,
        )

        pipeline = AutomatedFactorTrainingPipeline(factor_expressions=[])
        model = pipeline.load_model(method, training_date)

        if model is None:
            return jsonify({"success": False, "error": "Model not found"}), 404

        return jsonify({"success": True, "model": model})

    except Exception as e:
        logger.exception("Load model error")
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500
