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

# import agent.qlib_contrib.qlib_extend_ops
from flask import Blueprint, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
from backtest.qlib.single_alpha_backtest import backtest_by_single_alpha

from flask_cors import CORS
import utils.qlib_extend_ops as qlib_extend_ops

from utils.utils import (
    PersistentCache,
    cache_key,
    normalize_factors_from_expression_field,
    run_eval_with_timeout,
    run_check_with_timeout,
    run_batch_with_timeout,
    run_batch_beta_with_timeout,
    DEFAULT_INSTRUMENTS,
)

bp = Blueprint("factors", __name__, url_prefix="/factors")

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


# -----------------------------
# Helpers
# -----------------------------
def _fail_result(expr: str, market: str, start: str, end: str, msg: str):
    return {
        "success": False,
        "error": msg,
        "expression": expr,
        "market": market,
        "start_date": start,
        "end_date": end,
        "metrics": {
            "ic": 0.0,
            "rank_ic": 0.0,
            "ir": 0.0,
            "icir": 0.0,
            "rank_icir": 0.0,
            "turnover": 1.0,
            "n_dates": 0,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# -----------------------------
# Routes
# -----------------------------
@bp.route("/check", methods=["POST"])
def check():
    data = request.get_json(force=True, silent=True) or {}

    factors, err = normalize_factors_from_expression_field(data)
    if err:
        msg, etype = err
        return (
            jsonify(
                [
                    {
                        "success": False,
                        "name": "",
                        "expression": (
                            data.get("expression")
                            if isinstance(data.get("expression"), str)
                            else ""
                        ),
                        "error_message": msg,
                        "error_type": etype,
                    }
                ]
            ),
            400,
        )

    instruments = data.get("instruments", DEFAULT_INSTRUMENTS)
    start = data.get("start", DEFAULTS["check_start"])
    end = data.get("end", DEFAULTS["check_end"])
    timeout = int(data.get("timeout", DEFAULTS["timeout_check"]))

    results = []
    any_fail = False

    for f in factors:
        expr = f["expression"]
        name = f.get("name", "") or ""
        res = run_check_with_timeout(expr, instruments, start, end, timeout)

        if res.ok:
            payload = (
                res.payload
                if isinstance(res.payload, dict)
                else {"result": res.payload}
            )
            item = {"success": True, "name": name, "expression": expr, **payload}
        else:
            any_fail = True
            item = {
                "success": False,
                "name": name,
                "expression": expr,
                "error_message": res.payload,
                "error_type": res.error_type,
            }

        results.append(item)

    return jsonify(results), 200


@bp.route("/eval", methods=["POST"])
def eval_once():
    """
    Evaluate one or many expressions over a time range.

    POST JSON (supported):
    1) single no-name:
      {"expression": "Mean($close, 20)", ...}

    2) dict name->expr (one or many):
      {"expression": {"A":"Mean($close,20)", "B":"Std($close,20)"}, ...}

    3) list of expr (no names) OR mixed list:
      {"expression": ["Mean($close,20)", "Std($close,20)"], ...}
      {"expression": ["Mean($close,20)", {"B":"Std($close,20)"}], ...}

    Common fields:
    {
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "market": "csi300",
      "label": "close_return",
      "use_cache": true,
      "topk": 50,
      "n_drop": 5,
      "timeout": 120,

      "fast": true,                 # ✅ new: only eval IC, skip portfolio backtest
      "n_jobs_backtest": 4          # ✅ new: threads for backtest_by_single_alpha
    }

    Return: always a JSON list, even for single expression.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        # ---- parse factors (single / dict / list) ----
        factors, err = normalize_factors_from_expression_field(data)
        if err:
            msg, etype = err
            return (
                jsonify(
                    [
                        {
                            "success": False,
                            "name": "",
                            "expression": "",
                            "error": msg,
                            "error_type": etype,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                ),
                400,
            )

        start = data.get("start", DEFAULTS["start"])
        end = data.get("end", DEFAULTS["end"])
        market = (data.get("market", DEFAULTS["market"]) or DEFAULTS["market"]).lower()
        label = data.get("label", DEFAULTS["label"])
        use_cache = bool(data.get("use_cache", DEFAULTS.get("use_cache", True)))
        timeout = int(data.get("timeout", DEFAULTS["timeout_eval"]))
        topk = int(data.get("topk", 50))
        n_drop = int(data.get("n_drop", 5))

        # ✅ new params in POST JSON
        fast = bool(data.get("fast", False))
        n_jobs_backtest = int(data.get("n_jobs_backtest", 4))

        logger.info(
            "Evaluating %d expr(s) (market=%s, %s→%s, label=%s, fast=%s, n_jobs_backtest=%d)",
            len(factors),
            market,
            start,
            end,
            label,
            fast,
            n_jobs_backtest,
        )

        def _run_portfolio_backtest(expr_):
            # Import inside to avoid slow import per request startup issues

            market_map = {
                "csi300": ("csi300", "~/.qlib/qlib_data/cn_data", "cn", "SH000300"),
                "csi500": ("csi500", "~/.qlib/qlib_data/cn_data", "cn", "SH000905"),
                "csi1000": ("csi1000", "~/.qlib/qlib_data/cn_data", "cn", "SH000852"),
            }
            instruments, data_path, region, benchmark = market_map.get(
                market.lower(), market_map["csi300"]
            )

            analysis_df, report_normal, positions_normal = backtest_by_single_alpha(
                alpha_factor=expr_,
                topk=topk,
                n_drop=n_drop,
                start_time=start,
                end_time=end,
                data_path=data_path,
                instruments=instruments,
                region=region,
                BENCH=benchmark,
            )

            portfolio_metrics = None
            if analysis_df is not None and not analysis_df.empty:
                portfolio_metrics = {}
                for return_type in [
                    "benchmark",
                    "pure_return_without_cost",
                    "pure_return_with_cost",
                    "excess_return_without_cost",
                    "excess_return_with_cost",
                ]:
                    if return_type in analysis_df.index.get_level_values(0):
                        metrics_df = analysis_df.loc[return_type]
                        portfolio_metrics[return_type] = {
                            "mean_return": float(metrics_df.loc["mean", "risk"]) * 100,
                            "std": float(metrics_df.loc["std", "risk"]) * 100,
                            "annualized_return": float(
                                metrics_df.loc["annualized_return", "risk"]
                            )
                            * 100,
                            "information_ratio": float(
                                metrics_df.loc["information_ratio", "risk"]
                            ),
                            "max_drawdown": float(
                                metrics_df.loc["max_drawdown", "risk"]
                            )
                            * 100,
                        }
            return portfolio_metrics

        results = []

        # 先串行 eval（更稳），backtest 再并行（更快）
        indices_need_backtest = []

        for f in factors:
            expr = (f.get("expression") or "").strip()
            name = f.get("name", "") or ""

            if not expr:
                results.append(
                    {
                        "success": False,
                        "name": name,
                        "expression": expr,
                        "error": "Missing 'expression'",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            key = cache_key(expr, market, start, end, label, topk, n_drop)

            # cache hit: return directly
            if use_cache:
                cached = CACHE.get(key)
                if cached:
                    # cached is payload dict, unify output
                    item = {"name": name, "expression": expr, **cached}
                    results.append(item)
                    continue

            # main eval (IC etc.)
            res = run_eval_with_timeout(expr, market, start, end, label, timeout)

            if res.ok:
                payload = (
                    res.payload.copy()
                    if isinstance(res.payload, dict)
                    else {"result": res.payload}
                )
                item = {"name": name, "expression": expr, **payload}
                results.append(item)

                # fast=True => skip backtest
                if fast:
                    if use_cache:
                        # cache only eval payload
                        CACHE.set(key, payload)
                else:
                    # mark for backtest, cache later after portfolio_metrics filled
                    item["_cache_key"] = key
                    indices_need_backtest.append(len(results) - 1)

            else:
                fail_payload = _fail_result(
                    expr, market, start, end, f"{res.error_type}: {res.payload}"
                )
                item = {"name": name, "expression": expr, **fail_payload}
                results.append(item)

        # fast=True => done
        if fast or not indices_need_backtest:
            return jsonify(results), 200

        # parallel backtest
        max_workers = max(1, n_jobs_backtest)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            for idx in indices_need_backtest:
                expr_ = results[idx]["expression"]
                future = ex.submit(_run_portfolio_backtest, expr_)
                future_map[future] = idx

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    pm = future.result()
                    if pm:
                        results[idx]["portfolio_metrics"] = pm
                except Exception as e:
                    logger.warning(f"Portfolio backtest failed (idx={idx}): {e}")

        # write cache after portfolio_metrics ready
        if use_cache:
            for item in results:
                key = item.pop("_cache_key", None)
                if key and item.get("success") is True:
                    payload_to_cache = {
                        k: v for k, v in item.items() if k not in ("name", "expression")
                    }
                    CACHE.set(key, payload_to_cache)

        return jsonify(results), 200

    except Exception as e:
        logger.exception("eval error")
        return (
            jsonify(
                [
                    {
                        "success": False,
                        "error": f"{type(e).__name__}: {e}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            ),
            500,
        )


@bp.route("/eval_beta", methods=["POST"])
def eval_beta():
    """
    Beta: evaluate a batch of factors using a single compute_factor_data() call
    and GPU-accelerated (PyTorch) IC / RankIC computation.

    Unlike /eval (which evaluates factors one-by-one), this endpoint:
    - Packages ALL expressions into one Qlib data-loader call.
    - Computes IC and RankIC for all factors simultaneously on GPU.

    Recommended for large-scale evaluation (50+ factors).

    POST JSON:
    {
        "expression": str | dict | list,   // same formats as /eval
        "market":     "csi300",            // default
        "start":      "2023-01-01",        // default
        "end":        "2024-01-01",        // default
        "label":      "close_return",      // default
        "device":     "auto",              // "auto" | "cuda" | "cpu"
        "timeout":    600                  // seconds for the whole batch
    }

    Returns: JSON list, one item per factor (same schema as /eval).
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        factors, err = normalize_factors_from_expression_field(data)
        if err:
            msg, etype = err
            return (
                jsonify(
                    [
                        {
                            "success": False,
                            "name": "",
                            "expression": "",
                            "error": msg,
                            "error_type": etype,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                ),
                400,
            )

        start = data.get("start", DEFAULTS["start"])
        end = data.get("end", DEFAULTS["end"])
        market = (data.get("market", DEFAULTS["market"]) or DEFAULTS["market"]).lower()
        label = data.get("label", DEFAULTS["label"])
        device = data.get("device", "auto")
        timeout = int(data.get("timeout", DEFAULTS["timeout_batch"]))

        logger.info(
            "Beta batch eval: %d factor(s) (market=%s, %s→%s, label=%s, device=%s)",
            len(factors),
            market,
            start,
            end,
            label,
            device,
        )

        # Ensure unique names so columns don't collide in the feature DataFrame
        named_factors = []
        for i, f in enumerate(factors):
            name = (f.get("name") or "").strip() or f"factor_{i}"
            named_factors.append({"name": name, "expression": f["expression"]})

        res = run_batch_beta_with_timeout(
            named_factors, market, start, end, label, timeout, device
        )

        if not res.ok:
            error_msg = f"{res.error_type}: {res.payload}"
            logger.error("eval_beta subprocess failed: %s", error_msg)
            return jsonify([
                {
                    "success": False,
                    "name": f.get("name", ""),
                    "expression": f.get("expression", ""),
                    "error": error_msg,
                    "market": market,
                    "start_date": start,
                    "end_date": end,
                    "metrics": {"ic": 0.0, "icir": 0.0, "rank_ic": 0.0, "rank_icir": 0.0, "n_dates": 0},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for f in named_factors
            ]), 200

        payload = res.payload
        if not isinstance(payload, dict) or "results" not in payload:
            error_msg = "Unexpected response format from beta worker"
            logger.error("eval_beta: %s", error_msg)
            return jsonify([
                {
                    "success": False,
                    "name": f.get("name", ""),
                    "expression": f.get("expression", ""),
                    "error": error_msg,
                    "market": market,
                    "start_date": start,
                    "end_date": end,
                    "metrics": {"ic": 0.0, "icir": 0.0, "rank_ic": 0.0, "rank_icir": 0.0, "n_dates": 0},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for f in named_factors
            ]), 200

        # Normalise to /eval-compatible output schema
        results = []
        for r in payload["results"]:
            item = {
                "success": r.get("success", False),
                "name": r.get("name", ""),
                "expression": r.get("expression", ""),
                "market": market,
                "start_date": start,
                "end_date": end,
                "device": r.get("device", device),
                "metrics": r.get("metrics", {}),
                "timestamp": r.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
            }
            if not r.get("success", False):
                item["error"] = r.get("error", "Unknown error")
            results.append(item)

        return jsonify(results), 200

    except Exception as e:
        logger.exception("eval_beta error")
        error_msg = f"{type(e).__name__}: {e}"
        _named = locals().get("named_factors") or locals().get("factors") or []
        return jsonify([
            {
                "success": False,
                "name": f.get("name", "") if isinstance(f, dict) else "",
                "expression": f.get("expression", "") if isinstance(f, dict) else "",
                "error": error_msg,
                "metrics": {"ic": 0.0, "icir": 0.0, "rank_ic": 0.0, "rank_icir": 0.0, "n_dates": 0},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for f in _named
        ] or [{"success": False, "error": error_msg,
               "timestamp": datetime.now(timezone.utc).isoformat()}]), 200
