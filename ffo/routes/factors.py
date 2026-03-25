#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factor Evaluation API

Features:
- Incremental daily IC/RankIC cache (only compute missing dates)
- Factor scores saved to disk for portfolio backtest reuse
- Hard timeouts with kill-on-timeout using subprocesses
- Vectorized IC/RankIC, efficient batch eval
"""

import os
import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask_cors import CORS


from utils.utils import (
    PersistentCache,
    expr_hash,
    _normalize_expr,
    normalize_factors_from_expression_field,
    run_eval_with_timeout,
    run_check_with_timeout,
    run_batch_with_timeout,
    run_batch_check_with_timeout,
    run_portfolio_combine_with_timeout,
    DEFAULT_INSTRUMENTS,
)
from utils.factor_store import FactorStore
from utils.qlib_worker_pool import QlibWorkerPool
from config.manager import get_config

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

# Incremental daily IC cache + factor score storage
FACTOR_STORE = FactorStore()

# Legacy KV cache (kept for backward compatibility during migration)
CACHE = PersistentCache()

# Persistent qlib worker pool (one process per region, eager init at import)
def _build_worker_pool() -> QlibWorkerPool:
    """Build the worker pool from market configs — called once at import time."""
    cfg = get_config()
    region_configs: dict = {}
    for market_name in cfg.get("markets", {}):
        mcfg = cfg.get_market_config(market_name)
        r = mcfg["region"]
        if r not in region_configs:
            region_configs[r] = {"data_path": mcfg["data_path"], "region": r}
    workers_per_region = int(cfg.get("evaluation.workers_per_region", 4))
    return QlibWorkerPool(region_configs, workers_per_region=workers_per_region)


WORKER_POOL = _build_worker_pool()


def _get_worker_pool() -> QlibWorkerPool:
    """Return the shared worker pool (for use by web_app.py etc.)."""
    return WORKER_POOL


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
            "turnover": 0.0,
            "n_dates": 0,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _extract_portfolio_metrics(analysis_df):
    """Extract portfolio metrics dict from qlib analysis DataFrame."""
    if analysis_df is None or (hasattr(analysis_df, 'empty') and analysis_df.empty):
        return None
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
                ) * 100,
                "information_ratio": float(
                    metrics_df.loc["information_ratio", "risk"]
                ),
                "max_drawdown": float(
                    metrics_df.loc["max_drawdown", "risk"]
                ) * 100,
            }
    return portfolio_metrics or None


def _extract_portfolio_details(report_normal, positions_normal):
    """
    Extract detailed portfolio data from qlib backtest results.

    Returns JSON-serializable dict with:
    - daily_summary: daily account value, returns, turnover, costs, cash, benchmark
    - holdings: per-date stock holdings {date_str: [{stock, amount, weight}, ...]}
    - actions: buy/sell events detected by diffing consecutive holdings
    """
    details = {"daily_summary": [], "holdings": {}, "actions": []}

    # --- daily_summary from report_normal ---
    if report_normal is not None and not report_normal.empty:
        for idx, row in report_normal.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            details["daily_summary"].append({
                "date": date_str,
                "account_value": float(row.get("account", 0)),
                "stock_value": float(row.get("value", 0)),
                "cash": float(row.get("cash", 0)),
                "return": float(row.get("return", 0)),
                "turnover": float(row.get("turnover", 0)),
                "cost": float(row.get("cost", 0)),
                "bench_return": float(row.get("bench", 0)),
            })

    # --- holdings + actions from positions_normal ---
    if positions_normal:
        sorted_dates = sorted(positions_normal.keys())
        prev_amounts = {}

        for date_val in sorted_dates:
            pos = positions_normal[date_val]
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)

            stocks = pos.get_stock_list()
            weight_dict = pos.get_stock_weight_dict(only_stock=False)

            day_holdings = []
            curr_amounts = {}
            for stock in stocks:
                amount = pos.get_stock_amount(stock)
                weight = weight_dict.get(stock, 0.0)
                curr_amounts[stock] = amount
                day_holdings.append({
                    "stock": stock,
                    "amount": float(amount),
                    "weight": round(float(weight) * 100, 4),
                })

            # Sort by weight descending
            day_holdings.sort(key=lambda x: x["weight"], reverse=True)
            details["holdings"][date_str] = day_holdings

            # Detect actions by diffing with previous day
            curr_set = set(curr_amounts.keys())
            prev_set = set(prev_amounts.keys())

            for stock in curr_set - prev_set:
                details["actions"].append({
                    "date": date_str, "stock": stock,
                    "type": "buy", "amount": float(curr_amounts[stock]),
                })
            for stock in prev_set - curr_set:
                details["actions"].append({
                    "date": date_str, "stock": stock,
                    "type": "sell", "amount": float(prev_amounts[stock]),
                })
            for stock in curr_set & prev_set:
                diff = curr_amounts[stock] - prev_amounts[stock]
                if abs(diff) > 0.01:
                    details["actions"].append({
                        "date": date_str, "stock": stock,
                        "type": "buy" if diff > 0 else "sell",
                        "amount": float(abs(diff)),
                    })

            prev_amounts = curr_amounts

    return details


def _eval_factor_incremental(
    expr: str,
    eh: str,
    market: str,
    start: str,
    end: str,
    label: str,
    timeout: int,
    use_cache: bool,
    data_path: str = None,
    region: str = None,
    forward_n: int = 1,
):
    """
    Evaluate a single factor using the incremental daily IC cache.

    Returns (success: bool, result_dict: dict).
    The result_dict contains metrics, daily_metrics, and metadata.

    When forward_n > 1 the IC at each date is averaged over the next forward_n
    daily returns, giving a multi-horizon IC estimate.  The cache is keyed on
    cache_label = "{label}_fwd{n}" so results don't collide with n=1 entries.
    """
    # Use a distinct cache label so forward-n results don't overwrite 1-day results
    cache_label = label if forward_n <= 1 else f"{label}_fwd{forward_n}"

    if use_cache:
        missing = FACTOR_STORE.get_missing_ranges(eh, market, cache_label, start, end)
    else:
        missing = [(start, end)]

    if not missing:
        # Full cache hit — compute summary from cached daily IC
        daily_data = FACTOR_STORE.get_daily_ic(eh, market, cache_label, start, end)
        metrics = FactorStore.compute_summary(daily_data)
        return True, {
            "success": True,
            "expression": expr,
            "market": market,
            "start_date": start,
            "end_date": end,
            "metrics": metrics,
            "daily_metrics": daily_data,
            "cached": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Compute missing ranges (parallel if 2 sub-ranges: prefix + suffix)
    # Scores are saved to disk by the subprocess (avoids Queue deadlock with large data)
    scores_dir = str(FACTOR_STORE.scores_dir)
    all_new_daily = []
    eval_failed = False
    eval_error = ""

    def _eval_range(r_start, r_end):
        return run_eval_with_timeout(
            expr, market, r_start, r_end, label, timeout,
            data_path=data_path, region=region,
            scores_save_dir=scores_dir,
            forward_n=forward_n,
        )

    if len(missing) > 1:
        with ThreadPoolExecutor(max_workers=len(missing)) as pool:
            futures = {
                pool.submit(_eval_range, rs, re): (rs, re) for rs, re in missing
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res.ok and isinstance(res.payload, dict):
                    all_new_daily.extend(res.payload.get("daily_metrics", []))
                else:
                    eval_failed = True
                    eval_error = f"{res.error_type}: {res.payload}"
    else:
        rs, re = missing[0]
        res = _eval_range(rs, re)
        if res.ok and isinstance(res.payload, dict):
            all_new_daily.extend(res.payload.get("daily_metrics", []))
        else:
            eval_failed = True
            eval_error = f"{res.error_type}: {res.payload}"

    if eval_failed:
        return False, _fail_result(expr, market, start, end, eval_error)

    # Store new daily IC (use cache_label to namespace forward-n results)
    if all_new_daily:
        FACTOR_STORE.register_expression(eh, expr)
        FACTOR_STORE.put_daily_ic(eh, market, cache_label, all_new_daily)

    # Load full daily IC for the requested range and compute summary
    daily_data = FACTOR_STORE.get_daily_ic(eh, market, cache_label, start, end)
    metrics = FactorStore.compute_summary(daily_data)

    return True, {
        "success": True,
        "expression": expr,
        "market": market,
        "start_date": start,
        "end_date": end,
        "metrics": metrics,
        "daily_metrics": daily_data,
        "cached": len(missing) == 0,
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

    # Resolve market config for data_path/region
    market = (data.get("market", DEFAULTS["market"]) or DEFAULTS["market"]).lower()
    mcfg = get_config().get_market_config(market)
    check_data_path = mcfg["data_path"]
    check_region = mcfg["region"]

    # ================================================================
    # BATCH PATH: single subprocess for all factors (1 qlib.init())
    # ================================================================
    if len(factors) > 1:
        factor_defs = []
        for i, f in enumerate(factors):
            name = f.get("name", "") or f"__check_{i}__"
            factor_defs.append({"name": name, "expression": f["expression"]})

        batch_timeout = int(data.get("timeout", DEFAULTS["timeout_batch"]))
        res = run_batch_check_with_timeout(
            factor_defs, instruments, start, end, batch_timeout,
            data_path=check_data_path, region=check_region,
        )

        if res.ok and isinstance(res.payload, dict) and "results" in res.payload:
            return jsonify(res.payload["results"]), 200

        error_msg = f"{res.error_type}: {res.payload}" if not res.ok else "Invalid batch response"
        results = []
        for f in factors:
            results.append({
                "success": False,
                "name": f.get("name", "") or "",
                "expression": f["expression"],
                "error_message": error_msg,
                "error_type": res.error_type or "BATCH_ERROR",
            })
        return jsonify(results), 200

    # ================================================================
    # SINGLE FACTOR PATH
    # ================================================================
    results = []
    for f in factors:
        expr = f["expression"]
        name = f.get("name", "") or ""
        res = run_check_with_timeout(
            expr, instruments, start, end, timeout,
            data_path=check_data_path, region=check_region,
        )

        if res.ok:
            payload = (
                res.payload
                if isinstance(res.payload, dict)
                else {"result": res.payload}
            )
            item = {"success": True, "name": name, "expression": expr, **payload}
        else:
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

    Uses incremental daily IC cache: only computes dates not yet cached.
    Factor scores are saved to disk for portfolio backtest reuse.

    POST JSON:
      {"expression": "Mean($close, 20)", "market": "csi300", "start": "2023-01-01",
       "end": "2024-01-01", "label": "close_return", "fast": true, ...}

    Return: always a JSON list, even for single expression.
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
        use_cache = bool(data.get("use_cache", DEFAULTS.get("use_cache", True)))
        timeout = int(data.get("timeout", DEFAULTS["timeout_eval"]))
        topk = int(data.get("topk", 50))
        n_drop = int(data.get("n_drop", 5))
        forward_n = max(1, int(data.get("forward_n", 1)))

        # Cache key includes forward_n so forward-n results don't collide with n=1
        cache_label = label if forward_n <= 1 else f"{label}_fwd{forward_n}"

        fast = bool(data.get("fast", False))
        n_jobs_backtest = int(data.get("n_jobs_backtest", 4))

        # Resolve market config (data_path, region, benchmark, trade costs)
        mcfg = get_config().get_market_config(market)
        data_path = mcfg["data_path"]
        region = mcfg["region"]
        benchmark = mcfg.get("benchmark", "SH000300")
        instruments = mcfg.get("instruments", market)

        # Build exchange kwargs: trade costs and daily limit differ between CN and US.
        # CN: limit_threshold=0.095 (10% daily price limit), higher costs with stamp duty.
        # US: limit_threshold=None (no daily limit), lower costs without stamp duty.
        exchange_kwargs = {
            "freq": "day",
            "deal_price": "close",
            "limit_threshold": mcfg.get("limit_threshold", 0.095 if region == "cn" else None),
            "open_cost": mcfg.get("open_cost", 0.0005 if region == "cn" else 0.0001),
            "close_cost": mcfg.get("close_cost", 0.0015 if region == "cn" else 0.0001),
            "min_cost": mcfg.get("min_cost", 5 if region == "cn" else 0),
        }

        # Apply optional frontend overrides for exchange kwargs
        ek_overrides = data.get("exchange_kwargs")
        if isinstance(ek_overrides, dict):
            for key in ("open_cost", "close_cost", "min_cost", "limit_threshold"):
                if key in ek_overrides:
                    val = ek_overrides[key]
                    exchange_kwargs[key] = float(val) if val is not None else None

        logger.info(
            "Evaluating %d expr(s) (market=%s, %s→%s, label=%s, fast=%s)",
            len(factors), market, start, end, label, fast,
        )

        # ================================================================
        # FAST BATCH PATH: batch subprocess for uncached factors
        # ================================================================
        if fast and len(factors) > 1:
            results = [None] * len(factors)
            uncached_indices = []
            uncached_factor_defs = []

            # Phase 1: check incremental cache per factor
            for i, f in enumerate(factors):
                expr = (f.get("expression") or "").strip()
                name = f.get("name", "") or ""

                if not expr:
                    results[i] = {
                        "success": False, "name": name, "expression": expr,
                        "error": "Missing 'expression'",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    continue

                if use_cache:
                    eh = expr_hash(_normalize_expr(expr))
                    missing = FACTOR_STORE.get_missing_ranges(eh, market, cache_label, start, end)
                    if not missing:
                        # Full cache hit
                        daily_data = FACTOR_STORE.get_daily_ic(eh, market, cache_label, start, end)
                        metrics = FactorStore.compute_summary(daily_data)
                        results[i] = {
                            "name": name, "expression": expr,
                            "success": True, "market": market,
                            "start_date": start, "end_date": end,
                            "metrics": metrics,
                            "daily_metrics": daily_data,
                            "cached": True,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        continue

                # Uncached — collect for batch
                factor_name = f"__batch_{i}__"
                uncached_indices.append(i)
                uncached_factor_defs.append({"name": factor_name, "expression": expr})

            # Phase 2: batch evaluate all uncached factors
            if uncached_factor_defs:
                batch_timeout = int(data.get("timeout", DEFAULTS["timeout_batch"]))
                scores_dir = str(FACTOR_STORE.scores_dir)

                res = run_batch_with_timeout(
                    uncached_factor_defs, market, start, end, label, batch_timeout,
                    scores_save_dir=scores_dir,
                    data_path=data_path, region=region,
                    forward_n=forward_n,
                )

                if res.ok and isinstance(res.payload, dict) and "results" in res.payload:
                    batch_results = res.payload["results"]
                    daily_per_factor = res.payload.get("daily_metrics_per_factor", {})

                    for batch_idx, orig_idx in enumerate(uncached_indices):
                        f = factors[orig_idx]
                        expr = (f.get("expression") or "").strip()
                        name = f.get("name", "") or ""
                        eh = expr_hash(_normalize_expr(expr))
                        factor_name = f"__batch_{orig_idx}__"

                        if batch_idx < len(batch_results):
                            br = batch_results[batch_idx]
                            if br.get("success"):
                                # Store daily IC in incremental cache (keyed by cache_label)
                                daily_data = daily_per_factor.get(factor_name, [])
                                if daily_data and use_cache:
                                    FACTOR_STORE.register_expression(eh, expr)
                                    FACTOR_STORE.put_daily_ic(eh, market, cache_label, daily_data)

                                payload = {
                                    k: v for k, v in br.items() if k != "name"
                                }
                                payload["daily_metrics"] = daily_data
                                results[orig_idx] = {"name": name, **payload}
                            else:
                                results[orig_idx] = {
                                    "name": name, "expression": expr,
                                    **_fail_result(expr, market, start, end,
                                                   br.get("error", "Unknown batch error")),
                                }
                        else:
                            results[orig_idx] = {
                                "name": name, "expression": expr,
                                **_fail_result(expr, market, start, end,
                                               "Factor missing from batch result"),
                            }
                else:
                    error_msg = (
                        f"{res.error_type}: {res.payload}" if not res.ok
                        else "Invalid batch response"
                    )
                    for orig_idx in uncached_indices:
                        f = factors[orig_idx]
                        expr = (f.get("expression") or "").strip()
                        name = f.get("name", "") or ""
                        results[orig_idx] = {
                            "name": name, "expression": expr,
                            **_fail_result(expr, market, start, end, error_msg),
                        }

            return jsonify([r for r in results if r is not None]), 200

        # ================================================================
        # PER-FACTOR PATH: single factor, or fast=False (needs backtest)
        # ================================================================
        results = []
        indices_need_backtest = []
        expr_hashes = []  # parallel list tracking expr_hash per result

        for f in factors:
            expr = (f.get("expression") or "").strip()
            name = f.get("name", "") or ""

            if not expr:
                results.append({
                    "success": False, "name": name, "expression": expr,
                    "error": "Missing 'expression'",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                expr_hashes.append(None)
                continue

            eh = expr_hash(_normalize_expr(expr))
            expr_hashes.append(eh)

            # Use incremental cache
            ok, result_dict = _eval_factor_incremental(
                expr, eh, market, start, end, label, timeout, use_cache,
                data_path=data_path, region=region,
                forward_n=forward_n,
            )

            item = {"name": name, **result_dict}
            results.append(item)

            if ok and not fast:
                indices_need_backtest.append(len(results) - 1)

        if fast or not indices_need_backtest:
            return jsonify(results), 200

        # ---- Portfolio backtest using saved scores ----
        # data_path, region, benchmark, instruments already resolved above

        pool = _get_worker_pool()

        def _run_backtest_for_idx(idx):
            eh = expr_hashes[idx]
            expr_ = results[idx]["expression"]

            # Try using cached scores first
            scores_df = FACTOR_STORE.load_scores(eh, market, start, end) if eh else None
            if scores_df is not None:
                logger.info("Using cached scores for backtest: %s", eh[:12])
                analysis_df, report_normal, positions_normal = pool.submit_backtest(
                    region=region,
                    job_type="backtest_by_scores",
                    kwargs=dict(
                        factor_scores=scores_df,
                        topk=topk, n_drop=n_drop,
                        start_time=start, end_time=end,
                        data_path=data_path, region=region, BENCH=benchmark,
                        exchange_kwargs=exchange_kwargs,
                    ),
                )
            else:
                # Fallback: compute from expression
                logger.info("No cached scores, computing from expression for backtest")
                analysis_df, report_normal, positions_normal = pool.submit_backtest(
                    region=region,
                    job_type="backtest_by_single_alpha",
                    kwargs=dict(
                        alpha_factor=expr_,
                        topk=topk, n_drop=n_drop,
                        start_time=start, end_time=end,
                        data_path=data_path, instruments=instruments,
                        region=region, BENCH=benchmark,
                        exchange_kwargs=exchange_kwargs,
                    ),
                )
            pm = _extract_portfolio_metrics(analysis_df)
            details = _extract_portfolio_details(report_normal, positions_normal)
            return pm, details

        max_workers = max(1, n_jobs_backtest)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            for idx in indices_need_backtest:
                future = ex.submit(_run_backtest_for_idx, idx)
                future_map[future] = idx

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    pm, details = future.result()
                    if pm:
                        results[idx]["portfolio_metrics"] = pm
                    if details:
                        results[idx]["portfolio_details"] = details
                except Exception as e:
                    logger.warning("Portfolio backtest failed (idx=%d): %s", idx, e)

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


@bp.route("/portfolio", methods=["POST"])
def portfolio_combine():
    """
    No-train portfolio combine: z-score normalize + equal-weight average.

    Accepts a list of factor expressions, z-score normalizes each per date
    (cross-sectional), averages them into a combined signal, and computes
    IC/RankIC/ICIR for both per-factor and combined signals. Optionally
    runs a portfolio backtest (TopkDropout) on the combined signal.

    POST JSON:
      {
        "expression": ["Rank($close, 20)", "Mean($volume, 5)", ...],
        "market": "csi300",
        "start": "2023-01-01",
        "end": "2024-01-01",
        "label": "close_return",
        "fast": true,
        "topk": 50,
        "n_drop": 5,
        "timeout": 600,
        "forward_n": 1
      }

    Response:
      {
        "success": true,
        "n_factors": 3,
        "combined_metrics": {"ic": ..., "rank_ic": ..., "icir": ..., "rank_icir": ..., "n_dates": ...},
        "combined_daily_metrics": [{"date": ..., "ic": ..., "rank_ic": ...}, ...],
        "per_factor_results": [{"name": ..., "expression": ..., "metrics": {...}}, ...],
        "portfolio_metrics": {...},       // when fast=false
        "portfolio_details": {...},       // when fast=false
        "timestamp": "..."
      }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        factors, err = normalize_factors_from_expression_field(data)
        if err:
            msg, etype = err
            return jsonify({
                "success": False, "error": msg, "error_type": etype,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 400

        if len(factors) < 2:
            return jsonify({
                "success": False,
                "error": "Portfolio combine requires at least 2 factor expressions",
                "error_type": "INSUFFICIENT_FACTORS",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 400

        start = data.get("start", DEFAULTS["start"])
        end = data.get("end", DEFAULTS["end"])
        market = (data.get("market", DEFAULTS["market"]) or DEFAULTS["market"]).lower()
        label = data.get("label", DEFAULTS["label"])
        timeout = int(data.get("timeout", DEFAULTS["timeout_batch"]))
        topk = int(data.get("topk", 50))
        n_drop = int(data.get("n_drop", 5))
        forward_n = max(1, int(data.get("forward_n", 1)))
        fast = bool(data.get("fast", True))
        n_jobs_backtest = int(data.get("n_jobs_backtest", 4))

        # Resolve market config
        mcfg = get_config().get_market_config(market)
        data_path = mcfg["data_path"]
        region = mcfg["region"]
        benchmark = mcfg.get("benchmark", "SH000300")
        instruments = mcfg.get("instruments", market)

        exchange_kwargs = {
            "freq": "day",
            "deal_price": "close",
            "limit_threshold": mcfg.get("limit_threshold", 0.095 if region == "cn" else None),
            "open_cost": mcfg.get("open_cost", 0.0005 if region == "cn" else 0.0001),
            "close_cost": mcfg.get("close_cost", 0.0015 if region == "cn" else 0.0001),
            "min_cost": mcfg.get("min_cost", 5 if region == "cn" else 0),
        }

        # Apply optional frontend overrides for exchange kwargs
        ek_overrides = data.get("exchange_kwargs")
        if isinstance(ek_overrides, dict):
            for key in ("open_cost", "close_cost", "min_cost", "limit_threshold"):
                if key in ek_overrides:
                    val = ek_overrides[key]
                    exchange_kwargs[key] = float(val) if val is not None else None

        # Build factor defs and combined hash
        import hashlib
        factor_defs = []
        sorted_hashes = []
        for i, f in enumerate(factors):
            name = f.get("name", "") or f"__combine_{i}__"
            factor_defs.append({"name": name, "expression": f["expression"]})
            sorted_hashes.append(expr_hash(_normalize_expr(f["expression"])))
        sorted_hashes.sort()
        combined_hash = hashlib.blake2b(
            "|".join(sorted_hashes).encode(), digest_size=16
        ).hexdigest()

        scores_dir = str(FACTOR_STORE.scores_dir)

        logger.info(
            "Portfolio combine: %d factors (market=%s, %s→%s, fast=%s)",
            len(factors), market, start, end, fast,
        )

        res = run_portfolio_combine_with_timeout(
            factor_defs, market, start, end, label, timeout,
            scores_save_dir=scores_dir,
            combined_hash=combined_hash,
            data_path=data_path, region=region,
            forward_n=forward_n,
        )

        if not res.ok:
            return jsonify({
                "success": False,
                "error": f"{res.error_type}: {res.payload}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 200

        payload = res.payload
        if not isinstance(payload, dict) or not payload.get("success"):
            return jsonify({
                "success": False,
                "error": str(payload),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 200

        result = {
            "success": True,
            "n_factors": payload.get("n_factors", len(factors)),
            "n_valid_factors": payload.get("n_valid_factors", 0),
            "combined_metrics": payload.get("combined_metrics", {}),
            "combined_daily_metrics": payload.get("combined_daily_metrics", []),
            "per_factor_results": payload.get("per_factor_results", []),
            "market": market,
            "start_date": start,
            "end_date": end,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Portfolio backtest on combined signal
        if not fast:
            try:
                combined_scores = FACTOR_STORE.load_scores(
                    combined_hash, market, start, end
                )
                if combined_scores is not None:
                    pool = _get_worker_pool()
                    analysis_df, report_normal, positions_normal = pool.submit_backtest(
                        region=region,
                        job_type="backtest_by_scores",
                        kwargs=dict(
                            factor_scores=combined_scores,
                            topk=topk, n_drop=n_drop,
                            start_time=start, end_time=end,
                            data_path=data_path, region=region,
                            BENCH=benchmark,
                            exchange_kwargs=exchange_kwargs,
                        ),
                    )
                    pm = _extract_portfolio_metrics(analysis_df)
                    if pm:
                        result["portfolio_metrics"] = pm
                    details = _extract_portfolio_details(report_normal, positions_normal)
                    if details:
                        result["portfolio_details"] = details
                else:
                    logger.warning("No combined scores on disk for backtest")
            except Exception as e:
                logger.warning("Portfolio backtest failed: %s", e)

        return jsonify(result), 200

    except Exception as e:
        logger.exception("portfolio combine error")
        return jsonify({
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 500
