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
from datetime import datetime
from urllib.parse import unquote
import agent.qlib_contrib.qlib_extend_ops
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils import (
    PersistentCache,
    cache_key,
    run_eval_with_timeout,
    run_check_with_timeout,
    run_batch_with_timeout,
    DEFAULT_INSTRUMENTS,
)

# -----------------------------
# App / Logging
# -----------------------------
app = Flask(__name__)
CORS(app)

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
        "timestamp": datetime.utcnow().isoformat(),
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "Factor Evaluation API",
            "timestamp": datetime.utcnow().isoformat(),
            "cache": CACHE.stats(),
        }
    )

@app.route("/check", methods=["POST"])
def check():
    """
    Quick validation that an expression loads and isn't mostly NaN.

    Body JSON:
    {
      "expression": "...",
      "instruments": "CSI300"  (optional; default from utils.DEFAULT_INSTRUMENTS)
      "start": "YYYY-MM-DD"    (optional; default DEFAULTS['check_start'])
      "end": "YYYY-MM-DD"      (optional; default DEFAULTS['check_end'])
      "timeout": 30            (optional; seconds)
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    expr = (data.get("expression") or "").strip()
    if not expr:
        return jsonify({"success": False, "error_message": "Missing 'expression'", "error_type": "EMPTY_EXPR"}), 400

    instruments = data.get("instruments", DEFAULT_INSTRUMENTS)
    start = data.get("start", DEFAULTS["check_start"])
    end = data.get("end", DEFAULTS["check_end"])
    timeout = int(data.get("timeout", DEFAULTS["timeout_check"]))

    res = run_check_with_timeout(expr, instruments, start, end, timeout)

    print(f"Expression: {expr} Result: {res}")
    status = 200 if res.ok else 500
    return jsonify(res.payload if res.ok else {"success": False, "error_message": res.payload, "error_type": res.error_type}), status

@app.route("/eval", methods=["GET", "POST"])
def eval_once():
    """
    Evaluate a single expression over a time range.

    GET params:
      expr, start, end, market, label, use_cache=true|false, timeout
    POST JSON:
    {
      "expression": "...",
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "market": "csi300",
      "label": "close_return",
      "use_cache": true,
      "timeout": 120
    }
    """
    try:
        if request.method == "GET":
            expr = unquote((request.args.get("expression") or "").strip().strip("'").strip('"'))
            start = (request.args.get("start") or DEFAULTS["start"]).strip("'").strip('"')
            end = (request.args.get("end") or DEFAULTS["end"]).strip("'").strip('"')
            market = (request.args.get("market") or DEFAULTS["market"]).strip("'").strip('"').lower()
            label = (request.args.get("label") or DEFAULTS["label"]).strip()
            use_cache = (request.args.get("use_cache", "true").lower() == "true")
            timeout = int(request.args.get("timeout", DEFAULTS["timeout_eval"]))
        else:
            data = request.get_json(force=True, silent=True) or {}
            expr = (data.get("expression") or "").strip()
            start = data.get("start", DEFAULTS["start"])
            end = data.get("end", DEFAULTS["end"])
            market = data.get("market", DEFAULTS["market"]).lower()
            label = data.get("label", DEFAULTS["label"])
            use_cache = bool(data.get("use_cache", DEFAULTS["use_cache"]))
            timeout = int(data.get("timeout", DEFAULTS["timeout_eval"]))

        if not expr:
            return jsonify({"success": False, "error": "Missing 'expression'"}), 400

        key = cache_key(expr, market, start, end, label)
        if use_cache:
            cached = CACHE.get(key)
            if cached:
                return jsonify(cached), 200

        logger.info("Evaluating expr (market=%s, %s→%s, label=%s)", market, start, end, label)
        res = run_eval_with_timeout(expr, market, start, end, label, timeout)

        if res.ok:
            if use_cache:
                CACHE.set(key, res.payload)
            return jsonify(res.payload), 200
        else:
            return jsonify(_fail_result(expr, market, start, end, f"{res.error_type}: {res.payload}")), 200

    except Exception as e:
        logger.exception("eval error")
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}", "timestamp": datetime.utcnow().isoformat()}), 500

@app.route("/batch_eval", methods=["POST"])
def batch_eval():
    """
    Batch evaluate many expressions (faster than N single calls).

    Body JSON:
    {
      "factors": [{"name": "F1", "expression": "..."}, ...],
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "market": "csi300",
      "label": "close_return",    # will be turned into label spec for QlibDataLoader
      "timeout": 300
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    factors = data.get("factors") or []
    if not isinstance(factors, list) or not all(isinstance(f, dict) and "name" in f and "expression" in f for f in factors):
        return jsonify({"success": False, "error": "Invalid 'factors' format"}), 400

    start = data.get("start", DEFAULTS["start"])
    end = data.get("end", DEFAULTS["end"])
    market = (data.get("market", DEFAULTS["market"]) or "csi300").upper()  # instruments name for loader
    label = data.get("label", DEFAULTS["label"])
    timeout = int(data.get("timeout", DEFAULTS["timeout_batch"]))

    # QlibDataLoader expects a label expression. For common 'close_return', we use next day's return.
    # Adjust here if your setup defines labels differently.
    label_spec = {
        "close_return": "Ref($close, -1) / $close - 1",
    }.get(label, "Ref($close, -1) / $close - 1")

    res = run_batch_with_timeout(factors, market, start, end, label_spec, timeout)
    status = 200 if res.ok else 500
    if res.ok:
        return jsonify(res.payload), status
    return jsonify({"success": False, "error": f"{res.error_type}: {res.payload}", "timestamp": datetime.utcnow().isoformat()}), status

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    CACHE.clear()
    return jsonify({"success": True, "message": "Cache cleared", "timestamp": datetime.utcnow().isoformat()})

@app.route("/cache_stats", methods=["GET"])
def cache_stats():
    return jsonify({"success": True, "stats": CACHE.stats(), "timestamp": datetime.utcnow().isoformat()})

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9889"))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print(f"Starting Factor Evaluation API on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Example: http://localhost:{port}/eval?expr=%22Rank(Corr($close,$volume,10),252)%22&start=2023-01-01&end=2024-01-01&market=csi300")

    # threaded=True is fine: all heavy work is pushed to subprocesses with hard timeouts
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
