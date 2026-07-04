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

from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.utils import PersistentCache
from utils.assay_engine import engine_name, is_assay, health as assay_health

# Load routes
from routes.combinations import bp as combinations_bp
from routes.factors import bp as factors_bp

# -----------------------------
# App / Logging
# -----------------------------
app = Flask(__name__)
CORS(app)

app.register_blueprint(combinations_bp)
app.register_blueprint(factors_bp)

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
@app.route("/health", methods=["GET"])
def health():
    info = {
        "status": "healthy",
        "service": "Factor Evaluation API",
        "engine": engine_name(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache": CACHE.stats(),
    }
    # When running on the Assay engine, surface its reachability so callers
    # can tell whether the delegated backend is actually up.
    if is_assay():
        ok, assay_info = assay_health()
        info["assay"] = {"reachable": ok, **({"status": assay_info} if ok else {"error": assay_info})}
    return jsonify(info)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "19777"))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print(f"Starting Factor Evaluation API on port {port}")
    print(f"Debug mode: {debug}")
    print(
        f"Example: http://localhost:{port}/eval?expr=%22Rank(Corr($close,$volume,10),252)%22&start=2023-01-01&end=2024-01-01&market=csi300"
    )

    # threaded=True is fine: all heavy work is pushed to subprocesses with hard timeouts
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
