#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assay backend adapter.

FFO's factor-evaluation contract (POST /factors/eval, /factors/check,
/factors/portfolio) can be served by two interchangeable engines:

  * ``qlib``  — the native in-process Qlib worker pool (default)
  * ``assay`` — delegate to the Assay backtesting platform's REST API

This module implements the ``assay`` engine.  It is reached over HTTP because
Assay runs on Python >= 3.10 in its own virtual-env while FFO runs on the
Qlib (Python 3.9) interpreter, so the two cannot share a process.

The functions here return dictionaries shaped *exactly* like the ones the
native FFO routes return, so that every downstream consumer (searcher,
agent, benchmark, web UI) keeps working without any change:

    metrics = {"ic", "icir", "ir", "rank_ic", "rank_icir", "turnover", "n_dates"}

Engine selection and the Assay endpoint are read from config / environment:

    FFO_ENGINE=assay              # turn this engine on
    FFO_ASSAY_URL=http://127.0.0.1:8000
    FFO_ASSAY_TIMEOUT=180
    FFO_ASSAY_EXECUTION=close     # optional Assay fill model override
    FFO_ASSAY_ADJ=split           # optional Assay price-adjustment override

Qlib-style expressions (``$close``, ``Mean(...)``, ``Corr(...)``) are accepted
by Assay's parser directly, so no expression rewriting is needed.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

# Dual-import: the backend runs from the ffo/ dir (top-level `config` package),
# while AlphaBench imports this module as `ffo.utils.assay_engine` from the repo
# root. Support both so the agent layer can detect the active engine too.
try:
    from config.manager import get_config
except ModuleNotFoundError:  # imported as ffo.utils.assay_engine
    from ffo.config.manager import get_config

logger = logging.getLogger("FactorAPI.assay")

# Market name (FFO, lower-case) -> Assay universe id.  Used as a fallback when
# a market's config does not carry an explicit ``assay_universe`` key.
_UNIVERSE_FALLBACK = {
    "csi300": "CSI300",
    "csi500": "CSI500",
    "csi1000": "CSI1000",
    "sp500": "SP500",
    "nasdaq100": "NASDAQ100",
}


class AssayTransportError(Exception):
    """Raised when the Assay backend cannot be reached or returns garbage."""


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def engine_name() -> str:
    """Return the configured engine name (``qlib`` or ``assay``)."""
    return str(
        os.environ.get("FFO_ENGINE") or get_config().get("engine.backend", "qlib")
    ).strip().lower()


def is_assay() -> bool:
    return engine_name() == "assay"


def _assay_url() -> str:
    return str(
        os.environ.get("FFO_ASSAY_URL")
        or get_config().get("engine.assay_url", "http://127.0.0.1:8000")
    ).rstrip("/")


def _assay_timeout() -> int:
    return int(
        os.environ.get("FFO_ASSAY_TIMEOUT")
        or get_config().get("engine.assay_timeout", 180)
    )


def _assay_execution() -> Optional[str]:
    v = os.environ.get("FFO_ASSAY_EXECUTION") or get_config().get("engine.assay_execution")
    return str(v) if v else None


def _assay_adj() -> Optional[str]:
    v = os.environ.get("FFO_ASSAY_ADJ") or get_config().get("engine.assay_adj")
    return str(v) if v else None


def market_to_universe(market: str) -> str:
    """Map an FFO market name to an Assay universe id."""
    m = (market or "").lower()
    try:
        mcfg = get_config().get_market_config(m)
        u = mcfg.get("assay_universe")
        if u:
            return str(u)
    except Exception:  # pragma: no cover - config lookup is best-effort
        pass
    return _UNIVERSE_FALLBACK.get(m, market)


# --------------------------------------------------------------------------- #
# Low-level HTTP
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request(path: str, payload: dict, timeout: int) -> Tuple[int, Any]:
    """POST to Assay; return (status_code, parsed_json). Retries transient errors."""
    url = f"{_assay_url()}{path}"
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:  # connection / timeout
            last_err = e
            continue
        try:
            body = resp.json()
        except ValueError:
            raise AssayTransportError(
                f"non-JSON response from Assay ({resp.status_code}): {resp.text[:200]}"
            )
        return resp.status_code, body
    raise AssayTransportError(f"cannot reach Assay at {url}: {last_err}")


def health() -> Tuple[bool, dict]:
    """Probe the Assay backend. Returns (ok, info)."""
    url = f"{_assay_url()}/v1/system/status"
    try:
        resp = requests.get(url, timeout=10)
        info = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return resp.status_code == 200, info
    except requests.RequestException as e:
        return False, {"error": str(e)}


# --------------------------------------------------------------------------- #
# Result shaping (Assay FactorReport -> FFO metrics / result dict)
# --------------------------------------------------------------------------- #
def _num(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _metrics_from_report(rep: dict) -> Dict[str, Any]:
    """Translate an Assay FactorReport dict into FFO's metrics dict.

    Mirrors the keys produced by ``FactorStore.compute_summary``:
    ic, icir, ir, rank_ic, rank_icir, turnover, n_dates.
    """
    icir = _num(rep.get("icir"))
    return {
        "ic": _num(rep.get("ic")),
        "icir": icir,
        "ir": icir,  # backward-compatible alias (FactorResult maps ir -> icir)
        "rank_ic": _num(rep.get("rank_ic")),
        "rank_icir": _num(rep.get("rank_icir")),
        "turnover": _num(rep.get("turnover_1d")),
        "n_dates": int(rep.get("n_dates") or 0),
    }


def _daily_from_report(rep: dict) -> List[Dict[str, Any]]:
    """Best-effort per-date series -> FFO daily_metrics list.

    Assay does not return per-date IC series by default, in which case this
    yields an empty list (downstream consumers read aggregate ``metrics``,
    not ``daily_metrics``).
    """
    dates = rep.get("dates")
    ics = rep.get("ic_series")
    rics = rep.get("rank_ic_series")
    if not dates or not ics:
        return []
    out = []
    n = len(dates)
    for i in range(n):
        out.append({
            "date": dates[i],
            "ic": _num(ics[i]) if i < len(ics) else 0.0,
            "rank_ic": _num(rics[i]) if rics and i < len(rics) else 0.0,
        })
    return out


def _fail_item(expr: str, market: str, start: str, end: str, msg: str) -> Dict[str, Any]:
    """FFO-shaped failure result (mirrors routes.factors._fail_result)."""
    return {
        "success": False,
        "error": msg,
        "expression": expr,
        "market": market,
        "start_date": start,
        "end_date": end,
        "metrics": {
            "ic": 0.0, "rank_ic": 0.0, "ir": 0.0, "icir": 0.0,
            "rank_icir": 0.0, "turnover": 0.0, "n_dates": 0,
        },
        "timestamp": _now(),
    }


# --------------------------------------------------------------------------- #
# Core: evaluate one expression against Assay
# --------------------------------------------------------------------------- #
def _evaluate_report(
    expr: str, universe: str, start: str, end: str, forward_n: int, timeout: int,
    session_id: Optional[str] = None,
) -> Tuple[bool, Any]:
    """Evaluate one expression. Returns (ok, report_dict) or (False, error_message)."""
    payload: Dict[str, Any] = {
        "expr": expr,
        "universe": universe,
        "period": [start, end],
        "horizons": [int(forward_n)] if forward_n else None,
    }
    ex = _assay_execution()
    if ex:
        payload["execution"] = ex
    adj = _assay_adj()
    if adj:
        payload["adj"] = adj
    if session_id:
        payload["session_id"] = session_id

    status, body = _request("/v1/factor/evaluate", payload, timeout)

    if not isinstance(body, dict):
        return False, f"unexpected Assay response: {str(body)[:200]}"
    # Error envelope (transport-level / validation error)
    if "ic" not in body and body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return False, str(msg)
    # Per-factor failure mode (syntax / lookahead / constant / all-nan)
    fm = body.get("failure_mode")
    if fm:
        return False, f"{fm}: {body.get('suggestion') or fm}"
    return True, body


# --------------------------------------------------------------------------- #
# Public engine functions (called by routes.factors when engine == assay)
# --------------------------------------------------------------------------- #
def assay_eval(
    factors: List[Dict[str, str]],
    market: str,
    start: str,
    end: str,
    forward_n: int = 1,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """Evaluate factors via Assay, returning FFO-shaped /factors/eval results.

    ``factors`` is a list of ``{"name": str, "expression": str}``.
    Results preserve input order.
    """
    universe = market_to_universe(market)
    timeout = _assay_timeout()
    n = len(factors)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    def _work(i: int) -> Tuple[int, Dict[str, Any]]:
        f = factors[i]
        expr = (f.get("expression") or "").strip()
        name = f.get("name", "") or ""
        if not expr:
            return i, {
                "success": False, "name": name, "expression": expr,
                "error": "Missing 'expression'", "timestamp": _now(),
            }
        try:
            ok, rep = _evaluate_report(expr, universe, start, end, forward_n, timeout)
        except AssayTransportError as e:
            return i, {"name": name, **_fail_item(expr, market, start, end, str(e))}
        if not ok:
            return i, {"name": name, **_fail_item(expr, market, start, end, str(rep))}
        return i, {
            "name": name,
            "success": True,
            "expression": expr,
            "market": market,
            "start_date": start,
            "end_date": end,
            "metrics": _metrics_from_report(rep),
            "daily_metrics": _daily_from_report(rep),
            "cached": False,
            "timestamp": _now(),
        }

    workers = max(1, min(max_workers, n)) if n else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, item in pool.map(_work, range(n)):
            results[idx] = item

    return [r for r in results if r is not None]


def assay_check(factors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Syntax-check factors via Assay's lint endpoint.

    Returns FFO-shaped /factors/check results:
    ``{"success", "name", "expression", "error_message"?, "error_type"?}``
    """
    timeout = _assay_timeout()
    results: List[Dict[str, Any]] = []
    for f in factors:
        expr = (f.get("expression") or "").strip()
        name = f.get("name", "") or ""
        try:
            status, body = _request("/v1/factor/lint", {"expr": expr}, timeout)
        except AssayTransportError as e:
            results.append({
                "success": False, "name": name, "expression": expr,
                "error_message": str(e), "error_type": "ASSAY_UNREACHABLE",
            })
            continue

        diag = body.get("diagnostics") if isinstance(body, dict) else None
        diag = diag if isinstance(diag, dict) else {}
        ok = bool(diag.get("ok", diag.get("status") == "ok"))
        if ok:
            results.append({
                "success": True,
                "name": name or (body.get("canonical") or ""),
                "expression": expr,
                "canonical": body.get("canonical"),
                "fields": body.get("fields"),
                "operators": body.get("operators"),
            })
        else:
            errs = diag.get("errors") or []
            if errs and isinstance(errs[0], dict):
                msg = errs[0].get("message") or errs[0].get("title") or "Invalid expression"
            else:
                msg = diag.get("failure_mode") or "Invalid expression"
            results.append({
                "success": False, "name": name, "expression": expr,
                "error_message": msg,
                "error_type": diag.get("failure_mode") or "SYNTAX_ERROR",
            })
    return results


def assay_portfolio_combine(
    factors: List[Dict[str, str]],
    market: str,
    start: str,
    end: str,
    forward_n: int = 1,
) -> Dict[str, Any]:
    """Equal-weight z-score combine via Assay.

    Reproduces FFO's /factors/portfolio semantics ("z-score normalize +
    equal-weight average") by building the combined signal as an Assay
    expression ``(cs_zscore(c1) + cs_zscore(c2) + ...) / N`` over the
    canonical form of each valid factor, then evaluating it.
    """
    universe = market_to_universe(market)
    timeout = _assay_timeout()

    per_factor_results: List[Dict[str, Any]] = []
    canonical: List[str] = []
    n_valid = 0

    for f in factors:
        expr = (f.get("expression") or "").strip()
        name = f.get("name", "") or ""
        try:
            ok, rep = _evaluate_report(expr, universe, start, end, forward_n, timeout)
        except AssayTransportError as e:
            ok, rep = False, str(e)
        if ok:
            n_valid += 1
            canonical.append(rep.get("expr_canonical") or expr)
            per_factor_results.append({
                "name": name, "expression": expr,
                "metrics": _metrics_from_report(rep),
            })
        else:
            per_factor_results.append({
                "name": name, "expression": expr,
                "metrics": {
                    "ic": 0.0, "rank_ic": 0.0, "ir": 0.0, "icir": 0.0,
                    "rank_icir": 0.0, "turnover": 0.0, "n_dates": 0,
                },
                "error": str(rep),
            })

    combined_metrics: Dict[str, Any] = {}
    combined_daily: List[Dict[str, Any]] = []
    if len(canonical) >= 2:
        combo = "(" + " + ".join(f"cs_zscore({c})" for c in canonical) + ") / " + str(len(canonical))
        try:
            ok, rep = _evaluate_report(combo, universe, start, end, forward_n, timeout)
            if ok:
                combined_metrics = _metrics_from_report(rep)
                combined_daily = _daily_from_report(rep)
        except AssayTransportError as e:
            logger.warning("Assay combined eval failed: %s", e)

    return {
        "success": True,
        "n_factors": len(factors),
        "n_valid_factors": n_valid,
        "combined_metrics": combined_metrics,
        "combined_daily_metrics": combined_daily,
        "per_factor_results": per_factor_results,
        "market": market,
        "start_date": start,
        "end_date": end,
        "timestamp": _now(),
    }
