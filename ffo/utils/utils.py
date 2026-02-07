#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for Factor Evaluation API

Key features:
- Persistent SQLite cache with LRU eviction, expr hashing, and stats.
- Hard timeouts using subprocesses that get TERMINATED on exceed.
- Fast, vectorized IC / RankIC per-date, plus summaries.
- Self-contained workers to run inside child processes (safe to kill).
"""

from __future__ import annotations
import re

# import agent.qlib_contrib.qlib_extend_ops
import os
import json
import time
import math
import sqlite3
import hashlib
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from multiprocessing import Process, Queue
from dataclasses import dataclass

import pandas as pd
import numpy as np
import re
from zss import Node
import ast


# -----------------------------
# Config (env-overridable)
# -----------------------------
DEFAULT_PROVIDER_URI = os.environ.get(
    "QLIB_PROVIDER_URI", os.path.expanduser("~/.qlib/qlib_data/cn_data")
)
DEFAULT_REGION = os.environ.get("QLIB_REGION", "cn")
DEFAULT_INSTRUMENTS = os.environ.get("QLIB_INSTRUMENTS", "CSI300")

CACHE_PATH = os.environ.get("FACTOR_API_CACHE_PATH", "factor_cache.sqlite")
CACHE_MAX_ENTRIES = int(
    os.environ.get("FACTOR_API_CACHE_MAX_ENTRIES", "50000")
)  # LRU target
CACHE_PRUNE_BATCH = int(
    os.environ.get("FACTOR_API_CACHE_PRUNE_BATCH", "5000")
)  # delete this many when over

CPU_JOBS = max(1, int(os.environ.get("FACTOR_API_CPU_JOBS", str(os.cpu_count() or 4))))


# -----------------------------
# Helpers: hashing & keys
# -----------------------------
def expr_hash(expr: str) -> str:
    """Stable short hash for expressions (32 hex chars)."""
    h = hashlib.blake2b(expr.encode("utf-8"), digest_size=16)
    return h.hexdigest()


def cache_key(
    expr: str, market: str, start: str, end: str, label: str, topk: int, n_drop: int
) -> str:
    """Key = hash(expr) + params (keeps key short, ignores whitespace diffs)."""
    h = expr_hash(_normalize_expr(expr))
    return f"{h}|{market}|{start}|{end}|{label}|{topk}|{n_drop}"


def _normalize_expr(expr: str) -> str:
    return " ".join(expr.strip().split())


class FactorNode(Node):
    def __init__(self, label):
        super().__init__(label)
        self.label = label
        self._children = []

    def addkid(self, node):
        self._children.append(node)
        return super().addkid(node)

    def get_children(self):
        return self._children


class FactorParser:
    def __init__(self):
        self.param_map = {}
        self.param_counter = 1
        self.var_map = {}
        self.var_counter = 1

    def preprocess(self, expr):
        # Replace variables like $close
        def replace_var(match):
            var = match.group()
            safe = f"var{self.var_counter}"
            self.var_map[safe] = var
            self.var_counter += 1
            return safe

        expr = re.sub(r"\$\w+", replace_var, expr)

        # Replace parameters like {lag}
        def replace_param(match):
            param = match.group()
            if param not in self.param_map:
                cname = f"C{self.param_counter}"
                self.param_map[param] = cname
                self.param_counter += 1
            return self.param_map[param]

        expr = re.sub(r"\{\w+\}", replace_param, expr)
        return expr

    def parse(self, expr):
        pre_expr = self.preprocess(expr)
        tree = ast.parse(pre_expr, mode="eval")
        return self._convert(tree.body)

    def _convert(self, node):
        if isinstance(node, ast.BinOp):
            op_name = self._get_op_name(node.op)
            root = FactorNode(op_name)
            root.addkid(self._convert(node.left))
            root.addkid(self._convert(node.right))
            return root
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            root = FactorNode(func_name)
            for arg in node.args:
                root.addkid(self._convert(arg))
            return root
        elif isinstance(node, ast.Name):
            if node.id in self.var_map:
                return FactorNode(self.var_map[node.id])
            elif node.id in self.param_map.values():
                return FactorNode(node.id)
            else:
                return FactorNode(node.id)
        elif isinstance(node, ast.Constant):
            return FactorNode(str(node.value))
        elif isinstance(node, ast.UnaryOp):
            op_name = self._get_op_name(node.op)
            root = FactorNode(op_name)
            root.addkid(self._convert(node.operand))
            return root
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only simple comparisons supported")
            op_name = self._get_op_name(node.ops[0])
            root = FactorNode(op_name)
            root.addkid(self._convert(node.left))
            root.addkid(self._convert(node.comparators[0]))
            return root
        else:
            raise ValueError(f"Unsupported AST node: {node}")

    def _get_op_name(self, op):
        if isinstance(op, ast.Add):
            return "Add"
        elif isinstance(op, ast.Sub):
            return "Sub"
        elif isinstance(op, ast.Mult):
            return "Mul"
        elif isinstance(op, ast.Div):
            return "Div"
        elif isinstance(op, ast.USub):
            return "Neg"
        elif isinstance(op, ast.Gt):
            return "Gt"
        elif isinstance(op, ast.Lt):
            return "Lt"
        elif isinstance(op, ast.GtE):
            return "GtE"
        elif isinstance(op, ast.LtE):
            return "LtE"
        elif isinstance(op, ast.Eq):
            return "Eq"
        elif isinstance(op, ast.NotEq):
            return "NotEq"
        else:
            raise ValueError(f"Unsupported operator: {op}")

    # --- New Feature: Complexity Analysis ---
    def get_complexity(self, root: FactorNode):
        stats = {
            "node_count": 0,
            "depth": 0,
            "operator_count": 0,
            "function_count": 0,
            "var_count": 0,
            "param_count": 0,
        }

        vars_seen, params_seen = set(), set()

        def traverse(node, depth=1):
            stats["node_count"] += 1
            stats["depth"] = max(stats["depth"], depth)

            if node.label in [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Neg",
                "Gt",
                "Lt",
                "GtE",
                "LtE",
                "Eq",
                "NotEq",
            ]:
                stats["operator_count"] += 1
            elif node.label.startswith("C"):  # parameter
                params_seen.add(node.label)
            elif node.label.startswith("var"):  # variable
                vars_seen.add(node.label)
            elif (
                node.get_children()
            ):  # function call (if has children and not an operator)
                stats["function_count"] += 1

            for child in node.get_children():
                traverse(child, depth + 1)

        traverse(root)

        stats["var_count"] = len(vars_seen)
        stats["param_count"] = len(params_seen)

        # Composite score (simple heuristic)
        stats["complexity_score"] = (
            stats["node_count"]
            + 2 * stats["operator_count"]
            + 2 * stats["function_count"]
            + stats["depth"]
        )

        return stats


def print_tree(node, level=0):
    """
    Nicely print the tree.
    """
    print("  " * level + node.label)
    for child in node.children:
        print_tree(child, level + 1)


# -----------------------------
# Persistent cache (SQLite + JSON)
# -----------------------------
class PersistentCache:
    """Simple key->JSON persistent cache with LRU fields and auto-pruning."""

    def __init__(self, path: str = CACHE_PATH, max_entries: int = CACHE_MAX_ENTRIES):
        self.path = path
        self.max_entries = max_entries
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_access INTEGER NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kv_last_access ON kv(last_access)"
            )
        finally:
            conn.close()

    def get(self, k: str) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        conn = self._connect()
        try:
            cur = conn.execute("SELECT v, hits FROM kv WHERE k=?", (k,))
            row = cur.fetchone()
            if row is None:
                return None
            v_json, hits = row
            conn.execute(
                "UPDATE kv SET last_access=?, hits=? WHERE k=?", (now, hits + 1, k)
            )
            return json.loads(v_json)
        finally:
            conn.close()

    def set(self, k: str, obj: Dict[str, Any]) -> None:
        now = int(time.time())
        v_json = json.dumps(obj, ensure_ascii=False)
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv (k, v, created_at, last_access, hits) VALUES (?, ?, ?, ?, COALESCE((SELECT hits FROM kv WHERE k=?), 0))",
                (k, v_json, now, now, k),
            )
            # prune if too big (lightweight LRU)
            cur = conn.execute("SELECT COUNT(*) FROM kv")
            (n,) = cur.fetchone()
            if n > self.max_entries:
                to_delete = max(0, n - self.max_entries) + CACHE_PRUNE_BATCH
                conn.execute(
                    f"DELETE FROM kv WHERE k IN (SELECT k FROM kv ORDER BY last_access ASC LIMIT {to_delete})"
                )
        finally:
            conn.close()

    def clear(self):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM kv")
        finally:
            conn.close()

    def stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM kv").fetchone()
            (min_ts,) = conn.execute(
                "SELECT COALESCE(MIN(created_at),0) FROM kv"
            ).fetchone()
            (max_ts,) = conn.execute(
                "SELECT COALESCE(MAX(last_access),0) FROM kv"
            ).fetchone()
            return {
                "entries": n,
                "created_min": int(min_ts),
                "last_access_max": int(max_ts),
                "path": self.path,
            }
        finally:
            conn.close()


# -----------------------------
# Subprocess runner with hard timeout
# -----------------------------
@dataclass
class SubprocessResult:
    ok: bool
    payload: Any
    error_type: Optional[str] = None


def _subprocess_wrapper(q: Queue, target, args):
    """
    Wrapper function for subprocess execution.
    Moved to module level to be picklable in Python 3.12+
    """
    try:
        payload = target(*args)
        q.put((True, payload, None))
    except Exception as e:
        # pass back a trimmed error string + type
        q.put((False, f"{type(e).__name__}: {e}", type(e).__name__))


def _spawn_and_run(target, args: tuple, timeout: int) -> SubprocessResult:
    """
    Run `target(*args)` in a separate process, return its (ok, payload).
    If timeout exceeded, kill the process and return TIMEOUT.
    """
    q: Queue = Queue(maxsize=1)
    p = Process(target=_subprocess_wrapper, args=(q, target, args), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        # Hard kill and clean
        p.terminate()
        p.join(5)
        return SubprocessResult(
            ok=False,
            payload=f"Timeout: execution exceeded {timeout}s",
            error_type="TIMEOUT",
        )

    if q.empty():
        return SubprocessResult(
            ok=False,
            payload="No result returned from subprocess.",
            error_type="NO_RESULT",
        )

    ok, payload, err_type = q.get_nowait()
    return SubprocessResult(ok=ok, payload=payload, error_type=err_type)


# -----------------------------
# Vectorized IC / RankIC
# -----------------------------
def _daily_ic_rankic(
    feature_s: pd.Series, label_s: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute daily IC (Pearson) and RankIC (Spearman via ranks) by date.
    Both inputs are aligned Series with a MultiIndex containing 'datetime'.
    """
    df = pd.concat({"f": feature_s, "y": label_s}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    g = df.groupby(level="datetime", sort=False)

    # IC: corr per date
    ic_daily = g.apply(lambda x: x["f"].corr(x["y"]))

    # RankIC: rank within-date then Pearson corr
    def _rankic(x: pd.DataFrame) -> float:
        fr = x["f"].rank(method="average")
        yr = x["y"].rank(method="average")
        return fr.corr(yr)

    rankic_daily = g.apply(_rankic)
    return ic_daily, rankic_daily


def _ir(mean_val: float, std_val: float) -> float:
    if std_val is None or not np.isfinite(std_val) or std_val <= 0:
        return float(0)
    return float(mean_val / std_val)


def summarize_ic_rankic(
    ic_daily: pd.Series, rankic_daily: pd.Series
) -> Dict[str, float]:
    ic_mean = float(ic_daily.mean()) if len(ic_daily) else float(0)
    ic_std = float(ic_daily.std(ddof=1)) if len(ic_daily) > 1 else float(0)
    rankic_mean = float(rankic_daily.mean()) if len(rankic_daily) else float(0)
    rankic_std = float(rankic_daily.std(ddof=1)) if len(rankic_daily) > 1 else float(0)
    return {
        "ic": ic_mean,
        "icir": _ir(ic_mean, ic_std),
        "rank_ic": rankic_mean,
        "rank_icir": _ir(rankic_mean, rankic_std),
        "n_dates": int(len(ic_daily.index.unique())),
    }


# -----------------------------
# Child-process workers
# -----------------------------
def _child_eval_expr(
    expr: str, market: str, start: str, end: str, label: str
) -> Dict[str, Any]:
    """
    Runs inside child process. Heavy libs are imported here, so the parent can hard-kill safely.
    """
    import qlib
    from backtest.qlib.dataloader import compute_factor_data

    # Per-process init (safer across OS / forks)
    import logging

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=DEFAULT_PROVIDER_URI, region=DEFAULT_REGION)

    factor_list = [{"name": "api_factor", "expression": expr}]
    out = compute_factor_data(
        factor_list,
        label=label,
        instruments=market.lower(),
        start_time=start,
        end_time=end,
    )

    if (
        out is None
        or "feature" not in out
        or "label" not in out
        or "api_factor" not in out["feature"]
        or "LABEL" not in out["label"]
    ):
        raise RuntimeError("Missing feature/label output from compute_factor_data")

    f_s: pd.Series = out["feature"]["api_factor"]
    y_s: pd.Series = out["label"]["LABEL"]

    ic_d, rankic_d = _daily_ic_rankic(f_s, y_s)
    metrics = summarize_ic_rankic(ic_d, rankic_d)

    # Format daily metrics for response
    daily_metrics = []
    if not ic_d.empty and not rankic_d.empty:
        for date_val in ic_d.index.unique():
            daily_metrics.append(
                {
                    "date": (
                        date_val.strftime("%Y-%m-%d")
                        if hasattr(date_val, "strftime")
                        else str(date_val)
                    ),
                    "ic": (
                        float(ic_d.get(date_val, 0.0))
                        if date_val in ic_d.index
                        else 0.0
                    ),
                    "rank_ic": (
                        float(rankic_d.get(date_val, 0.0))
                        if date_val in rankic_d.index
                        else 0.0
                    ),
                }
            )

    return {
        "success": True,
        "expression": expr,
        "market": market,
        "start_date": start,
        "end_date": end,
        "metrics": {
            "ic": metrics["ic"],
            "rank_ic": metrics["rank_ic"],
            "ir": metrics["icir"],  # backward compatible alias
            "icir": metrics["icir"],
            "rank_icir": metrics["rank_icir"],
            "turnover": 0.0,
            "n_dates": metrics["n_dates"],
        },
        "daily_metrics": daily_metrics,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


def _child_check_expr(
    expr: str, instruments: str, start_time: str, end_time: str
) -> Dict[str, Any]:
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader

    import logging

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=DEFAULT_PROVIDER_URI, region=DEFAULT_REGION)

    cfg = {"feature": ([expr], ["test_expr"])}
    dl = QlibDataLoader(config=cfg)
    try:
        df = dl.load(instruments=instruments, start_time=start_time, end_time=end_time)
        feat = df.get("feature", pd.DataFrame())
    except Exception as e:
        error_message = str(e)
        # Read error message from Qlib
        if re.search(r"missing \d+ required positional argument", error_message):
            return {
                "success": False,
                "error_message": error_message,
                "error_type": "INVALID_PARA",
            }

        elif re.search(r"The operator \[.*?\] is not registered", error_message):
            return {
                "success": False,
                "error_message": error_message,
                "error_type": "UNREGISTERED_OPERATOR",
            }
        else:
            return {
                "success": False,
                "error_message": error_message,
                "error_type": "UNKNOWN_ERROR",
            }

    if feat.empty:
        return {
            "success": False,
            "error_message": "Empty feature matrix",
            "error_type": "EMPTY_DATA",
        }

    nan_ratio = float(feat.isna().mean().mean())
    if nan_ratio > 0.01:
        return {
            "success": False,
            "error_message": f"High NaN ratio: {nan_ratio:.2%}",
            "error_type": "HIGH_NAN_RATIO",
        }
    return {"success": True, "nan_ratio": nan_ratio}


def _child_batch_eval(
    factors: List[Dict[str, str]],
    instruments: str,
    start: str,
    end: str,
    label_spec: str,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Efficient batch IC/RankIC:
    - Load all factor columns + one returns label in a single dataloader call
    - For each date, compute corrwith across columns (Pearson), and Spearman by ranking
    """
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader
    import os

    # Configure parallel loading
    if n_jobs > 1:
        os.environ["QLIB_ENABLE_PARALLEL"] = str(n_jobs)
    else:
        os.environ["QLIB_ENABLE_PARALLEL"] = "0"

    import logging

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=DEFAULT_PROVIDER_URI, region=DEFAULT_REGION)

    names = [f["name"] for f in factors]
    fields = [f["expression"] for f in factors]

    cfg = {"feature": (fields, names), "label": ([label_spec], ["RET"])}
    dl = QlibDataLoader(config=cfg)
    data = dl.load(instruments=instruments, start_time=start, end_time=end)

    if "feature" not in data or "label" not in data or data["feature"].empty:
        raise RuntimeError("Missing data for batch evaluation")

    X = data["feature"]  # MultiIndex (datetime, instrument) columns per factor
    y = data["label"]["RET"]  # Series on same MI

    # Align and drop NA once
    df = X.join(y.rename("RET"), how="inner").dropna()
    if df.empty:
        raise RuntimeError("Empty aligned feature/label after dropna")

    # Split back
    X = df.drop(columns=["RET"])
    y = df["RET"]

    # Group by date
    groups = X.index.get_level_values("datetime")
    uniq_dates = np.unique(groups)

    # Pre-allocate collectors
    ic_rows = []
    ric_rows = []

    # Process each date; vectorized within the date
    for d in uniq_dates:
        mask = groups == d
        Xd = X.loc[mask]
        yd = y.loc[mask]

        # IC: Pearson correlation column-wise
        ic = Xd.corrwith(yd, axis=0)
        ic.index.name = "factor"
        ic_rows.append(ic)

        # RankIC: rank features and y within date, then Pearson
        Xr = Xd.rank(method="average")
        yr = yd.rank(method="average")
        ric = Xr.corrwith(yr, axis=0)
        ric.index.name = "factor"
        ric_rows.append(ric)

    ic_table = pd.concat(ic_rows, axis=1).T  # shape: n_dates x n_factors
    ric_table = pd.concat(ric_rows, axis=1).T

    # Summaries
    results: List[Dict[str, Any]] = []
    for nm in X.columns:
        ic_series = ic_table[nm].dropna()
        ric_series = ric_table[nm].dropna()
        ic_mean = float(ic_series.mean()) if len(ic_series) else float(0)
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float(0)
        ric_mean = float(ric_series.mean()) if len(ric_series) else float(0)
        ric_std = float(ric_series.std(ddof=1)) if len(ric_series) > 1 else float(0)

        # find expression
        expr = next((f["expression"] for f in factors if f["name"] == nm), "")

        results.append(
            {
                "name": nm,
                "expression": expr,
                "success": True,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "metrics": {
                    "ic": ic_mean,
                    "icir": _ir(ic_mean, ic_std),
                    "rank_ic": ric_mean,
                    "rank_icir": _ir(ric_mean, ric_std),
                    "n_dates": int(len(ic_series.index)),
                },
            }
        )

    return {
        "success": True,
        "count": len(results),
        "results": results,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


# -----------------------------
# Public API (used by server)
# -----------------------------
def run_eval_with_timeout(
    expr: str, market: str, start: str, end: str, label: str, timeout: int
) -> SubprocessResult:
    return _spawn_and_run(_child_eval_expr, (expr, market, start, end, label), timeout)


def run_check_with_timeout(
    expr: str, instruments: str, start: str, end: str, timeout: int
) -> SubprocessResult:
    return _spawn_and_run(_child_check_expr, (expr, instruments, start, end), timeout)


def run_batch_with_timeout(
    factors: List[Dict[str, str]],
    instruments: str,
    start: str,
    end: str,
    label_spec: str,
    timeout: int,
    n_jobs: int = 1,
) -> SubprocessResult:
    return _spawn_and_run(
        _child_batch_eval,
        (factors, instruments, start, end, label_spec, n_jobs),
        timeout,
    )


def normalize_factors_from_expression_field(data: dict):
    """
    Normalize request data into a list of factors:
    Returns: List[{"name": str, "expression": str}]

    Supported "expression":
      - "xxx"                                   (single, no name)
      - {"n1": "e1", "n2": "e2"}                (one or many, with name)
      - ["e1", "e2", ...]                       (many, no name)
      - (optional) mixed list: ["e1", {"n2":"e2"}]
    """
    expr_field = data.get("expression", None)
    if expr_field is None:
        expr_field = data.get("expr", None)  # optional legacy alias

    if expr_field is None:
        return None, ("Missing 'expression'", "EMPTY_EXPR")

    def _push(name: str, expr: str):
        expr = (expr or "").strip()
        if not expr:
            raise ValueError("EMPTY_EXPR")
        return {"name": name or "", "expression": expr}

    factors = []

    # case 1: string
    if isinstance(expr_field, str):
        expr = expr_field.strip()
        if not expr:
            return None, ("Missing 'expression'", "EMPTY_EXPR")
        return [{"name": "", "expression": expr}], None

    # case 2: dict name -> expr
    if isinstance(expr_field, dict):
        if len(expr_field) == 0:
            return None, ("Missing 'expression'", "EMPTY_EXPR")
        for name, expr in expr_field.items():
            if not isinstance(expr, str):
                return None, (
                    "Invalid 'expression' dict value (must be string)",
                    "BAD_EXPR_FORMAT",
                )
            expr = expr.strip()
            if not expr:
                return None, (f"Empty expression for name='{name}'", "EMPTY_EXPR")
            factors.append({"name": str(name), "expression": expr})
        return factors, None

    # case 3: list
    if isinstance(expr_field, list):
        if len(expr_field) == 0:
            return None, ("Missing 'expression'", "EMPTY_EXPR")

        for i, item in enumerate(expr_field):
            # list of strings
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    return None, (f"Empty expression at index {i}", "EMPTY_EXPR")
                factors.append({"name": "", "expression": s})
                continue

            # list of dicts: {"name": "expr"} (allow one or many pairs)
            if isinstance(item, dict):
                if len(item) == 0:
                    return None, (f"Empty dict at index {i}", "BAD_EXPR_FORMAT")
                for name, expr in item.items():
                    if not isinstance(expr, str):
                        return None, (
                            f"Invalid expression at index {i} (must be string)",
                            "BAD_EXPR_FORMAT",
                        )
                    expr = expr.strip()
                    if not expr:
                        return None, (
                            f"Empty expression for name='{name}' at index {i}",
                            "EMPTY_EXPR",
                        )
                    factors.append({"name": str(name), "expression": expr})
                continue

            return None, (
                f"Invalid item type in expression list at index {i}",
                "BAD_EXPR_FORMAT",
            )

        return factors, None

    return None, (
        "Invalid 'expression' type (must be string, dict, or list)",
        "BAD_EXPR_FORMAT",
    )
