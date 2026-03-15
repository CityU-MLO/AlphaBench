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
    expr: str, market: str, start: str, end: str, label: str, topk: int, n_drop: int,
    fast: bool = True,
) -> str:
    """Key = hash(expr) + params (keeps key short, ignores whitespace diffs)."""
    h = expr_hash(_normalize_expr(expr))
    return f"{h}|{market}|{start}|{end}|{label}|{topk}|{n_drop}|fast={int(fast)}"


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
        # pass back error message (without type prefix) + type separately
        q.put((False, str(e), type(e).__name__))


def _spawn_and_run(target, args: tuple, timeout: int) -> SubprocessResult:
    """
    Run `target(*args)` in a separate process, return its (ok, payload).
    If timeout exceeded, kill the process and return TIMEOUT.

    NOTE: We read from the Queue BEFORE joining the process.
    This avoids a deadlock when the result is large (e.g., pickled DataFrames):
    the subprocess blocks in q.put() waiting for the pipe buffer to drain,
    but p.join() waits for the process to exit — creating a deadlock.
    """
    q: Queue = Queue(maxsize=1)
    p = Process(target=_subprocess_wrapper, args=(q, target, args), daemon=True)
    p.start()

    # Read result from queue first (with timeout) to avoid deadlock
    try:
        ok, payload, err_type = q.get(timeout=timeout)
    except Exception:
        # Timeout or empty — kill the process
        if p.is_alive():
            p.terminate()
            p.join(5)
        return SubprocessResult(
            ok=False,
            payload=f"Timeout: execution exceeded {timeout}s",
            error_type="TIMEOUT",
        )

    # Now safe to join — process should exit quickly after queue is drained
    p.join(10)
    if p.is_alive():
        p.terminate()
        p.join(5)

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
    expr: str, market: str, start: str, end: str, label: str,
    return_scores: bool = False,
    data_path: str = None, region: str = None,
    scores_save_dir: str = None,
    forward_n: int = 1,
) -> Dict[str, Any]:
    """
    Runs inside child process. Heavy libs are imported here, so the parent can hard-kill safely.

    Args:
        return_scores: If True, include pickled factor scores DataFrame in the result
            (sent via Queue — may deadlock with very large DataFrames).
        data_path: Qlib data directory (overrides DEFAULT_PROVIDER_URI).
        region: Qlib region (overrides DEFAULT_REGION).
        scores_save_dir: If provided, save factor scores to disk at
            {scores_save_dir}/{expr_hash}/{market}.pkl (preferred over return_scores).
        forward_n: Number of forward days over which to average IC/RankIC.
            forward_n=1 (default): compute IC against next-day return only.
            forward_n=n: compute IC against each of the n forward daily returns,
            then average — giving a smoother, multi-horizon IC estimate.
    """
    import pickle as _pickle
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader

    # Per-process init (safer across OS / forks)
    import logging

    provider_uri = data_path or DEFAULT_PROVIDER_URI
    qlib_region = region or DEFAULT_REGION

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=provider_uri, region=qlib_region)
    import utils.qlib_extend_ops as _ext_ops; _ext_ops.register()  # re-register after qlib.init() reset

    # Build label spec(s) depending on forward_n
    _empty_result = {
        "success": True,
        "expression": expr,
        "market": market,
        "start_date": start,
        "end_date": end,
        "metrics": {
            "ic": 0.0, "rank_ic": 0.0, "ir": 0.0,
            "icir": 0.0, "rank_icir": 0.0, "turnover": 0.0, "n_dates": 0,
        },
        "daily_metrics": [],
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }

    if forward_n <= 1:
        # Single label — load directly via QlibDataLoader (avoids dataloader.py
        # re-initialising qlib with CN defaults at module-import time)
        label_expr = _LABEL_MAP.get(label, _LABEL_MAP["close_return"])
        cfg = {
            "feature": ([expr], ["api_factor"]),
            "label": ([label_expr], ["LABEL"]),
        }
        dl = QlibDataLoader(config=cfg)
        try:
            out = dl.load(instruments=market.lower(), start_time=start, end_time=end)
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger("_child_eval_expr").warning(
                "Exception during factor computation: %s", _exc,
            )
            return _empty_result

        if out is None or out.empty or "feature" not in out or "label" not in out:
            import logging as _logging
            _logging.getLogger("_child_eval_expr").warning(
                "No data for %s in [%s, %s] — likely non-trading days", expr[:60], start, end,
            )
            return _empty_result

        feat = out["feature"]
        lbl = out["label"]
        if "api_factor" not in feat.columns or "LABEL" not in lbl.columns:
            return _empty_result

        f_s: pd.Series = feat["api_factor"]
        y_s: pd.Series = lbl["LABEL"]
        ic_d, rankic_d = _daily_ic_rankic(f_s, y_s)

    else:
        # Multi-day forward IC: average IC over forward_n consecutive daily returns
        fwd_labels = _build_forward_label_exprs(forward_n)
        label_names_fw = [nm for nm, _ in fwd_labels]
        label_fields_fw = [ex for _, ex in fwd_labels]

        cfg = {
            "feature": ([expr], ["api_factor"]),
            "label": (label_fields_fw, label_names_fw),
        }
        dl = QlibDataLoader(config=cfg)
        try:
            out = dl.load(instruments=market.lower(), start_time=start, end_time=end)
        except Exception:
            return _empty_result

        if out is None or out.empty or "feature" not in out or "label" not in out:
            return _empty_result

        feat = out["feature"]
        if "api_factor" not in feat.columns:
            return _empty_result

        f_s = feat["api_factor"]
        lbl = out["label"]

        # Compute IC/RankIC against each forward day, then average per date
        ic_ds: List[pd.Series] = []
        rankic_ds: List[pd.Series] = []
        for nm in label_names_fw:
            if nm not in lbl.columns:
                continue
            ic_d_k, rankic_d_k = _daily_ic_rankic(f_s, lbl[nm])
            if not ic_d_k.empty:
                ic_ds.append(ic_d_k)
                rankic_ds.append(rankic_d_k)

        if not ic_ds:
            return _empty_result

        ic_d = pd.concat(ic_ds, axis=1).mean(axis=1)
        rankic_d = pd.concat(rankic_ds, axis=1).mean(axis=1)

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

    result = {
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

    # Save factor scores to disk (preferred — avoids large Queue transfers)
    # Merge with existing scores to preserve data from previous incremental runs
    if scores_save_dir:
        import os as _os
        import tempfile as _tempfile
        scores_df = f_s.to_frame("score")
        eh = expr_hash(_normalize_expr(expr))
        score_dir = _os.path.join(scores_save_dir, eh)
        _os.makedirs(score_dir, exist_ok=True)
        score_path = _os.path.join(score_dir, f"{market.lower()}.pkl")

        # Merge with existing scores if present
        if _os.path.exists(score_path):
            try:
                existing = pd.read_pickle(score_path)
                scores_df = pd.concat([existing, scores_df])
                scores_df = scores_df[~scores_df.index.duplicated(keep="last")]
                scores_df = scores_df.sort_index()
            except Exception:
                pass  # overwrite on merge failure

        # Atomic write: temp file + rename
        fd, tmp_path = _tempfile.mkstemp(dir=score_dir, suffix=".pkl.tmp")
        try:
            _os.close(fd)
            scores_df.to_pickle(tmp_path)
            _os.replace(tmp_path, score_path)
        except Exception:
            if _os.path.exists(tmp_path):
                _os.unlink(tmp_path)
            raise
        result["scores_saved"] = True
    elif return_scores:
        # Fallback: send via Queue (may deadlock with very large DataFrames)
        scores_df = f_s.to_frame("score")
        result["factor_scores_bytes"] = _pickle.dumps(scores_df)

    return result


def _child_check_expr(
    expr: str, instruments: str, start_time: str, end_time: str,
    data_path: str = None, region: str = None,
) -> Dict[str, Any]:
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader

    import logging

    provider_uri = data_path or DEFAULT_PROVIDER_URI
    qlib_region = region or DEFAULT_REGION

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=provider_uri, region=qlib_region)
    import utils.qlib_extend_ops as _ext_ops; _ext_ops.register()  # re-register after qlib.init() reset

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


def _check_single_column(col_name: str, feat_col: pd.Series) -> Dict[str, Any]:
    """Check a single feature column for validity (used by batch check)."""
    if feat_col.empty or feat_col.dropna().empty:
        return {
            "success": False,
            "error_message": "Empty feature matrix",
            "error_type": "EMPTY_DATA",
        }
    nan_ratio = float(feat_col.isna().mean())
    if nan_ratio > 0.01:
        return {
            "success": False,
            "error_message": f"High NaN ratio: {nan_ratio:.2%}",
            "error_type": "HIGH_NAN_RATIO",
        }
    return {"success": True, "nan_ratio": nan_ratio}


def _classify_qlib_error(error_message: str) -> str:
    """Classify a Qlib error message into an error type."""
    if re.search(r"missing \d+ required positional argument", error_message):
        return "INVALID_PARA"
    elif re.search(r"The operator \[.*?\] is not registered", error_message):
        return "UNREGISTERED_OPERATOR"
    return "UNKNOWN_ERROR"


def _child_batch_check(
    factors: List[Dict[str, str]],
    instruments: str,
    start_time: str,
    end_time: str,
    data_path: str = None, region: str = None,
) -> Dict[str, Any]:
    """
    Batch syntax check: load ALL expressions in a single QlibDataLoader call.
    If the batch load fails (e.g. one bad expression), fall back to checking
    each expression individually — still only one qlib.init().
    """
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader
    import logging

    provider_uri = data_path or DEFAULT_PROVIDER_URI
    qlib_region = region or DEFAULT_REGION

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=provider_uri, region=qlib_region)
    import utils.qlib_extend_ops as _ext_ops; _ext_ops.register()  # re-register after qlib.init() reset

    names = [f["name"] for f in factors]
    fields = [f["expression"] for f in factors]

    # Try loading all expressions at once
    cfg = {"feature": (fields, names)}
    dl = QlibDataLoader(config=cfg)
    try:
        df = dl.load(instruments=instruments, start_time=start_time, end_time=end_time)
        feat = df.get("feature", pd.DataFrame())
    except Exception:
        feat = None  # batch load failed, fall back to per-expression check

    results = []

    if feat is not None and not feat.empty:
        # Batch load succeeded — check each column
        for nm, expr in zip(names, fields):
            if nm in feat.columns:
                check = _check_single_column(nm, feat[nm])
                results.append({"name": nm, "expression": expr, **check})
            else:
                results.append({
                    "name": nm, "expression": expr,
                    "success": False, "error_message": "Column missing from batch load",
                    "error_type": "EMPTY_DATA",
                })
    else:
        # Batch load failed or empty — check each expression individually
        # (still uses the same qlib.init(), so no redundant init overhead)
        for nm, expr in zip(names, fields):
            single_cfg = {"feature": ([expr], [nm])}
            single_dl = QlibDataLoader(config=single_cfg)
            try:
                single_df = single_dl.load(
                    instruments=instruments,
                    start_time=start_time,
                    end_time=end_time,
                )
                single_feat = single_df.get("feature", pd.DataFrame())
            except Exception as e:
                error_message = str(e)
                results.append({
                    "name": nm, "expression": expr,
                    "success": False, "error_message": error_message,
                    "error_type": _classify_qlib_error(error_message),
                })
                continue

            if single_feat.empty or nm not in single_feat.columns:
                results.append({
                    "name": nm, "expression": expr,
                    "success": False, "error_message": "Empty feature matrix",
                    "error_type": "EMPTY_DATA",
                })
            else:
                check = _check_single_column(nm, single_feat[nm])
                results.append({"name": nm, "expression": expr, **check})

    return {"success": True, "count": len(results), "results": results}


_LABEL_MAP = {
    "close_return": "Ref($close, -1)/$close - 1",
    "close_return_lag": "Ref($close, -2)/Ref($close, -1) - 1",
    "close": "Ref($close, -1)",
}


def _build_forward_label_exprs(forward_n: int) -> List[Tuple[str, str]]:
    """
    Build (name, qlib_expr) pairs for n consecutive forward daily returns.

    Day k return = price change from close of day k-1 to close of day k (forward).
      k=1: Ref($close,-1)/$close - 1
      k=2: Ref($close,-2)/Ref($close,-1) - 1
      ...
      k=n: Ref($close,-n)/Ref($close,-(n-1)) - 1

    Args:
        forward_n: Number of forward days (>=1).

    Returns:
        List of (label_name, qlib_expression) tuples.
    """
    labels = []
    for k in range(1, forward_n + 1):
        if k == 1:
            expr = "Ref($close, -1)/$close - 1"
        else:
            expr = f"Ref($close, -{k})/Ref($close, -{k - 1}) - 1"
        labels.append((f"RET_d{k}", expr))
    return labels


def _child_batch_eval(
    factors: List[Dict[str, str]],
    instruments: str,
    start: str,
    end: str,
    label_spec: str,
    scores_save_dir: str = None,
    data_path: str = None, region: str = None,
    forward_n: int = 1,
) -> Dict[str, Any]:
    """
    Efficient batch IC/RankIC:
    - Load all factor columns + label(s) in a single dataloader call
    - For each date, compute corrwith across columns (Pearson), and Spearman by ranking
    - When forward_n > 1, loads n forward daily return labels and averages IC per date

    Args:
        scores_save_dir: If provided, save each factor's scores to disk at
            {scores_save_dir}/{expr_hash}/{market}.pkl
        data_path: Qlib data directory (overrides DEFAULT_PROVIDER_URI).
        region: Qlib region (overrides DEFAULT_REGION).
        forward_n: Number of forward days to average IC over (default 1 = next-day only).
    """
    import os as _os
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader
    import logging

    provider_uri = data_path or DEFAULT_PROVIDER_URI
    qlib_region = region or DEFAULT_REGION

    logging.getLogger("qlib.Initialization").setLevel(logging.WARNING)
    qlib.init(provider_uri=provider_uri, region=qlib_region)
    import utils.qlib_extend_ops as _ext_ops; _ext_ops.register()  # re-register after qlib.init() reset

    names = [f["name"] for f in factors]
    fields = [f["expression"] for f in factors]

    # Build label expressions
    if forward_n <= 1:
        label_expr = _LABEL_MAP.get(label_spec)
        if label_expr is None:
            raise ValueError(
                f"Unsupported label: {label_spec}. "
                f"Supported: {list(_LABEL_MAP.keys())}"
            )
        label_fields = [label_expr]
        label_names = ["RET"]
    else:
        fwd_labels = _build_forward_label_exprs(forward_n)
        label_names = [nm for nm, _ in fwd_labels]
        label_fields = [ex for _, ex in fwd_labels]

    cfg = {"feature": (fields, names), "label": (label_fields, label_names)}
    dl = QlibDataLoader(config=cfg)
    data = dl.load(instruments=instruments, start_time=start, end_time=end)

    if "feature" not in data or "label" not in data or data["feature"].empty:
        raise RuntimeError("Missing data for batch evaluation")

    X = data["feature"]  # MultiIndex (datetime, instrument) columns per factor
    Y = data["label"]    # DataFrame with one or more label columns

    # Save factor scores to disk (before dropna, to preserve full cross-section)
    # Merges with existing scores to preserve data from previous incremental runs
    if scores_save_dir:
        import tempfile as _tmpfile
        for nm in X.columns:
            expr = next((f["expression"] for f in factors if f["name"] == nm), "")
            if expr:
                eh = expr_hash(_normalize_expr(expr))
                score_dir = _os.path.join(scores_save_dir, eh)
                _os.makedirs(score_dir, exist_ok=True)
                score_path = _os.path.join(score_dir, f"{instruments}.pkl")
                scores_df = X[[nm]].rename(columns={nm: "score"})

                # Merge with existing scores if present
                if _os.path.exists(score_path):
                    try:
                        existing = pd.read_pickle(score_path)
                        scores_df = pd.concat([existing, scores_df])
                        scores_df = scores_df[~scores_df.index.duplicated(keep="last")]
                        scores_df = scores_df.sort_index()
                    except Exception:
                        pass  # overwrite on merge failure

                # Atomic write
                fd, tmp_path = _tmpfile.mkstemp(dir=score_dir, suffix=".pkl.tmp")
                try:
                    _os.close(fd)
                    scores_df.to_pickle(tmp_path)
                    _os.replace(tmp_path, score_path)
                except Exception:
                    if _os.path.exists(tmp_path):
                        _os.unlink(tmp_path)
                    raise

    # Align features with all label columns and drop NA
    df = X.join(Y, how="inner").dropna()
    if df.empty:
        raise RuntimeError("Empty aligned feature/label after dropna")

    # Split back
    X = df[X.columns]
    Y = df[Y.columns]

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
        Yd = Y.loc[mask]

        if forward_n <= 1:
            # Single label — fast vectorized path
            yd = Yd.iloc[:, 0]
            ic = Xd.corrwith(yd, axis=0)
            Xr = Xd.rank(method="average")
            yr = yd.rank(method="average")
            ric = Xr.corrwith(yr, axis=0)
        else:
            # Multi-day forward: average IC/RankIC across all label columns
            ic_mats: List[pd.Series] = []
            ric_mats: List[pd.Series] = []
            Xr = Xd.rank(method="average")  # rank features once
            for col in Yd.columns:
                yd_k = Yd[col]
                ic_mats.append(Xd.corrwith(yd_k, axis=0))
                yr_k = yd_k.rank(method="average")
                ric_mats.append(Xr.corrwith(yr_k, axis=0))
            ic = pd.concat(ic_mats, axis=1).mean(axis=1)
            ric = pd.concat(ric_mats, axis=1).mean(axis=1)

        ic.index.name = "factor"
        ric.index.name = "factor"
        ic_rows.append(ic)
        ric_rows.append(ric)

    ic_table = pd.concat(ic_rows, axis=1).T  # shape: n_dates x n_factors
    ric_table = pd.concat(ric_rows, axis=1).T

    # Format date strings for daily metrics
    date_strs = [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        for d in uniq_dates
    ]

    # Summaries + daily metrics per factor
    results: List[Dict[str, Any]] = []
    daily_metrics_per_factor: Dict[str, List[Dict]] = {}

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
                "market": instruments,
                "start_date": start,
                "end_date": end,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "metrics": {
                    "ic": ic_mean,
                    "ir": _ir(ic_mean, ic_std),
                    "icir": _ir(ic_mean, ic_std),
                    "rank_ic": ric_mean,
                    "rank_icir": _ir(ric_mean, ric_std),
                    "turnover": 0.0,
                    "n_dates": int(len(ic_series.index)),
                },
            }
        )

        # Build daily metrics for this factor
        factor_daily = []
        for i, d_str in enumerate(date_strs):
            d_ts = uniq_dates[i]
            ic_val = float(ic_table.iloc[i][nm]) if d_ts in ic_series.index else 0.0
            ric_val = float(ric_table.iloc[i][nm]) if d_ts in ric_series.index else 0.0
            factor_daily.append({"date": d_str, "ic": ic_val, "rank_ic": ric_val})
        daily_metrics_per_factor[nm] = factor_daily

    return {
        "success": True,
        "count": len(results),
        "results": results,
        "daily_metrics_per_factor": daily_metrics_per_factor,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


# -----------------------------
# Public API (used by server)
# -----------------------------
def run_eval_with_timeout(
    expr: str, market: str, start: str, end: str, label: str, timeout: int,
    return_scores: bool = False,
    data_path: str = None, region: str = None,
    scores_save_dir: str = None,
    forward_n: int = 1,
) -> SubprocessResult:
    return _spawn_and_run(
        _child_eval_expr,
        (expr, market, start, end, label, return_scores, data_path, region, scores_save_dir, forward_n),
        timeout,
    )


def run_check_with_timeout(
    expr: str, instruments: str, start: str, end: str, timeout: int,
    data_path: str = None, region: str = None,
) -> SubprocessResult:
    return _spawn_and_run(
        _child_check_expr, (expr, instruments, start, end, data_path, region), timeout
    )


def run_batch_check_with_timeout(
    factors: List[Dict[str, str]],
    instruments: str,
    start: str,
    end: str,
    timeout: int,
    data_path: str = None, region: str = None,
) -> SubprocessResult:
    return _spawn_and_run(
        _child_batch_check,
        (factors, instruments, start, end, data_path, region),
        timeout,
    )


def run_batch_with_timeout(
    factors: List[Dict[str, str]],
    instruments: str,
    start: str,
    end: str,
    label_spec: str,
    timeout: int,
    scores_save_dir: str = None,
    data_path: str = None, region: str = None,
    forward_n: int = 1,
) -> SubprocessResult:
    return _spawn_and_run(
        _child_batch_eval,
        (factors, instruments, start, end, label_spec, scores_save_dir, data_path, region, forward_n),
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
