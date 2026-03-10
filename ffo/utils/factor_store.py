"""
Incremental Daily IC Cache with Factor Score Storage.

Replaces the simple key-value PersistentCache with a daily-granularity cache
that supports incremental updates:

- Daily IC/RankIC stored per (expr_hash, market, label, date) in SQLite
- Raw factor scores stored on disk as pickle files per (expr_hash, market)
- On repeated evaluations, only missing date ranges are computed
- Saved scores are reused for portfolio backtests (via backtest_by_scores)

Storage layout:
    cache_data/
        factor_perf.sqlite          # Daily IC/RankIC
        factor_scores/
            <expr_hash>/
                <market>.pkl        # pd.DataFrame MI:(datetime, instrument) col:"score"
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("FactorStore")


def _day_before(date_str: str) -> str:
    """Return the calendar day before date_str (YYYY-MM-DD)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - timedelta(days=1)).strftime("%Y-%m-%d")


def _day_after(date_str: str) -> str:
    """Return the calendar day after date_str (YYYY-MM-DD)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


class FactorStore:
    """
    Persistent incremental cache for factor evaluation.

    - SQLite stores daily IC/RankIC per (expr_hash, market, label, date)
    - Pickle files store raw factor scores per (expr_hash, market)
    - Coverage queries determine which date ranges still need computation
    """

    def __init__(self, cache_dir: str = "./cache_data"):
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "factor_perf.sqlite"
        self.scores_dir = self.cache_dir / "factor_scores"
        self._lock = threading.Lock()

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # ------------------------------------------------------------------ #
    #  DB setup                                                            #
    # ------------------------------------------------------------------ #

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS expressions (
                    expr_hash   TEXT PRIMARY KEY,
                    expression  TEXT NOT NULL,
                    created_at  INTEGER DEFAULT (strftime('%s','now'))
                );

                CREATE TABLE IF NOT EXISTS daily_ic (
                    expr_hash   TEXT    NOT NULL,
                    market      TEXT    NOT NULL,
                    label       TEXT    NOT NULL,
                    date        TEXT    NOT NULL,
                    ic          REAL,
                    rank_ic     REAL,
                    PRIMARY KEY (expr_hash, market, label, date)
                );

                CREATE INDEX IF NOT EXISTS idx_daily_ic_range
                    ON daily_ic(expr_hash, market, label, date);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    # ------------------------------------------------------------------ #
    #  Expression metadata                                                 #
    # ------------------------------------------------------------------ #

    def register_expression(self, expr_hash: str, expression: str) -> None:
        """Store expression text for a given hash (for admin/debugging)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO expressions (expr_hash, expression) VALUES (?, ?)",
                (expr_hash, expression),
            )

    # ------------------------------------------------------------------ #
    #  Coverage queries                                                    #
    # ------------------------------------------------------------------ #

    def get_coverage(
        self, expr_hash: str, market: str, label: str, start: str, end: str
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Returns (min_date, max_date, count) of cached daily IC entries
        within the requested [start, end] range.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(date), MAX(date), COUNT(*)
                FROM daily_ic
                WHERE expr_hash = ? AND market = ? AND label = ?
                  AND date BETWEEN ? AND ?
                """,
                (expr_hash, market, label, start, end),
            ).fetchone()
        return (row[0], row[1], row[2]) if row else (None, None, 0)

    def get_missing_ranges(
        self, expr_hash: str, market: str, label: str, start: str, end: str
    ) -> List[Tuple[str, str]]:
        """
        Determine which sub-ranges of [start, end] need computation.

        Uses min/max of cached dates to find prefix and/or suffix gaps.
        Assumes evals always cover contiguous date ranges (no internal gaps).

        Returns:
            List of (range_start, range_end) tuples. Empty if fully cached.
        """
        min_d, max_d, count = self.get_coverage(expr_hash, market, label, start, end)

        if count == 0:
            # No cached data in [start, end] — compute the full range.
            # We intentionally don't try to extend from global data outside
            # this range, because gap-filling with non-trading-day boundaries
            # can produce empty sub-ranges that cause subprocess errors.
            return [(start, end)]

        missing = []
        if min_d > start:
            missing.append((start, _day_before(min_d)))
        if max_d < end:
            missing.append((_day_after(max_d), end))
        return missing

    # ------------------------------------------------------------------ #
    #  Daily IC read / write                                               #
    # ------------------------------------------------------------------ #

    def get_daily_ic(
        self, expr_hash: str, market: str, label: str, start: str, end: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve daily IC and RankIC values for [start, end].

        Returns:
            List of {"date": str, "ic": float, "rank_ic": float} sorted by date.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, ic, rank_ic
                FROM daily_ic
                WHERE expr_hash = ? AND market = ? AND label = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                (expr_hash, market, label, start, end),
            ).fetchall()
        return [
            {"date": r[0], "ic": r[1], "rank_ic": r[2]}
            for r in rows
        ]

    def put_daily_ic(
        self,
        expr_hash: str,
        market: str,
        label: str,
        daily_data: List[Dict[str, Any]],
    ) -> int:
        """
        Insert or replace daily IC/RankIC values.

        Args:
            daily_data: List of {"date": str, "ic": float, "rank_ic": float}.

        Returns:
            Number of rows inserted/updated.
        """
        if not daily_data:
            return 0

        rows = [
            (expr_hash, market, label, d["date"], d.get("ic"), d.get("rank_ic"))
            for d in daily_data
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO daily_ic
                    (expr_hash, market, label, date, ic, rank_ic)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------ #
    #  Factor scores on disk                                               #
    # ------------------------------------------------------------------ #

    def _scores_path(self, expr_hash: str, market: str) -> Path:
        """Return the filesystem path for the scores pickle file."""
        d = self.scores_dir / expr_hash
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{market}.pkl"

    def has_scores(self, expr_hash: str, market: str) -> bool:
        p = self._scores_path(expr_hash, market)
        return p.exists() and p.stat().st_size > 0

    def save_scores(
        self,
        expr_hash: str,
        market: str,
        scores_df: pd.DataFrame,
        merge: bool = True,
    ) -> None:
        """
        Save factor scores to disk.

        Args:
            scores_df: DataFrame with MultiIndex (datetime, instrument).
            merge: If True and file exists, concatenate and deduplicate.
        """
        path = self._scores_path(expr_hash, market)

        with self._lock:
            if merge and path.exists():
                try:
                    existing = pd.read_pickle(path)
                    scores_df = pd.concat([existing, scores_df])
                    scores_df = scores_df[~scores_df.index.duplicated(keep="last")]
                    scores_df = scores_df.sort_index()
                except Exception as e:
                    logger.warning("Failed to merge scores, overwriting: %s", e)

            # Atomic write: temp file + rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".pkl.tmp"
            )
            try:
                os.close(fd)
                scores_df.to_pickle(tmp_path)
                os.replace(tmp_path, str(path))
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    def load_scores(
        self,
        expr_hash: str,
        market: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load factor scores from disk, optionally filtering to [start, end].

        Returns:
            DataFrame with MultiIndex (datetime, instrument), or None.
        """
        path = self._scores_path(expr_hash, market)
        if not path.exists():
            return None
        try:
            df = pd.read_pickle(path)
        except Exception as e:
            logger.warning("Corrupted scores file %s, removing: %s", path, e)
            try:
                path.unlink()
            except OSError:
                pass
            return None

        if start is not None or end is not None:
            dates = df.index.get_level_values("datetime")
            mask = pd.Series(True, index=df.index)
            if start is not None:
                mask &= dates >= pd.Timestamp(start)
            if end is not None:
                mask &= dates <= pd.Timestamp(end)
            df = df.loc[mask]

        return df if not df.empty else None

    # ------------------------------------------------------------------ #
    #  Summary computation                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_summary(daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary metrics from daily IC/RankIC values.

        Args:
            daily_data: List of {"date": str, "ic": float, "rank_ic": float}.

        Returns:
            {"ic", "icir", "rank_ic", "rank_icir", "n_dates", "turnover"}
        """
        if not daily_data:
            return {
                "ic": 0.0, "icir": 0.0,
                "rank_ic": 0.0, "rank_icir": 0.0,
                "turnover": 0.0, "n_dates": 0,
            }

        ics = np.array([d["ic"] for d in daily_data if d["ic"] is not None], dtype=float)
        rics = np.array([d["rank_ic"] for d in daily_data if d["rank_ic"] is not None], dtype=float)

        # Filter NaN
        ics = ics[~np.isnan(ics)]
        rics = rics[~np.isnan(rics)]

        def _ir(vals):
            if len(vals) < 2:
                return 0.0
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1))
            return m / s if s > 0 else 0.0

        return {
            "ic": float(np.mean(ics)) if len(ics) else 0.0,
            "icir": _ir(ics),
            "ir": _ir(ics),  # backward-compatible alias
            "rank_ic": float(np.mean(rics)) if len(rics) else 0.0,
            "rank_icir": _ir(rics),
            "turnover": 0.0,
            "n_dates": len(ics),
        }

    # ------------------------------------------------------------------ #
    #  Admin                                                               #
    # ------------------------------------------------------------------ #

    def get_store_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._connect() as conn:
            n_expr = conn.execute("SELECT COUNT(*) FROM expressions").fetchone()[0]
            n_daily = conn.execute("SELECT COUNT(*) FROM daily_ic").fetchone()[0]

        # Count score files
        n_score_files = sum(1 for _ in self.scores_dir.rglob("*.pkl"))

        return {
            "expressions": n_expr,
            "daily_ic_entries": n_daily,
            "score_files": n_score_files,
            "db_path": str(self.db_path),
            "scores_dir": str(self.scores_dir),
        }

    def clear(self, expr_hash: Optional[str] = None) -> None:
        """Clear all cache data, or for a specific expression."""
        with self._connect() as conn:
            if expr_hash:
                conn.execute("DELETE FROM daily_ic WHERE expr_hash = ?", (expr_hash,))
                conn.execute("DELETE FROM expressions WHERE expr_hash = ?", (expr_hash,))
                score_dir = self.scores_dir / expr_hash
                if score_dir.exists():
                    import shutil
                    shutil.rmtree(score_dir, ignore_errors=True)
            else:
                conn.execute("DELETE FROM daily_ic")
                conn.execute("DELETE FROM expressions")
                import shutil
                if self.scores_dir.exists():
                    shutil.rmtree(self.scores_dir, ignore_errors=True)
                    self.scores_dir.mkdir(parents=True, exist_ok=True)
