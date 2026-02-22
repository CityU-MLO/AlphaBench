"""
Persistent factor cache manager with expression hashing and date range support.
Stores daily IC values for efficient subset queries.
"""

import hashlib
import json
import os
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FactorCacheManager:
    """
    Manages persistent caching of factor evaluation results.

    Features:
    - Expression hashing for consistent cache keys
    - Daily IC/RankIC storage for subset queries
    - SQLite for metadata, pickle for data
    - Efficient date range subset retrieval
    """

    def __init__(self, cache_dir: str = "./cache_data"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite for metadata
        self.db_path = self.cache_dir / "factor_cache.db"
        self._init_database()

        # Data directory for pickle files
        self.data_dir = self.cache_dir / "factor_data"
        self.data_dir.mkdir(exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_cache (
                expr_hash TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                data_file TEXT NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expr_hash TEXT NOT NULL,
                market TEXT NOT NULL,
                date DATE NOT NULL,
                ic REAL,
                rank_ic REAL,
                icir REAL,
                rank_icir REAL,
                turnover REAL,
                FOREIGN KEY (expr_hash) REFERENCES factor_cache(expr_hash),
                UNIQUE(expr_hash, market, date)
            )
        """
        )

        # Create indexes for fast queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_entries_lookup 
            ON cache_entries(expr_hash, market, date)
        """
        )

        # Query history table for storing evaluation activity
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expression TEXT NOT NULL,
                market TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                ic REAL,
                rank_ic REAL,
                icir REAL,
                rank_icir REAL,
                query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_history_timestamp
            ON query_history(query_timestamp DESC)
        """
        )

        conn.commit()
        conn.close()

    def _hash_expression(self, expr: str) -> str:
        """Generate SHA256 hash of expression for cache key."""
        return hashlib.sha256(expr.encode()).hexdigest()

    def _get_data_file_path(self, expr_hash: str) -> Path:
        """Get path to pickle file for given expression hash."""
        return self.data_dir / f"{expr_hash}.pkl"

    def has_cache(self, expr: str, market: str, start_date: str, end_date: str) -> bool:
        """
        Check if cache exists for given parameters.
        Returns True if the requested date range is fully covered by cache.
        """
        expr_hash = self._hash_expression(expr)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Check if we have data covering the requested range
        cursor.execute(
            """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM cache_entries
            WHERE expr_hash = ? AND market = ?
        """,
            (expr_hash, market.lower()),
        )

        result = cursor.fetchone()
        conn.close()

        if result and result[0] and result[1]:
            cached_start = result[0]
            cached_end = result[1]
            return cached_start <= start_date and cached_end >= end_date

        return False

    def get_cache_coverage(
        self, expr: str, market: str, start_date: str, end_date: str
    ) -> Tuple[bool, Optional[str], Optional[str], List[Tuple[str, str]]]:
        """
        Check cache coverage and return missing date ranges.

        Returns:
            - is_fully_cached: True if entire range is cached
            - cached_start: Start date of cached data (if any)
            - cached_end: End date of cached data (if any)
            - missing_ranges: List of (start, end) tuples for missing date ranges
        """
        expr_hash = self._hash_expression(expr)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get all cached dates for this expression and market
        cursor.execute(
            """
            SELECT DISTINCT date
            FROM cache_entries
            WHERE expr_hash = ? AND market = ?
            ORDER BY date
        """,
            (expr_hash, market.lower()),
        )

        cached_dates = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not cached_dates:
            return False, None, None, [(start_date, end_date)]

        # Convert to pandas for easier date range analysis
        cached_dates_pd = pd.to_datetime(cached_dates)
        requested_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Find missing dates
        missing_dates = requested_range.difference(cached_dates_pd)

        if len(missing_dates) == 0:
            return True, cached_dates[0], cached_dates[-1], []

        # Group consecutive missing dates into ranges
        missing_ranges = []
        if len(missing_dates) > 0:
            missing_dates = sorted(missing_dates)
            range_start = missing_dates[0]
            prev_date = missing_dates[0]

            for date in missing_dates[1:]:
                if (date - prev_date).days > 1:
                    # End of a missing range
                    missing_ranges.append(
                        (
                            range_start.strftime("%Y-%m-%d"),
                            prev_date.strftime("%Y-%m-%d"),
                        )
                    )
                    range_start = date
                prev_date = date

            # Add the last range
            missing_ranges.append(
                (range_start.strftime("%Y-%m-%d"), prev_date.strftime("%Y-%m-%d"))
            )

        return False, cached_dates[0], cached_dates[-1], missing_ranges

    def get_cache(
        self,
        expr: str,
        market: str,
        start_date: str,
        end_date: str,
        label: str = "close_return",
    ) -> Optional[Dict]:
        """
        Retrieve cached results for given parameters.
        Can return subset of cached data if available.
        """
        expr_hash = self._hash_expression(expr)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Update access stats
        cursor.execute(
            """
            UPDATE factor_cache 
            SET last_accessed = CURRENT_TIMESTAMP, 
                access_count = access_count + 1
            WHERE expr_hash = ?
        """,
            (expr_hash,),
        )

        # Retrieve daily metrics for date range
        cursor.execute(
            """
            SELECT date, ic, rank_ic, icir, rank_icir, turnover
            FROM cache_entries
            WHERE expr_hash = ? AND market = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """,
            (expr_hash, market.lower(), start_date, end_date),
        )

        results = cursor.fetchall()
        conn.commit()
        conn.close()

        if not results:
            return None

        # Build result dictionary
        df = pd.DataFrame(
            results, columns=["date", "ic", "rank_ic", "icir", "rank_icir", "turnover"]
        )

        # Calculate summary statistics
        ic_values = df["ic"].dropna()
        rank_ic_values = df["rank_ic"].dropna()

        metrics = {
            "ic": float(ic_values.mean()) if len(ic_values) > 0 else 0.0,
            "rank_ic": float(rank_ic_values.mean()) if len(rank_ic_values) > 0 else 0.0,
            "icir": (
                float(ic_values.mean() / ic_values.std())
                if len(ic_values) > 1 and ic_values.std() > 0
                else 0.0
            ),
            "rank_icir": (
                float(rank_ic_values.mean() / rank_ic_values.std())
                if len(rank_ic_values) > 1 and rank_ic_values.std() > 0
                else 0.0
            ),
            "turnover": float(df["turnover"].mean()) if "turnover" in df else 0.0,
            "n_dates": len(results),
        }

        # Load additional data from pickle if needed
        data_file = self._get_data_file_path(expr_hash)
        daily_data = {}
        if data_file.exists():
            try:
                with open(data_file, "rb") as f:
                    stored_data = pickle.load(f)
                    daily_data = stored_data.get("daily_data", {})
            except Exception as e:
                logger.warning(f"Failed to load pickle data: {e}")

        return {
            "success": True,
            "expression": expr,
            "market": market,
            "start_date": start_date,
            "end_date": end_date,
            "metrics": metrics,
            "daily_metrics": df.to_dict("records"),
            "daily_data": daily_data,
            "from_cache": True,
            "timestamp": datetime.now().isoformat(),
        }

    def set_cache(
        self,
        expr: str,
        market: str,
        start_date: str,
        end_date: str,
        result: Dict,
        daily_ic: Optional[pd.Series] = None,
        daily_rankic: Optional[pd.Series] = None,
    ):
        """
        Store evaluation results in cache.

        Args:
            expr: Factor expression
            market: Market identifier
            start_date: Start date
            end_date: End date
            result: Evaluation result dictionary
            daily_ic: Optional daily IC values
            daily_rankic: Optional daily RankIC values
        """
        expr_hash = self._hash_expression(expr)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Insert or update factor metadata
        cursor.execute(
            """
            INSERT OR REPLACE INTO factor_cache 
            (expr_hash, expression, data_file)
            VALUES (?, ?, ?)
        """,
            (expr_hash, expr, f"{expr_hash}.pkl"),
        )

        # Store daily metrics if provided
        if daily_ic is not None and daily_rankic is not None:
            # Ensure we have datetime index
            if isinstance(daily_ic.index, pd.MultiIndex):
                dates = daily_ic.index.get_level_values("datetime").unique()
            else:
                dates = daily_ic.index

            for date in dates:
                date_str = (
                    date.strftime("%Y-%m-%d")
                    if hasattr(date, "strftime")
                    else str(date)
                )

                ic_val = (
                    float(daily_ic.get(date, np.nan)) if daily_ic is not None else None
                )
                rankic_val = (
                    float(daily_rankic.get(date, np.nan))
                    if daily_rankic is not None
                    else None
                )

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (expr_hash, market, date, ic, rank_ic, icir, rank_icir, turnover)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        expr_hash,
                        market.lower(),
                        date_str,
                        ic_val,
                        rankic_val,
                        result["metrics"].get("icir"),
                        result["metrics"].get("rank_icir"),
                        result["metrics"].get("turnover", 0.0),
                    ),
                )

        conn.commit()
        conn.close()

        # Store full data in pickle
        data_file = self._get_data_file_path(expr_hash)
        pickle_data = {
            "result": result,
            "daily_ic": daily_ic,
            "daily_rankic": daily_rankic,
            "market": market,
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": datetime.now().isoformat(),
        }

        with open(data_file, "wb") as f:
            pickle.dump(pickle_data, f)

    def get_cached_expressions(
        self, limit: int = 100, order_by: str = "last_accessed"
    ) -> List[Dict]:
        """
        Get list of cached expressions with metadata.

        Args:
            limit: Maximum number of results
            order_by: Sort field (last_accessed, access_count, created_at)

        Returns:
            List of expression metadata dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        valid_orders = ["last_accessed", "access_count", "created_at"]
        if order_by not in valid_orders:
            order_by = "last_accessed"

        # Also get aggregated metrics for each expression
        cursor.execute(
            f"""
            SELECT 
                fc.expr_hash, 
                fc.expression, 
                fc.created_at, 
                fc.last_accessed, 
                fc.access_count,
                COUNT(DISTINCT ce.market) as market_count,
                COUNT(ce.id) as data_points
            FROM factor_cache fc
            LEFT JOIN cache_entries ce ON fc.expr_hash = ce.expr_hash
            GROUP BY fc.expr_hash, fc.expression, fc.created_at, fc.last_accessed, fc.access_count
            ORDER BY fc.{order_by} DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "expr_hash": row[0],
                "expression": row[1],
                "created_at": row[2],
                "last_accessed": row[3],
                "access_count": row[4],
                "market_count": row[5],
                "data_points": row[6],
            }
            for row in results
        ]

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM factor_cache")
        num_expressions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        num_entries = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(access_count) FROM factor_cache")
        total_accesses = cursor.fetchone()[0] or 0

        # Get additional stats
        cursor.execute("SELECT COUNT(DISTINCT market) FROM cache_entries")
        num_markets = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT MIN(date), MAX(date) 
            FROM cache_entries 
            WHERE date IS NOT NULL
        """
        )
        date_range = cursor.fetchone()
        min_date = date_range[0] if date_range and date_range[0] else None
        max_date = date_range[1] if date_range and date_range[1] else None

        conn.close()

        # Calculate disk usage
        total_size = (
            sum(f.stat().st_size for f in self.data_dir.glob("*.pkl"))
            if self.data_dir.exists()
            else 0
        )
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Calculate average data points per expression
        avg_data_points = num_entries / num_expressions if num_expressions > 0 else 0

        return {
            "num_expressions": num_expressions,
            "num_entries": num_entries,
            "total_accesses": total_accesses,
            "data_size_mb": total_size / (1024 * 1024),
            "db_size_mb": db_size / (1024 * 1024),
            "total_size_mb": (total_size + db_size) / (1024 * 1024),
            "num_markets": num_markets,
            "avg_data_points_per_expr": avg_data_points,
            "date_range": {"min": min_date, "max": max_date},
        }

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries.

        Args:
            older_than_days: Only clear entries older than this many days
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if older_than_days:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cursor.execute(
                """
                SELECT expr_hash, data_file FROM factor_cache
                WHERE last_accessed < ?
            """,
                (cutoff_date.isoformat(),),
            )

            to_delete = cursor.fetchall()

            for expr_hash, data_file in to_delete:
                # Delete pickle file
                file_path = self.data_dir / data_file
                if file_path.exists():
                    file_path.unlink()

                # Delete from database
                cursor.execute(
                    "DELETE FROM cache_entries WHERE expr_hash = ?", (expr_hash,)
                )
                cursor.execute(
                    "DELETE FROM factor_cache WHERE expr_hash = ?", (expr_hash,)
                )
        else:
            # Clear everything
            cursor.execute("DELETE FROM cache_entries")
            cursor.execute("DELETE FROM factor_cache")

            # Delete all pickle files
            for file in self.data_dir.glob("*.pkl"):
                file.unlink()

        conn.commit()
        conn.close()

        logger.info("Cache cleared successfully")

    def merge_cache_results(
        self,
        cached_result: Dict,
        new_result: Dict,
        full_start_date: str,
        full_end_date: str,
    ) -> Dict:
        """
        Merge cached results with newly computed results.

        Args:
            cached_result: Result from cache
            new_result: Newly computed result
            full_start_date: Full requested start date
            full_end_date: Full requested end date

        Returns:
            Merged result dictionary
        """
        # Combine daily metrics
        cached_daily = pd.DataFrame(cached_result.get("daily_metrics", []))
        new_daily = pd.DataFrame(new_result.get("daily_metrics", []))

        if not cached_daily.empty and not new_daily.empty:
            # Combine and sort by date
            all_daily = pd.concat([cached_daily, new_daily], ignore_index=True)
            all_daily["date"] = pd.to_datetime(all_daily["date"])
            all_daily = all_daily.drop_duplicates(subset=["date"]).sort_values("date")

            # Filter to requested date range
            mask = (all_daily["date"] >= full_start_date) & (
                all_daily["date"] <= full_end_date
            )
            all_daily = all_daily[mask]

            # Recalculate summary metrics
            ic_values = all_daily["ic"].dropna()
            rank_ic_values = all_daily["rank_ic"].dropna()

            metrics = {
                "ic": float(ic_values.mean()) if len(ic_values) > 0 else 0.0,
                "rank_ic": (
                    float(rank_ic_values.mean()) if len(rank_ic_values) > 0 else 0.0
                ),
                "icir": (
                    float(ic_values.mean() / ic_values.std())
                    if len(ic_values) > 1 and ic_values.std() > 0
                    else 0.0
                ),
                "rank_icir": (
                    float(rank_ic_values.mean() / rank_ic_values.std())
                    if len(rank_ic_values) > 1 and rank_ic_values.std() > 0
                    else 0.0
                ),
                "turnover": (
                    float(all_daily["turnover"].mean())
                    if "turnover" in all_daily
                    else 0.0
                ),
                "n_dates": len(all_daily),
            }

            # Convert dates back to string for JSON serialization
            all_daily["date"] = all_daily["date"].dt.strftime("%Y-%m-%d")

            return {
                "success": True,
                "expression": cached_result.get(
                    "expression", new_result.get("expression")
                ),
                "market": cached_result.get("market", new_result.get("market")),
                "start_date": full_start_date,
                "end_date": full_end_date,
                "metrics": metrics,
                "daily_metrics": all_daily.to_dict("records"),
                "from_cache": "partial",
                "timestamp": datetime.now().isoformat(),
            }
        elif not new_daily.empty:
            return new_result
        else:
            return cached_result

    def get_factor_history(self, expr: str, market: Optional[str] = None) -> List[Dict]:
        """
        Get complete evaluation history for a factor expression.

        Args:
            expr: Factor expression
            market: Optional market filter

        Returns:
            List of evaluation records with dates and metrics
        """
        expr_hash = self._hash_expression(expr)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if market:
            cursor.execute(
                """
                SELECT market, date, ic, rank_ic, icir, rank_icir, turnover
                FROM cache_entries
                WHERE expr_hash = ?  AND market = ?
                ORDER BY market, date
            """,
                (expr_hash, market.lower()),
            )
        else:
            cursor.execute(
                """
                SELECT market, date, ic, rank_ic, icir, rank_icir, turnover
                FROM cache_entries
                WHERE expr_hash = ?
                ORDER BY market, date
            """,
                (expr_hash,),
            )

        results = cursor.fetchall()
        conn.close()

        # Group by market
        history = {}
        for row in results:
            market_name = row[0]
            if market_name not in history:
                history[market_name] = []

            history[market_name].append(
                {
                    "date": row[1],
                    "ic": row[2],
                    "rank_ic": row[3],
                    "icir": row[4],
                    "rank_icir": row[5],
                    "turnover": row[6],
                }
            )

        return history

    def get_latest_evaluation(
        self, expr: str, market: str = "csi300"
    ) -> Optional[Dict]:
        """
        Get the most recent evaluation for an expression.

        Args:
            expr: Factor expression
            market: Market (default: csi300)

        Returns:
            Most recent evaluation data or None
        """
        expr_hash = self._hash_expression(expr)

        # Get metrics from database for this expression and market
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get aggregated metrics for the expression
        cursor.execute(
            """
            SELECT 
                AVG(ic) as avg_ic,
                AVG(rank_ic) as avg_rank_ic,
                AVG(icir) as avg_icir,
                AVG(rank_icir) as avg_rank_icir,
                AVG(turnover) as avg_turnover,
                COUNT(*) as n_dates,
                MAX(date) as last_date,
                MIN(date) as first_date
            FROM cache_entries
            WHERE expr_hash = ? AND market = ?
        """,
            (expr_hash, market.lower()),
        )

        result = cursor.fetchone()

        if not result or result[0] is None:
            conn.close()
            return None

        # Extract values
        ic_mean = float(result[0]) if result[0] is not None else 0.0
        rank_ic_mean = float(result[1]) if result[1] is not None else 0.0

        # Use ICIR from database if available, otherwise use a default
        icir = float(result[2]) if result[2] is not None else 0.0
        rank_icir = float(result[3]) if result[3] is not None else 0.0

        # Build metrics dictionary
        metrics = {
            "ic": ic_mean,
            "rank_ic": rank_ic_mean,
            "icir": icir,
            "rank_icir": rank_icir,
            "turnover": float(result[4]) if result[4] is not None else 0.0,
            "n_dates": result[5],
            "last_date": result[6],
            "first_date": result[7],
        }

        conn.close()

        return {"success": True, "expr": expr, "market": market, "metrics": metrics}

    def add_to_query_history(
        self,
        expression: str,
        market: str,
        start_date: str,
        end_date: str,
        ic: float,
        rank_ic: float,
        icir: Optional[float] = None,
        rank_icir: Optional[float] = None,
    ):
        """
        Add a factor evaluation to query history.

        Args:
            expression: Factor expression
            market: Market identifier
            start_date: Start date
            end_date: End date
            ic: IC value
            rank_ic: Rank IC value
            icir: Optional ICIR value
            rank_icir: Optional Rank ICIR value
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO query_history 
            (expression, market, start_date, end_date, ic, rank_ic, icir, rank_icir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                expression,
                market.lower(),
                start_date,
                end_date,
                ic,
                rank_ic,
                icir,
                rank_icir,
            ),
        )

        conn.commit()
        conn.close()

    def get_query_history(
        self, limit: int = 20, search_query: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent query history with optional search filter.

        Args:
            limit: Maximum number of records to return
            search_query: Optional search string to filter expressions

        Returns:
            List of query history records
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if search_query:
            cursor.execute(
                """
                SELECT id, expression, market, start_date, end_date, 
                       ic, rank_ic, icir, rank_icir, query_timestamp
                FROM query_history
                WHERE expression LIKE ?
                ORDER BY query_timestamp DESC
                LIMIT ?
            """,
                (f"%{search_query}%", limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, expression, market, start_date, end_date, 
                       ic, rank_ic, icir, rank_icir, query_timestamp
                FROM query_history
                ORDER BY query_timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "expression": row[1],
                "market": row[2],
                "start_date": row[3],
                "end_date": row[4],
                "ic": row[5],
                "rank_ic": row[6],
                "icir": row[7],
                "rank_icir": row[8],
                "query_timestamp": row[9],
            }
            for row in results
        ]

    def get_top_factors_from_history(
        self, limit: int = 5, metric: str = "rank_ic"
    ) -> dict:
        """
        Get top performing factors from query history.

        Args:
            limit: Maximum number of top factors to return
            metric: Metric to sort by ('ic', 'rank_ic', 'icir', 'rank_icir')

        Returns:
            Dictionary with top_factors list, total_factors count, and showing_count
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Validate metric
        valid_metrics = ["ic", "rank_ic", "icir", "rank_icir"]
        if metric not in valid_metrics:
            metric = "rank_ic"

        # Get unique factors with their best performance for the selected metric
        # Group by expression and take the record with the best metric value
        query = f"""
            WITH RankedFactors AS (
                SELECT 
                    expression,
                    market,
                    ic,
                    rank_ic,
                    icir,
                    rank_icir,
                    query_timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY expression 
                        ORDER BY {metric} DESC, query_timestamp DESC
                    ) as rn
                FROM query_history
                WHERE {metric} IS NOT NULL
            )
            SELECT 
                expression,
                market,
                ic,
                rank_ic,
                icir,
                rank_icir,
                query_timestamp,
                (SELECT COUNT(*) FROM query_history qh2 WHERE qh2.expression = RankedFactors.expression) as evaluation_count
            FROM RankedFactors
            WHERE rn = 1
            ORDER BY {metric} DESC
            LIMIT ?
        """

        cursor.execute(query, (limit * 2,))  # Get more than needed to filter
        results = cursor.fetchall()

        # Get total unique factors
        cursor.execute("SELECT COUNT(DISTINCT expression) FROM query_history")
        total_factors = cursor.fetchone()[0]

        conn.close()

        # Format results
        top_factors = []
        for row in results[:limit]:
            top_factors.append(
                {
                    "expression": row[0],
                    "market": row[1],
                    "metrics": {
                        "ic": float(row[2]) if row[2] is not None else 0.0,
                        "rank_ic": float(row[3]) if row[3] is not None else 0.0,
                        "icir": float(row[4]) if row[4] is not None else None,
                        "rank_icir": float(row[5]) if row[5] is not None else None,
                    },
                    "last_evaluated": row[6],
                    "evaluation_count": row[7],
                }
            )

        return {
            "success": True,
            "top_factors": top_factors,
            "total_factors": total_factors,
            "showing_count": len(top_factors),
        }
