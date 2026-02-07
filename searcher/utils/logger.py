"""
Logging utilities for AlphaBench search system.
Follows EvoAlpha's SearchLogger pattern.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class SearchLogger:
    """
    Specialized logger for search operations.
    """

    def __init__(self, name: str = "AlphaBench", log_file: Optional[str] = None):
        self.logger = setup_logger(name, log_file)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def log_round_start(self, round_num: int, total_rounds: int):
        self.logger.info("=" * 70)
        self.logger.info(f"Round {round_num}/{total_rounds}")
        self.logger.info("=" * 70)

    def log_round_end(self, round_num: int, stats: dict):
        self.logger.info("=" * 70)
        self.logger.info(f"Round {round_num} Complete")
        self.logger.info("=" * 70)

        self.logger.info(f"  Generated: {stats.get('generated', 0)}")
        self.logger.info(f"  Evaluated: {stats.get('evaluated', 0)}")
        self.logger.info(f"  Accepted: {stats.get('accepted', 0)}")
        self.logger.info(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        self.logger.info(f"  Time: {stats.get('elapsed_time', 0):.1f}s")

        if stats.get("accepted", 0) > 0:
            self.logger.info(f"  Best IC: {stats.get('best_ic', 0):.4f}")
            self.logger.info(f"  Best RankIC: {stats.get('best_rank_ic', 0):.4f}")
            self.logger.info(f"  Best ICIR: {stats.get('best_icir', 0):.4f}")

        pool_stats = stats.get("pool_stats", {})
        if pool_stats and pool_stats.get("total_factors", 0) > 0:
            self.logger.info(f"  Factor Pool:")
            self.logger.info(f"    Total: {pool_stats.get('total_factors', 0)}")
            self.logger.info(f"    Top 10% IC: {pool_stats.get('top_10pct_avg_ic', 0):.4f}")
            self.logger.info(f"    Top 10% RankIC: {pool_stats.get('top_10pct_avg_rank_ic', 0):.4f}")
            self.logger.info(f"    Overall IC: {pool_stats.get('overall_avg_ic', 0):.4f}")

        self.logger.info("=" * 70 + "\n")

    def log_generation(self, operation: str, num_factors: int, success: bool):
        status = "OK" if success else "FAIL"
        self.logger.info(f"[{status}] Generated {num_factors} factors via {operation}")

    def log_evaluation(self, num_factors: int, num_success: int, elapsed_time: float):
        self.logger.info(
            f"Evaluated {num_factors} factors in {elapsed_time:.1f}s "
            f"({num_success} successful)"
        )

    def log_search_start(self, task_id: str, config: dict):
        self.logger.info("=" * 70)
        self.logger.info(f"Starting search: {task_id}")
        self.logger.info("=" * 70)

    def log_search_end(self, task_id: str, stats: dict):
        self.logger.info("=" * 70)
        self.logger.info(f"Search complete: {task_id}")
        self.logger.info("=" * 70)
        self.logger.info(f"  Total generated: {stats.get('total_generated', 0)}")
        self.logger.info(f"  Total accepted: {stats.get('total_accepted', 0)}")
        self.logger.info(f"  Best IC: {stats.get('best_ic', 0):.4f}")
        self.logger.info(f"  Best RankIC: {stats.get('best_rank_ic', 0):.4f}")
        self.logger.info(f"  Pool size: {stats.get('pool_size', 0)}")
        self.logger.info(f"  Total time: {stats.get('total_time', 0):.1f}s")
