"""
Logging utilities for AlphaBench search system.

Console output: plain messages only (no timestamps).
File output:    full timestamps + log levels.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure a logger with clean console output and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    return logger


class SearchLogger:
    """
    Logger for search operations with structured progress output.

    Console shows clean messages with no timestamps.
    Log files (if configured) include full timestamps.
    """

    def __init__(self, name: str = "AlphaBench", log_file: Optional[str] = None):
        self.logger = setup_logger(name, log_file)

    # ── Raw log methods ──────────────────────────────────────────────── #

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(f"[WARN] {msg}")

    def error(self, msg: str):
        self.logger.error(f"[ERROR] {msg}")

    def debug(self, msg: str):
        self.logger.debug(msg)

    # ── Structured progress helpers ───────────────────────────────────── #

    def section(self, title: str):
        """Print a major section header (double rule)."""
        self.logger.info("")
        self.logger.info("═" * 60)
        self.logger.info(f"  {title}")
        self.logger.info("═" * 60)

    def round_header(self, round_num: int, total_rounds: int, algo: str = ""):
        """Print a round divider with round number."""
        tag = f" · {algo}" if algo else ""
        self.logger.info("")
        self.logger.info("─" * 60)
        self.logger.info(f"  Round {round_num}/{total_rounds}{tag}")
        self.logger.info("─" * 60)

    def factor_table(
        self,
        factors: List[Dict[str, Any]],
        title: str = "New candidates",
        max_show: int = 10,
    ):
        """Print a compact table of factors with IC / RankIC / ICIR."""
        if not factors:
            return
        self.logger.info(f"\n  {title} ({len(factors)}):")
        self.logger.info(
            f"    {'Name':<30} {'IC':>8}  {'RankIC':>8}  {'ICIR':>8}  Tag"
        )
        self.logger.info(f"    {'─'*30} {'─'*8}  {'─'*8}  {'─'*8}  ────────")
        for f in factors[:max_show]:
            m    = f.get("metrics") or {}
            ic   = m.get("ic",      float("nan"))
            ric  = m.get("rank_ic", float("nan"))
            icir = m.get("icir",    float("nan"))
            tag  = (f.get("provenance") or "")[:8]
            name = (f.get("name") or "")[:30]
            self.logger.info(
                f"    {name:<30} {self._fmt(ic):>8}  {self._fmt(ric):>8}"
                f"  {self._fmt(icir):>8}  {tag}"
            )
        if len(factors) > max_show:
            self.logger.info(f"    … {len(factors) - max_show} more")
        self.logger.info("")

    def pool_status(self, pool: List[Dict[str, Any]], label: str = "Pool"):
        """Print a one-line pool summary (RankIC first)."""
        if not pool:
            self.logger.info(f"  {label}: empty")
            return
        ics  = [f["metrics"].get("ic",      0.0) for f in pool if f.get("metrics")]
        rics = [f["metrics"].get("rank_ic", 0.0) for f in pool if f.get("metrics")]
        top_ric  = max(rics) if rics else 0.0
        mean_ric = sum(rics) / len(rics) if rics else 0.0
        top_ic   = max(ics)  if ics  else 0.0
        self.logger.info(
            f"  {label} [{len(pool)}]"
            f"  top RankIC={top_ric:.4f}  mean RankIC={mean_ric:.4f}  top IC={top_ic:.4f}"
        )

    def mining_summary(
        self,
        factors: List[Dict[str, Any]],
        title: str = "Mining Summary — Top Factors",
        max_show: int = 15,
    ):
        """Print a rich summary of the best discovered factors (with expressions)."""
        if not factors:
            self.logger.info("  No factors to summarize.")
            return
        ranked = sorted(
            factors,
            key=lambda f: f.get("metrics", {}).get("rank_ic", float("-inf")),
            reverse=True,
        )
        self.section(title)
        self.logger.info(
            f"  {'#':>3}  {'Name':<26}  {'RankIC':>8}  {'IC':>8}  {'ICIR':>7}  Expression"
        )
        self.logger.info(f"  {'─'*3}  {'─'*26}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*45}")
        for rank, f in enumerate(ranked[:max_show], 1):
            m    = f.get("metrics") or {}
            ric  = m.get("rank_ic", float("nan"))
            ic   = m.get("ic",      float("nan"))
            icir = m.get("icir",    float("nan"))
            name = (f.get("name") or "")[:26]
            expr = (f.get("expression") or "")[:45]
            self.logger.info(
                f"  {rank:>3}  {name:<26}  {self._fmt(ric):>8}  {self._fmt(ic):>8}"
                f"  {self._fmt(icir):>7}  {expr}"
            )
        if len(ranked) > max_show:
            self.logger.info(f"  … {len(ranked) - max_show} more — see final_pool.jsonl")
        self.logger.info("")

    # ── Internal ─────────────────────────────────────────────────────── #

    @staticmethod
    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "     nan"
        return f"{float(v):.4f}"
