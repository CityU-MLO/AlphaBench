"""Base evaluation dataset builders for T2 (Ranking & Scoring)."""

from .config import BaseEvalConfig, RegimeConfig, RankingConfig, ScoringConfig
from .ranking_builder import build_ranking_dataset
from .scoring_builder import build_scoring_dataset
from .stats import (
    compute_full_stats,
    classify_signal,
    assign_scores,
    load_table,
)

__all__ = [
    "BaseEvalConfig",
    "RegimeConfig",
    "RankingConfig",
    "ScoringConfig",
    "build_ranking_dataset",
    "build_scoring_dataset",
    "compute_full_stats",
    "classify_signal",
    "assign_scores",
    "load_table",
]
