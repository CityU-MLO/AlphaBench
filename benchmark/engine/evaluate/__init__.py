"""
AlphaBench Evaluate Engine

Covers both T2 (ranking/scoring) and T4 (atomic: binary_noise, pairwise_select).
"""

# T2: Factor evaluation (ranking and scoring)
from .benchmark_main import (
    benchmark_ranking_performance,
    benchmark_scoring_performance,
    run_atomic_benchmark,
    run_t4_from_config,
)
from .factor_eval import (
    evaluate_performance_ranking,
    evaluate_performance_scoring,
    ndcg_at_k,
    precision,
)
from .report import generate_report, compare_runs

# T4: Atomic evaluation (binary noise & pairwise select)
from .atomic_prompts import (
    build_noise_system_prompt,
    build_noise_user_prompt,
    build_pairwise_system_prompt,
    build_pairwise_user_prompt,
)
from .atomic_metrics import (
    compute_binary_metrics,
    compute_pairwise_metrics,
    format_report as format_atomic_report,
    normalize_noise_label,
    normalize_ab_label,
)
from .atomic_infer import run_atomic_infer

__all__ = [
    # T2
    "benchmark_ranking_performance",
    "benchmark_scoring_performance",
    "evaluate_performance_ranking",
    "evaluate_performance_scoring",
    "ndcg_at_k",
    "precision",
    "generate_report",
    "compare_runs",
    # T4
    "run_atomic_benchmark",
    "run_t4_from_config",
    "run_atomic_infer",
    "build_noise_system_prompt",
    "build_noise_user_prompt",
    "build_pairwise_system_prompt",
    "build_pairwise_user_prompt",
    "compute_binary_metrics",
    "compute_pairwise_metrics",
    "format_atomic_report",
    "normalize_noise_label",
    "normalize_ab_label",
]
