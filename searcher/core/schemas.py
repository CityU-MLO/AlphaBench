"""
Data schemas for AlphaBench factor search system.
Follows EvoAlpha's architecture, simplified for open-source use (no database dependency).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FactorCandidate:
    """
    In-memory representation of a factor candidate.

    Attributes:
        name: Unique factor identifier
        expression: Qlib expression string
        reason: Explanation for why this factor was generated
        doc_type: "origin" or "search"
        meta: Metadata about generation (type, parents, etc.)
        provenance: Generation context (agent_id, round, timestamp)
    """
    name: str
    expression: str
    reason: str = ""
    doc_type: str = "search"
    meta: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "expression": self.expression,
            "reason": self.reason,
            "type": self.doc_type,
            "meta": self.meta,
            "provenance": self.provenance,
        }


@dataclass
class EvaluatedFactor:
    """
    Factor with evaluation metrics.

    Attributes:
        candidate: Original factor candidate
        metrics: IC-based metrics (always present)
        portfolio_metrics: Portfolio backtest metrics (optional)
        backtest_mode: "fast" or "full"
        success: Whether evaluation succeeded
        error: Error message if evaluation failed
    """
    candidate: FactorCandidate
    metrics: Dict[str, float] = field(default_factory=dict)
    portfolio_metrics: Optional[Dict[str, Any]] = None
    backtest_mode: str = "fast"
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = self.candidate.to_dict()
        result["metrics"] = self.metrics
        result["portfolio_metrics"] = self.portfolio_metrics
        result["backtest_mode"] = self.backtest_mode
        result["success"] = self.success
        result["error"] = self.error
        return result

    def get_ic(self) -> float:
        """Get IC value, returns 0 if missing."""
        return self.metrics.get("ic", 0.0)

    def get_rank_ic(self) -> float:
        """Get Rank IC value, returns 0 if missing."""
        return self.metrics.get("rank_ic", 0.0)

    def get_icir(self) -> float:
        """Get ICIR value, returns 0 if missing."""
        return self.metrics.get("icir", 0.0)

    def passes_threshold(self, min_ic: float = 0.02, min_rank_ic: float = 0.02) -> bool:
        """Check if factor passes minimum thresholds."""
        return abs(self.get_ic()) >= min_ic and abs(self.get_rank_ic()) >= min_rank_ic


@dataclass
class SearchRoundResult:
    """
    Results from a single search round.

    Attributes:
        round_num: Round number
        candidates: Generated candidates
        evaluated: Evaluated factors
        accepted: Factors that passed thresholds
        rejected: Factors that didn't pass
        elapsed_time: Time taken for the round
    """
    round_num: int
    candidates: List[FactorCandidate]
    evaluated: List[EvaluatedFactor]
    accepted: List[EvaluatedFactor]
    rejected: List[EvaluatedFactor]
    elapsed_time: float

    def get_best_factor(self) -> Optional[EvaluatedFactor]:
        """Get the best factor from this round."""
        if not self.accepted:
            return None
        return max(self.accepted, key=lambda f: abs(f.get_ic()))

    def get_summary(self) -> Dict[str, Any]:
        """Get round summary statistics."""
        summary = {
            "round": self.round_num,
            "generated": len(self.candidates),
            "evaluated": len(self.evaluated),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "success_rate": len(self.accepted) / len(self.evaluated) if self.evaluated else 0,
            "elapsed_time": self.elapsed_time,
        }

        if self.accepted:
            ics = [abs(f.get_ic()) for f in self.accepted]
            rank_ics = [abs(f.get_rank_ic()) for f in self.accepted]
            icirs = [abs(f.get_icir()) for f in self.accepted]

            summary.update({
                "best_ic": max(ics) if ics else 0,
                "best_rank_ic": max(rank_ics) if rank_ics else 0,
                "best_icir": max(icirs) if icirs else 0,
                "avg_ic": sum(ics) / len(ics) if ics else 0,
                "avg_rank_ic": sum(rank_ics) / len(rank_ics) if rank_ics else 0,
                "avg_icir": sum(icirs) / len(icirs) if icirs else 0,
            })
        else:
            summary.update({
                "best_ic": 0, "best_rank_ic": 0, "best_icir": 0,
                "avg_ic": 0, "avg_rank_ic": 0, "avg_icir": 0,
            })

        return summary


@dataclass
class SeedFactor:
    """
    Seed factor for generation.

    Attributes:
        name: Factor name
        expression: Factor expression
        metrics: Performance metrics
        weight: Sampling weight
    """
    name: str
    expression: str
    metrics: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedFactor":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            expression=data.get("expression", ""),
            metrics=data.get("metrics", {}),
            weight=data.get("weight", 1.0),
        )

    def get_ic(self) -> float:
        """Get IC value."""
        return self.metrics.get("ic", 0.0)


@dataclass
class GenerationRequest:
    """
    Request to generate new factors.

    Attributes:
        operation: "mutation" or "crossover"
        seeds: Seed factors to use
        num_factors: Number of factors to generate
        temperature: Sampling temperature for LLM
        context: Additional context for generation
    """
    operation: str  # "mutation" or "crossover"
    seeds: List[SeedFactor]
    num_factors: int = 1
    temperature: float = 0.7
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """
    Result of factor generation.

    Attributes:
        request: Original request
        candidates: Generated candidates
        success: Whether generation succeeded
        error: Error message if failed
        attempts: Number of LLM calls made
        elapsed_time: Time taken
    """
    request: GenerationRequest
    candidates: List[FactorCandidate]
    success: bool = True
    error: str = ""
    attempts: int = 1
    elapsed_time: float = 0.0
