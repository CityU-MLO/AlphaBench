"""
Search controller for AlphaBench factor search.

Follows EvoAlpha's SearchController architecture but simplified:
- No MongoDB dependency → uses in-memory FactorPool with JSONL persistence
- No task management → single search run
- Same core loop: sample seeds → generate (mutation/crossover) → evaluate → filter → store
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    EvaluatedFactor,
    FactorCandidate,
    GenerationRequest,
    SearchRoundResult,
    SeedFactor,
)


class FactorPool:
    """
    In-memory factor pool with JSONL file persistence.
    Replaces MongoDB SearchDatabase from EvoAlpha with zero external dependencies.
    """

    def __init__(self, save_dir: str = "./results"):
        self.factors: List[Dict[str, Any]] = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._expressions: set = set()  # for fast dedup

    def add_seeds(self, factors: List[Dict[str, Any]]) -> int:
        """
        Add seed factors to the pool.

        Args:
            factors: List of factor dicts with at least 'name' and 'expression'

        Returns:
            Number of new factors added (skips duplicates by expression)
        """
        added = 0
        for f in factors:
            expr = f.get("expression", "")
            if expr and expr not in self._expressions:
                self.factors.append(f)
                self._expressions.add(expr)
                added += 1
        return added

    def store_results(self, results: List[Dict[str, Any]]) -> None:
        """Store evaluated factor results into the pool."""
        for r in results:
            expr = r.get("expression", "")
            if expr and expr not in self._expressions:
                self.factors.append(r)
                self._expressions.add(expr)

    def get_seeds(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get top factors by |IC| as seeds for next round.

        Args:
            limit: Maximum number of seeds to return

        Returns:
            Top factors sorted by |IC| descending
        """
        valid = [f for f in self.factors if f.get("metrics")]
        sorted_factors = sorted(
            valid,
            key=lambda x: abs(x.get("metrics", {}).get("ic", 0)),
            reverse=True,
        )
        return sorted_factors[:limit]

    def get_top_factors(self, limit: int = 100000) -> List[Dict[str, Any]]:
        """Get all factors with metrics, sorted by |IC| descending."""
        valid = [f for f in self.factors if f.get("metrics")]
        return sorted(
            valid,
            key=lambda x: abs(x.get("metrics", {}).get("ic", 0)),
            reverse=True,
        )[:limit]

    def get_all_expressions(self) -> set:
        """Return set of all expressions in the pool."""
        return set(self._expressions)

    @property
    def size(self) -> int:
        return len(self.factors)

    def save(self, filename: str = "factor_pool.jsonl") -> Path:
        """Save entire pool to JSONL file."""
        fpath = self.save_dir / filename
        with fpath.open("w", encoding="utf-8") as f:
            for factor in self.factors:
                f.write(json.dumps(factor, default=str, ensure_ascii=False) + "\n")
        return fpath

    def load(self, filepath: str) -> int:
        """Load factors from JSONL file into pool."""
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                factor = json.loads(line)
                expr = factor.get("expression", "")
                if expr and expr not in self._expressions:
                    self.factors.append(factor)
                    self._expressions.add(expr)
                    loaded += 1
        return loaded


# ---------------------------------------------------------------------------
# Sampling helper (inlined from EvoAlpha utils/sampler.py for simplicity)
# ---------------------------------------------------------------------------

def _sample_seeds_ic_weighted(
    factors: List[Dict[str, Any]],
    num_samples: int,
) -> List[SeedFactor]:
    """IC-weighted sampling without numpy dependency."""
    import random

    if not factors:
        return []
    num_samples = min(num_samples, len(factors))

    weights = []
    for f in factors:
        ic = abs(f.get("metrics", {}).get("ic", 0))
        weights.append(max(ic, 0.001))

    total = sum(weights)
    weights = [w / total for w in weights]

    chosen_indices = []
    remaining_indices = list(range(len(factors)))
    remaining_weights = list(weights)

    for _ in range(num_samples):
        if not remaining_indices:
            break
        total_w = sum(remaining_weights)
        if total_w <= 0:
            idx_pos = random.randint(0, len(remaining_indices) - 1)
        else:
            r = random.random() * total_w
            cum = 0
            idx_pos = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if cum >= r:
                    idx_pos = i
                    break

        chosen_indices.append(remaining_indices[idx_pos])
        remaining_indices.pop(idx_pos)
        remaining_weights.pop(idx_pos)

    return [SeedFactor.from_dict(factors[i]) for i in chosen_indices]


class SearchController:
    """
    Main controller for factor search process (AlphaBench simplified version).

    Orchestrates:
    - Seed sampling from in-memory pool
    - Parallel factor generation (mutation + crossover)
    - Parallel factor evaluation via FFO
    - Result storage in pool + file
    - Statistics tracking

    Key differences from EvoAlpha's SearchController:
    - Uses FactorPool (in-memory + JSONL) instead of MongoDB
    - No task management system
    - Simpler configuration
    """

    def __init__(
        self,
        pool: FactorPool,
        breeder,  # FactorBreeder
        evaluator,  # FactorEvaluator
        config: Dict[str, Any],
        logger=None,
    ):
        """
        Initialize controller.

        Args:
            pool: In-memory factor pool
            breeder: Factor breeder agent (LLM-based generation)
            evaluator: Factor evaluator agent (FFO-based backtesting)
            config: Search configuration dict
            logger: Optional logger
        """
        self.pool = pool
        self.breeder = breeder
        self.evaluator = evaluator
        self.config = config

        # Use a simple print-based logger if none provided
        if logger is None:
            from ..utils.logger import SearchLogger
            logger = SearchLogger("AlphaBench")
        self.logger = logger

        # Parse config
        self.num_rounds = config.get("num_rounds", 10)
        self.mutation_rate = config.get("mutation_rate", 0.3)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.min_ic = config.get("min_ic", 0.02)
        self.min_rank_ic = config.get("min_rank_ic", 0.02)

        # Elite pool settings
        self.elite_size = config.get("seed_limit", 50)

        # Adaptive threshold parameters
        self.adaptive_threshold = config.get("adaptive_threshold", True)
        self.threshold_mode = config.get("threshold_mode", "or")
        self.adaptive_threshold_ratio = config.get("adaptive_threshold_ratio", 0.8)
        self.start_rounds = config.get("start_rounds", 2)
        self.diversity_rounds = config.get("diversity_rounds", 2)

        # Batch settings
        self.window_size = config.get("window_size", 5)
        self.factors_per_batch = config.get("factors_per_batch", 10)
        self.num_workers = config.get("num_workers", 3)
        self.batch_max_retries = config.get("batch_max_retries", 3)
        self.batch_failure_threshold = config.get("batch_failure_threshold", 0.7)

        self.factors_per_round = self.factors_per_batch * self.num_workers

        # State tracking
        self.round_results: List[SearchRoundResult] = []
        self.total_generated = 0
        self.total_accepted = 0
        self.best_ic = 0.0
        self.best_rank_ic = 0.0

    def run_search(self) -> Dict[str, Any]:
        """
        Run complete search process.

        Returns:
            Final search statistics
        """
        start_time = time.time()
        self.logger.log_search_start("alphabench", self.config)

        # Check FFO API health
        if not self.evaluator.check_api_health():
            raise RuntimeError("FFO API is not available")

        status = "running"

        try:
            for round_num in range(1, self.num_rounds + 1):
                self.logger.log_round_start(round_num, self.num_rounds)

                round_result = self._run_round(round_num)
                self.round_results.append(round_result)

                self._update_statistics(round_result)
                pool_stats = self._calculate_pool_statistics()

                round_summary = round_result.get_summary()
                round_summary["pool_stats"] = pool_stats

                # Save audit + summary after each round
                self._save_audit_file(round_num, round_result)
                self._save_text_summary(round_num, round_result, pool_stats)

                # Save pool checkpoint
                self.pool.save(f"pool_round_{round_num:03d}.jsonl")

                self.logger.log_round_end(round_num, round_summary)

            status = "completed"

        except KeyboardInterrupt:
            self.logger.info("Search interrupted by user")
            status = "interrupted"

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            status = "failed"
            raise

        finally:
            elapsed_time = time.time() - start_time

            final_stats = {
                "status": status,
                "total_generated": self.total_generated,
                "total_accepted": self.total_accepted,
                "total_rounds": len(self.round_results),
                "best_ic": self.best_ic,
                "best_rank_ic": self.best_rank_ic,
                "total_time": elapsed_time,
                "pool_size": self.pool.size,
                "success_rate": (
                    self.total_accepted / self.total_generated
                    if self.total_generated > 0
                    else 0
                ),
            }

            self.logger.log_search_end("alphabench", final_stats)

            # Save final pool
            final_path = self.pool.save("final_pool.jsonl")
            self.logger.info(f"Final pool saved to: {final_path}")

        return final_stats

    def _run_round(self, round_num: int) -> SearchRoundResult:
        """Execute a single search round."""
        start_time = time.time()

        # Step 1: Sample seeds from pool
        seeds = self.pool.get_seeds(limit=self.elite_size)
        self.logger.info(f"Sampled {len(seeds)} seeds from pool")

        if not seeds:
            raise ValueError("No seeds in pool — cannot start search round")

        # Step 2 & 3: Generate and evaluate candidates (batch mode)
        evaluated = self._generate_candidates(seeds, round_num)
        candidates = [ev.candidate for ev in evaluated]
        self.logger.info(f"Generated and evaluated {len(candidates)} candidates")

        # Step 4: Filter by thresholds
        accepted, rejected = self._filter_by_threshold(evaluated, round_num)
        self.logger.info(f"Accepted {len(accepted)}/{len(evaluated)} factors")

        # Step 5: Store ALL factors into pool
        all_factor_dicts = []
        for factor in accepted:
            d = factor.to_dict()
            d["status"] = "accepted"
            all_factor_dicts.append(d)

        for factor in rejected:
            d = factor.to_dict()
            d["status"] = "rejected"
            all_factor_dicts.append(d)

        # Store accepted factors into pool (rejected are saved to audit only)
        accepted_dicts = [f.to_dict() for f in accepted]
        self.pool.store_results(accepted_dicts)

        elapsed_time = time.time() - start_time

        return SearchRoundResult(
            round_num=round_num,
            candidates=candidates,
            evaluated=evaluated,
            accepted=accepted,
            rejected=rejected,
            elapsed_time=elapsed_time,
        )

    def _generate_candidates(
        self, seeds: List[Dict[str, Any]], round_num: int
    ) -> List[EvaluatedFactor]:
        """
        Batch generation: multiple workers in parallel.
        Each worker generates a batch then evaluates it.
        """
        num_mutation_workers = max(1, int(self.num_workers * self.mutation_rate))
        num_crossover_workers = self.num_workers - num_mutation_workers

        self.logger.info(
            f"Launching {self.num_workers} workers: "
            f"{num_mutation_workers} mutation + {num_crossover_workers} crossover"
        )

        worker_tasks = []
        for i in range(num_mutation_workers):
            worker_tasks.append({
                "operation": "mutation",
                "batch_id": f"mutation_{i}",
                "seeds": seeds,
                "round_num": round_num,
            })
        for i in range(num_crossover_workers):
            worker_tasks.append({
                "operation": "crossover",
                "batch_id": f"crossover_{i}",
                "seeds": seeds,
                "round_num": round_num,
            })

        all_evaluated = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._generate_and_evaluate_batch,
                    operation=task["operation"],
                    seeds=task["seeds"],
                    batch_id=task["batch_id"],
                    round_num=task["round_num"],
                ): task
                for task in worker_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    evaluated = future.result()
                    all_evaluated.extend(evaluated)
                    self.logger.info(
                        f"Worker {task['batch_id']} completed with {len(evaluated)} factors"
                    )
                except Exception as e:
                    self.logger.error(f"Worker {task['batch_id']} failed: {e}")

        self.logger.info(
            f"All {len(worker_tasks)} workers completed, total {len(all_evaluated)} factors"
        )
        return all_evaluated

    def _generate_and_evaluate_batch(
        self,
        operation: str,
        seeds: List[Dict[str, Any]],
        batch_id: str,
        round_num: int,
    ) -> List[EvaluatedFactor]:
        """Generate and evaluate a single batch with retry logic."""
        for retry_attempt in range(self.batch_max_retries):
            # Sample seeds for this batch
            batch_seeds = _sample_seeds_ic_weighted(seeds, self.window_size)

            # Create generation request
            req = GenerationRequest(
                operation=operation,
                seeds=batch_seeds,
                num_factors=self.factors_per_batch,
                temperature=self.config.get("temperature", 0.7),
                context={
                    "round_num": round_num,
                    "batch_id": batch_id,
                    "retry_attempt": retry_attempt,
                },
            )

            # Generate factors
            result = self.breeder.generate(req)

            if not result.success:
                self.logger.warning(
                    f"Batch {batch_id} generation failed "
                    f"(attempt {retry_attempt + 1}/{self.batch_max_retries}): {result.error}"
                )
                if retry_attempt < self.batch_max_retries - 1:
                    continue
                else:
                    return []

            candidates = result.candidates
            self.logger.info(f"Batch {batch_id} generated {len(candidates)} candidates")

            # Deduplicate against existing pool
            existing_exprs = self.pool.get_all_expressions()
            unique_candidates = []
            for c in candidates:
                if c.expression not in existing_exprs:
                    unique_candidates.append(c)
                    existing_exprs.add(c.expression)

            if not unique_candidates:
                self.logger.warning(
                    f"Batch {batch_id} has no unique candidates, retrying..."
                )
                continue

            # Evaluate the batch
            evaluated = self.evaluator.evaluate_batch(unique_candidates, parallel=True)
            self.logger.info(f"Batch {batch_id} evaluated {len(evaluated)} candidates")

            # Check failure rate
            num_failed = sum(1 for ev in evaluated if not ev.success)
            failure_rate = num_failed / len(evaluated) if evaluated else 1.0

            if (
                failure_rate > self.batch_failure_threshold
                and retry_attempt < self.batch_max_retries - 1
            ):
                self.logger.warning(
                    f"Batch {batch_id} failure rate {failure_rate:.1%} "
                    f"exceeds threshold, retrying..."
                )
                continue

            return evaluated

        return []

    def _filter_by_threshold(
        self, evaluated: List[EvaluatedFactor], round_num: int
    ) -> tuple:
        """
        Filter factors by performance thresholds with 3-phase strategy:
        1. START PHASE: Accept ALL factors
        2. DIVERSITY PHASE: Accept by absolute IC
        3. SELECTION PHASE: Use adaptive thresholds
        """
        accepted = []
        rejected = []

        # START PHASE
        if round_num <= self.start_rounds:
            self.logger.info(
                f"Round {round_num} - START PHASE: Accepting ALL factors"
            )
            for factor in evaluated:
                if not factor.success:
                    rejected.append(factor)
                else:
                    accepted.append(factor)
            return accepted, rejected

        # DIVERSITY PHASE
        if round_num <= self.start_rounds + self.diversity_rounds:
            self.logger.info(
                f"Round {round_num} - DIVERSITY PHASE: Accepting by |IC|"
            )
            for factor in evaluated:
                if not factor.success:
                    rejected.append(factor)
                    continue
                ic = abs(factor.get_ic())
                rank_ic = abs(factor.get_rank_ic())
                if self.threshold_mode == "and":
                    passes = (ic >= self.min_ic) and (rank_ic >= self.min_rank_ic)
                else:
                    passes = (ic >= self.min_ic) or (rank_ic >= self.min_rank_ic)
                if passes:
                    accepted.append(factor)
                else:
                    rejected.append(factor)
            return accepted, rejected

        # SELECTION PHASE
        effective_min_ic = self.min_ic
        effective_min_rank_ic = self.min_rank_ic

        if self.adaptive_threshold:
            pool_factors = self.pool.get_top_factors()
            if pool_factors:
                ics = [abs(f.get("metrics", {}).get("ic", 0)) for f in pool_factors]
                rank_ics = [abs(f.get("metrics", {}).get("rank_ic", 0)) for f in pool_factors]
                avg_ic = sum(ics) / len(ics)
                avg_rank_ic = sum(rank_ics) / len(rank_ics)
                effective_min_ic = max(self.min_ic, avg_ic * self.adaptive_threshold_ratio)
                effective_min_rank_ic = max(
                    self.min_rank_ic, avg_rank_ic * self.adaptive_threshold_ratio
                )
                self.logger.info(
                    f"Adaptive thresholds - IC: {effective_min_ic:.4f}, "
                    f"RankIC: {effective_min_rank_ic:.4f}"
                )

        for factor in evaluated:
            if not factor.success:
                rejected.append(factor)
                continue
            ic = abs(factor.get_ic())
            rank_ic = abs(factor.get_rank_ic())
            if self.threshold_mode == "and":
                passes = (ic >= effective_min_ic) and (rank_ic >= effective_min_rank_ic)
            else:
                passes = (ic >= effective_min_ic) or (rank_ic >= effective_min_rank_ic)
            if passes:
                accepted.append(factor)
            else:
                rejected.append(factor)

        return accepted, rejected

    def _calculate_pool_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics from the factor pool."""
        all_factors = self.pool.get_top_factors()
        if not all_factors:
            return {"total_factors": 0}

        valid = []
        for f in all_factors:
            m = f.get("metrics", {})
            ic = m.get("ic")
            rank_ic = m.get("rank_ic")
            icir = m.get("icir")
            if ic is None or rank_ic is None:
                continue
            try:
                if math.isnan(ic) or math.isnan(rank_ic):
                    continue
            except TypeError:
                continue
            valid.append({"ic": abs(ic), "rank_ic": abs(rank_ic), "icir": abs(icir or 0)})

        if not valid:
            return {"total_factors": 0}

        ics = sorted([v["ic"] for v in valid], reverse=True)
        rank_ics = sorted([v["rank_ic"] for v in valid], reverse=True)
        icirs = sorted([v["icir"] for v in valid], reverse=True)

        n = len(valid)
        top10 = max(1, int(n * 0.1))
        top50 = max(1, int(n * 0.5))

        return {
            "total_factors": n,
            "top_10pct_avg_ic": sum(ics[:top10]) / top10,
            "top_10pct_avg_rank_ic": sum(rank_ics[:top10]) / top10,
            "top_10pct_avg_icir": sum(icirs[:top10]) / top10,
            "top_50pct_avg_ic": sum(ics[:top50]) / top50,
            "top_50pct_avg_rank_ic": sum(rank_ics[:top50]) / top50,
            "top_50pct_avg_icir": sum(icirs[:top50]) / top50,
            "overall_avg_ic": sum(ics) / n,
            "overall_avg_rank_ic": sum(rank_ics) / n,
            "overall_avg_icir": sum(icirs) / n,
        }

    def _save_audit_file(self, round_num: int, round_result: SearchRoundResult) -> None:
        """Save all factors from a round to audit JSONL file."""
        try:
            audit_dir = self.pool.save_dir / f"round_{round_num:03d}"
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_file = audit_dir / "all_factors.jsonl"

            with audit_file.open("w", encoding="utf-8") as f:
                for factor in round_result.accepted + round_result.rejected:
                    d = factor.to_dict()
                    d["status"] = "accepted" if factor in round_result.accepted else "rejected"
                    f.write(json.dumps(d, default=str, ensure_ascii=False) + "\n")

            self.logger.info(f"Saved audit file: {audit_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save audit file: {e}")

    def _save_text_summary(
        self, round_num: int, round_result: SearchRoundResult, pool_stats: Dict
    ) -> None:
        """Save human-readable text summary of the round."""
        try:
            audit_dir = self.pool.save_dir / f"round_{round_num:03d}"
            audit_dir.mkdir(parents=True, exist_ok=True)
            summary_file = audit_dir / "round_summary.txt"

            with summary_file.open("w", encoding="utf-8") as f:
                f.write(f"{'=' * 70}\n")
                f.write(f"ROUND {round_num} SUMMARY\n")
                f.write(f"{'=' * 70}\n\n")

                f.write(f"Generation:\n")
                f.write(f"  Evaluated: {len(round_result.evaluated)} factors\n")
                f.write(f"  Accepted:  {len(round_result.accepted)} factors\n")
                f.write(f"  Rejected:  {len(round_result.rejected)} factors\n")
                rate = (
                    len(round_result.accepted) / len(round_result.evaluated) * 100
                    if round_result.evaluated
                    else 0
                )
                f.write(f"  Success:   {rate:.1f}%\n")
                f.write(f"  Time:      {round_result.elapsed_time:.2f}s\n\n")

                f.write(f"Factor Pool:\n")
                f.write(f"  Total factors: {pool_stats.get('total_factors', 0)}\n")
                f.write(f"  Avg IC:  {pool_stats.get('overall_avg_ic', 0):.4f}\n")
                f.write(f"  Avg RankIC: {pool_stats.get('overall_avg_rank_ic', 0):.4f}\n")
                f.write(f"  Top 10% IC: {pool_stats.get('top_10pct_avg_ic', 0):.4f}\n\n")

                if round_result.accepted:
                    f.write(f"Top Accepted Factors (by IC):\n")
                    f.write(f"{'-' * 70}\n")
                    sorted_accepted = sorted(
                        round_result.accepted,
                        key=lambda x: abs(x.get_ic()),
                        reverse=True,
                    )[:10]
                    for i, factor in enumerate(sorted_accepted, 1):
                        f.write(
                            f"{i:2d}. {factor.candidate.name:30s} "
                            f"IC={factor.get_ic():7.4f} "
                            f"RankIC={factor.get_rank_ic():7.4f}\n"
                        )
                        f.write(f"    {factor.candidate.expression[:65]}\n")

                f.write(f"\n{'=' * 70}\n")

            self.logger.info(f"Saved text summary: {summary_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save text summary: {e}")

    def _update_statistics(self, round_result: SearchRoundResult) -> None:
        """Update running statistics."""
        self.total_generated += len(round_result.evaluated)
        self.total_accepted += len(round_result.accepted)

        for factor in round_result.accepted:
            ic = abs(factor.get_ic())
            rank_ic = abs(factor.get_rank_ic())
            if ic > abs(self.best_ic):
                self.best_ic = factor.get_ic()
            if rank_ic > abs(self.best_rank_ic):
                self.best_rank_ic = factor.get_rank_ic()

    def get_summary(self) -> Dict[str, Any]:
        """Get search summary statistics."""
        return {
            "total_rounds": len(self.round_results),
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "success_rate": (
                self.total_accepted / self.total_generated
                if self.total_generated > 0
                else 0
            ),
            "best_ic": self.best_ic,
            "best_rank_ic": self.best_rank_ic,
            "pool_size": self.pool.size,
            "round_summaries": [r.get_summary() for r in self.round_results],
        }
