#!/usr/bin/env python3
"""
AlphaBench Factor Search — Main Entry Point

Usage:
    # Cold start (LLM generates initial factors):
    python start_search.py --config config.yaml

    # Warm start (load factors from file):
    python start_search.py --config config.yaml --seed-file factors.txt

    # Resume from a previous pool checkpoint:
    python start_search.py --config config.yaml --resume results/search_001/final_pool.jsonl

Pipeline:
    1. Load YAML config
    2. Cold/warm start: build initial factor pool
    3. Evaluate initial pool via FFO server
    4. Run evolutionary search (mutation + crossover)
    5. Save results
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure AlphaBench root is on sys.path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_ALPHABENCH_ROOT = _SCRIPT_DIR.parent
if str(_ALPHABENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALPHABENCH_ROOT))

from searcher.config.config import load_config_from_yaml
from searcher.core.controller import FactorPool, SearchController
from searcher.agents.breeder import create_breeder
from searcher.agents.evaluator import create_evaluator
from searcher.utils.logger import SearchLogger


# ---------------------------------------------------------------------------
# Seed loading helpers
# ---------------------------------------------------------------------------

def load_seeds_from_file(filepath: str) -> list:
    """
    Load seed factors from a file.

    Supports:
    - Text file: one expression per line
    - JSON file: list of {"name": ..., "expression": ...}
    - JSONL file: one JSON object per line

    Returns:
        List of {"name": str, "expression": str}
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {filepath}")

    factors = []

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    factors.append({
                        "name": item.get("name", f"seed_{i}"),
                        "expression": item.get("expression", item.get("qlib_expression_default", "")),
                    })
                elif isinstance(item, str):
                    factors.append({"name": f"seed_{i}", "expression": item})
        elif isinstance(data, dict):
            for name, expr in data.items():
                if isinstance(expr, str):
                    factors.append({"name": name, "expression": expr})

    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                factors.append({
                    "name": obj.get("name", f"seed_{i}"),
                    "expression": obj.get("expression", ""),
                })

    else:  # txt or any other text format
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                factors.append({"name": f"seed_{i}", "expression": line})

    return [f for f in factors if f.get("expression")]


def cold_start_generate(config, logger) -> list:
    """
    Cold start: use LLM to generate initial random factors.

    Returns:
        List of {"name": str, "expression": str}
    """
    from agent.generator_qlib_search import call_qlib_search

    logger.info("Cold start: generating initial factors via LLM...")

    instruction = """Generate diverse alpha factors for stock ranking.
Cover different ideas: momentum, mean reversion, volatility, volume dynamics, cross-variable relations.
Each factor should be a valid Qlib expression using only: $close, $open, $high, $low, $volume.
"""

    n_initial = 30
    result = call_qlib_search(
        instruction=instruction,
        model=config.searching.model.name,
        N=n_initial,
        verbose=True,
        temperature=config.searching.model.temperature,
        enable_reason=True,
    )

    factors = []
    for f in result.get("factors", []):
        factors.append({
            "name": f.get("name", ""),
            "expression": f.get("expression", ""),
        })

    logger.info(f"Cold start: generated {len(factors)} initial factors")
    return factors


def warm_start_from_alpha158(logger) -> list:
    """
    Warm start from built-in alpha158 factor library.

    Returns:
        List of {"name": str, "expression": str}
    """
    from factors.lib.alpha158 import load_factors_alpha158

    logger.info("Warm start: loading alpha158 factors...")
    standard_factors, compile_factors = load_factors_alpha158(
        exclude_var="vwap", collection=["kbar", "rolling"]
    )

    factors = []
    for name, item in compile_factors.items():
        expr = item.get("qlib_expression_default", "")
        if expr:
            factors.append({"name": name, "expression": expr})

    logger.info(f"Loaded {len(factors)} alpha158 factors")
    return factors


def evaluate_initial_pool(factors, evaluator, logger):
    """
    Evaluate a list of seed factors via FFO and attach metrics.

    Args:
        factors: List of {"name": str, "expression": str}
        evaluator: FactorEvaluator instance
        logger: Logger

    Returns:
        factors with "metrics" attached
    """
    from searcher.core.schemas import FactorCandidate

    logger.info(f"Evaluating {len(factors)} initial factors via FFO...")

    candidates = [
        FactorCandidate(name=f["name"], expression=f["expression"], doc_type="origin")
        for f in factors
    ]

    evaluated = evaluator.evaluate_batch(candidates, parallel=True)

    result = []
    for ev in evaluated:
        d = ev.candidate.to_dict()
        d["metrics"] = ev.metrics
        d["success"] = ev.success
        if ev.success:
            result.append(d)
        else:
            logger.warning(f"Failed to evaluate seed: {ev.candidate.name}")

    logger.info(f"Successfully evaluated {len(result)}/{len(factors)} initial factors")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AlphaBench Factor Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cold start
  python start_search.py --config config.yaml

  # Warm start with factor file
  python start_search.py --config config.yaml --seed-file seeds.txt

  # Warm start with built-in alpha158
  python start_search.py --config config.yaml --alpha158

  # Resume from checkpoint
  python start_search.py --config config.yaml --resume results/final_pool.jsonl
        """,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML config file")
    parser.add_argument("--seed-file", type=str, help="Seed factors file (txt/json/jsonl)")
    parser.add_argument("--alpha158", action="store_true", help="Use alpha158 as seed factors")
    parser.add_argument("--resume", type=str, help="Resume from a pool checkpoint JSONL")
    parser.add_argument("--log", type=str, help="Log file path")

    args = parser.parse_args()

    # Setup logger
    logger = SearchLogger("AlphaBench", log_file=args.log)
    logger.info("=" * 70)
    logger.info("AlphaBench Factor Search")
    logger.info("=" * 70)

    try:
        # Load configuration
        config_path = args.config
        if not Path(config_path).is_absolute():
            config_path = str(_SCRIPT_DIR / config_path)

        logger.info(f"Loading config from: {config_path}")
        config = load_config_from_yaml(config_path)

        # Initialize evaluator (for both initial eval and search)
        evaluator = create_evaluator(
            config_dict={
                "ffo_url": config.backtesting.get_api_url(),
                "market": config.backtesting.market,
                "period_start": config.backtesting.period_start,
                "period_end": config.backtesting.period_end,
                "top_k": config.backtesting.top_k,
                "n_drop": config.backtesting.n_drop,
                "fast": config.backtesting.fast,
                "n_jobs": config.backtesting.n_jobs,
            },
            logger=logger,
        )
        logger.info(f"Evaluator: FFO at {config.backtesting.ffo_server}")

        # Initialize factor pool
        pool = FactorPool(save_dir=config.savedir)

        # Determine start mode and load seeds
        if args.resume:
            # Resume from checkpoint
            logger.info(f"Resuming from checkpoint: {args.resume}")
            loaded = pool.load(args.resume)
            logger.info(f"Loaded {loaded} factors from checkpoint")

        else:
            # Determine seed source
            seed_file = args.seed_file or config.searching.algo.seed_file

            if seed_file:
                # Warm start: load from file
                logger.info(f"Warm start: loading seeds from {seed_file}")
                factors = load_seeds_from_file(seed_file)
            elif args.alpha158:
                # Warm start: built-in alpha158
                factors = warm_start_from_alpha158(logger)
            else:
                # Cold start: generate via LLM
                factors = cold_start_generate(config, logger)

            if not factors:
                logger.error("No seed factors available — cannot start search")
                return 1

            # Evaluate initial pool via FFO
            evaluated_factors = evaluate_initial_pool(factors, evaluator, logger)

            if not evaluated_factors:
                logger.error("All initial factor evaluations failed — check FFO server")
                return 1

            # Add to pool
            added = pool.add_seeds(evaluated_factors)
            logger.info(f"Added {added} seed factors to pool (pool size: {pool.size})")

            # Save initial pool
            pool.save("initial_pool.jsonl")

        # Initialize breeder
        breeder = create_breeder(
            config_dict={
                "name": config.searching.model.name,
                "key": config.searching.model.resolve_key(),
                "base_url": config.searching.model.base_url,
                "temperature": config.searching.model.temperature,
                "enable_reason": True,
            },
            logger=logger,
        )
        logger.info(f"Breeder: {config.searching.model.name}")

        # Initialize controller
        controller = SearchController(
            pool=pool,
            breeder=breeder,
            evaluator=evaluator,
            config={
                "num_rounds": config.searching.num_rounds,
                "mutation_rate": config.searching.mutation_rate,
                "crossover_rate": config.searching.crossover_rate,
                "window_size": config.searching.window_size,
                "factors_per_batch": config.searching.factors_per_batch,
                "num_workers": config.searching.num_workers,
                "batch_max_retries": config.searching.batch_max_retries,
                "batch_failure_threshold": config.searching.batch_failure_threshold,
                "seed_limit": config.searching.algo.seed_top_k,
                "min_ic": config.searching.min_ic,
                "min_rank_ic": config.searching.min_rank_ic,
                "adaptive_threshold": config.searching.adaptive_threshold,
                "threshold_mode": config.searching.threshold_mode,
                "adaptive_threshold_ratio": config.searching.adaptive_threshold_ratio,
                "start_rounds": config.searching.start_rounds,
                "diversity_rounds": config.searching.diversity_rounds,
                "temperature": config.searching.model.temperature,
            },
            logger=logger,
        )
        logger.info("Controller initialized")

        # Run search
        logger.info("")
        logger.info("Starting factor search...")
        logger.info(f"  Rounds: {config.searching.num_rounds}")
        logger.info(f"  Window size: {config.searching.window_size}")
        logger.info(f"  Factors/batch: {config.searching.factors_per_batch}")
        logger.info(f"  Workers: {config.searching.num_workers}")
        logger.info(
            f"  Factors/round: {config.searching.factors_per_batch * config.searching.num_workers}"
        )
        logger.info(f"  Mutation: {config.searching.mutation_rate:.0%}")
        logger.info(f"  Crossover: {config.searching.crossover_rate:.0%}")
        logger.info("")

        final_stats = controller.run_search()

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Search Complete!")
        logger.info("=" * 70)
        logger.info(f"  Status: {final_stats['status']}")
        logger.info(f"  Total generated: {final_stats['total_generated']}")
        logger.info(f"  Total accepted: {final_stats['total_accepted']}")
        logger.info(f"  Success rate: {final_stats['success_rate']:.1%}")
        logger.info(f"  Best IC: {final_stats['best_ic']:.4f}")
        logger.info(f"  Best RankIC: {final_stats['best_rank_ic']:.4f}")
        logger.info(f"  Pool size: {final_stats['pool_size']}")
        logger.info(f"  Total time: {final_stats['total_time']:.1f}s")

        return 0

    except KeyboardInterrupt:
        logger.info("\nSearch interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
