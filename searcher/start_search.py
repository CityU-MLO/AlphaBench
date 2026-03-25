#!/usr/bin/env python3
"""
AlphaBench Factor Search — Main Entry Point

Runs the full factor discovery pipeline:
  1. Cold / warm / resume start to build initial seed pool
  2. Baseline evaluation of seeds via FFO server
  3. Search algorithm (CoT / EA / ToT) iterates, evaluating every candidate via FFO
  4. Results (best factor + final pool) are saved to disk

Usage
─────
  # Cold start (LLM generates initial factors):
  python start_search.py --config search_config.yaml

  # Warm start (load seeds from file):
  python start_search.py --config search_config.yaml --seed-file factors.txt

  # Warm start from built-in alpha158:
  python start_search.py --config search_config.yaml --alpha158

  # Resume from a previous pool checkpoint:
  python start_search.py --config search_config.yaml --resume results/final_pool.jsonl

YAML config format
──────────────────
  searching:
    algo:
      name: ea           # "ea" | "cot" | "tot"
      param:
        rounds: 10
        N: 30
        mutation_rate: 0.4
        crossover_rate: 0.6

    model:
      name: deepseek-chat
      temperature: 0.7

  backtesting:
    ffo_server: "127.0.0.1:19777"
    market: csi300
    benchmark: SH000300
    search_start: "2016-01-01"
    search_end: "2021-01-01"
    top_k: 30
    n_drop: 1
    fast: true

  verification:
    enabled: true
    val_start: "2021-01-01"
    val_end: "2022-01-01"
    test_start: "2022-01-01"
    test_end: "2025-01-01"

  savedir: "./results"
"""

import argparse
import sys
from pathlib import Path

# Ensure AlphaBench root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_ALPHABENCH_ROOT = _SCRIPT_DIR.parent
if str(_ALPHABENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALPHABENCH_ROOT))

from searcher.config.config import load_config_from_yaml
from searcher.pipeline import SearchPipeline
from searcher.utils.logger import SearchLogger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AlphaBench Factor Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cold start
  python start_search.py --config search_config.yaml

  # Warm start with factor file
  python start_search.py --config search_config.yaml --seed-file seeds.txt

  # Warm start with built-in alpha158
  python start_search.py --config search_config.yaml --alpha158

  # Resume from checkpoint
  python start_search.py --config search_config.yaml --resume results/final_pool.jsonl
        """,
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML configuration file (default: config.yaml)")
    parser.add_argument("--seed-file", type=str, default=None,
                        help="Seed factors file (.txt / .json / .jsonl)")
    parser.add_argument("--alpha158", action="store_true",
                        help="Use built-in alpha158 as seed factors")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a pool checkpoint JSONL file")
    parser.add_argument("--log", type=str, default=None,
                        help="Optional log file path")
    parser.add_argument("--savedir", type=str, default=None,
                        help="Override the save directory from config")

    args = parser.parse_args()

    logger = SearchLogger("AlphaBench", log_file=args.log)

    try:
        # Resolve config path relative to this script's directory
        config_path = args.config
        if not Path(config_path).is_absolute():
            candidate = _SCRIPT_DIR / config_path
            if candidate.exists():
                config_path = str(candidate)

        logger.info(f"Loading config from: {config_path}")
        config = load_config_from_yaml(config_path)

        # Allow CLI override of savedir
        if args.savedir:
            config.savedir = args.savedir

        # Run pipeline
        pipeline = SearchPipeline(config, logger=logger)
        results = pipeline.run(
            seed_file=args.seed_file,
            alpha158=args.alpha158,
            resume=args.resume,
        )

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
