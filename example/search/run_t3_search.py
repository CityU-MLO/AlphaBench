"""
Run T3 Factor Searching

Runs LLM-driven factor search strategies over a pool of seed alpha factors,
attempting to discover novel factors with better predictive performance.

Three search algorithms are supported (individually or in combination):

  CoT (Chain-of-Thought) — Iteratively refines a seed factor over multiple
      rounds by prompting the LLM to reason about possible improvements.

  ToT (Tree-of-Thought) — Builds a search tree by branching each candidate
      into N alternatives at each round, then selecting the best.

  EA  (Evolutionary Algorithm) — Maintains a population of factors and uses
      LLM-driven mutation and crossover to evolve better factors across
      multiple generations.

Which algorithms run is controlled by a YAML config file.  An example config
is provided in example/search/configs/search_csi300.yaml.

Prerequisites:
  - A running FFO (Factor Fitness Oracle) server for factor evaluation.
    Configure its address in ffo/config/ffo.yaml.
  - Seed factors are loaded from the Alpha158 factor library (built-in).

Output layout:
  runs/T3/<model>_<cot>/
    factor_seed_metrics.json     ← baseline performance of seed factors
    CoT/
      <factor_name>/             ← per-factor CoT search history
        round_0.json
        round_1.json
        ...
    ToT/
      <factor_name>/
        ...
    EA/
      population_<gen>.pkl       ← population snapshot per generation
      summary.json               ← best factors found

Usage:
  # Default config (CSI300, EA only)
  python example/search/run_t3_search.py \\
      --config example/search/configs/search_csi300.yaml

  # Override model from CLI
  python example/search/run_t3_search.py \\
      --config example/search/configs/search_csi300.yaml \\
      --model_name gpt-4.1

  # Use local vLLM server
  python example/search/run_t3_search.py \\
      --config example/search/configs/search_csi300.yaml \\
      --model_name Qwen2.5-72B-Instruct --model_local true --local_port 8000

  # Target SP500 market
  python example/search/run_t3_search.py \\
      --config example/search/configs/search_csi300.yaml \\
      --market sp500 --save_dir runs/T3/sp500_ea
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from benchmark.engine import run_searching_benchmark


_DEFAULT_CONFIG = "example/search/configs/search_csi300.yaml"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run T3 factor searching (CoT / ToT / EA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help=f"Path to YAML config file. Default: {_DEFAULT_CONFIG}",
    )
    # CLI overrides (mirror benchmark_searching.py load_config)
    p.add_argument("--model_name",  default=None,
                   help="Override model name from config.")
    p.add_argument("--model_local", default=None, type=lambda x: x.lower() == "true",
                   help="Override model.local flag (true/false).")
    p.add_argument("--local_port",  default=None, type=int,
                   help="Override model.local_port from config.")
    p.add_argument("--market",      default=None,
                   help="Override target market (e.g. csi300, sp500).")
    p.add_argument("--save_dir",    default=None,
                   help="Override output directory.")
    return p.parse_args()


def _load_config(args) -> dict:
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        print(
            "  → Copy and edit the example config:\n"
            f"    cp example/search/configs/search_csi300.yaml {args.config}"
        )
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model_name is not None:
        config.setdefault("model", {})["name"] = args.model_name
    if args.model_local is not None:
        config.setdefault("model", {})["local"] = args.model_local
    if args.local_port is not None:
        config.setdefault("model", {})["local_port"] = args.local_port
    if args.market is not None:
        config["market"] = args.market
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir

    # Set default save_dir based on model + market if not already set
    if "save_dir" not in config:
        model_name = config.get("model", {}).get("name", "unknown")
        market     = config.get("market", "csi300")
        config["save_dir"] = f"runs/T3/{model_name}_{market}"

    return config


def main():
    args   = _parse_args()
    config = _load_config(args)

    model_name = config.get("model", {}).get("name", "?")
    market     = config.get("market", "csi300")
    save_dir   = config["save_dir"]

    # Summarize which algorithms are enabled
    enabled = [
        alg for alg in ("cot", "tot", "ea")
        if config.get(alg, {}).get("enable", False)
    ]

    print(f"[T3 Search] model={model_name}  market={market}")
    print(f"  algorithms: {enabled or ['none — check config']}")
    print(f"  output:     {save_dir}")

    if not enabled:
        print("\n[WARN] No search algorithm is enabled in the config.")
        print("  → Set at least one of cot.enable / tot.enable / ea.enable to true.")

    # ---- Run searching benchmark ----
    run_searching_benchmark(config)

    print(f"\n[Done] Search results saved to: {save_dir}")


if __name__ == "__main__":
    main()
