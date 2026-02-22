"""
AlphaBench - Unified Benchmark Runner

Usage:
  python benchmark/run_benchmark.py --config config/benchmark.yaml
  python benchmark/run_benchmark.py --config config/benchmark.yaml --tasks t1 t2
  python benchmark/run_benchmark.py --config config/benchmark.yaml --tasks t3

Tasks:
  t1  Factor generation (Text2Alpha, Directional Mining, Stability)
  t2  Factor evaluation (Ranking, Scoring)
  t3  Factor searching (CoT, ToT, EA)
  all Run all tasks in sequence (default)
"""

import os
import sys
import argparse
import yaml
from datetime import datetime


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaBench - Unified Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/benchmark.yaml",
        help="Path to the benchmark config file (default: ./config/benchmark.yaml)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["t1", "t2", "t3", "all"],
        default=["all"],
        metavar="TASK",
        help="Tasks to run: t1, t2, t3, all (default: all)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TASK_LABELS = {
    "t1": "Factor Generation  (Text2Alpha, Directional Mining, Stability)",
    "t2": "Factor Evaluation  (Ranking, Scoring)",
    "t3": "Factor Searching   (CoT, ToT, EA)",
}

_TASK_CONFIG_KEYS = {
    "t1": "T1_GENERATION",
    "t2": "T2_EVALUATION",
    "t3": "T3_SEARCHING",
}


def _llm_kwargs(config):
    """Extract shared LLM settings from top-level config."""
    return {
        "model": config.get("BACKEND_LLM", "deepseek-chat"),
        "local": config.get("BACKEND_SERVICE", "online") == "local",
        "local_port": config.get("BACKEND_LLM_PORT", 8000),
        "enable_cot": config.get("ENABLE_COT", False),
    }


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_t1(config, result_dir):
    """T1 - Factor Generation: build instructions then run LLM generation."""
    from benchmark.engine import build_instructions, start_running_LLM_generation

    llm = _llm_kwargs(config)
    t1_cfg = config.get("T1_GENERATION", {})
    data_path = t1_cfg.get("data_path", "./benchmark/data/generate")

    instr_dir = os.path.join(result_dir, "T1", "instructions")
    out_dir = os.path.join(result_dir, "T1", "outputs")
    os.makedirs(instr_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print("[T1] Building instructions...")
    build_instructions(data_path=data_path, save_dir=instr_dir)

    print("[T1] Running LLM generation...")
    start_running_LLM_generation(
        model=llm["model"],
        enable_cot=llm["enable_cot"],
        data_path=instr_dir,
        save_dir=out_dir,
        local_model=llm["local"],
        local_port=llm["local_port"],
    )
    print("[T1] Done.")


def run_t2(config, result_dir):
    """T2 - Factor Evaluation: ranking and scoring benchmarks."""
    from benchmark.engine import (
        benchmark_ranking_performance,
        benchmark_scoring_performance,
    )

    llm = _llm_kwargs(config)
    t2_cfg = config.get("T2_EVALUATION", {})
    data_path = t2_cfg.get("data_path", "./benchmark/data/evaluation")

    out_dir = os.path.join(result_dir, "T2")
    os.makedirs(out_dir, exist_ok=True)

    print("[T2] Running ranking benchmark...")
    benchmark_ranking_performance(
        data_path=data_path,
        output_path=out_dir,
        enable_CoT=llm["enable_cot"],
        model=llm["model"],
        local=llm["local"],
        local_port=llm["local_port"],
    )

    print("[T2] Running scoring benchmark...")
    benchmark_scoring_performance(
        data_path=data_path,
        output_path=out_dir,
        enable_CoT=llm["enable_cot"],
        model=llm["model"],
        local=llm["local"],
        local_port=llm["local_port"],
    )
    print("[T2] Done.")


def run_t3(config, result_dir):
    """T3 - Factor Searching: CoT, ToT, and/or EA search."""
    from benchmark.engine import run_searching_benchmark

    t3_cfg = config.get("T3_SEARCHING", {})
    search_config = {
        "model": {
            "name": config.get("BACKEND_LLM", "deepseek-chat"),
            "local": config.get("BACKEND_SERVICE", "online") == "local",
            "local_port": config.get("BACKEND_LLM_PORT", 8000),
            "temperature": config.get("BACKEND_LLM_TEMPERATURE", 1.5),
        },
        "market": t3_cfg.get("market", "csi300"),
        "save_dir": os.path.join(result_dir, "T3"),
        "cot": t3_cfg.get("cot", {"enable": False}),
        "tot": t3_cfg.get("tot", {"enable": False}),
        "ea": t3_cfg.get("ea", {"enable": True}),
    }
    os.makedirs(search_config["save_dir"], exist_ok=True)

    print("[T3] Running searching benchmark...")
    run_searching_benchmark(search_config)
    print("[T3] Done.")


_TASK_RUNNERS = {
    "t1": run_t1,
    "t2": run_t2,
    "t3": run_t3,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Configure environment
    qlib_data_path = config.get("QLIB_DATA_PATH", "~/.qlib/qlib_data/cn_data")
    os.environ["QLIB_DATA_PATH"] = qlib_data_path

    # Create timestamped result directory
    model_name = config.get("BACKEND_LLM", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_save_path = config.get("RESULT_SAVE_PATH", "./runs")
    result_dir = os.path.join(result_save_path, f"{model_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Resolve task list
    tasks = args.tasks
    if "all" in tasks:
        tasks = ["t1", "t2", "t3"]

    print("=" * 60)
    print("AlphaBench - Benchmark Run")
    print(f"  Config  : {args.config}")
    print(f"  Model   : {model_name}")
    print(f"  CoT     : {config.get('ENABLE_COT', False)}")
    print(f"  Tasks   : {', '.join(t.upper() for t in tasks)}")
    print(f"  Results : {result_dir}")
    print("=" * 60)

    for task in tasks:
        task_cfg = config.get(_TASK_CONFIG_KEYS[task], {})
        if not task_cfg.get("enable", True):
            print(f"\n[{task.upper()}] Skipped (disabled in config).")
            continue

        print(f"\n--- {task.upper()}: {_TASK_LABELS[task]} ---")
        _TASK_RUNNERS[task](config, result_dir)

    print(f"\nAll tasks completed. Results saved to: {result_dir}")
