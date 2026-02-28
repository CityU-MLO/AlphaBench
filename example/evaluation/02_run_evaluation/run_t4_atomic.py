"""
Run T4 Atomic Evaluation

Runs LLM inference on the atomic evaluation JSONL datasets and computes metrics.

Two tasks:
  binary_noise    — classify each factor expression as "signal" or "noise"
  pairwise_select — choose the better factor between two presented expressions

Expected data layout (built by build_atomic_*.py):
  benchmark/data/evaluate/atomic/noise_csi300/
    train.jsonl  val.jsonl  test.jsonl  manifest.json

  benchmark/data/evaluate/atomic/pairwise_csi300/
    train.jsonl  val.jsonl  test.jsonl  manifest.json

Output per task × split:
  runs/T4/<model>_<cot>/<task>/<split>/
    results.jsonl     ← predicted label per record
    metrics.json      ← accuracy, precision, recall, F1
    report.txt        ← human-readable summary
    infer.log         ← inference log
    llm_output.json   ← raw LLM JSON responses

Usage:
  # Both tasks, test split only (default)
  python example/02_run_evaluation/run_t4_atomic.py

  # Only noise task
  python example/02_run_evaluation/run_t4_atomic.py --tasks noise

  # Only pairwise task with CoT
  python example/02_run_evaluation/run_t4_atomic.py --tasks pairwise --cot

  # All splits (train + val + test)
  python example/02_run_evaluation/run_t4_atomic.py --splits train val test

  # Local vLLM server
  python example/02_run_evaluation/run_t4_atomic.py \
      --model Qwen2.5-72B-Instruct --local --local_port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.engine.evaluate import run_atomic_benchmark


_NOISE_DIR    = "benchmark/data/evaluate/atomic/noise_csi300"
_PAIRWISE_DIR = "benchmark/data/evaluate/atomic/pairwise_csi300"
_OUTPUT_ROOT  = "runs/T4"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run T4 atomic evaluation (binary_noise + pairwise_select).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--noise_dir",    default=_NOISE_DIR,
                   help=f"Directory with noise JSONL splits. Default: {_NOISE_DIR}")
    p.add_argument("--pairwise_dir", default=_PAIRWISE_DIR,
                   help=f"Directory with pairwise JSONL splits. Default: {_PAIRWISE_DIR}")
    p.add_argument("--output_root",  default=_OUTPUT_ROOT,
                   help=f"Root output directory. Default: {_OUTPUT_ROOT}")
    p.add_argument("--model",        default="gpt-4.1",
                   help="LLM model name (default: gpt-4.1).")
    p.add_argument("--cot",          action="store_true",
                   help="Enable chain-of-thought prompting.")
    p.add_argument("--market_prompt", default="auto",
                   choices=["general", "us", "cn", "auto"],
                   help="Market context in prompts (default: auto).")
    p.add_argument("--local",        action="store_true",
                   help="Use a local vLLM server.")
    p.add_argument("--local_port",   type=int, default=8000,
                   help="Port of local LLM server (default: 8000).")
    p.add_argument("--num_workers",  type=int, default=8,
                   help="Parallel workers for batch LLM calls (default: 8).")
    p.add_argument("--tasks", nargs="+", choices=["noise", "pairwise", "all"],
                   default=["all"],
                   help="Which tasks to run (default: all).")
    p.add_argument("--splits", nargs="+", default=["test"],
                   choices=["train", "val", "test"],
                   help="Which splits to evaluate (default: test).")
    p.add_argument("--temperature",  type=float, default=0.3,
                   help="Sampling temperature (default: 0.3).")
    p.add_argument("--save_prompts", action="store_true",
                   help="Persist prompts in results for debugging.")
    return p.parse_args()


def main():
    args = _parse_args()

    run_tag = f"{args.model}_{args.cot}"

    tasks = args.tasks
    if "all" in tasks:
        tasks = ["noise", "pairwise"]

    infer_kwargs = dict(
        model=args.model,
        cot=args.cot,
        market_prompt=args.market_prompt,
        num_workers=args.num_workers,
        temperature=args.temperature,
        local=args.local,
        local_port=args.local_port,
        splits=args.splits,
        save_prompts=args.save_prompts,
    )

    all_summaries = {}

    if "noise" in tasks:
        if not os.path.isdir(args.noise_dir):
            print(f"[ERROR] Noise data dir not found: {args.noise_dir}")
            print("  → Run build_atomic_noise_dataset.py first.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"[T4] Binary Noise Classification  model={args.model}  cot={args.cot}")
        print(f"     data:    {args.noise_dir}")
        print(f"     splits:  {args.splits}")
        print("=" * 60)

        noise_out = os.path.join(args.output_root, run_tag, "binary_noise")
        summary = run_atomic_benchmark(
            task="binary_noise",
            data_dir=args.noise_dir,
            output_dir=noise_out,
            **infer_kwargs,
        )
        all_summaries["binary_noise"] = summary
        _print_summary(summary)

    if "pairwise" in tasks:
        if not os.path.isdir(args.pairwise_dir):
            print(f"[ERROR] Pairwise data dir not found: {args.pairwise_dir}")
            print("  → Run build_atomic_pairwise_dataset.py first.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"[T4] Pairwise Selection  model={args.model}  cot={args.cot}")
        print(f"     data:    {args.pairwise_dir}")
        print(f"     splits:  {args.splits}")
        print("=" * 60)

        pairwise_out = os.path.join(args.output_root, run_tag, "pairwise_select")
        summary = run_atomic_benchmark(
            task="pairwise_select",
            data_dir=args.pairwise_dir,
            output_dir=pairwise_out,
            **infer_kwargs,
        )
        all_summaries["pairwise_select"] = summary
        _print_summary(summary)

    # ---- Write combined summary ----
    combined_out = os.path.join(args.output_root, run_tag)
    combined_path = os.path.join(combined_out, "t4_summary.json")
    os.makedirs(combined_out, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        # Serialize only the metrics part (exclude non-serializable objects)
        slim = {}
        for task_name, s in all_summaries.items():
            slim[task_name] = {
                "splits": {
                    split: {"metrics": v.get("metrics", {})}
                    for split, v in s.get("splits", {}).items()
                }
            }
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Combined T4 summary → {combined_path}")


def _print_summary(summary: dict) -> None:
    task = summary.get("task", "?")
    print(f"\n[Summary] task={task}")
    for split_name, res in summary.get("splits", {}).items():
        metrics = res.get("metrics", {})
        acc = metrics.get("accuracy", metrics.get("acc", "N/A"))
        print(f"  {split_name}: accuracy={acc}")


if __name__ == "__main__":
    main()
