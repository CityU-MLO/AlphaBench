"""
Run T1 Factor Generation

Builds instruction files from the benchmark data and calls an LLM to generate
alpha factor expressions.  Three sub-tasks are covered:

  Text2Alpha       — translate a natural-language description into a factor expression.
  Directional Mining — generate factors that fit a given theme / direction tag.
  Stability        — generate paraphrase variants of the same factor instruction.

Generated outputs are saved as pickle files under the run directory and can
be evaluated for fitness and diversity with eval_t1_fitness.py.

Expected input data (already present in this repo):
  benchmark/data/generate/
    T1_Text2Alpha.json                ← Text2Alpha instructions per difficulty level
    T1_DirectionalMining.json         ← Directional-Mining instructions per level
    T1_Text2Alpha_stability.json      ← Stability synonym groups
    T1_DirectionalMining_creativity.json ← Directional-Mining creativity prompts
    tags_system.json                  ← Tag descriptions for directional mining

Output layout:
  runs/T1/<model>_<cot>/
    instructions/
      T1_1_easy_instruction.pkl       ← Text2Alpha instructions (per level)
      T1_1_medium_instruction.pkl
      T1_1_hard_instruction.pkl
      T1_2_easy_instruction.pkl       ← Directional-Mining instructions
      ...
      T1_stability_0.pkl              ← Stability synonym groups
      ...
    outputs/
      T1_1_easy_results.pkl           ← LLM generation results (per level)
      T1_2_easy_results.pkl
      T1_stability_0_results.pkl
      ...

Usage:
  # Default (deepseek-chat, no CoT)
  python example/generate/run_t1_generate.py

  # GPT-4.1 with chain-of-thought
  python example/generate/run_t1_generate.py --model gpt-4.1 --cot

  # Local vLLM server
  python example/generate/run_t1_generate.py \\
      --model Qwen2.5-72B-Instruct --local --local_port 8000

  # Skip instruction building (instructions already exist)
  python example/generate/run_t1_generate.py --skip_build

  # Only build instructions, do not run LLM
  python example/generate/run_t1_generate.py --build_only
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.engine import (
    build_instructions,
    start_running_LLM_generation,
)


_DATA_DIR    = "benchmark/data/generate"
_OUTPUT_ROOT = "runs/T1"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run T1 factor generation with an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_dir",    default=_DATA_DIR,
                   help=f"Directory containing T1 benchmark data. Default: {_DATA_DIR}")
    p.add_argument("--output_root", default=_OUTPUT_ROOT,
                   help=f"Root output directory. Default: {_OUTPUT_ROOT}")
    p.add_argument("--model",       default="deepseek-chat",
                   help="LLM model name (default: deepseek-chat)")
    p.add_argument("--cot",         action="store_true",
                   help="Enable chain-of-thought prompting.")
    p.add_argument("--local",       action="store_true",
                   help="Use a local vLLM server.")
    p.add_argument("--local_port",  type=int, default=8000,
                   help="Port of local LLM server (default: 8000).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Parallel workers for batch LLM calls (default: 4).")
    p.add_argument("--skip_build",  action="store_true",
                   help="Skip instruction building (use existing instruction pkl files).")
    p.add_argument("--build_only",  action="store_true",
                   help="Only build instruction files; skip LLM generation.")
    return p.parse_args()


def main():
    args = _parse_args()

    # ---- Build paths ----
    run_dir   = os.path.join(args.output_root, f"{args.model}_{args.cot}")
    instr_dir = os.path.join(run_dir, "instructions")
    out_dir   = os.path.join(run_dir, "outputs")

    os.makedirs(instr_dir, exist_ok=True)
    os.makedirs(out_dir,   exist_ok=True)

    # ---- Validate data directory ----
    required_files = [
        "T1_Text2Alpha.json",
        "T1_DirectionalMining.json",
        "T1_Text2Alpha_stability.json",
    ]
    for fname in required_files:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[ERROR] Required data file not found: {fpath}")
            sys.exit(1)

    print(f"[T1 Generate] model={args.model}  cot={args.cot}")
    print(f"  data:   {args.data_dir}")
    print(f"  output: {run_dir}")

    # ---- Step 1: Build instructions ----
    if not args.skip_build:
        print("\n[Step 1] Building instruction files...")
        build_instructions(
            data_path=args.data_dir,
            save_dir=instr_dir,
        )
        print(f"[Done] Instructions saved to: {instr_dir}")
    else:
        print("\n[Step 1] Skipping instruction build (--skip_build).")
        if not any(f.endswith(".pkl") for f in os.listdir(instr_dir)):
            print(f"[ERROR] No .pkl instruction files found in: {instr_dir}")
            print("  → Remove --skip_build to build instructions first.")
            sys.exit(1)

    if args.build_only:
        print("\n[build_only] Skipping LLM generation.")
        return

    # ---- Step 2: Run LLM generation ----
    print("\n[Step 2] Running LLM generation...")
    start_running_LLM_generation(
        num_workers=args.num_workers,
        data_path=instr_dir,
        save_dir=out_dir,
        model=args.model,
        enable_cot=args.cot,
        local_model=args.local,
        local_port=args.local_port,
    )
    print(f"\n[Done] Generation results saved to: {out_dir}")
    print(
        "\nNext step: run eval_t1_fitness.py to evaluate correctness and diversity.\n"
        f"  python example/generate/eval_t1_fitness.py --run_dir {run_dir}"
    )


if __name__ == "__main__":
    main()
