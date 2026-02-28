"""
Evaluate T1 Generation — Fitness and Diversity

Loads the LLM-generated factor results from a prior run_t1_generate.py run and
computes two categories of metrics:

  Fitness   — LLM-as-judge correctness score per sub-task level (Text2Alpha,
              Directional Mining).  A judge model evaluates whether each
              generated expression faithfully follows its instruction.

  Diversity — Structural and output-space diversity of the creative (directional
              mining) generated factors.  Measured via AST tree-edit distance
              and pairwise IC correlation.

Expected run directory layout (produced by run_t1_generate.py):
  runs/T1/<model>_<cot>/
    instructions/
      T1_1_easy_instruction.pkl
      T1_1_medium_instruction.pkl
      ...
    outputs/
      T1_1_easy_results.pkl
      T1_1_medium_results.pkl
      ...
      T1_2_easy_results.pkl
      ...
      T1_creative_<idx>_results.pkl   ← for creativity/diversity eval
      T1_DirectionalMining_creativity.json

Output:
  runs/T1/<model>_<cot>/outputs/scores/
    eval_fitness_results_<judge_model>.pkl   ← per-prefix correctness list
    eval_fitness_responses_<judge_model>.pkl ← raw LLM judge responses
    creativity_results.json                 ← AST distance + correlation stats

Usage:
  # Default: use deepseek-chat as judge
  python example/generate/eval_t1_fitness.py \\
      --run_dir runs/T1/deepseek-chat_False

  # Use a different judge model
  python example/generate/eval_t1_fitness.py \\
      --run_dir runs/T1/gpt-4.1_True --judge_model gpt-4.1

  # Skip creativity eval (faster)
  python example/generate/eval_t1_fitness.py \\
      --run_dir runs/T1/deepseek-chat_False --skip_diversity
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.engine.generate.eval_fitness import start_eval_fitness
from benchmark.engine.generate.benchmark_main import run_creativity_eval


def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate T1 generation fitness and diversity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run_dir", required=True,
                   help="Run directory produced by run_t1_generate.py "
                        "(e.g. runs/T1/deepseek-chat_False).")
    p.add_argument("--judge_model", default="deepseek-chat",
                   help="LLM to use as judge for fitness evaluation "
                        "(default: deepseek-chat).")
    p.add_argument("--skip_fitness",   action="store_true",
                   help="Skip fitness (correctness) evaluation.")
    p.add_argument("--skip_diversity", action="store_true",
                   help="Skip creativity/diversity evaluation.")
    p.add_argument("--creativity_n",   type=int, default=30,
                   help="Number of creativity prompt groups to evaluate (default: 30).")
    p.add_argument("--num_workers",    type=int, default=8,
                   help="Parallel workers for fitness judge calls (default: 8).")
    return p.parse_args()


def main():
    args = _parse_args()

    instr_dir  = os.path.join(args.run_dir, "instructions")
    out_dir    = os.path.join(args.run_dir, "outputs")
    scores_dir = os.path.join(out_dir, "scores")

    # ---- Validate run directory ----
    for d, label in [(instr_dir, "instructions"), (out_dir, "outputs")]:
        if not os.path.isdir(d):
            print(f"[ERROR] {label} directory not found: {d}")
            print("  → Run run_t1_generate.py first.")
            sys.exit(1)

    os.makedirs(scores_dir, exist_ok=True)

    print(f"[T1 Eval] run_dir     = {args.run_dir}")
    print(f"          judge_model = {args.judge_model}")

    # ---- Fitness evaluation ----
    if not args.skip_fitness:
        print("\n[Step 1] Fitness evaluation (LLM-as-judge)...")
        start_eval_fitness(
            instruction_dir=instr_dir,
            outputs_dir=out_dir,
            model=args.judge_model,
        )
        results_path = os.path.join(
            scores_dir, f"eval_fitness_results_{args.judge_model}.pkl"
        )
        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            print("\n  Fitness summary:")
            for prefix, labels in results.items():
                n_total   = len(labels)
                n_correct = sum(1 for r in labels if r == "correct")
                acc = n_correct / n_total if n_total else 0.0
                print(f"    {prefix:<25}  correct={n_correct}/{n_total}  acc={acc:.3f}")
        print(f"[Done] Fitness results → {scores_dir}")
    else:
        print("\n[Step 1] Skipping fitness evaluation (--skip_fitness).")

    # ---- Diversity / creativity evaluation ----
    if not args.skip_diversity:
        print("\n[Step 2] Diversity evaluation (AST distance + IC correlation)...")
        creativity_results = run_creativity_eval(
            data_path=instr_dir,
            save_dir=out_dir,
            N_SIZE=args.creativity_n,
            num_workers=args.num_workers,
        )

        # Persist results as JSON
        out_json = os.path.join(scores_dir, "creativity_results.json")
        serializable = {}
        for idx, stats in creativity_results.items():
            serializable[str(idx)] = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in stats.items()
                if k not in {"dist_pairs", "dist_matrix", "corr_pairs", "corr_matrix"}
            }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        # Print summary
        dist_vals  = [s["dist"]["mean_dist"]  for s in creativity_results.values()
                      if "dist" in s and s["dist"].get("mean_dist") is not None]
        corr_vals  = [s["corr"]["mean_abs_corr"] for s in creativity_results.values()
                      if "corr" in s and s["corr"].get("mean_abs_corr") is not None]
        if dist_vals:
            import statistics
            print(f"\n  AST mean_dist  :  avg={statistics.mean(dist_vals):.3f}")
        if corr_vals:
            print(f"  IC mean_abs_corr:  avg={statistics.mean(corr_vals):.3f}  "
                  f"(diversity={1 - statistics.mean(corr_vals):.3f})")
        print(f"[Done] Creativity results → {out_json}")
    else:
        print("\n[Step 2] Skipping diversity evaluation (--skip_diversity).")

    print(f"\n[All done] Evaluation results in: {scores_dir}")


if __name__ == "__main__":
    main()
