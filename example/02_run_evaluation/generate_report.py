"""
Generate Performance Reports

Loads existing result files from a run directory and produces:
  - Ranking summary table (Precision@K, NDCG@K per regime × setting)
  - Scoring summary table (MAE per score dimension)
  - Classification report (Precision/Recall/F1 for Positive/Negative/Noise)
  - Markdown report
  - CSV exports

Can also compare multiple model runs side-by-side.

Expected run directory layout (produced by run_t2_*.py):
  runs/T2/<model>_<cot>/
    ranking_results.json    (from run_t2_ranking.py)
    scoring_results.json    (from run_t2_scoring.py)

Usage:
  # Single run report
  python example/02_run_evaluation/generate_report.py \
      --run_dir runs/T2/gpt-4.1_False

  # Compare multiple models
  python example/02_run_evaluation/generate_report.py \
      --compare \
      --run_dirs "GPT-4.1=runs/T2/gpt-4.1_False" \
                 "GPT-4.1 CoT=runs/T2/gpt-4.1_True" \
                 "Deepseek=runs/T2/deepseek-chat_False" \
      --output_dir runs/comparison

  # Custom test-case paths
  python example/02_run_evaluation/generate_report.py \
      --run_dir runs/T2/gpt-4.1_False \
      --ranking_cases benchmark/data/evaluate/built/ranking/all_env_scenarios.json \
      --scoring_cases benchmark/data/evaluate/built/scoring/alphabench_testset.json
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.engine.evaluate import generate_report, compare_runs


_DEFAULT_RANKING_CASES = "benchmark/data/evaluate/built/ranking/all_env_scenarios.json"
_DEFAULT_SCORING_CASES = "benchmark/data/evaluate/built/scoring/alphabench_testset.json"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate T2 performance reports from existing result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single run
    p.add_argument("--run_dir", default=None,
                   help="Run directory containing ranking_results.json and/or "
                        "scoring_results.json.")

    # Multi-run comparison
    p.add_argument("--compare", action="store_true",
                   help="Generate a comparison table across multiple runs.")
    p.add_argument("--run_dirs", nargs="+", default=[],
                   metavar="LABEL=PATH",
                   help='Space-separated "Label=path" pairs for --compare mode.')

    # Shared options
    p.add_argument("--ranking_cases", default=_DEFAULT_RANKING_CASES,
                   help=f"Path to ranking test cases JSON. Default: {_DEFAULT_RANKING_CASES}")
    p.add_argument("--scoring_cases", default=_DEFAULT_SCORING_CASES,
                   help=f"Path to scoring test cases JSON. Default: {_DEFAULT_SCORING_CASES}")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (defaults to <run_dir>/report/ or runs/comparison/).")
    return p.parse_args()


def _parse_run_dirs(pairs: list[str]) -> dict[str, str]:
    """Parse 'Label=path' strings into {label: path}."""
    result = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"[WARN] Skipping malformed run_dir spec (expected 'Label=path'): {pair!r}")
            continue
        label, path = pair.split("=", 1)
        result[label.strip()] = path.strip()
    return result


def main():
    args = _parse_args()

    if args.compare:
        # ---- Multi-run comparison mode ----
        run_dirs = _parse_run_dirs(args.run_dirs)
        if not run_dirs:
            print("[ERROR] --compare requires at least one --run_dirs 'Label=path' entry.")
            sys.exit(1)

        output_dir = args.output_dir or "runs/comparison"
        print(f"[Compare] {len(run_dirs)} runs → {output_dir}")
        for label, path in run_dirs.items():
            print(f"  {label}: {path}")

        compare_runs(
            run_dirs=run_dirs,
            ranking_cases_path=args.ranking_cases,
            scoring_cases_path=args.scoring_cases,
            output_dir=output_dir,
            verbose=True,
        )

    else:
        # ---- Single run mode ----
        if not args.run_dir:
            print("[ERROR] Provide --run_dir <path> or use --compare mode.")
            sys.exit(1)
        if not os.path.isdir(args.run_dir):
            print(f"[ERROR] Run directory not found: {args.run_dir}")
            sys.exit(1)

        output_dir = args.output_dir or os.path.join(args.run_dir, "report")
        print(f"[Report] run_dir={args.run_dir}")
        print(f"         output ={output_dir}")

        result = generate_report(
            run_dir=args.run_dir,
            ranking_cases_path=args.ranking_cases,
            scoring_cases_path=args.scoring_cases,
            output_dir=output_dir,
            verbose=True,
        )

        print(f"\n[Done] Report files:")
        for k, v in result.get("output_paths", {}).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
