#!/bin/bash
# ============================================================================
# AlphaBench Searcher — Run All Three Algorithms Sequentially
# ============================================================================
#
# This script runs EA, CoT, and ToT searches one after another,
# then runs test-period backtesting on each result.
#
# Prerequisites:
#   1. FFO server running at 127.0.0.1:19777
#   2. API keys set in environment or config/api_keys.yaml
#
# Usage:
#   cd AlphaBench/searcher
#   bash run_all_algos.sh
#
#   # Or with alpha158 warm start:
#   bash run_all_algos.sh --alpha158
#
#   # Or with a custom seed file:
#   bash run_all_algos.sh --seed-file seeds.txt
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXTRA_ARGS="$@"

echo "============================================================"
echo " AlphaBench — Running All Three Search Algorithms"
echo "============================================================"
echo ""

# ── 1. EA Search ────────────────────────────────────────────────────────────
# echo "============================================================"
# echo " [1/3] Running EA (Evolutionary Algorithm) ..."
# echo "============================================================"
# python start_search.py --config configs/ea_config.yaml $EXTRA_ARGS
# echo ""
# echo " EA search complete. Results in ./results/ea_search/"
# echo ""

# ── 2. CoT Search ──────────────────────────────────────────────────────────
echo "============================================================"
echo " [2/3] Running CoT (Chain-of-Thought) ..."
echo "============================================================"
python start_search.py --config configs/cot_config.yaml $EXTRA_ARGS
echo ""
echo " CoT search complete. Results in ./results/cot_search/"
echo ""

# ── 3. ToT Search ──────────────────────────────────────────────────────────
echo "============================================================"
echo " [3/3] Running ToT (Tree-of-Thought) ..."
echo "============================================================"
python start_search.py --config configs/tot_config.yaml $EXTRA_ARGS
echo ""
echo " ToT search complete. Results in ./results/tot_search/"
echo ""

# ── 4. Test-period backtesting on all results ──────────────────────────────
echo "============================================================"
echo " Running test-period backtesting on all results ..."
echo "============================================================"

for algo_dir in ea_search cot_search tot_search; do
    result_dir="./results/$algo_dir"
    if [ -f "$result_dir/final_pool.jsonl" ]; then
        echo ""
        echo "--- Test backtest: $algo_dir ---"
        python run_test_backtest.py \
            --results-dir "$result_dir" \
            --top-n 50 \
            --drop-k 5 \
            --n-jobs 4
    else
        echo "Skipping $algo_dir — no final_pool.jsonl found"
    fi
done

echo ""
echo "============================================================"
echo " All done! Results:"
echo "   EA:  ./results/ea_search/"
echo "   CoT: ./results/cot_search/"
echo "   ToT: ./results/tot_search/"
echo "============================================================"
