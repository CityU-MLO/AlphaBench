import argparse
import ast
import csv
from datetime import datetime
from itertools import count
import math
import os
import json
import pickle
import pandas as pd
import shutil

import numpy as np
from pathlib import Path
from agent.qlib_contrib.qlib_valid import test_qlib_operator
from agent.generator_qlib import call_gen_qlib_factors, batch_call_gen_qlib_factors

from agent.qlib_contrib.qlib_expr_parsing import FactorParser

from benchmark.engine.utils import (
    get_factor_performance_batch,
    similarity_factor_output,
)
from benchmark.engine.generate.eval_fitness import start_eval_fitness
from benchmark.engine.generate.eval_diversity import (
    eval_factor_ast_distance,
    eval_pairwise_factor_similarity,
)
from benchmark.engine.evaluate.factor_eval import (
    evaluate_performance_scoring,
    evaluate_performance_ranking,
)
import re

from agent.llm_client import batch_call_llm, call_llm
from factors.lib.alpha158 import load_factors_alpha158


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaBench T1 generation pipeline")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("./runs/T2_deepseek_cot"),
        help="Base directory to save instructions, outputs, and scores.",
    )
    parser.add_argument(
        "--enable_cot",
        action="store_true",
        help="Enable Chain-of-Thought generation for LLM calls.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--local_model",
        action="store_true",
        help="Use a local LLM server for generation.",
    )
    parser.add_argument(
        "--local_port",
        type=int,
        default=8000,
        help="Port number for the local LLM server.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


RANKING_SYSTEM_PROMPTS_BASE = """
You are an expert quantitative researcher. 
You are given N candidate alpha factors, each with an expression and a short description. 
Your task is to identify the top-k factors that are most likely to have true predictive ability 
for stock returns and perform well in trading.

Definition of a good factor:
- It should demonstrate strong predictive power, reflected by absolute higher Information Coefficient (IC) 
  and RankIC in backtests, notice in this case we care about absolute values, because no matter positive or negative, the valus is higer, the better performance.
- It should capture meaningful and economically intuitive patterns (e.g., momentum, reversal, volatility, liquidity).
- It should be robust and generalizable, avoiding spurious correlations or overfitting to noise.
- Its mathematical form should align with known signal categories and provide stable signals.

The given example is ID with their expressions:
-1: expr
-2: expr
....

"""

SCORING_SYSTEM_PROMPTS_BASE = """
You are an expert quantitative researcher. This is a **zero-shot factor scoring** task.
You will ONLY receive a factor expression (symbolic formula). You must infer its likely behavior
from the expression structure (operators, windows, normalization, delays), and output scores.

### Goal
For each factor expression, predict:
- "signal": "Positive" | "Negative" | "Noise"
- "scores": 1–5 for each dimension:
  • Performance  (higher expected predictive strength is better)
  • Stability    (lower variability / more robustness is better)
  • WinRate      (likelihood of correct directional calls is higher)
  • Skewness     (right-tail / positive-outlier tendency is better)

Output **ONLY** a single JSON object per factor with the schema below. No extra text.

### Assumptions & semantics
- Variables: $close, $open, $high, $low, $volume (daily).
- Lag/Delay: Delay(x, k) or Ref(x, k) uses k-day lag (k>0 means past). Any non-positive lag (k<=0) is data leakage → severe penalty.
- Ranking/Normalization: Rank(.), ZScore(.), Normalize(.), Division by Std/Vol are stabilizers.
- Windows: Larger windows imply smoother/stabler signals; overly small windows (<5) are noisy.
- Composition risk: Excessive nesting, unstable operations (Exp, Pow with large exponents, division by small Std) reduce Stability and WinRate.
- Economic archetypes (heuristics):
  • Momentum/Trend (e.g., MA cross, ROC, breakout, positive weights on recent returns) → Positive
  • Mean-reversion (e.g., raw ZScore(price vs MA), deviation-from-mean without a negative sign) → Negative
  • Volatility-carry (stable realized vol or ATR changes with sensible smoothing) → usually Positive
  • Pure price level, raw difference without horizon or smoothing → Noise (unless strongly structured)
  • If the expression is nearly constant or trivially transforms a constant → Noise

### Direction (signal) inference rules
Decide sign based on how the factor value co-moves with expected future return:
- If factor increases with recent positive returns (trend), eg. is -ZScore(deviation) for overbought/oversold buy-low logic → "Positive".
- If factor is raw overbought measure, eg. ZScore(close - MA_N) (high value = overbought → expect negative future return) → "Negative".
- If expression is ambiguous, level-only, or dominated by noise/unstable ops → "Noise".
- Any data leakage (Ref/Delay with k<=0) → set "signal"="Noise" and minimum scores.

### Dimension scoring (1–5) from expression heuristics
Score each dimension independently using these qualitative cues:

1) Performance (higher is better)
   + Higher: classic motifs (trend, mean-reversion, carry), sensible windows (10–60), use of ranking/normalization, combination with volume/vol for conditioning, simple & interpretable structure.
   – Lower: level-only, arbitrary composites with no economic story, extreme nonlinearity (Exp, high Pow), tiny windows, algebraic cancelations, obvious leakage flags.
   Heuristic tiers:
     5: canonical well-formed signal (e.g., multi-window momentum or robust z-scored reversal) with stabilization
     4: solid single-idea factor with some smoothing/normalization
     3: plausible but simplistic or slightly noisy
     2: weakly specified, likely low edge
     1: meaningless/unstable/leaky

2) Stability (higher is better when smoother/robust)
   + Higher: Rank/ZScore/Normalize, long windows, moving averages, EWMA, median filters, winsorization/clipping.
   – Lower: division by tiny Std, nested differences, tiny windows (<5), Exp/Pow explosions, many unbounded mult/div chains.
   Map: very smooth & normalized = 5; moderate smoothing = 3–4; noisy/unstable = 1–2.

3) WinRate (higher is better)
   Proxy from construction:
   + Higher: stable signals with incremental edges (trend with smoothing, carry), rank-based thresholds, ensemble/averaging → more frequent small wins.
   – Lower: tail-dependent or trigger-based signals (hard thresholds, breakout-only) → lower hit-rate but bigger payoffs.
   Map: smoothed trend/carry with rank/zscore = 4–5; noisy thresholds = 2–3; chaotic = 1.

4) Skewness (higher prefers right-tail outcomes)
   + Higher: breakout/regime/trend-follow signals that occasionally capture large moves; conservative normalization limiting downside.
   – Lower: mean-reversion with crash risk, leverage via division/Exp causing left tails.
   Map: breakout/trend with controls = 4–5; balanced smooth factors = 3; reversal with downside/tight division = 1–2.

### Hard penalties
- Any lookahead/leakage (Ref/Delay k<=0, using future bars) → "signal":"Noise"; all scores=1.
- Expressions that are constant or nearly constant → "Noise"; all scores=1.


"""


# Example: build a CoT-enabled system prompt by replacing the Instructions block
def get_system_prompts_ranking(enable_CoT=False) -> str:
    if enable_CoT:
        instructions = """
        Instructions:
        1. Analyze each factor’s mathematical structure and description, judging whether it encodes a valid trading signal.
        2. Evaluate the predictive potential by considering:
        - Signal type and financial intuition (does it make sense economically?).
        - Likelihood of stable predictive power across different regimes.
        - Complexity and risk of overfitting (simpler and interpretable forms are preferred if effective).
        3. Compare factors against each other and determine which ones are most likely to generate reliable alpha.
        4. The final output must be exactly k factor IDs, formatted as a JSON list, e.g.: [1, 3, 4, 5].
        5. Think step-by-step and reason about the factors before making a decision, here is steps:
        - First, I will analyze each factor's mathematical structure and description.
        - Then, I will evaluate the predictive potential by considering the signal type and financial intuition.
        - Next, I will compare factors against each other and determine which ones are most likely to generate reliable alpha.
        - Finally, I will output the factor IDs that I believe have predictive power and are the strongest among the set.
        6. Please only return the JSON in above format
        {{
            "analysis": here is your analysis steps.
            "results": [1, 2, 3, 4]
        }}
        """
    else:
        instructions = """
        Instructions:
        1. Analyze each factor’s mathematical structure and description, judging whether it encodes a valid trading signal.
        2. Evaluate the predictive potential by considering:
        - Signal type and financial intuition (does it make sense economically?).
        - Likelihood of stable predictive power across different regimes.
        - Complexity and risk of overfitting (simpler and interpretable forms are preferred if effective).
        3. Compare factors against each other and determine which ones are most likely to generate reliable alpha.
        4. Only return the factor IDs that you believe have predictive power and are the strongest among the set.
        5. The final output must be exactly k factor IDs, formatted as a JSON list, e.g.: [1, 3, 4, 5].
        6. Please only return the JSON list of IDs, without any additional text or explanation, output:
        {{
            "results": [1, 2, 3, 4]
        }}
        """
    return RANKING_SYSTEM_PROMPTS_BASE + instructions


def get_system_prompts_scoring(enable_CoT=False) -> str:
    if enable_CoT:
        instructions = """
        ### Think then output JSON only
        Steps:
        1) Parse the expression: list operators, windows, lags, normalizers; detect leakage.
        2) Classify archetype (trend / reversal / volatility / volume-conditional / other).
        3) Decide signal direction from archetype and signs.
        4) Assign each score 1–5 using the mapping above, citing 1–2 key cues per dimension.
        5) Emit a SINGLE JSON object exactly matching the schema. No extra text.
        Don't write too long

        ### Output
        For each factor, output a JSON object with:
        {{
        "analysis": Please write your analysis steps here.
        "signal": "Positive" | "Negative" | "Noise"
        "scores": {
            "Performance": int (1–5),
            "Stability": int (1–5),
            "WinRate": int (1–5),
            "Skewness": int (1–5)
        }
        }}

        The scores must reflect **relative ranking among all factors in the same environment** (quintiles).  
        Do not generate explanations, only structured JSON output.
        """
    else:
        instructions = """
        ### Output
        For each factor, output a JSON object with:
        {{
        "signal": "Positive" | "Negative" | "Noise"
        "scores": {
            "Performance": int (1–5),
            "Stability": int (1–5),
            "WinRate": int (1–5),
            "Skewness": int (1–5)
        }}

        The scores must reflect **relative ranking among all factors in the same environment** (quintiles).  
        Do not generate explanations, only structured JSON output.
        """
    return SCORING_SYSTEM_PROMPTS_BASE + instructions


def benchmark_ranking_performance(
    data_path="./benchmark/data/evaluation",
    output_path="./runs/T2_Evaluate",
    enable_CoT=False,
    model="gpt-4.1",
    local=False,
    local_port=8000,
    num_workers=4,
):
    factor_lib_path = os.path.join(data_path, "factor_pool.json")
    with open(factor_lib_path, "r") as f:
        factor_lib = json.load(f)

    test_case_path = os.path.join(data_path, "all_env_scenarios.json")
    with open(test_case_path, "r") as f:
        test_cases = json.load(f)

    # Start build instruction to evaluate
    instructions = []
    for case in test_cases:
        case_type = case["scenario"]
        market_environment = case["environment"]
        factors = {name: factor_lib[name] for name in case["factors"]}
        instruction = "Now we have the following factors:\n"

        for idx, (name, factor) in enumerate(factors.items(), start=1):
            instruction += f" - {idx}: {factor}\n"

        if market_environment == "bear":
            instruction += "The market is bearish, so we expect factors that perform well in down markets."
        elif market_environment == "bull":
            instruction += "The market is bullish, so we expect factors that perform well in up markets."
        else:
            instruction += "We hope that factors can perform well in both up and down markets or neutral scenarios."

        instruction += f"Please rank these factors based on their expected performance in the {case_type} scenario."
        instructions.append(instruction)

    results = batch_call_llm(
        instructions,
        model=model,  # replace with the model you want
        num_workers=num_workers,
        system_prompt=get_system_prompts_ranking(enable_CoT=enable_CoT),
        json_output=True,  # output as JSON
        verbose=True,  # show progress bar
        latency=0.5,
        temperature=0.5,
        local=local,
        local_port=local_port,
    )

    with open(os.path.join(output_path, "ranking_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def benchmark_scoring_performance(
    data_path="./benchmark/data/evaluation",
    output_path="./runs/T2_Evaluate",
    enable_CoT=False,
    model="gpt-4.1",
    local=False,
    local_port=8000,
    num_workers=4,
):
    factor_lib_path = os.path.join(data_path, "factor_pool.json")
    with open(factor_lib_path, "r") as f:
        factor_lib = json.load(f)

    test_case_path = os.path.join(data_path, "alphabench_testset.json")
    with open(test_case_path, "r") as f:
        test_cases = json.load(f)

    # Start build instruction to evaluate
    instructions = []
    for case in test_cases["items"]:
        pass
        env = case["env"]
        factor_name = case["factor"]
        factor_expr = factor_lib[factor_name]

        if "Bear" in env:
            instruction = f"Please evaluate the factor '{factor_name}' with expression '{factor_expr}' in the Bear market environment. "
        elif "Bull" in env:
            instruction = f"Please evaluate the factor '{factor_name}' with expression '{factor_expr}' in the Bull market environment. "
        else:
            instruction = f"Please evaluate the factor '{factor_name}' with expression '{factor_expr}' in the long term or normal environment. "

        instructions.append(instruction)

    results = batch_call_llm(
        instructions,
        model=model,  # replace with the model you want
        num_workers=num_workers,
        system_prompt=get_system_prompts_scoring(enable_CoT=enable_CoT),
        json_output=True,  # output as JSON
        verbose=True,  # show progress bar
        latency=0.5,
        temperature=0.5,
        local=local,
        local_port=local_port,
    )

    with open(os.path.join(output_path, "scoring_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # import pdb; pdb.set_trace()


if __name__ == "__main__":

    args = parse_args()

    model = args.model
    local = args.local_model
    local_port = args.local_port
    enable_CoT_status = args.enable_cot
    save_dir = args.base_dir
    num_workers = 8

    print(f"Running benchmark for model: {model}, enable_CoT: {enable_CoT_status}")
    outdir = os.path.join(save_dir, f"{model}_{enable_CoT_status}")
    os.makedirs(outdir, exist_ok=True)
    benchmark_ranking_performance(
        model=model,
        output_path=outdir,
        enable_CoT=enable_CoT_status,
        local=local,
        local_port=local_port,
        num_workers=num_workers,
    )
    benchmark_scoring_performance(
        model=model,
        output_path=outdir,
        enable_CoT=enable_CoT_status,
        local=local,
        local_port=local_port,
        num_workers=num_workers,
    )

    print("Complete all benchmark.")
