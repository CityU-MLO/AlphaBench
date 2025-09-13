import ast
import json
from typing import List, Dict, Any, Tuple
import pickle
import os
from agent.llm_client import batch_call_llm

import json
from typing import Any, Dict

# --- Criteria blocks ---------------------------------------------------------

TEXT2ALPHA_CRITERIA = """
1. **Variable Validity**
   Must only use: $close, $open, $high, $low, $volume. Any other variable → invalid.

2. **Operator Validity (Strict)**
   Must only use allowed operators/functions. For example, all explicitly required ops must appear
   with correct arguments (e.g., `Mean(..., 20)` for a 20-day MA; z-score must divide
   by `Std(..., N)` or equivalent). No extra, missing, or wrong ops.

3. **Instruction Faithfulness (Strict)**
   The expression must exactly match the instruction’s intent:
   - Windows: exact (e.g., “20-day” ⇒ `..., 20`).
   - Direction/sign: match required ordering or sign.
   - Normalization/ratio: present when requested and applied to the specified terms.
   - Multi-factor requests: generate exactly the requested count and each must match
     its sub-instruction.
   Any deviation → invalid.
"""

DIRECTIONAL_MINING_CRITERIA = """
1. **Variable Validity**
   Must only use: $close, $open, $high, $low, $volume. Any other variable → invalid.

2. **Operator Validity (Relaxed)**
   Must only use allowed operators/functions and keep them sensible. Exact windows
   and precise parameter values may vary.

3. **Theme Consistency (Relaxed but Required)**
   The factor(s) must clearly belong to the requested theme, here is some examples:
   - Momentum/Trend: use returns, slope, ROC, moving averages, breakout/range signals.
   - Mean-reversion: z-scores vs. moving average, deviation-from-mean, reversal cues.
   - Volatility: Std/Var/ATR/high–low ranges; rolling dispersion.
   - Liquidity/Activity: volume/turnover/participation metrics.
   ...
   For rest of cases, you can analyze from the instruction.
   If multiple factors are requested under a theme, they should not be near-duplicates
   (e.g., different primary operator, window family, or normalization).
"""

# --- Template that injects the appropriate criteria block --------------------

GENERATION_FACTOR_EVAL_TEMPLATE = """
You are an expert quantitative researcher who evaluates alpha factor generation.

Your role is to judge whether the generated factor (in JSON format) faithfully follows the given instruction,
with task-specific criteria depending on the task type: {type_name}.
Think step by step privately before giving the final judgement.

### Input
- Task Type: {type_name}
- Instruction: {instruction}
- Generated Factor (JSON): {factor}

### Evaluation Criteriaå
{criteria_block}

### Output
First, think silently to yourself. Then return ONLY the final decision in strict JSON:

{{
  "reason": "<short explanation of why it is correct or incorrect>",
  "result": "correct" or "incorrect"
}}
""".strip()


def _criteria_for_type(task_type: str) -> str:
    t = (task_type or "").strip().lower()
    if t in {"text2alpha", "text-to-alpha", "text2alpha generation"}:
        return TEXT2ALPHA_CRITERIA
    if t in {"directional mining", "directional", "theme generation"}:
        return DIRECTIONAL_MINING_CRITERIA
    # Fallback: be conservative—default to Text2Alpha strictness
    return TEXT2ALPHA_CRITERIA


def build_prompt(type_name: str, instruction: str, factor_json: Dict[str, Any]) -> str:
    """Build evaluation prompt for a single case with task-specific criteria."""
    criteria_block = _criteria_for_type(type_name)
    # Normalize display name
    display_type = (
        "Text2Alpha" if criteria_block is TEXT2ALPHA_CRITERIA
        else "Directional Mining" if criteria_block is DIRECTIONAL_MINING_CRITERIA
        else (type_name or "Text2Alpha")
    )
    return GENERATION_FACTOR_EVAL_TEMPLATE.format(
        type_name=display_type,
        instruction=instruction,
        factor=json.dumps(factor_json, indent=2, ensure_ascii=False),
        criteria_block=criteria_block,
    )


def judge_batch(
    cases: List[Tuple[str, Dict[str, Any]]],
    model: str = "deepseek-chat",
    verbose: bool = True,
    type_name: str = "Text2Alpha",
    num_workers=8,
) -> Tuple[List[str], float]:
    """
    Evaluate a batch of (instruction, factor_json) pairs in one shot using batch_call_llm.
    Returns:
        - List of "correct"/"incorrect"
        - Accuracy score
    """
    prompts = [build_prompt(type_name, inst, fac) for inst, fac in cases]

    responses = batch_call_llm(
        prompts,
        model=model,
        verbose=verbose,
        json_output=True,
        temperature=0,
        num_workers=num_workers,
        service_provider='default'
    )

    results = []
    for resp in responses:
        try:
            r = resp.strip().lower()
            r = ast.literal_eval(r)
            results.append("incorrect" if "incorrect" in r["result"] else "correct")
        except Exception as e:
            print(f"Error parsing response: {resp}\nError: {e}")
            results.append("incorrect")

    return results, responses


def start_eval_fitness(
    instruction_dir="./runs/T1_Generate/instructions",
    outputs_dir="./runs/T1_Generate/outputs",
    model="deepseek-chat"
):
    ckp_file_prefix = [
        "T1_1_easy",
        "T1_1_hard",
        "T1_1_medium",
        "T1_2_easy",
        "T1_2_hard",
        "T1_2_medium",
    ]

    results = {}
    results_calc = {}

    responses = {}
    for prefix in ckp_file_prefix:
        if prefix not in ["T1_1_easy", "T1_1_medium", "T1_1_hard"]:
            task_name = "Text2Alpha"
        else:
            task_name = "Directional Mining"
            
        dump_path = os.path.join(outputs_dir, "scores")

        instruction_path = f"{instruction_dir}/{prefix}_instruction.pkl"
        output_path = f"{outputs_dir}/{prefix}_results.pkl"

        with open(instruction_path, "rb") as f:
            instructions = pickle.load(f)

        with open(output_path, "rb") as f:
            factors = pickle.load(f)

        cases = [
            (inst, fac["content"])
            for inst, fac in zip(instructions, factors)
            if fac["success"]
        ]
        print("Evaluating", len(cases), "cases for", prefix)
        results[prefix], responses[prefix] = judge_batch(
            cases, model=model, num_workers=8, type_name=task_name
        )

        os.makedirs(dump_path, exist_ok=True)
        with open(os.path.join(dump_path, f"eval_fitness_results_{model}.pkl"), "wb") as f:
            pickle.dump(results, f)

        with open(os.path.join(dump_path, f"eval_fitness_responses_{model}.pkl"), "wb") as f:
            pickle.dump(responses, f)


# Example usage
if __name__ == "__main__":

    # instruction_dir = "./runs/T1_Generate/instructions"
    # outputs_dir = "./runs/T1_Generate/outputs"
    start_eval_fitness()

    # print("Results:", results)
    # print("Accuracy:", acc)
