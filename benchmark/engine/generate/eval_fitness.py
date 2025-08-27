import ast
import json
from typing import List, Dict, Any, Tuple
import pickle
import os
from agent.llm_client import batch_call_llm

GENERATION_FACTOR_EVAL_PROMPTS = """
You are an expert quantitative researcher who evaluates alpha factor generation.

Your role is to judge whether the generated factor (in JSON format) faithfully follows the given instruction, 
regardless of whether the task is Text2Alpha generation (natural language → formula) or Directional Mining (generate diverse factors under a theme). 
Think step by step before giving the final judgement.

### Input
- Instruction: {instruction}
- Generated Factor (JSON): {factor}

### Evaluation Criteria
1. **Variable Validity**  
   Must only use allowed variables: $close, $open, $high, $low, $volume.  
   If any other variable appears, it is invalid.

2. **Operator Validity**  
   Must only use allowed operators/functions.  
   Must also satisfy any explicit requirements (e.g., Mean, Std, normalization, ratio, rolling window).  
   If operator usage is missing, extra, or wrong, it is invalid.

3. **Instruction Faithfulness**  
   Expression must reflect the intent of the instruction.  
   Examples:  
   - If instruction requires a 20-day moving average, check for `Mean(..., 20)`.  
   - If instruction requires normalization/z-score, check scaling by Std.  
   - If instruction requires ratio (e.g., high/low), check numerator and denominator match.  
   - If instruction asks for multiple factors under a theme (e.g., volatility), verify all generated factors belong to that theme.  
   Any deviation means invalid.

4. **JSON Format Consistency**  
   Output must be valid JSON with required keys (`expression.template`, `expression.parameters`).  
   If missing or malformed, it is invalid.

### Output
First, reason step by step to yourself (chain-of-thought).  
Then return the final decision in **strict JSON format**:

```json
{{
  "reason": "<short explanation of why it is correct or incorrect>",
  "result": "correct" or "incorrect"
}}
```
"""


def build_prompt(instruction: str, factor_json: Dict[str, Any]) -> str:
    """Build evaluation prompt for a single case"""
    return GENERATION_FACTOR_EVAL_PROMPTS.format(
        instruction=instruction, factor=json.dumps(factor_json, indent=2)
    )


def judge_batch(
    cases: List[Tuple[str, Dict[str, Any]]],
    model: str = "deepseek-chat",
    verbose: bool = True,
    num_workers=8,
) -> Tuple[List[str], float]:
    """
    Evaluate a batch of (instruction, factor_json) pairs in one shot using batch_call_llm.
    Returns:
        - List of "correct"/"incorrect"
        - Accuracy score
    """
    prompts = [build_prompt(inst, fac) for inst, fac in cases]

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
            results.append("correct" if "correct" in r["result"] else "incorrect")
        except Exception as e:
            print(f"Error parsing response: {resp}\nError: {e}")
            results.append("incorrect")

    return results, responses


def start_eval_fitness(
    instruction_dir="./runs/T1_Generate/instructions",
    outputs_dir="./runs/T1_Generate/outputs",
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
            cases, model="deepseek-chat", num_workers=8
        )

        # with open("./runs/T1_Generate/eval_fitness_results.pkl", "wb") as f:
        #     pickle.dump(results, f)
        os.makedirs(dump_path, exist_ok=True)
        with open(os.path.join(dump_path, "eval_fitness_results.pkl"), "wb") as f:
            pickle.dump(results, f)

        with open(os.path.join(dump_path, "eval_fitness_responses.pkl"), "wb") as f:
            pickle.dump(responses, f)


# Example usage
if __name__ == "__main__":

    # instruction_dir = "./runs/T1_Generate/instructions"
    # outputs_dir = "./runs/T1_Generate/outputs"
    start_eval_fitness()

    # print("Results:", results)
    # print("Accuracy:", acc)
