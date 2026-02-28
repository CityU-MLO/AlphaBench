import time
import json
import re

from jsonschema import validate, ValidationError
from pathlib import Path
import random

from tqdm import tqdm
import uuid

import random
from agent.llm_client import call_llm
from agent.robust.valid import is_valid_template_expression
from agent.qlib_contrib.qlib_valid import test_qlib_operator
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from factors.lib.alpha158 import load_factors_alpha158
from agent.prompts_qlib_instruction import QLIB_GENERATE_INSTRUCTION

from ffo.client.factor_eval_client import (
    FactorEvalClient,
    evaluate_factor_via_api,
    batch_evaluate_factors_via_api,
    check_factor_via_api,
)
import json
import random
import time
from typing import Any, Dict, List, Tuple, Optional, Set

DEFAULT_API_URL = "http://localhost:9888"
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

SYSTEM_SEARCHER_PROMPT = f"""
You are an expert quantitative researcher who designs alpha factors for stock ranking. 
Your job is a **search task**: plan → generate → evaluate heuristically → refine/prune → diversify, then output candidates.
Please generate N new factors in JSON list format. 

### Requirements:
1. **Output Format**
   - Return a JSON list. Each element should be an object with:
     - key: factor name (string, CamelCase, short but descriptive)
     - value: full Qlib expression (string)

2. **Allowed Variables**
   - Only use variables prefixed with `$`: $close, $open, $high, $low, $volume.

3. **Operators and Functions**
3. Use only CamelCase function names and operators (Qlib style). Supported functions include: 
And please pay attention that all ( and ) are used correctly and closed properly. More details about operators:

{QLIB_GENERATE_INSTRUCTION}

4. **Expression Rules**
   - Always use function calls (e.g., Div(x, y)) instead of arithmetic symbols (+, -, *, /).
   - Parentheses must always be properly closed.
   - Do not use invalid or undefined functions.
   - Ensure all rolling window parameters are positive integers.
   - No missing or NaN parameters.

5. **Factor Naming**
   - Each factor must have a unique and descriptive CamelCase name (e.g., Momentum20, VolumeSpikeRatio).
   - Names should briefly describe the intuition of the factor.

6. **Diversity**
   - The N generated factors should cover different ideas: momentum, mean reversion, volatility, volume dynamics, cross-variable relations, etc.

### Universal Search Process (algorithm-agnostic)
A. **Plan**
   - Decompose candidates into roles: **Core signal** (trend/mean-revert/range), **Normalizer** (vol/range/level), **Conditioner/Smoother** (volume, Rank, Mean).
   - Sketch 3–5 distinct blueprints before writing expressions.

B. **Generate**
   - For each blueprint, emit 1–2 concise expressions (2–4 ops main chain).
   - Align windows: prefer (short, long) pairs like (5–10, 30–60) to stabilize behavior.
   - Encourage price ⨁ volume combinations in part of the set.

C. **Heuristic Evaluate (no backtest here)**
   - Prefer normalized forms: Div(core, Add(Std(..., L) or Mean(..., L), 1e-12)).
   - Smooth noisy cores: Mean(..., short L) or Rank(...).
   - Avoid brittle constructs (e.g., deep nesting, stacked Power, Rank inside Rank inside Rank).

D. **Refine / Prune**
   - If two candidates share the same skeleton, keep the cleaner one.
   - Replace fragile pieces (e.g., raw Delta) with stabilized variants (e.g., Div(Delta, Add(Std, 1e-12))).

E. **Diversify**
   - Ensure at least one price+volume candidate and one range/volatility‑scaled candidate.
   - Vary operator families (Sub/Delta vs. Sub($high,$low); Std vs. Mean; Rank vs. Mean smoothers).

F. **Validate (pre‑output self‑check)**
   - Count equals **N**.
   - Only allowed variables/operators; windows in [2, 120].
   - All denominators have epsilon.
   - Depth ≤ 6; expressions are parsable Qlib style.
   - Names unique; expressions materially different from each other.
   
"""


def get_system_searcher_prompt(enable_reason) -> str:
    if enable_reason:
        extra_instruction = """
        7. **Reasoning**
        - If possible, include a short reasoning for each factor.
        - This can help understand the intuition behind the factor.
        - The reasoning should be a brief text explaining the factor's purpose or expected behavior.
        
        ### Output:
        - Return only the JSON list as specified, with exactly N factors
        Example:
        {{ "generated":
        [
            {{"name": "Momentum20", "reason": "Measures 20-day price momentum as percentage change.", "expression": "Div(Sub($close, Ref($close, 20)), Ref($close, 20))"}},
            {{"name": "VolumeVolatility10", "reason": "Captures short-term fluctuations in trading volume.", "expression": "Std($volume, 10)"}}
        ]
        }}"""

    else:
        extra_instruction = """
        ### Output:
        - Return only a JSON list with exactly N factors.
        - Each factor must be an object with two keys: "name" and "expression".

        Example:
        {{ "generated": 
        [
            {{"name": "Momentum20", "expression": "Div(Sub($close, Ref($close, 20)), Ref($close, 20))"}},
            {{"name": "VolumeVolatility10", "expression": "Std($volume, 10)"}}
        ]
        }}"""

    return SYSTEM_SEARCHER_PROMPT + extra_instruction


def _extract_expression(payload: Any) -> Optional[str]:
    """
    Extract a Qlib expression string from many possible shapes.
    Supported:
      - str
      - {"qlib_expression" | "expression" | "expression" | "value": str}
      - {"expression": {"template": str}}
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for k in ("qlib_expression", "expression", "expression", "value"):
            v = payload.get(k)
            if isinstance(v, str):
                return v.strip()
        exp = payload.get("expression")
        if isinstance(exp, dict):
            tpl = exp.get("template")
            if isinstance(tpl, str):
                return tpl.strip()
    return None


def _normalize_llm_output(parsed_output: dict) -> list[dict]:
    """
    Enforce the unified schema:
        {
          "generated": [
            {"name": str, "expression": str, "reason": Optional[str]},
            ...
          ]
        }
    Returns a cleaned list of items or [] if schema invalid.
    """
    if not isinstance(parsed_output, dict):
        return [], 0

    # hanlde case when single return:
    if (
        isinstance(parsed_output, dict)
        and "name" in parsed_output
        and "expression" in parsed_output
    ):
        cleaned = [
            {
                "name": parsed_output["name"],
                "expression": parsed_output["expression"],
                "reason": parsed_output.get("reason", ""),
            }
        ]
        accept_rate = 1.0 if cleaned else 0.0
        return cleaned, accept_rate

    items = parsed_output.get("generated")
    if not isinstance(items, list):
        return [], 0

    cleaned = []
    accept_rate = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        expr = it.get("expression")
        reason = it.get("reason", None)

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(expr, str) or not expr.strip():
            continue
        if reason is not None and not isinstance(reason, str):
            # If reason is non-string, drop it rather than failing the item
            reason = None

        cleaned.append(
            {
                "name": name.strip() + "_" + str(uuid.uuid4())[:6],
                "expression": expr.strip(),
                **({"reason": reason.strip()} if reason else {}),
            }
        )
    accept_rate = len(cleaned) / len(items)
    return cleaned, accept_rate


def _unique_name(base: str, used: Set[str]) -> str:
    """Ensure factor name is unique by appending suffixes if needed."""
    name = base or "Factor"
    if name not in used:
        return name
    n = 2
    while f"{name}_{n}" in used:
        n += 1
    return f"{name}_{n}"


# --- main ------------------------------------------------------------------


def call_qlib_search(
    instruction: str,
    model: str = "deepseek-chat",
    N: int = 10,
    max_try: int = 5,
    avoid_repeat: bool = False,
    verbose: bool = False,
    debug_mode: bool = False,
    temperature: float = 1.0,
    enable_reason: bool = False,
    local: bool = False,
    local_port: int = 8000,
) -> Dict[str, Any]:
    """
    Generate and validate up to N Qlib-executable factors via an LLM.
    Accumulates passing factors across attempts, deduplicates by expression,
    avoids repetition when requested, and preserves optional per-factor reasoning.

    Returns:
      {
        "success": bool,                # True iff we filled N
        "results": {name: expression, ...},   # backward-compatible mapping
        "factors": [                    # detailed list with reasoning
          {"name": str, "expression": str, "reason": str}, ...
        ],
        "trynum": int
      }
    """
    base_instruction = instruction
    # name -> {"expression": str, "reason": str}
    collected: Dict[str, Dict[str, str]] = {}
    used_exprs: Set[str] = set()
    used_names: Set[str] = set()
    expr_to_name: Dict[str, str] = {}  # for backfilling reasons on duplicates
    last_error: Optional[str] = None

    def _anti_repeat_block() -> str:
        if not avoid_repeat or not used_exprs:
            return ""
        expr_list = list(used_exprs)
        if len(expr_list) > 20:
            expr_list = expr_list[:20] + ["..."]
        return (
            "\nDo NOT repeat any of the following expressions already accepted:\n"
            + "\n".join(f"- {e}" for e in expr_list)
            + "\n"
        )

    min_N = int(N * 0.5) if N > 1 else 1
    attempt = 0
    quality_info = {}
    quality_info["request_num"] = 0
    quality_info["generated_num"] = 0
    quality_info["accepted_num"] = 0
    quality_info["first_accept_rate"] = []
    quality_info["error_record"] = []
    quality_info["output_format_error"] = 0
    for attempt in range(1, max_try + 1):
        time.sleep(0.5)  # small backoff to mitigate rate limits

        quality_info["request_num"] += 1
        response = call_llm(
            instruction,
            model=model,
            system_prompt=get_system_searcher_prompt(enable_reason=enable_reason),
            json_output=True,
            temperature=temperature,
            local=local,
            local_port=local_port,
            service_provider="default",
        )

        # print(response)
        if verbose and debug_mode:
            print(f"[{attempt}/{max_try}] Raw LLM output:\n{response}\n")

        # Parse JSON
        try:
            parsed_output = json.loads(response)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            instruction = (
                f"Your last output was not valid JSON.\n"
                f"Error: {e}\nOutput was: {response}\n\n"
                f"Please return a JSON object ONLY.\n"
                'Recall format: {"generated": [{"name": xxx, "expression": xxx, ..}, ...]}'
                f"one key is 'generated', and value is a list of objects with fields name/expression[/reason]."
                f"{_anti_repeat_block()}"
            )
            quality_info["output_format_error"] += 1
            continue

        # Normalize to (name, expr, reason?)
        items, accept_rate = _normalize_llm_output(parsed_output)
        quality_info["first_accept_rate"].append(accept_rate)

        if not items:
            last_error = "No factors could be parsed from your JSON."

            instruction = f"""
            Base instruction:
            {base_instruction}

            The last output contained no usable factors.
            Return a JSON object with a single key 'generated' that maps to a list of factor objects.
            Each object MUST have 'name' (string) and 'expression' (string); 'reason' is optional.
            Do not include any extra keys or text outside of valid JSON.

            Schema (informal):
            { '{ "generated": [ { "name": str, "reason"?: str, "expression": str }, ... ] }' if enable_reason else '{ "generated": [ { "name": str, "expression": str }, ... ] }' }

            {_anti_repeat_block()}
            """

            quality_info["output_format_error"] += 1
            continue

        # Validate & collect
        for record in items:
            quality_info["generated_num"] += 1

            raw_name, expr, reason = (
                record["name"],
                record["expression"],
                record.get("reason", ""),
            )
            expr = (expr or "").strip()
            if not expr:
                continue

            # Duplicate expression handling:
            if expr in used_exprs:
                # If we already have this expr but no reason stored, backfill if new reason exists.
                prev_name = expr_to_name.get(expr)
                if prev_name and reason and not collected[prev_name]["reason"]:
                    collected[prev_name]["reason"] = reason.strip()
                continue

            # Validate expression via API
            try:
                result = check_factor_via_api(expr)
                # check_factor_via_api returns List[Dict]; normalise to single dict
                if isinstance(result, list):
                    result = result[0] if result else {}

            except Exception as e:
                if verbose:
                    print(f"[WARN] check_factor_via_api failed: {e}")
                quality_info["error_record"].append((expr, "API_ERROR"))
                continue

            if isinstance(result, dict) and result.get("success"):
                name = _unique_name((raw_name or "Factor").strip(), used_names)
                collected[name] = {"expression": expr, "reason": (reason or "").strip()}
                used_names.add(name)
                used_exprs.add(expr)
                expr_to_name[expr] = name

                quality_info["accepted_num"] += 1
                if len(collected) >= N:
                    break  # early stop

            elif isinstance(result, dict) and not result.get("success"):
                quality_info["error_record"].append((expr, result))

        if len(collected) >= N:
            break  # done

        # Prepare next self-healing instruction
        instruction = (
            f"Origin instruction: {base_instruction}"
            "Regenerate factors.\n"
            "Return ONLY JSON (object or list). Allowed vars: $close, $open, $high, $low, $volume. "
            "Use Qlib-style operators only. If available, include a short 'reason' per factor.\n"
            f"Target remaining: {max(0, N - len(collected))}.\n"
            f"{_anti_repeat_block()}"
        )

    # If overfilled (rare, but possible), sample down to N.
    if len(collected) >= N:
        items = list(collected.items())
        sampled = random.sample(items, N)
        collected = dict(sampled)

        # Rebuild helper maps after sampling
        used_exprs = {v["expression"] for v in collected.values()}
        used_names = set(collected.keys())
        expr_to_name = {v["expression"]: k for k, v in collected.items()}

    # Build return payloads
    results_mapping = {name: info["expression"] for name, info in collected.items()}
    factors_detailed = [
        {"name": name, "expression": info["expression"], "reason": info["reason"] or ""}
        for name, info in collected.items()
    ]

    success = len(collected) >= min_N
    return {
        "success": success,
        "results": results_mapping,  # backward-compatible: {name: expr}
        "factors": factors_detailed,  # new: includes optional reasoning
        "trynum": attempt if attempt else 0,
        "quality": quality_info,
    }


if __name__ == "__main__":

    # standard_factors, compile_factors = load_factors_alpha158(exclude_var="vwap")
    # parsed_factor_pool = [{"name": factor.get("name"), "expression": factor.get('qlib_expression_default')} for factor in compile_factors.values()]

    # sample_n = 10
    # sample_factors = random.sample(list(parsed_factor_pool), sample_n)
    # factor_performance = batch_evaluate_factors_via_api(sample_factors)

    # filtered_performance = [{
    #     "name": result["name"],
    #     "exxpression": result["expression"],
    #     "metrics": {k: v for k, v in result["metrics"].items() if k in ["ir", "ic", "rank_ic", "rank_icir", "icir"]}} for result in factor_performance
    # ]

    # instructions = []
    # for result in filtered_performance:
    #     instruction = f"Improve this new Qlib factor based on the following performance metrics, we hope you can generate N=5 candidates:\n"
    #     instruction += f"Factor Name: {result['name']}\n"
    #     instruction += f"Expression: {result['exxpression']}\n"
    #     instruction += f"Metrics: {json.dumps(result['metrics'], indent=2)}\n"
    #     instructions.append(instruction)

    # for instruction in instructions:
    #     print(f"Generated instruction:\n{instruction}\n")
    #     result = call_qlib_search(instruction, model="deepseek-chat", N=5, verbose=True)
    #     if result["success"]:
    #         print(f"Generated factors: {json.dumps(result['content'], indent=2)}\n")
    #     else:
    #         print(f"Failed to generate factors: {result['content']}\n")

    sample_instruction = """
    Improve this new Qlib factor based on the following performance metrics, we hope you can generate N=5 candidates:
    Factor Name: KLEN
    Expression: ($high - $low) / $open
    Metrics: {
    "ic": -0.0014161287108436227,
    "icir": -1.7819754541390813,
    "ir": -0.11454971879720688,
    "rank_ic": -0.03059507872342784,
    "rank_icir": -5.622098781147838
    }    
    """

    # No reasoning, just generate factors
    result = call_qlib_search(
        sample_instruction,
        model="deepseek-chat",
        N=8,
        verbose=True,
        enable_reason=False,
    )
    print(result)

    # Enable reasoning, generate factors with explanations
    result = call_qlib_search(
        sample_instruction, model="deepseek-chat", N=8, verbose=True, enable_reason=True
    )
    print(result)

    # import pdb; pdb.set_trace()  # Debugging point to inspect the result
    # Example usage
    # instruction = "Generate a new Qlib factor for stock ranking."
    # result = call_qlib_search(instruction, model="deepseek-chat", N=5, verbose=True)
    # print(result)
    # Test single factor evaluation
    # test_expr = "Rank(Corr($close, $volume, 10), 252)"

    # print(f"\nChecking if test factor expression is valid")
    # check_expr_lists = ["Rank(Corr($close, $volume, 10), 252)",
    #                     "Rank(Correlation($close, $volume, 10), 252)",
    #                     "Rank(Corr($close, $volume), 252)"]
    # for expr in check_expr_lists:
    #     result = check_factor_via_api(expr)
    #     if not result["success"]:
    #         print(f"Factor expression is invalid: {result}")
    #     else:
    #         print(f"Factor expression is valid: {expr}")

    # print(f"\nEvaluating test factor: {test_expr}")

    # result = evaluate_factor_via_api(test_expr)
    # if result["success"]:
    #     print(f"Evaluation successful!")
    #     print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
    # else:
    #     print(f"Evaluation failed: {result.get('error', 'Unknown error')}")

    # # Test batch evaluation
    # test_factors = [
    #     {"name": "factor1", "expression": "Rank($close, 20)"},
    #     {"name": "factor2", "expression": "Mean($volume, 10)"},
    #     {"name": "factor3", "expression": "Corr($close, $volume, 30)"},
    # ]

    # print(f"\nBatch evaluating {len(test_factors)} factors...")
    # results = batch_evaluate_factors_via_api(test_factors)

    # for i, result in enumerate(results):
    #     if result.get("success", False):
    #         print(
    #             f"{test_factors[i]['name']}: Rank IC = {result['metrics']['rank_ic']:.4f}"
    #         )
    #     else:
    #         print(
    #             f"{test_factors[i]['name']}: Failed - {result.get('error', 'Unknown error')}"
    #         )
