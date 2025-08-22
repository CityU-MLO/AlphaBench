import time
import json
import re

from jsonschema import validate, ValidationError
from pathlib import Path
import random

from tqdm import tqdm

import random
from agent.llm_client import call_llm
from agent.robust.valid import is_valid_template_expression
from agent.qlib_contrib.qlib_valid import test_qlib_operator
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from factors.lib.alpha158 import load_factors_alpha158

from api.factor_eval_client import (
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
Please generate N new factors in JSON list format. 

### Requirements:
1. **Output Format**
   - Return a JSON list. Each element should be an object with:
     - key: factor name (string, CamelCase, short but descriptive)
     - value: full Qlib expression (string)

2. **Allowed Variables**
   - Only use variables prefixed with `$`: $close, $open, $high, $low, $volume.

3. **Operators and Functions**
   - Use only CamelCase function names, Qlib style.
   - Supported operators:
     Add(x, y), Sub(x, y), Mul(x, y), Div(x, y)
     Power(x, y), Log(x), Abs(x), Sign(x), Delta(x, n)
     And(x, y), Or(x, y), Not(x)
     Exp(x), Sqrt(x), Tan(x)
     Greater(x, y), Less(x, y), Gt(x, y), Ge(x, y), Lt(x, y), Le(x, y), Eq(x, y), Ne(x, y)
   - Supported rolling functions (with integer window n > 0):
     Mean(x, n), Std(x, n), Var(x, n), Max(x, n), Min(x, n), Skew(x, n), Kurt(x, n),
     Sum(x, n), Med(x, n), Mad(x, n), Count(x, n),
     EMA(x, n), WMA(x, n), Corr(x, y, n), Cov(x, y, n),
     Slope(x, n), Rsquare(x, n), Resi(x, n)
   - Ranking & Conditional:
     Rank(x, n), Quantile(x, n), Ref(x, n), IdxMax(x, n), IdxMin(x, n),
     If(cond, x, y), Mask(cond, x), Clip(x, a, b)

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
        {{
            "Momentum20": {{"reason":..., "expr": "Div(Sub($close, Ref($close, 20)), Ref($close, 20))"},
            "VolumeVolatility10": {{"reason":..., "expr": "Std($volume, 10)"}
        }}
        
        """
    else:
        extra_instruction = """
        ### Output:
        - Return only the JSON list as specified, with exactly N factors
        Example:
        {{
            "Momentum20": "Div(Sub($close, Ref($close, 20)), Ref($close, 20))",
            "VolumeVolatility10": "Std($volume, 10)"
        }}
        """

    return SYSTEM_SEARCHER_PROMPT + extra_instruction


def _extract_expression(payload: Any) -> Optional[str]:
    """
    Extract a Qlib expression string from many possible shapes.
    Supported:
      - str
      - {"qlib_expression" | "expression" | "expr" | "value": str}
      - {"expression": {"template": str}}
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for k in ("qlib_expression", "expression", "expr", "value"):
            v = payload.get(k)
            if isinstance(v, str):
                return v.strip()
        exp = payload.get("expression")
        if isinstance(exp, dict):
            tpl = exp.get("template")
            if isinstance(tpl, str):
                return tpl.strip()
    return None


def _extract_reason(payload: Any) -> Optional[str]:
    """
    Extract an optional reasoning text if provided by the LLM.
    Looks for common keys and simple nested shapes.
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        for k in (
            "reason",
            "rationale",
            "explanation",
            "why",
            "analysis",
            "justification",
        ):
            v = payload.get(k)
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, dict):
                # e.g., {"reason": {"text": "..."}}
                text = v.get("text")
                if isinstance(text, str):
                    return text.strip()
    return None


def _normalize_llm_output(parsed_output: Any) -> List[Tuple[str, str, Optional[str]]]:
    """
    Normalize LLM output into a list of (name, expr, reason) tuples.

    Accepts:
      - {"Momentum20": "Div(...)", "StdVol": "Std(...)"}
      - [{"Momentum20": "Div(...)"}, {"StdVol": "Std(...)"}, ...]
      - [{"name": "Momentum20", "expression": "Div(...)", "reason": "..."} , ...]
      - "Div(...)"  -> becomes ("Factor_1", "Div(...)", None)
    """
    triplets: List[Tuple[str, str, Optional[str]]] = []

    def push(name_guess: Optional[str], payload: Any, idx: int):
        # Single-key dict like {"Momentum20": "Div(...)"}
        if isinstance(payload, dict) and "name" not in payload:
            if len(payload) == 1:
                (k, v), = payload.items()
                name = str(k)
                expr = _extract_expression(v)
                if expr:
                    reason = _extract_reason(v)  # usually none in this shape
                    triplets.append((name, expr, reason))
                return
            # general dict (maybe has expression/reason keys)
            expr = _extract_expression(payload)
            if expr:
                name = (name_guess or f"Factor_{idx}").strip()
                reason = _extract_reason(payload)
                triplets.append((name, expr, reason))
            return

        # Explicit "name"
        if isinstance(payload, dict) and "name" in payload:
            name = str(payload.get("name") or name_guess or f"Factor_{idx}").strip()
            expr = _extract_expression(payload)
            if expr:
                reason = _extract_reason(payload)
                triplets.append((name, expr, reason))
            return

        # Plain string expression
        if isinstance(payload, str):
            triplets.append(
                ((name_guess or f"Factor_{idx}").strip(), payload.strip(), None)
            )
            return

        # Fallback
        expr = _extract_expression(payload)
        if expr:
            triplets.append(
                (
                    (name_guess or f"Factor_{idx}").strip(),
                    expr,
                    _extract_reason(payload),
                )
            )

    if isinstance(parsed_output, dict):
        for i, (k, v) in enumerate(parsed_output.items(), 1):
            name = str(k).strip()
            expr = _extract_expression(v)
            if expr:
                reason = _extract_reason(v)
                triplets.append((name, expr, reason))
    elif isinstance(parsed_output, (list, tuple)):
        for i, item in enumerate(parsed_output, 1):
            if isinstance(item, dict) and "name" not in item and len(item) == 1:
                (k, v), = item.items()
                name = str(k).strip()
                expr = _extract_expression(v)
                if expr:
                    reason = _extract_reason(v)
                    triplets.append((name, expr, reason))
            else:
                push(name_guess=None, payload=item, idx=i)
    elif isinstance(parsed_output, str):
        triplets.append(("Factor_1", parsed_output.strip(), None))

    return triplets


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
    enable_reason: bool = True,
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
        "results": {name: expr, ...},   # backward-compatible mapping
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

    attempt = 0
    for attempt in range(1, max_try + 1):
        time.sleep(0.1)  # small backoff to mitigate rate limits

        response = call_llm(
            instruction,
            model=model,
            system_prompt=get_system_searcher_prompt(enable_reason=enable_reason),
            json_output=True,
            temperature=temperature,
            local=local,
            local_port=local_port,
        )

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
                f'Preferred format: {{"FactorName": "<QlibExpression>", ...}} '
                f"or a list of objects with fields name/expression[/reason]."
                f"{_anti_repeat_block()}"
            )
            continue

        # Normalize to (name, expr, reason?)
        items = _normalize_llm_output(parsed_output)
        if not items:
            last_error = "No factors could be parsed from your JSON."
            instruction = (
                f"Base instruction:\n{base_instruction}\n\n"
                f"The last output contained no usable factors.\n"
                f"Return a JSON object mapping factor names to Qlib expressions, "
                f"optionally with 'reason'. Example:\n"
                f'{{"Momentum20": "Div(Sub($close, Ref($close, 20)), Ref($close, 20))"}}\n'
                f"or\n"
                f'[{{"name": "Momentum20", "expression": "Div(...)", "reason": "..."}}]\n'
                f"{_anti_repeat_block()}"
            )
            continue

        # Validate & collect
        for raw_name, expr, reason in items:
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
            except Exception as e:
                if verbose:
                    print(f"[WARN] check_factor_via_api failed: {e}")
                continue

            if isinstance(result, dict) and result.get("success"):
                name = _unique_name((raw_name or "Factor").strip(), used_names)
                collected[name] = {"expression": expr, "reason": (reason or "").strip()}
                used_names.add(name)
                used_exprs.add(expr)
                expr_to_name[expr] = name

                if len(collected) >= N:
                    break  # early stop

        if len(collected) >= N:
            break  # done

        # Prepare next self-healing instruction
        instruction = (
            "Regenerate factors.\n"
            "Return ONLY JSON (object or list). Allowed vars: $close, $open, $high, $low, $volume. "
            "Use Qlib-style operators only. If available, include a short 'reason' per factor.\n"
            f"Target remaining: {max(0, N - len(collected))}.\n"
            f"{_anti_repeat_block()}"
        )

    # If overfilled (rare, but possible), sample down to N.
    if len(collected) > N:
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

    success = len(collected) >= N
    return {
        "success": success,
        "results": results_mapping,  # backward-compatible: {name: expr}
        "factors": factors_detailed,  # new: includes optional reasoning
        "trynum": attempt if attempt else 0,
    }


if __name__ == "__main__":

    # standard_factors, compile_factors = load_factors_alpha158(exclude_var="vwap")
    # parsed_factor_pool = [{"name": factor.get("name"), "expr": factor.get('qlib_expression_default')} for factor in compile_factors.values()]

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
    #     {"name": "factor1", "expr": "Rank($close, 20)"},
    #     {"name": "factor2", "expr": "Mean($volume, 10)"},
    #     {"name": "factor3", "expr": "Corr($close, $volume, 30)"},
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
