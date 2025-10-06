import time
import json
import re
from uuid import uuid4

from jsonschema import validate, ValidationError
from pathlib import Path
import random

from tqdm import tqdm

from agent.llm_client import call_llm
from agent.robust.valid import is_valid_template_expression
from agent.qlib_contrib.qlib_valid import test_qlib_operator
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from agent.prompts_qlib_instruction import QLIB_GENERATE_INSTRUCTION
from api.factor_eval_client import check_factor_via_api
from agent.qlib_contrib.qlib_expr_parsing import FactorParser, print_tree

DEFAULT_API_URL = "http://localhost:9888"
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

TAG_CATEGORY_PATH = "./factors/registry/factor_categories.json"
factor_categories_dict = json.loads(Path(TAG_CATEGORY_PATH).read_text())


headers = [
    "Please follow the instruction to generate.",
    "Kindly follow instructions and generate output.",
    "Please process as per the instruction below.",
    "Follow the below instruction and generate.",
    "Kindly generate according to the following instruction.",
    "Please complete the task as instructed below.",
    "Generate as per the instruction provided.",
    "Please proceed with generating according to instructions.",
    "Ensure to follow instruction and generate.",
    "Kindly follow the given instruction to produce output.",
]


def prepend_random_header(prompt):
    header = random.choice(headers)
    return f"{header}\n{prompt}"


SYSTEM_GENERATOR_PROMPT = f"""
You are an expert quantitative researcher who designs alpha factors for stock ranking. Please generate a new factor in JSON format with the following requirements:

1. The factor must be defined using a mathematical expression in a string template, provided under "expression".
2. Allowed Variables
- Use ONLY variables prefixed with `$`: $close, $open, $high, $low, $volume  
- Numeric constants are allowed (e.g., 1.0, 1e-12, 20).

3. Use only CamelCase function names and operators (Qlib style). Supported functions include: 
And please pay attention that all ( and ) are used correctly and closed properly. More details about operators:

4. You factor should be executed in 30 seconds, please control the complexity, never generate too deep and complex factor, the max depth should be 4.

{QLIB_GENERATE_INSTRUCTION}

4. Self check list:
- Variables ∈ {{$close,$open,$high,$low,$volume}} only  
- Functions all ∈ allowed list, with correct arity  
- All n parameters are positive integers  
- Parentheses close properly  
- Any Div uses denominator + 1e-12 when needed, no negative Sqrt


Output must be a single valid JSON object. Example:
"""


def get_system_prompt(enable_cot=False):
    """
    Get the system prompt for the LLM.
    
    Args:
        enable_cot (bool): Whether to enable Chain of Thought reasoning.
        
    Returns:
        str: The system prompt.
    """
    cot_field = (
        """"CoT": "This is your chain of thought thinking process. Limited length, don't write too much" """
        if enable_cot
        else ""
    )
    output_format = f"""
    {{
        "name": "meanreversion_short_term",
        {cot_field}
        "expression": "Div(Sum($close, 5), Sum($volume, 10))"
    }}"""

    system_prompt = SYSTEM_GENERATOR_PROMPT + output_format
    if enable_cot:
        return (
            system_prompt
            + "\n\nPlease use Chain of Thought reasoning to generate the factor, and write your thinking process in the 'CoT' field of the output JSON. The CoT progress includes (1) What sub item you want to use (2) How to combine them step by step (3) Format the final expression in the 'expression.template' field. Make sure don't write too much in CoT."
        )
    else:
        return system_prompt


# ──────────────────────────────────────────────────────────────────────────────
# 1. Single-instruction helper
# ──────────────────────────────────────────────────────────────────────────────
def call_gen_qlib_factors(
    instruction: str,
    model: str = "deepseek-chat",
    max_try: int = 5,
    avoid_repeat: bool = False,
    verbose: bool = False,
    debug_mode: bool = False,
    enable_cot: bool = False,
    temperature: float = 1.0,
    local: bool = False,
    local_port: int = 8000,
) -> Dict[str, Any]:
    """
    Generate a Qlib-executable factor from an LLM, validating and retrying
    if necessary.

    Returns
    -------
    dict with keys
    • success : bool
    • content : parsed factor dict   (when success=True)
               or last error string (when success=False)
    • trynum  : int   # how many attempts were actually made
    """
    base_instruction = instruction
    if avoid_repeat:
        instruction = prepend_random_header(instruction)

    last_response, last_error = None, None
    error_records = []
    for attempt in range(1, max_try + 1):
        time.sleep(0.1)  # Avoid rate limiting
        response = call_llm(
            instruction,
            model=model,
            system_prompt=get_system_prompt(enable_cot=enable_cot),
            json_output=True,
            temperature=temperature,
            local=local,
            local_port=local_port,
            service_provider="default",
        )

        # print(response)

        if verbose and debug_mode:
            print(f"[{attempt}/{max_try}] Raw LLM output:\n{response}\n")

        # ── JSON parsing ────────────────────────────────────────────────────
        try:
            # Clean up common LLM formatting like ```json ... ```
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # remove leading and trailing fences
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            cleaned = cleaned.strip()
            parsed_output = json.loads(cleaned)

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            instruction = (
                f"Previous output could not be parsed as JSON.\n"
                f"Error: {e}\nOutput was: {response}\n\n"
                f"Please fix and regenerate."
            )
            continue

        # ── Schema validation ───────────────────────────────────────────────
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "expression": {"type": "string"},
            },
            "required": ["name", "expression"],
        }

        try:
            validate(instance=parsed_output, schema=schema)
            expr = parsed_output["expression"]
            stack = []
            for ch in expr:
                if ch == "(":
                    stack.append(ch)
                elif ch == ")":
                    if not stack:
                        raise ValidationError(
                            "Unmatched closing parenthesis ) in expression"
                        )
                    stack.pop()
            if stack:
                raise ValidationError("Unmatched opening parenthesis ( in expression")

            # Depth check
            fs = FactorParser()
            ast = fs.parse(expr)
            depth = fs.get_complexity(ast)["depth"]
            if depth > 5:
                last_error = f"Too complex factor with depth {depth}"
                instruction = (
                    f"Previous output was too complex.\n"
                    f"Error: {last_error}\nOutput was: {json.dumps(parsed_output, indent=2)}\n\n"
                    f"Please simplify and regenerate."
                )
                error_records.append("COMPLEX")
                continue

        except ValidationError as e:
            last_error = f"Schema validation error: {e.message}"
            instruction = (
                f"Previous output did not conform to the required schema.\n"
                f"Error: {e.message}\nOutput was: {json.dumps(parsed_output, indent=2)}\n\n"
                f"Please fix and regenerate."
            )
            error_records.append("WRONG_SCHEMA")
            continue

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            instruction = (
                f"Previous output caused an unexpected error.\n"
                f"Error: {e}\nOutput was: {json.dumps(parsed_output, indent=2)}\n\n"
                f"Please fix and regenerate."
            )
            error_records.append("UNKNOWN_ERROR")
            continue

        # ── Qlib runtime check ────────────────────────────────────────
        expr_default = parsed_output["expression"]

        try:
            result = check_factor_via_api(expr_default)
        except Exception as e:
            if verbose:
                print(f"[WARN] check_factor_via_api failed: {e}")
            continue

        if verbose and debug_mode:
            print(f"[{attempt}/{max_try}] Qlib validation result: {result}")
        if result.get("success"):
            parsed_output["name"] = parsed_output["name"] + "_" + str(uuid4())[:8]
            # if verbose:
            #     print(f"[{attempt}/{max_try}] Factor generated successfully: {parsed_output['name']}")
            return {
                "success": True,
                "content": parsed_output,
                "trynum": attempt,
                "error_records": error_records,
            }
        else:
            try:
                qlib_msg = result["error_message"]
                error_records.append(result["error_type"])
            except Exception as e:
                qlib_msg = f"Unknown error during Qlib validation"
                error_records.append("UNKNOWN_ERROR")

        last_error = f"Qlib validation failed: {qlib_msg}"

        # ── Prepare next prompt (self-healing retry) ───────────────────────
        instruction = (
            f"Base instruction:\n{base_instruction}\n\n"
            f"The last generated factor was invalid: {last_error}\n\n"
            f"Last attempt:\n{parsed_output}\n\n"
            f"Please correct the issues and regenerate a valid factor."
        )
        last_response = response

    # ── Final failure ────────────────────────────────────────────────────────
    if verbose:
        print(
            f"[!] Failed to generate a valid factor after {max_try} attempts.\n"
            f"Last error: {last_error}\nLast response:\n{last_response}"
        )

    return {
        "success": False,
        "content": last_error or last_response,
        "trynum": max_try,
        "error_records": error_records,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-threaded batch helper
# ──────────────────────────────────────────────────────────────────────────────
def batch_call_gen_qlib_factors(
    instructions: List[str], num_workers: int = 4, verbose: bool = False, **kwargs
) -> List[Dict[str, Any]]:
    """
    Parallel version of `call_gen_qlib_factors`.

    Parameters
    ----------
    instructions : list[str]
        Prompts in the original order.
    num_workers  : int
        Thread pool size.
    verbose      : bool
        If True, show a tqdm progress bar.

    Other keyword arguments are passed straight through to
    `call_gen_qlib_factors`.

    Returns
    -------
    list[dict] –  Results in the same order as `instructions`.
    """
    results: List[Tuple[int, Dict[str, Any]]] = [None] * len(instructions)

    def _worker(idx: int, prompt: str) -> Tuple[int, Dict[str, Any]]:
        res = call_gen_qlib_factors(prompt, verbose=verbose, **kwargs)
        return idx, res

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_worker, i, instr): i for i, instr in enumerate(instructions)
        }

        iterator = tqdm(as_completed(futures), total=len(futures), disable=not verbose)
        for future in iterator:
            idx, result = future.result()
            results[idx] = result
            if verbose:
                iterator.set_description(f"Done {idx+1}/{len(futures)}")

    return results


if __name__ == "__main__":
    instructions = [
        "Design a factor that ranks stocks by z-scored 10-day cumulative return, then multiplies the rank by the inverse of rolling volume volatility (default return window = 10, volume vol window = 20).",
        "Generate a signal that outputs 1 if close is above its 30-day rolling maximum and today’s volume exceeds its 90th percentile, otherwise −1; the result is then standardized cross-sectionally each day (default price window = 30, volume window = 30).",
        "Create a factor equal to the beta of 15-day price returns regressed on 15-day volume changes, divided by the rolling standard deviation of those betas (default window = 15).",
        "Compute the percentage distance of close from its 60-day rolling median, winsorize at ±3σ, multiply by volume percentile rank, and finally rank the result cross-sectionally (default median window = 60).",
        "Construct an indicator that counts consecutive up closes longer than 3 days while volume is below its 20-day SMA; output the count divided by rolling ATR (default ATR window = 14).",
        "Return the skewness of 21-day close returns minus the skewness of 21-day log-volume changes, then divide by their pooled standard deviation (default window = 21).",
    ]
    for instruction in instructions:
        output = call_gen_qlib_factors(
            instruction,
            model="gemini-2.5-pro",
            verbose=True,
            max_try=5,
            debug_mode=True,
        )
        print(output)
