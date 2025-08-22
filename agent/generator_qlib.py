import time
import json
import re

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

1. The factor must be defined using a mathematical expression in a string template, provided under "expression.template".
2. Allowed variables (must be prefixed with '$') include: $close, $open, $high, $low, $volume.
3. Use only CamelCase function names and operators (Qlib style). Supported functions include: Mean, Std, Max, Min, Ref, Abs, Rank, Sum, Delay.
And please pay attention that all ( and ) are used correctly and closed properly. More details about operators:

Arithmetic & Logical Operators:
NOTICE x and y are two variable
    Add(x, y), Sub(x, y), Mul(x, y), Div(x, y)
    Power(x, y), Log(x), Abs(x), Sign(x), Delta(x, n)
    And(x, y), Or(x, y), Not(x)

Special Meth Operator:
Exp(x), Sqrt(x), Tan(x)

Get max or min item between two variable (x, y): Greater(x, y), Less(x, y)
Compare two variable (x, y), return bool: Gt(x, y): x > y, Ge(x, y): x ≥ y, Lt(x, y): x < y, Le(x, y): x ≤ y, Eq(x, y): x == y, Ne(x, y): x ≠ y

Rolling Statistical Functions: 
NOTICE where n is the time period length in rolling, is parameter
    Mean(x, n), Std(x, n), Var(x, n)
    Max(x, n), Min(x, n) (Notice this is rolling get max/min value in last n steps, don't use it to compare two variable)
    Skew(x, n), Kurt(x, n)
    Sum(x, n), Med(x, n), Mad(x, n), Count(x, n)
    EMA(x, n), WMA(x, n)
    Corr(x, y, n), Cov(x, y, n)
    Clip(x, a, b): Clip x to [a, b] (if a or b is None, it means no limit for that side)

Regression & Decomposition:
For above, notice never use N<=0, it is forbidden.
    Slope(x, n), Rsquare(x, n), Resi(x, n)
    Ranking & Quantile:
    Rank(x, n), Quantile(x, n)
    Index & Conditional Logic:
    Ref(x, n): value of x n steps ago
    IdxMax(x, n), IdxMin(x, n): index of max/min in last n steps
    If(cond, x, y): if condition is true, return x; else return y
    Mask(cond, x): x where cond is true, otherwise NaN

For arithmetic operations, do NOT use symbols. Instead, use:
    Add for +, Sub for -, Mul for *, Div for /

5. Any parameter in the expression must be written as {{param_name}}. You must define all such parameters in "expression.parameters".
6. In "expression.parameters", specify each parameter's:
   - type: "int" or "float"
   - range: [min_value, max_value]
   - default: default_value
   - never: Don't use any NaN in parameter field
   
   
7. Do NOT include "param_config". All necessary defaults must be in "expression.parameters".
8. Make sure all parameters used in the expression are defined, and vice versa.

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
        """"CoT": "This is your chain of thought thinking process." """
        if enable_cot
        else ""
    )
    output_format = f"""
    {{
        "name": "meanreversion_short_term",
        {cot_field}
        "expression": {{
            "template": "Div(Sum($close, {{windows_1}}), Sum($volume, {{windows_2}}))",
            "parameters": {{
                "window_1": {{"type": "int", "range": [3, 60], "default": 10}},
                "window_2": {{"type": "int", "range": [3, 60], "default": 15}}
            }}
        }}
    }}"""

    system_prompt = SYSTEM_GENERATOR_PROMPT + output_format
    if enable_cot:
        return (
            system_prompt
            + "\n\nPlease use Chain of Thought reasoning to generate the factor, and write your thinking process in the 'CoT' field of the output JSON. The CoT progress includes (1) What sub item you want to use (2) How to combine them step by step (3) Format the final expression in the 'expression.template' field. Make sure don't write too much in CoT."
        )
    else:
        return system_prompt


# Template schema to validate against
factor_schema = {
    "type": "object",
    "required": ["name", "expression"],
    "properties": {
        "name": {"type": "string"},
        "expression": {
            "type": "object",
            "required": ["template", "parameters"],
            "properties": {
                "template": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                            "type": "object",
                            "required": ["type", "range", "default"],
                            "properties": {
                                "type": {"type": "string", "enum": ["int", "float"]},
                                "range": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "default": {"type": "number"},
                            },
                        }
                    },
                },
            },
        },
    },
}


def validate_all(parsed_output):
    # JSON schema check
    try:
        validate(instance=parsed_output, schema=factor_schema)
    except ValidationError as e:
        return False, f"Schema validation failed: {e.message}"

    # Expression template check
    from json import JSONDecodeError

    try:
        template = parsed_output["expression"]["template"]
    except (KeyError, TypeError):
        return False, "Missing or malformed 'expression.template'"

    result = is_valid_template_expression(template)
    if not result["valid"]:
        return False, f"Expression template error: {result['error']}"

    # Parameter consistency check
    template_params = set(re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", template))
    defined_params = set(parsed_output["expression"]["parameters"].keys())
    if template_params != defined_params:
        missing = template_params - defined_params
        extra = defined_params - template_params
        error_parts = []
        if missing:
            error_parts.append(f"Missing parameter definitions: {sorted(missing)}")
        if extra:
            error_parts.append(f"Unused parameters: {sorted(extra)}")
        return False, "; ".join(error_parts)

    return True, None


def apply_parameters_to_template(
    template: str, param_specs: dict, parameters: dict = None
) -> str:
    """
    Validate and apply parameters to the factor template.

    Args:
        template (str): The factor expression template with {param} placeholders.
        param_specs (dict): Dict of param_name → {type, range, default}.
        parameters (dict, optional): Optional overrides.

    Returns:
        str: The filled-in template string.
    """
    final_params = {}

    if parameters is None:
        parameters = {}

    for name, spec in param_specs.items():
        if name in parameters:
            value = parameters[name]
            expected_type = spec["type"]
            if expected_type == "int" and not isinstance(value, int):
                raise TypeError(
                    f"Parameter '{name}' must be int, got {type(value).__name__}"
                )
            if expected_type == "float" and not isinstance(value, (int, float)):
                raise TypeError(
                    f"Parameter '{name}' must be float, got {type(value).__name__}"
                )
            if "range" in spec:
                low, high = spec["range"]
                if not (low <= value <= high):
                    raise ValueError(
                        f"Parameter '{name}' must be in range [{low}, {high}], got {value}"
                    )
            final_params[name] = value
        else:
            final_params[name] = spec["default"]

    # Check for extra parameters not in spec
    for name in parameters:
        if name not in param_specs:
            raise ValueError(
                f"Unexpected parameter '{name}' not defined in the factor spec."
            )

    try:
        filled = template.format(**final_params)
    except KeyError as e:
        raise ValueError(f"Missing parameter for formatting: {e}")

    return filled


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
        )

        if verbose and debug_mode:
            print(f"[{attempt}/{max_try}] Raw LLM output:\n{response}\n")

        # ── JSON parsing ────────────────────────────────────────────────────
        try:
            parsed_output = json.loads(response)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            instruction = (
                f"Previous output could not be parsed as JSON.\n"
                f"Error: {e}\nOutput was: {response}\n\n"
                f"Please fix and regenerate."
            )
            continue

        # ── Schema / expression validation ─────────────────────────────────
        is_valid, error_msg = validate_all(parsed_output)
        if not is_valid:
            last_error = f"Schema/expr validation failed: {error_msg}"

        else:
            # ── Qlib runtime check ────────────────────────────────────────
            expr_default = apply_parameters_to_template(
                parsed_output["expression"]["template"],
                parsed_output["expression"]["parameters"],
            )

            qlib_status, qlib_msg = test_qlib_operator(
                expr_default, verbose=verbose, timeout=90
            )

            if qlib_status:
                return {"success": True, "content": parsed_output, "trynum": attempt}
            last_error = f"Qlib validation failed: {qlib_msg}"

        # ── Prepare next prompt (self-healing retry) ───────────────────────
        instruction = (
            f"Base instruction:\n{base_instruction}\n\n"
            f"The last generated factor was invalid: {last_error}\n\n"
            f"Last attempt:\n{json.dumps(parsed_output, indent=2)}\n\n"
            f"Please correct the issues and regenerate a valid factor."
        )
        last_response = response

    # ── Final failure ────────────────────────────────────────────────────────
    if verbose:
        print(
            f"[!] Failed to generate a valid factor after {max_try} attempts.\n"
            f"Last error: {last_error}\nLast response:\n{last_response}"
        )

    return {"success": False, "content": last_error or last_response, "trynum": max_try}


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
