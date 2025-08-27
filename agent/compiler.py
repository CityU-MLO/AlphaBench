import json
from agent.llm_client import call_llm
from agent.qlib_contrib.qlib_ops import operator_docs as QLIB_OPERATOR_DOCS
from agent.qlib_contrib.qlib_valid import test_qlib_operator
from agent.robust.valid import is_valid_template_expression
import re

COMPILE_QLIB_SYS_PROMPT = """
You are a compiler that converts financial factor expressions into Qlib-compatible Python expressions.
Only return the Qlib expression (no explanation or extra text).
Operators supported are:
Qlib Expression Rules:
- Supported variables: $open, $high, $low, $close, $volume
- Expressions are written in a functional style using provided operators
- Each operator has specific input argument rules (see operator_docs)

Examples:
1. Expression: "MA close in past 5 days"
   Output: Mean($close, 5)

2. Expression: "Diff between open and close prices"
   Output: Sub($open, $close)

3. Expression: "High price / low price"
   Output: Div($high, $low)

Operators supported are:
{operator_docs}

Please output in json with:
name: The factor name, copy from given factor record
qlib_expression: The Qlib-compatible expression string
qlib_expression_default: The default Qlib-compatible expression string with default parameters applied

Example:
{{
    "name": "VWAP_REF",
    "qlib_expression": "Ref($vwap, {{window}}) / $close",
    "qlib_expression_default": "Ref($vwap, 1) / $close"
}},
""".strip().format(
    operator_docs=QLIB_OPERATOR_DOCS
)


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


def compile_factor_to_qlib_expr(
    factor_record: dict,
    model="deepseek-chat",
    max_try=5,
    verbose=False,
    debug_mode=False,
) -> tuple[str, bool]:
    """
    Compile a factor expression using LLM, validate the template, apply default parameters,
    and test whether the final expression passes Qlib validation.

    Args:
        factor_record (dict): The factor definition with 'template' and 'parameters'.
        model (str): LLM model name.
        max_try (int): Number of attempts to get a valid compiled template from LLM.

    Returns:
        (str, bool): Final Qlib-compatible expression and whether it passed validation.
    """
    template = factor_record["expression"]["template"]
    param_specs = factor_record["expression"].get("parameters", {})
    expected_name = factor_record.get("name", "").strip()

    last_error = None
    compiled_data = None

    for attempt in range(1, max_try + 1):
        user_prompt = (
            f"Convert the following factor template into Qlib-compatible JSON format.\n"
            f"Template:\n{factor_record}"
        )

        if last_error:
            user_prompt += f"\n\nThe previous output was invalid due to: {last_error}\nPlease fix it."

        try:
            llm_output = call_llm(
                prompt=user_prompt,
                model=model,
                json_output=True,
                system_prompt=COMPILE_QLIB_SYS_PROMPT,
            ).strip()

            if verbose:
                print(f"[Attempt {attempt}] LLM output: {llm_output}")

            compiled_data = json.loads(llm_output)

            # Check required keys
            if not all(
                k in compiled_data
                for k in ("name", "qlib_expression", "qlib_expression_default")
            ):
                last_error = f"Missing required keys in output: {compiled_data.keys()}"
                if verbose:
                    print(f"[Attempt {attempt}] {last_error}")
                continue

            # Check name match (if expected provided)
            if expected_name and compiled_data["name"].strip() != expected_name:
                last_error = f"Name mismatch: expected '{expected_name}', got '{compiled_data['name']}'"
                if verbose:
                    print(f"[Attempt {attempt}] {last_error}")
                continue

            # Validate qlib_expression template
            is_valid = is_valid_template_expression(compiled_data["qlib_expression"])
            if not is_valid["valid"]:
                last_error = f"Invalid qlib_expression template: {is_valid['error']}"
                if verbose:
                    print(f"[Attempt {attempt}] {last_error}")
                continue

            # Apply default parameters to qlib_expression
            try:
                expr_applied = apply_parameters_to_template(
                    compiled_data["qlib_expression"], param_specs, parameters=None
                )
            except Exception as e:
                last_error = f"Failed to apply parameters: {e}"
                if verbose:
                    print(f"[Attempt {attempt}] {last_error}")
                continue

            # Check if applied expression matches qlib_expression_default
            if expr_applied != compiled_data["qlib_expression_default"]:
                last_error = (
                    f"Default expression mismatch.\n"
                    f"Expected: {compiled_data['qlib_expression_default']}\n"
                    f"Got: {expr_applied}"
                )
                if verbose:
                    print(f"[Attempt {attempt}] {last_error}")
                continue

            # Test qlib_expression_default
            qlib_status, qlib_msg = test_qlib_operator(
                compiled_data["qlib_expression_default"]
            )
            if not qlib_status:
                last_error = "qlib_expression_default failed Qlib validation."
                if verbose:
                    print(f"[Attempt {attempt}] {last_error}")
                continue

            # Success
            return compiled_data, True

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            print(f"[Attempt {attempt}] {last_error}")
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            print(f"[Attempt {attempt}] {last_error}")

    # If all attempts fail
    error_message = """LLM failed to produce a valid Qlib expression after {max_try} attempts.\n
        Last error: {last_error}"""
    return error_message, False
