def is_valid_template_expression(template: str):
    """
    Validates a factor expression template with strict and detailed error reporting.
    Catches:
    - Unmatched or unbalanced parentheses
    - Unmatched, nested, or malformed curly braces
    """
    # 1. Check parentheses balance
    paren_count = 0
    for idx, char in enumerate(template):
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
            if paren_count < 0:
                return {"valid": False, "error": f"Unmatched ')' at position {idx}"}
    if paren_count != 0:
        return {
            "valid": False,
            "error": f"Mismatched parentheses: {paren_count} unclosed '('",
        }

    # 2. Check for valid and non-nested curly braces using a state machine
    i = 0
    while i < len(template):
        if template[i] == "{":
            start = i
            end = template.find("}", i)
            if end == -1:
                return {"valid": False, "error": f"Unmatched '{{' at position {start}"}
            if "{" in template[start + 1 : end]:
                return {"valid": False, "error": f"Nested '{{' at position {start}"}
            if "}" in template[end + 1 : end + 2]:
                return {"valid": False, "error": f"Extra '}}' at position {end+1}"}
            i = end  # skip to closing brace
        elif template[i] == "}":
            # stray closing brace
            if i == 0 or template[i - 1] != "}":
                return {"valid": False, "error": f"Unmatched '}}' at position {i}"}
        i += 1

    return {"valid": True}


if __name__ == "__main__":
    import pandas as pd

    print("Check validation of template expressions:")
    # Define a list of test cases with expected outcomes
    test_cases = [
        # Valid cases
        ("DIV(SUB($close, MEAN($close, {window_1})), STD($close, {window_2}))", True),
        ("ADD($open, $close)", True),
        ("IF(GT($close, {threshold}), $volume, 0)", True),
        ("MEAN(ABS(SUB($high, $low)), {window})", True),
        ("MASK(GT($volume, {min_vol}), SUM($close, {window}))", True),
        # Invalid parentheses
        (
            "DIV(SUB($close, MEAN($close, {window_1})), STD($close, {window_2})",
            False,
        ),  # missing closing )
        ("ADD($open, $close))", False),  # extra closing )
        ("(MUL($close, $volume)", False),  # missing closing )
        # Invalid braces
        ("MEAN($close, {{window}})", False),  # nested braces
        ("SUM($close, {window)", False),  # missing closing }
        ("SUB($close, window})", False),  # missing opening {
        ("DIV($close, {window_1}})", False),  # extra }
        # Mixed issues
        ("DIV((($close), {window}", False),  # unbalanced both
        ("IF(GT($close, {threshold}), $volume, 0", False),  # missing closing )
    ]

    # Run validation function on each test case
    test_results = [
        {"template": t, "expected": e, "result": is_valid_template_expression(t)}
        for t, e in test_cases
    ]
    print(pd.DataFrame(test_results).to_string(index=False))
