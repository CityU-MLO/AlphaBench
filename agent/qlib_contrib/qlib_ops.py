from collections import OrderedDict
import pandas as pd

ops_list = [
    "ChangeInstrument",
    "Rolling",
    "Ref",
    "Max",
    "Min",
    "Sum",
    "Mean",
    "Std",
    "Var",
    "Skew",
    "Kurt",
    "Med",
    "Mad",
    "Slope",
    "Rsquare",
    "Resi",
    "Rank",
    "Quantile",
    "Count",
    "EMA",
    "WMA",
    "Corr",
    "Cov",
    "Delta",
    "Abs",
    "Sign",
    "Log",
    "Power",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Greater",
    "Less",
    "And",
    "Or",
    "Not",
    "Gt",
    "Ge",
    "Lt",
    "Le",
    "Eq",
    "Ne",
    "Mask",
    "IdxMax",
    "IdxMin",
    "If",
    "Feature",
    "PFeature",
    "TResample",
]

categories = OrderedDict(
    [
        (
            "Element-wise (Unary)",
            {"Abs", "Sign", "Log", "Not", "Mask", "ChangeInstrument"},
        ),
        (
            "Pair-wise (Binary)",
            {
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Power",
                "Gt",
                "Ge",
                "Lt",
                "Le",
                "Eq",
                "Ne",
                "Greater",
                "Less",
                "And",
                "Or",
            },
        ),
        ("Triple-wise", {"If"}),
        (
            "Rolling (Single Series)",
            {
                "Rolling",
                "Ref",
                "Max",
                "Min",
                "Sum",
                "Mean",
                "Std",
                "Var",
                "Skew",
                "Kurt",
                "Med",
                "Mad",
                "Slope",
                "Rsquare",
                "Resi",
                "Rank",
                "Quantile",
                "Count",
                "EMA",
                "WMA",
                "Delta",
                "IdxMax",
                "IdxMin",
            },
        ),
        ("Rolling (Pair Series)", {"Corr", "Cov"}),
        ("Time-Aware", {"TResample"}),
        ("Base", {"Feature", "PFeature"}),
    ]
)

# Categorize any uncategorized ones
categorized_ops = set()
for ops in categories.values():
    categorized_ops |= ops
uncategorized_ops = set(ops_list) - categorized_ops


# Generate structured documentation templates for all operators
operator_docs = []

template_map = {
    "Element-wise (Unary)": "Unary operator: {name}(x) – {desc}",
    "Pair-wise (Binary)": "Binary operator: {name}(x, y) – {desc}",
    "Triple-wise": "Ternary operator: {name}(cond, x, y) – {desc}",
    "Rolling (Single Series)": "Rolling operator: {name}(x, N) – {desc}",
    "Rolling (Pair Series)": "Pair rolling operator: {name}(x, y, N) – {desc}",
    "Time-Aware": "Time-aware operator: {name}(x, freq, func) – {desc}",
    "Base": "Base expression: {name} – {desc}",
}

description_stub = {
    "Abs": "Returns absolute value.",
    "Sign": "Returns the sign (-1, 0, +1).",
    "Log": "Natural logarithm.",
    "Not": "Bitwise NOT (logical NOT).",
    "Mask": "Applies instrument mask.",
    "ChangeInstrument": "Switches feature calculation to another instrument.",
    "Add": "Addition.",
    "Sub": "Subtraction.",
    "Mul": "Multiplication.",
    "Div": "Division.",
    "Power": "Exponentiation (x^y).",
    "Gt": "x > y comparison.",
    "Ge": "x >= y comparison.",
    "Lt": "x < y comparison.",
    "Le": "x <= y comparison.",
    "Eq": "x == y equality check.",
    "Ne": "x != y inequality check.",
    "Greater": "Element-wise maximum.",
    "Less": "Element-wise minimum.",
    "And": "Logical AND.",
    "Or": "Logical OR.",
    "If": "Returns x if cond is true, else y.",
    "Rolling": "Base rolling operator.",
    "Ref": "Shift by N steps.",
    "Max": "Rolling max.",
    "Min": "Rolling min.",
    "Sum": "Rolling sum.",
    "Mean": "Rolling mean (moving average).",
    "Std": "Rolling standard deviation.",
    "Var": "Rolling variance.",
    "Skew": "Rolling skewness.",
    "Kurt": "Rolling kurtosis.",
    "Med": "Rolling median.",
    "Mad": "Rolling mean absolute deviation.",
    "Slope": "Rolling linear regression slope.",
    "Rsquare": "Rolling R^2 from regression.",
    "Resi": "Rolling residuals from regression.",
    "Rank": "Rolling percentile rank.",
    "Quantile": "Rolling quantile.",
    "Count": "Rolling count of non-NaNs.",
    "EMA": "Exponential moving average.",
    "WMA": "Weighted moving average.",
    "Delta": "Difference from N steps ago.",
    "IdxMax": "Rolling index of max value.",
    "IdxMin": "Rolling index of min value.",
    "Corr": "Rolling correlation.",
    "Cov": "Rolling covariance.",
    "TResample": "Resample time series to given frequency.",
    "Feature": "Basic expression for a feature.",
    "PFeature": "Point-in-time aware feature expression.",
}

for category, ops in categories.items():
    if not ops:
        continue
    for op in sorted(ops):
        desc = description_stub.get(op, "No description available.")
        doc_line = template_map[category].format(name=op, desc=desc)
        operator_docs.append(f"- {doc_line}")
