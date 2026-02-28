"""
Prompt builders for atomic evaluation tasks.

Task 1 – Pairwise Select:  Given expressions A and B, pick the better (non-noise) one.
Task 2 – Binary Noise:     Given an expression, classify as 'noise' or 'signal'.

Market-style guidance mirrors the rebuttal scripts:
    general  — no market disclosed
    us       — S&P 500 specific guidance
    cn       — China A-shares (CSI 300) specific guidance
    auto     — both style cards; model uses the "Market:" line to pick
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Shared market-style copy
# ---------------------------------------------------------------------------

_COMMON_FACTOR_GUIDE = """You are a quantitative research assistant evaluating equity alpha factor definitions.

Judging principles:
• Economic meaning: trend / mean-reversion / regime & volatility / liquidity & pressure / risk premia.
• Directionality: slope, sign, fast-vs-slow comparisons, asymmetric up/down behavior.
• Robustness: smoothing (EMA), stabilizers (epsilon), vol/turnover scaling, ranking/z-score; avoid 1-day spikes.
• Multi-horizon structure: interactions between fast/slow windows or multi-scale logic.
• Cross-sectional comparability: rank/z-score/vol-scaling to compare across names.
• Nonlinear / conditional logic: logs, clipping, gating, conditional combinations.
• Structured price–volume interactions (with smoothing), vol-of-vol, regime filters.

Mark as noise when:
• Pure algebra on raw OHLCV (Add/Sub/Mul/Div only) or a single rolling stat with no structure.
• Indicator clones (plain RSI/MACD-style) without normalization or meaningful interaction.
• Scale-dependent outputs (price/volume levels) without normalization or ranking.
• Microstructure artifacts (open/close spikes) with no smoothing.

Important: Do NOT decide solely by operator presence or expression length. Prioritize economic meaning, directionality, robustness, and multi-horizon structure."""

_US_MARKET_STYLE = """US large-cap (S&P 500) market context:
• Microstructure: decimalized quotes, no daily price limits, widespread short-selling, deep liquidity, pre/post-market sessions, common HFT activity.
• Typical pitfalls: 1-day gap/open artifacts; raw ranges without scaling; indicator clones; overfit microstructure wiggles.
• More credible patterns: multi-horizon momentum vs short-term reversal; regime filters (vol/turnover); ranking & volatility scaling; structured price–volume interactions with smoothing."""

_CN_MARKET_STYLE = """China A-shares (CSI 300) market context:
• Microstructure: daily price limits (≈±10%), midday break, T+1 selling rule, short-selling constraints, higher retail participation, frequent limit-up/down events.
• Typical pitfalls: raw gaps/limit touches without normalization; noon-break artifacts; scale effects without cross-sectional comparability.
• More credible patterns: regime-aware momentum/reversal that handle limits; turnover/volatility filters; rank/vol-scaling; fast/slow window interactions with robust smoothing."""


def _market_style_block(market_prompt: str) -> str:
    mp = (market_prompt or "auto").lower()
    if mp == "us":
        return "\n" + _US_MARKET_STYLE
    if mp == "cn":
        return "\n" + _CN_MARKET_STYLE
    if mp == "auto":
        return (
            "\nApply the relevant market context based on the 'Market:' line in the user message:\n"
            + _US_MARKET_STYLE + "\n" + _CN_MARKET_STYLE
        )
    return ""  # "general" — no market exposed


# ---------------------------------------------------------------------------
# Task 2: Binary Noise Classification
# ---------------------------------------------------------------------------

_NOISE_TASK_DESCRIPTION = """Task: Decide whether the factor expression below is a **noise signal** or a **signal**.

Return only:
- 'noise'   = uninformative / spurious signal unlikely to have repeatable alpha
- 'signal'  = plausibly informative factor with interpretable economic meaning"""

_NOISE_JSON_SCHEMA_COT = """
Output constraints: Return valid JSON only.

JSON schema (chain-of-thought):
{
  "analysis": "<concise reasoning, ≤80 words>",
  "prediction": "noise" | "signal"
}"""

_NOISE_JSON_SCHEMA_NOCOT = """
Output constraints: Return valid JSON only.

JSON schema:
{
  "prediction": "noise" | "signal"
}
Think silently. Do not include explanations."""


def build_noise_system_prompt(cot: bool, market_prompt: str = "auto") -> str:
    core = _COMMON_FACTOR_GUIDE + "\n\n" + _NOISE_TASK_DESCRIPTION
    core += _market_style_block(market_prompt)
    core += _NOISE_JSON_SCHEMA_COT if cot else _NOISE_JSON_SCHEMA_NOCOT
    return core


def build_noise_user_prompt(
    row: Dict[str, Any],
    cot: bool,
    market_prompt: str = "auto",
    max_expr_chars: int = 2000,
) -> str:
    expr = str(row.get("expression", ""))
    if len(expr) > max_expr_chars:
        expr = expr[:max_expr_chars] + " ...[truncated]"

    market = str(row.get("market", ""))
    window = row.get("window", {})
    wstr = ""
    if isinstance(window, dict):
        wstr = f'{window.get("start", "?")} → {window.get("end", "?")}'

    show_market = (market_prompt.lower() != "general")
    lines = ["You are evaluating whether this alpha factor is noise or signal:\n"]
    if show_market and market:
        lines.append(f"Market: {market}")
    if wstr:
        lines.append(f"Window: {wstr}")
    lines.append(f"\nExpression:\n{expr}\n")

    task_line = (
        'Reason in "analysis" first, then give a one-word "prediction" (noise or signal).'
        if cot else
        'Output only the required JSON with the "prediction" field.'
    )
    lines.append(task_line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1: Pairwise Factor Selection
# ---------------------------------------------------------------------------

_PAIRWISE_TASK_DESCRIPTION = """Task: Compare two equity alpha factor expressions and pick the **better** (more signal-like, less noise-like) one.

Return only:
- 'A' = Candidate A is better (more likely non-noise / informative)
- 'B' = Candidate B is better"""

_PAIRWISE_JSON_SCHEMA_COT = """
Output constraints: Return valid JSON only.

JSON schema (chain-of-thought):
{
  "analysis": "<concise reasoning comparing A vs B, ≤80 words>",
  "prediction": "A" | "B"
}"""

_PAIRWISE_JSON_SCHEMA_NOCOT = """
Output constraints: Return valid JSON only.

JSON schema:
{
  "prediction": "A" | "B"
}
Think silently. Do not include explanations."""


def build_pairwise_system_prompt(cot: bool, market_prompt: str = "auto") -> str:
    core = _COMMON_FACTOR_GUIDE + "\n\n" + _PAIRWISE_TASK_DESCRIPTION
    core += _market_style_block(market_prompt)
    core += _PAIRWISE_JSON_SCHEMA_COT if cot else _PAIRWISE_JSON_SCHEMA_NOCOT
    return core


def build_pairwise_user_prompt(
    row: Dict[str, Any],
    cot: bool,
    market_prompt: str = "auto",
    max_expr_chars: int = 2000,
) -> str:
    def _trunc(s: str) -> str:
        s = str(s or "")
        return s if len(s) <= max_expr_chars else s[:max_expr_chars] + " ...[truncated]"

    expr_a = _trunc(row.get("A", ""))
    expr_b = _trunc(row.get("B", ""))
    market = str(row.get("market", ""))
    window = row.get("window", {})
    wstr = ""
    if isinstance(window, dict):
        wstr = f'{window.get("start", "?")} → {window.get("end", "?")}'

    show_market = (market_prompt.lower() != "general")
    lines = ["You are helping decide which of two factor definitions is more likely a useful alpha (non-noise):\n"]
    if show_market and market:
        lines.append(f"Market: {market}")
    if wstr:
        lines.append(f"Window: {wstr}")

    lines.append("\nCandidate A\n-----------")
    lines.append(f"Expression:\n{expr_a}")
    lines.append("\nCandidate B\n-----------")
    lines.append(f"Expression:\n{expr_b}")
    lines.append("")

    task_line = (
        'Reason in "analysis" first, then give a one-letter "prediction" (A or B).'
        if cot else
        'Output only the required JSON with the "prediction" field.'
    )
    lines.append(task_line)
    return "\n".join(lines)
