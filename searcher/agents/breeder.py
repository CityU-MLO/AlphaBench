"""
Breeder agent for generating new factors via mutation and crossover.

Follows EvoAlpha's FactorBreeder pattern but simplified:
- Reuses AlphaBench's existing call_qlib_search for LLM interaction
- No LLM call logger (uses standard logging)
- No scheduler-executor 2-step workflow (single-step prompt)
- Builds mutation/crossover prompts inline (same style as EA_searcher.py)
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.schemas import (
    FactorCandidate,
    GenerationRequest,
    GenerationResult,
    SeedFactor,
)

import sys
from pathlib import Path

# Add AlphaBench root to path so we can import agent module
_ALPHABENCH_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ALPHABENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALPHABENCH_ROOT))

from agent.generator_qlib_search import call_qlib_search
from agent.llm_client import call_llm
from agent.prompts_qlib_instruction import QLIB_GENERATE_INSTRUCTION


@dataclass
class BreederConfig:
    """
    Configuration for breeder agent.

    Attributes:
        model_name: LLM model name
        api_key: API key
        base_url: API base URL
        temperature: Sampling temperature
        max_retries: Maximum retries per generation
        enable_reason: Whether to include reasoning in LLM output
        local: Use local model server
        local_port: Local model server port
    """
    model_name: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_retries: int = 3
    enable_reason: bool = True
    local: bool = False
    local_port: int = 8000


def _seed_block_json(seeds: List[SeedFactor]) -> str:
    """Build a compact JSON list with {name, expression, metrics} for LLM prompt."""
    compact = []
    for s in seeds:
        entry = {"name": s.name, "expression": s.expression}
        if s.metrics:
            entry["metrics"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in s.metrics.items()
                if k in ("ic", "rank_ic", "icir", "rank_icir")
            }
        compact.append(entry)
    return json.dumps(compact, ensure_ascii=False, separators=(",", ": "), indent=2)


class FactorBreeder:
    """
    Agent for generating new factors via LLM.

    Supports:
    - Mutation: Modify existing factors
    - Crossover: Combine two factors
    - Parallel generation via call_qlib_search
    """

    def __init__(self, config: BreederConfig, logger=None):
        self.config = config
        if logger is None:
            from ..utils.logger import SearchLogger
            logger = SearchLogger("Breeder")
        self.logger = logger

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate new factors based on request.

        Args:
            request: Generation request (mutation or crossover)

        Returns:
            Generation result with candidates
        """
        start_time = time.time()

        if request.operation == "mutation":
            result = self._generate_mutation(request)
        elif request.operation == "crossover":
            result = self._generate_crossover(request)
        else:
            raise ValueError(f"Unknown operation: {request.operation}")

        result.elapsed_time = time.time() - start_time
        status = "✓" if result.success else "✗"
        self.logger.info(
            f"{status} Generated {len(result.candidates)} factors via {request.operation}"
        )
        return result

    def _generate_mutation(self, request: GenerationRequest) -> GenerationResult:
        """Generate factors via mutation."""
        seed_block = _seed_block_json(request.seeds)
        n = request.num_factors
        round_num = request.context.get("round_num", 0)

        prompt = self._compose_mutation_prompt(seed_block, n, round_num)

        return self._call_search(request, prompt, n)

    def _generate_crossover(self, request: GenerationRequest) -> GenerationResult:
        """Generate factors via crossover."""
        seed_block = _seed_block_json(request.seeds)
        n = request.num_factors
        round_num = request.context.get("round_num", 0)

        prompt = self._compose_crossover_prompt(seed_block, n, round_num)

        return self._call_search(request, prompt, n)

    def _call_search(
        self, request: GenerationRequest, prompt: str, n: int
    ) -> GenerationResult:
        """Call the LLM search function and parse results."""
        try:
            result = call_qlib_search(
                instruction=prompt,
                model=self.config.model_name,
                N=n,
                verbose=False,
                temperature=request.temperature or self.config.temperature,
                enable_reason=self.config.enable_reason,
                local=self.config.local,
                local_port=self.config.local_port,
                max_try=self.config.max_retries,
            )

            if not result.get("success") and not result.get("factors"):
                return GenerationResult(
                    request=request,
                    candidates=[],
                    success=False,
                    error="LLM returned no usable factors",
                )

            # Convert to FactorCandidate objects
            candidates = []
            for f in result.get("factors", []):
                name = f.get("name", f"factor_{uuid.uuid4().hex[:6]}")
                expr = f.get("expression", "")
                if not expr:
                    continue
                candidates.append(
                    FactorCandidate(
                        name=name,
                        expression=expr,
                        reason=f.get("reason", ""),
                        doc_type="search",
                        meta={"operation": request.operation},
                        provenance={
                            "round": request.context.get("round_num"),
                            "batch_id": request.context.get("batch_id"),
                        },
                    )
                )

            return GenerationResult(
                request=request,
                candidates=candidates,
                success=len(candidates) > 0,
                attempts=result.get("trynum", 1),
            )

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return GenerationResult(
                request=request,
                candidates=[],
                success=False,
                error=str(e),
            )

    def _compose_mutation_prompt(self, seed_block: str, n: int, round_id: int) -> str:
        """Build mutation prompt — same style as EA_searcher.py."""
        return f"""You are a quantitative researcher. Your task is to **mutate** existing alpha factors.

Round: {round_id}
Goal: Propose **exactly {n}** mutated candidates that are likely to improve the information coefficient (IC) while remaining valid Qlib-style expressions.

Seed factors (JSON; each item has "name", "expression", "metrics"):
{seed_block}

What to do (Mutation):
- Tweak window lengths (e.g., 5→7, 10→12, 20→18) to control smoothness and responsiveness.
- Replace or insert nearby operators while preserving the core signal type (momentum / mean-reversion / volatility / liquidity).
- Normalize signals to reduce scale effects (e.g., divide by rolling Std or use Rank).
- Add light regularization tricks (e.g., small epsilon in denom, clipping via Min/Max) to improve numerical stability.
- Keep expressions parsable and balanced (all parentheses closed), and variables limited to: $close, $open, $high, $low, $volume.
- Do NOT invent new variables or unsupported ops.
- Try to use more diverse operators and window sizes than the seeds, don't only adjust parameters.

Examples (illustrative only; you must produce new ones):
- From: Mean(Sub($close, Ref($close, 1)), 10)
  To:   Div(Mean(Sub($close, Ref($close, 1)), 12), Add(Std($close, 60), 1e-12))
- From: Rank(Sub($high, $low))
  To:   Rank(Div(Sub($high, $low), Add(Mean(Sub($close, Ref($close, 1)), 20), 1e-12)))

Output format:
Return a JSON array of length {n}. Each item MUST be an object with:
  - "name": a short unique name (string)
  {"- 'reason': 1–2 sentences explaining the mutation (string)" if self.config.enable_reason else ''}
  - "expression": the full Qlib-style expression (string)

No extra text. Output ONLY the JSON array.
"""

    def _compose_crossover_prompt(self, seed_block: str, n: int, round_id: int) -> str:
        """Build crossover prompt — same style as EA_searcher.py."""
        return f"""You are a quantitative researcher. Your task is to **crossover** existing alpha factors.

Round: {round_id}
Goal: Propose **exactly {n}** crossover candidates by combining complementary parts of the seed expressions to improve robustness and IC.

Seed factors (JSON; each item has "name", "expression", "metrics"):
{seed_block}

What "crossover" means here
- **Pick good parts from good factors**: identify sub-expressions that plausibly drive performance (e.g., momentum cores, volatility/volume normalizers, range/volatility proxies, smoothers, gates/filters).
- **Recombine** complementary parts across seeds to form concise, novel expressions (not minor edits or concatenations).

How to identify & extract good parts
1) Rank seeds by metrics (prefer higher RankIC/ICIR and stability). Skim top seeds first.
2) Decompose expressions into roles:
   - Core signal (e.g., Sub/Delta/Range/Momentum on $close/$high/$low)
   - Normalizer (e.g., Std/Mean/Rank with safe epsilon in denominators)
   - Volume or regime component (e.g., Mean($volume, L), Rank(...))
   - Smoother (e.g., Mean(..., L), Rank(...))
3) Extract the **short, reusable subchains** (2–4 ops) that carry the behavior (trend, mean-revert, breakout) or the stabilizer (vol/volume scaling).

Examples (illustrative only; you must produce new ones):
- From A: Mean(Sub($close, Ref($close, 1)), 10)
  From B: Div($volume, Add(Mean($volume, 60), 1e-12))
  To:     Div(Mean(Sub($close, Ref($close, 1)), 12), Add(Mean($volume, 60), 1e-12))

- From A: Rank(Sub($high, $low))
  From B: Div(Sub($close, Ref($close, 1)), Add(Std($close, 30), 1e-12))
  To:     Rank(Div(Sub($high, $low), Add(Std($close, 30), 1e-12)))

Output format:
Return a JSON array of length {n}. Each item MUST be an object with:
  - "name": a short unique name (string)
  {"- 'reason': 1–2 sentences explaining the crossover (string)" if self.config.enable_reason else ''}
  - "expression": the full Qlib-style expression (string)

No extra text. Output ONLY the JSON array.
"""


def create_breeder(config_dict: Dict[str, Any], logger=None) -> FactorBreeder:
    """
    Factory function to create a FactorBreeder from config dict.

    Args:
        config_dict: Configuration dictionary with keys:
            - name: Model name
            - key: API key
            - base_url: API base URL
            - temperature: Sampling temperature
            - enable_reason: Whether to include reasoning
            - local: Use local model
            - local_port: Local model port
        logger: Optional logger

    Returns:
        Configured FactorBreeder instance
    """
    config = BreederConfig(
        model_name=config_dict.get("name", "deepseek-chat"),
        api_key=config_dict.get("key", ""),
        base_url=config_dict.get("base_url", ""),
        temperature=config_dict.get("temperature", 0.7),
        max_retries=config_dict.get("max_retries", 3),
        enable_reason=config_dict.get("enable_reason", True),
        local=config_dict.get("local", False),
        local_port=config_dict.get("local_port", 8000),
    )
    return FactorBreeder(config, logger=logger)
