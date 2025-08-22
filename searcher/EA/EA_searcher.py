import os
import time
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
import uuid

from factors.lib.alpha158 import load_factors_alpha158
from agent.generator_qlib_search import call_qlib_search
from api.factor_eval_client import batch_evaluate_factors_via_api

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_metric(f: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(f.get("metrics", {}).get(key, default))


def _format_metrics(f: Dict[str, Any]) -> Dict[str, float]:
    # Keep the schema flexible: include what exists, with common aliases
    m = f.get("metrics", {}) or {}
    out = {}
    for k in ("ic", "rank_ic", "icir", "rank_icir", "winrate", "stability"):
        if k in m:
            out[k] = m[k]
    return out


def _rank_by_ic(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(pool, key=lambda x: _safe_metric(x, "ic", -1e9), reverse=True)


def _select_seed_pool(
    pool: List[Dict[str, Any]], top_k: int = 12
) -> List[Dict[str, Any]]:
    ranked = _rank_by_ic(pool)
    return ranked[: max(1, min(top_k, len(ranked)))]


def _seed_block_json(seeds: List[Dict[str, Any]]) -> str:
    """Builds a compact JSON list with {name, expression, metrics}."""
    compact = []
    for s in seeds:
        compact.append(
            {
                "name": s.get("name"),
                "expression": s.get("expr"),  # <- keep key "expression" for LLM
                "metrics": _format_metrics(s),
            }
        )
    return json.dumps(compact, ensure_ascii=False, separators=(",", ": "), indent=2)


class EA_Searcher:
    """
    Evolutionary Algorithm (EA) style factor searcher.

    Differences from your previous version:
      - Removed the contextual instruction summary: we only run MUTATION and CROSSOVER.
      - The prompts sent to the LLM are full text blocks (triple-quoted), not concatenated pieces.
      - Each prompt embeds an explicit JSON seed block with fields: name, expression, metrics.
      - Prompts include concise "what to do" guidelines PLUS concrete examples.
      - Keeps the pool size constant (equal to the initial pool size) while iterating.

    Required external callables:
      - batch_evaluate_factors_via_api(factors: List[Dict[str,str]]) -> List[Dict]
      - call_qlib_search(instruction: str, **kwargs) -> Dict[str, Any]
    """

    def __init__(
        self,
        *,
        batch_evaluate_factors_fn,
        search_fn,
        model: str = "deepseek-chat",
        temperature: float = 1.0,
        enable_reason: bool = True,
        local: bool = False,
        local_port: int = 8000,
        save_dir: str = "./runs/ea_search",
        seeds_top_k: int = 12,  # how many seeds to show the LLM each round
    ) -> None:
        self.batch_evaluate_factors_fn = batch_evaluate_factors_fn
        self.search_fn = search_fn

        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.local = bool(local)
        self.local_port = int(local_port)

        self.save_dir = save_dir
        self.seeds_top_k = int(seeds_top_k)
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------ Public API -------------------------------- #
    def search_population(
        self,
        pool: List[Dict[str, str]],
        *,
        mutation_rate: float,
        crossover_rate: float,
        N: int,
        rounds: int,
        verbose: bool = True,
        save_pickle: bool = True,
    ) -> Dict[str, Any]:
        """
        Evolutionary search on a factor pool.

        Args:
            pool: initial factor pool as [{"name": str, "expr": str}]
            mutation_rate: fraction of N assigned to mutation (0~1)
            crossover_rate: fraction of N assigned to crossover (0~1)
            N: total new candidates to request from LLM each round
            rounds: number of EA iterations
        """
        # Evaluate initial pool
        if verbose:
            print("Evaluating initial pool …")
        perf = self.batch_evaluate_factors_fn(pool)
        for i, p in enumerate(perf):
            pool[i]["metrics"] = p.get("metrics", {})

        current_pool = _rank_by_ic(pool)
        baseline_ic = sum(_safe_metric(f, "ic", 0.0) for f in current_pool) / max(
            len(current_pool), 1
        )
        if verbose:
            print(f"Baseline mean IC: {baseline_ic:.6f}")

        history = []

        for r in range(1, rounds + 1):
            if verbose:
                print("\n" + "=" * 80)
                print(f"Round {r}/{rounds}")

            # --- Select seed factors (top by IC) and embed metrics ---
            seeds = _select_seed_pool(current_pool, top_k=self.seeds_top_k)
            seed_block = _seed_block_json(seeds)

            # Split budget
            n_mut = max(0, int(N * mutation_rate))
            n_cross = max(0, int(N * crossover_rate))
            # Adjust to exactly N
            while n_mut + n_cross < N:
                n_mut += 1
            while n_mut + n_cross > N and n_cross > 0:
                n_cross -= 1

            # --- Build prompts (full text blocks) ---
            mut_prompt = self._compose_mutation_prompt(seed_block, n_mut, round_id=r)
            cross_prompt = self._compose_crossover_prompt(
                seed_block, n_cross, round_id=r
            )

            if verbose:
                print("\n[MUTATION PROMPT]\n", mut_prompt)
                print("\n[CROSSOVER PROMPT]\n", cross_prompt)

            # --- Call LLM for mutation ---
            mut_candidates = []
            if n_mut > 0:
                start_time = time.time()
                mut_payload = self.search_fn(
                    instruction=mut_prompt,
                    model=self.model,
                    N=n_mut,
                    verbose=verbose,
                    temperature=self.temperature,
                    enable_reason=self.enable_reason,
                    local=self.local,
                    local_port=self.local_port,
                )
                mut_escaped = time.time() - start_time
                mut_candidates = self._extract_candidates(
                    mut_payload, provenance="mutation"
                )

            # --- Call LLM for crossover ---
            cross_candidates = []
            if n_cross > 0:
                start_time = time.time()
                cross_payload = self.search_fn(
                    instruction=cross_prompt,
                    model=self.model,
                    N=n_cross,
                    verbose=verbose,
                    temperature=self.temperature,
                    enable_reason=self.enable_reason,
                    local=self.local,
                    local_port=self.local_port,
                )
                cross_escaped = time.time() - start_time
                cross_candidates = self._extract_candidates(
                    cross_payload, provenance="crossover"
                )

            candidates = mut_candidates + cross_candidates

            def get_unique_set(factor_pool):
                unique_candidates = []
                seen_exprs = set()

                for cand in factor_pool:
                    expr = cand["expr"]
                    if expr not in seen_exprs:
                        unique_candidates.append(cand)
                        seen_exprs.add(expr)
                return unique_candidates

            # candidates = unique_candidates
            unique_candidates = get_unique_set(candidates)
            # --- Evaluate new candidates ---
            eval_input = [
                {"name": c["name"], "expr": c["expr"]}
                for c in unique_candidates
                if c.get("expr")
            ]
            results = self.batch_evaluate_factors_fn(eval_input)
            for i, res in enumerate(results):
                unique_candidates[i]["metrics"] = res.get("metrics", {})

            # --- Update pool: keep constant size (same as initial) ---
            # Keep the best from (old + new)
            combined = current_pool + unique_candidates

            combined = get_unique_set(combined)

            combined = _rank_by_ic(combined)
            current_pool = combined[: len(pool)]

            history.append(
                {
                    "round": r,
                    "mutation_prompt": mut_prompt,
                    "crossover_prompt": cross_prompt,
                    "mutation_candidates": mut_candidates,
                    "crossover_candidates": cross_candidates,
                    "cross_escaped_time": cross_escaped if n_cross > 0 else None,
                    "mut_escaped_time": mut_escaped if n_mut > 0 else None,
                    "candidates": candidates,
                    "unique_candidates": unique_candidates,
                    "pool_size": len(current_pool),
                }
            )

            if verbose:
                top_ic = (
                    _safe_metric(current_pool[0], "ic", float("nan"))
                    if current_pool
                    else float("nan")
                )
                mean_ic = sum(_safe_metric(f, "ic", 0.0) for f in current_pool) / max(
                    len(current_pool), 1
                )
                print(f"Round {r} — top IC: {top_ic:.6f}, mean IC: {mean_ic:.6f}")

        summary = {
            "baseline_ic": baseline_ic,
            "history": history,
            "final_pool": current_pool,
        }

        if save_pickle:
            fname = os.path.join(self.save_dir, f"ea_search_{int(time.time())}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(summary, f)
            summary["save_path"] = fname
        return summary

    # ----------------------- Prompt builders (full text) ----------------------- #
    def _compose_mutation_prompt(self, seed_block: str, n: int, round_id: int) -> str:
        """
        A single self-contained text block. Includes:
          - What you get (seed list with name/expression/metrics)
          - What to do (mutation rules)
          - Output schema + constraints
          - Examples
        """
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

Examples (illustrative only; you must produce new ones):
- From: Mean(Sub($close, Ref($close, 1)), 10)
  To:   Div(Mean(Sub($close, Ref($close, 1)), 12), Add(Std($close, 60), 1e-12))
- From: Rank(Sub($high, $low))
  To:   Rank(Div(Sub($high, $low), Add(Mean(Sub($close, Ref($close, 1)), 20), 1e-12)))

Output format:
Return a JSON array of length {n}. Each item MUST be an object with:
  - "name": a short unique name (string)
  - "expression": the full Qlib-style expression (string)
  - "reason": 1–2 sentences explaining the mutation (string)

No extra text. Output ONLY the JSON array.
"""

    def _compose_crossover_prompt(self, seed_block: str, n: int, round_id: int) -> str:
        """
        A single self-contained text block. Includes:
          - What you get (seed list with name/expression/metrics)
          - What to do (crossover rules)
          - Output schema + constraints
          - Examples
        """
        return f"""You are a quantitative researcher. Your task is to **crossover** existing alpha factors.

Round: {round_id}
Goal: Propose **exactly {n}** crossover candidates by combining complementary parts of the seed expressions to improve robustness and IC.

Seed factors (JSON; each item has "name", "expression", "metrics"):
{seed_block}

What to do (Crossover):
- Fuse complementary operator chains (e.g., momentum core + volatility normalization).
- Align and blend window sizes; when merging, prefer stable combos like (short, long) = (5–10, 30–60).
- Combine volume- and price-based components to reduce overfitting to one modality.
- Keep expressions concise and valid; avoid excessively deep nesting or unsupported ops.
- Variables allowed: $close, $open, $high, $low, $volume.

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
  - "expression": the full Qlib-style expression (string)
  - "reason": 1–2 sentences explaining which traits were combined and why (string)

No extra text. Output ONLY the JSON array.
"""

    # ------------------------------ Internals --------------------------------- #
    def _extract_candidates(
        self, payload: Any, provenance: str
    ) -> List[Dict[str, Any]]:
        """
        Accepts two shapes (both common in LLM tooling):
          1) {"factors": [{"name":..., "expression":..., "reason":...}, ...]}
          2) Direct JSON array: [{"name":..., "expression":..., "reason":...}, ...]
        """
        out: List[Dict[str, Any]] = []
        items = []
        if (
            isinstance(payload, dict)
            and "factors" in payload
            and isinstance(payload["factors"], list)
        ):
            items = payload["factors"]
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and "choices" in payload:
            # Some clients return raw text JSON in choices[0].message.content
            try:
                text = payload["choices"][0]["message"]["content"]
                items = json.loads(text)
            except Exception:
                items = []
        for it in items:
            name = it.get("name")
            expr = it.get("expression") or it.get("expr")
            if not (name and expr):
                continue
            random_suffix = str(uuid.uuid4())[:6]
            out.append(
                {
                    "name": name + "_" + random_suffix,
                    "expr": expr,
                    "reason": it.get("reason", ""),
                    "provenance": provenance,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Example main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load factor pool
    standard_factors, compile_factors = load_factors_alpha158(
        exclude_var="vwap", collection=["kbar", "rolling"]
    )
    print(
        "Load {} standard factors and {} compiled factors.".format(
            len(standard_factors), len(compile_factors)
        )
    )

    parsed_factor_pool = [
        {"name": f.get("name"), "expr": f.get("qlib_expression_default")}
        for f in compile_factors.values()
    ]

    # Evaluate baseline metrics (IC, RankIC, etc.) via your API
    perf = batch_evaluate_factors_via_api(parsed_factor_pool)
    for i, p in enumerate(perf):
        parsed_factor_pool[i]["metrics"] = p.get("metrics", {})

    # Build and run EA searcher
    searcher = EA_Searcher(
        batch_evaluate_factors_fn=batch_evaluate_factors_via_api,
        search_fn=call_qlib_search,
        model="deepseek-chat",
        temperature=1.55,
        enable_reason=True,
        local=False,
        local_port=8000,
        save_dir="./runs/ea_search_unique",
        seeds_top_k=12,
    )

    summary = searcher.search_population(
        parsed_factor_pool,
        mutation_rate=0.5,
        crossover_rate=0.5,
        N=20,
        rounds=15,
        verbose=True,
        save_pickle=True,
    )
    print("Final pool size:", len(summary["final_pool"]))
    if "save_path" in summary:
        print("Saved summary:", summary["save_path"])
