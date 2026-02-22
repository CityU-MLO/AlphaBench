"""
Evolutionary Algorithm (EA) factor search algorithm.

EA_Searcher:  population-based mutation + crossover loop.
EAAlgo:       BaseAlgo wrapper that feeds the full seed pool.
"""

import os
import time
import json
import pickle
from typing import Any, Dict, List, Optional
import uuid

from .base import BaseAlgo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_metric(f: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(f.get("metrics", {}).get(key, default))


def _format_metrics(f: Dict[str, Any]) -> Dict[str, float]:
    m = f.get("metrics", {}) or {}
    return {k: m[k] for k in ("ic", "rank_ic", "icir", "rank_icir", "winrate", "stability") if k in m}


def _rank_by_ic(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(pool, key=lambda x: _safe_metric(x, "ic", -1e9), reverse=True)


def _select_seed_pool(pool: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
    ranked = _rank_by_ic(pool)
    return ranked[: max(1, min(top_k, len(ranked)))]


def _seed_block_json(seeds: List[Dict[str, Any]]) -> str:
    compact = [
        {
            "name": s.get("name"),
            "expression": s.get("expression", ""),
            "metrics": _format_metrics(s),
        }
        for s in seeds
    ]
    return json.dumps(compact, ensure_ascii=False, separators=(",", ": "), indent=2)


def _get_unique_set(factor_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique, seen = [], set()
    for f in factor_pool:
        expr = f.get("expression", "")
        if expr and expr not in seen:
            unique.append(f)
            seen.add(expr)
    return unique


# ---------------------------------------------------------------------------
# EA_Searcher — core implementation (moved from EA/EA_searcher.py)
# ---------------------------------------------------------------------------

class EA_Searcher:
    """
    Evolutionary Algorithm (EA) style factor searcher.

    Runs mutation and crossover LLM calls each generation, evaluates offspring
    via the injected batch_evaluate_fn, and maintains a constant-size pool of
    the best factors found so far.

    Required external callables:
      - batch_evaluate_fn(factors: List[Dict]) -> List[Dict]
      - search_fn(instruction, model, N, **kw) -> Dict
    """

    def __init__(
        self,
        *,
        batch_evaluate_fn,
        search_fn,
        model: str = "deepseek-chat",
        temperature: float = 1.0,
        enable_reason: bool = True,
        local: bool = False,
        local_port: int = 8000,
        save_dir: str = "./runs/ea_search",
        seeds_top_k: int = 12,
    ) -> None:
        self.batch_evaluate_fn = batch_evaluate_fn
        self.search_fn = search_fn
        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.local = bool(local)
        self.local_port = int(local_port)
        self.save_dir = save_dir
        self.seeds_top_k = int(seeds_top_k)
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_log_dir = os.path.join(self.save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

    def search_population(
        self,
        pool: List[Dict[str, Any]],
        *,
        mutation_rate: float,
        crossover_rate: float,
        N: int,
        rounds: int,
        verbose: bool = True,
        save_pickle: bool = True,
        pool_size: int = 30,
    ) -> Dict[str, Any]:
        """
        Evolutionary search on a factor pool.

        Args:
            pool:           Initial factor pool [{"name": str, "expression": str, "metrics": {...}}]
            mutation_rate:  Fraction of N assigned to mutation (0~1)
            crossover_rate: Fraction of N assigned to crossover (0~1)
            N:              Total new candidates per round
            rounds:         Number of generations
            pool_size:      Maximum pool size (kept constant)
        """
        current_pool = _rank_by_ic(pool)
        baseline_ic = (
            sum(_safe_metric(f, "ic", 0.0) for f in current_pool) / max(len(current_pool), 1)
        )
        if verbose:
            print(f"Baseline mean IC: {baseline_ic:.6f}")

        history = []

        for r in range(1, rounds + 1):
            if verbose:
                print("\n" + "=" * 80)
                print(f"Round {r}/{rounds}")

            seeds = _select_seed_pool(current_pool, top_k=self.seeds_top_k)
            seed_block = _seed_block_json(seeds)

            n_mut = max(0, int(N * mutation_rate))
            n_cross = max(0, int(N * crossover_rate))
            while n_mut + n_cross < N:
                n_mut += 1
            while n_mut + n_cross > N and n_cross > 0:
                n_cross -= 1

            mut_prompt = self._compose_mutation_prompt(seed_block, n_mut, round_id=r)
            cross_prompt = self._compose_crossover_prompt(seed_block, n_cross, round_id=r)

            if verbose:
                print("\n[MUTATION PROMPT]\n", mut_prompt)
                print("\n[CROSSOVER PROMPT]\n", cross_prompt)

            # LLM: mutation
            mut_candidates, mut_elapsed = [], 0.0
            if n_mut > 0:
                t0 = time.time()
                payload = self.search_fn(
                    instruction=mut_prompt,
                    model=self.model,
                    N=n_mut,
                    verbose=verbose,
                    temperature=self.temperature,
                    enable_reason=self.enable_reason,
                    local=self.local,
                    local_port=self.local_port,
                )
                mut_elapsed = time.time() - t0
                mut_candidates = self._extract_candidates(payload, provenance="mutation")

            # LLM: crossover
            cross_candidates, cross_elapsed = [], 0.0
            if n_cross > 0:
                t0 = time.time()
                payload = self.search_fn(
                    instruction=cross_prompt,
                    model=self.model,
                    N=n_cross,
                    verbose=verbose,
                    temperature=self.temperature,
                    enable_reason=self.enable_reason,
                    local=self.local,
                    local_port=self.local_port,
                )
                cross_elapsed = time.time() - t0
                cross_candidates = self._extract_candidates(payload, provenance="crossover")

            candidates = mut_candidates + cross_candidates
            unique_candidates = _get_unique_set(candidates)

            # Evaluate new candidates
            eval_input = [
                {"name": c["name"], "expression": c["expression"]}
                for c in unique_candidates
                if c.get("expression")
            ]
            results = self.batch_evaluate_fn(eval_input)
            for i, res in enumerate(results):
                unique_candidates[i]["metrics"] = res.get("metrics", {})

            # Update pool: keep best (old + new), maintain constant size
            combined = _get_unique_set(current_pool + unique_candidates)
            combined = _rank_by_ic(combined)
            current_pool = combined[:pool_size]

            history.append({
                "round": r,
                "mutation_prompt": mut_prompt,
                "crossover_prompt": cross_prompt,
                "mutation_candidates": mut_candidates,
                "crossover_candidates": cross_candidates,
                "mut_elapsed_time": mut_elapsed,
                "cross_elapsed_time": cross_elapsed,
                "candidates": candidates,
                "unique_candidates": unique_candidates,
            })

            if verbose:
                top_ic = _safe_metric(current_pool[0], "ic") if current_pool else 0.0
                mean_ic = (
                    sum(_safe_metric(f, "ic", 0.0) for f in current_pool) / max(len(current_pool), 1)
                )
                print(f"Round {r} — top IC: {top_ic:.6f}, mean IC: {mean_ic:.6f}")

        summary = {
            "baseline_ic": baseline_ic,
            "history": history,
            "final_pool": current_pool,
            "best": current_pool[0] if current_pool else {},
        }

        if save_pickle:
            fname = os.path.join(self.save_dir, f"ea_search_{int(time.time())}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(summary, f)
            summary["save_path"] = fname

        return summary

    # ------------------------------------------------------------------ #
    def _compose_mutation_prompt(self, seed_block: str, n: int, round_id: int) -> str:
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
  {"- 'reason': 1-2 sentences explaining the mutation (string)" if self.enable_reason else ''}
  - "expression": the full Qlib-style expression (string)

No extra text. Output ONLY the JSON array.
"""

    def _compose_crossover_prompt(self, seed_block: str, n: int, round_id: int) -> str:
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
3) Extract the **short, reusable subchains** (2-4 ops) that carry the behavior (trend, mean-revert, breakout) or the stabilizer (vol/volume scaling).

Examples (illustrative only; you must produce new ones):
- From A: Mean(Sub($close, Ref($close, 1)), 10)
  From B: Div($volume, Add(Mean($volume, 60), 1e-12))
  To:     Div(Mean(Sub($close, Ref($close, 1)), 12), Add(Mean($volume, 60), 1e-12))

Output format:
Return a JSON array of length {n}. Each item MUST be an object with:
  - "name": a short unique name (string)
  {"- 'reason': 1-2 sentences explaining the crossover (string)" if self.enable_reason else ''}
  - "expression": the full Qlib-style expression (string)

No extra text. Output ONLY the JSON array.
"""

    def _extract_candidates(self, payload: Any, provenance: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        items = []

        if isinstance(payload, dict):
            if payload.get("quality"):
                log_name = provenance + "_" + time.strftime("log_%Y%m%d_%H%M%S.json")
                with open(os.path.join(self.save_log_dir, log_name), "w") as f:
                    json.dump(payload["quality"], f)
            if isinstance(payload.get("factors"), list):
                items = payload["factors"]
            elif "choices" in payload:
                try:
                    items = json.loads(payload["choices"][0]["message"]["content"])
                except Exception:
                    items = []
        elif isinstance(payload, list):
            items = payload

        for it in items:
            name = it.get("name")
            expr = it.get("expression", "")
            if not (name and expr):
                continue
            suffix = str(uuid.uuid4())[:6]
            out.append({
                "name": name + "_" + suffix,
                "expression": expr,
                "reason": it.get("reason", ""),
                "provenance": provenance,
            })
        return out


# ---------------------------------------------------------------------------
# EAAlgo — BaseAlgo wrapper
# ---------------------------------------------------------------------------

class EAAlgo(BaseAlgo):
    """
    EA BaseAlgo adapter.

    Runs EA_Searcher on the full seed pool.
    Config keys (under searching.algo.param):
      rounds:         generations (default: 10)
      N:              new candidates per round (default: 30)
      mutation_rate:  fraction for mutation (default: 0.4)
      crossover_rate: fraction for crossover (default: 0.6)
      pool_size:      max pool size (default: 30)
      seeds_top_k:    seeds shown to LLM each round (default: 12)
      model:          LLM model name
      temperature:    sampling temperature (default: 1.0)
      enable_reason:  include reasoning in output (default: True)
    """

    name = "ea"

    def run(self, seeds: List[Dict[str, Any]], save_dir: str) -> Dict[str, Any]:
        rounds = self.config.get("rounds", 10)
        N = self.config.get("N", 30)
        mutation_rate = float(self.config.get("mutation_rate", 0.4))
        crossover_rate = float(self.config.get("crossover_rate", 0.6))
        pool_size = int(self.config.get("pool_size", max(len(seeds), 30)))
        seeds_top_k = int(self.config.get("seeds_top_k", 12))
        model = self.config.get("model", "deepseek-chat")
        temperature = float(self.config.get("temperature", 1.0))
        enable_reason = bool(self.config.get("enable_reason", True))

        searcher = EA_Searcher(
            batch_evaluate_fn=self.batch_evaluate_fn,
            search_fn=self.search_fn,
            model=model,
            temperature=temperature,
            enable_reason=enable_reason,
            save_dir=save_dir,
            seeds_top_k=seeds_top_k,
        )

        summary = searcher.search_population(
            pool=seeds,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            N=N,
            rounds=rounds,
            pool_size=pool_size,
            verbose=True,
            save_pickle=True,
        )

        return {
            "best": summary.get("best", {}),
            "history": summary.get("history", []),
            "final_pool": summary.get("final_pool", []),
        }
