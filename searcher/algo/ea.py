"""
Evolutionary Algorithm (EA) factor search algorithm.

EA_Searcher:  population-based mutation + crossover loop.
              Mutation and crossover LLM calls run in parallel each round.
              Next round's LLM generation is pre-submitted while current
              round's evaluation runs (pipeline parallelism).
EAAlgo:       BaseAlgo wrapper that feeds the full seed pool.
"""

import json
import os
import pickle
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAlgo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_metric(f: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(f.get("metrics", {}).get(key, default))


def _format_metrics(f: Dict[str, Any]) -> Dict[str, float]:
    m = f.get("metrics", {}) or {}
    return {k: m[k] for k in ("ic", "rank_ic", "icir", "rank_icir", "winrate", "stability") if k in m}


def _rank_by_rank_ic(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort pool by RankIC descending (primary metric)."""
    return sorted(pool, key=lambda x: _safe_metric(x, "rank_ic", -1e9), reverse=True)


def _select_seed_pool(pool: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
    ranked = _rank_by_rank_ic(pool)
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
# EA_Searcher — core implementation
# ---------------------------------------------------------------------------

class EA_Searcher:
    """
    Evolutionary Algorithm (EA) style factor searcher.

    Each round:
      1) Select top seeds from the current pool.
      2) Build mutation and crossover prompts.
      3) Call both LLMs **in parallel** via ThreadPoolExecutor(max_workers=2).
      4) Evaluate all unique offspring via FFO (parallel batch).
      5) Merge offspring with pool, keep top pool_size factors.

    Pipeline parallelism: next round's LLM generation is pre-submitted
    immediately after this round's LLM results arrive, so it runs in
    parallel with the current round's batch evaluation.

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
        accept_threshold: float = 0.0,
        logger=None,
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
        self.accept_threshold = float(accept_threshold)
        self.logger = logger
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
        save_pickle: bool = True,
        pool_size: int = 30,
    ) -> Dict[str, Any]:
        """
        Evolutionary search on a factor pool.

        Args:
            pool:           Initial factor pool [{"name", "expression", "metrics"}].
            mutation_rate:  Fraction of N assigned to mutation (0~1).
            crossover_rate: Fraction of N assigned to crossover (0~1).
            N:              Total new candidates per round.
            rounds:         Number of generations.
            pool_size:      Maximum pool size (kept constant).
        """
        current_pool = _rank_by_rank_ic(pool)
        baseline_rank_ic = (
            sum(_safe_metric(f, "rank_ic", 0.0) for f in current_pool) / max(len(current_pool), 1)
        )
        self._log(
            f"  Baseline  {len(current_pool)} factors  "
            f"mean RankIC={baseline_rank_ic:.4f}  accept_threshold={self.accept_threshold:.4f}"
        )

        # Pre-compute mut/cross split (fixed for all rounds)
        n_mut = max(0, int(N * mutation_rate))
        n_cross = max(0, int(N * crossover_rate))
        while n_mut + n_cross < N:
            n_mut += 1
        while n_mut + n_cross > N and n_cross > 0:
            n_cross -= 1

        history = []

        # Use a persistent executor so the pre-submitted future for the next
        # round's LLM call runs while current round's eval is blocking.
        with ThreadPoolExecutor(max_workers=4) as ex:
            # Pre-submit round 1's LLM before entering the loop
            seeds_r1 = _select_seed_pool(current_pool, top_k=self.seeds_top_k)
            fut_llm = ex.submit(
                self._run_llm_round,
                _seed_block_json(seeds_r1), n_mut, n_cross, 1,
            )

            for r in range(1, rounds + 1):
                if self.logger and hasattr(self.logger, "round_header"):
                    self.logger.round_header(r, rounds, "EA")

                # ── Wait for this round's LLM results ──────────────────────
                self._log(f"  LLM  mutation({n_mut}) + crossover({n_cross}) — waiting for results …")
                t_wait = time.time()
                mut_candidates, cross_candidates, llm_elapsed = fut_llm.result()
                wait_elapsed = time.time() - t_wait
                self._log(
                    f"  LLM  done (total {llm_elapsed:.1f}s, wait {wait_elapsed:.1f}s)  —  "
                    f"mut: {len(mut_candidates)}  cross: {len(cross_candidates)}"
                )

                # ── Pre-submit next round's LLM immediately ─────────────────
                # Uses current pool (before this round's update) — approximate
                # but fine, as eval typically dominates and next-round seeding
                # benefits from pipeline overlap more than from exact pool state.
                if r < rounds:
                    seeds_next = _select_seed_pool(current_pool, top_k=self.seeds_top_k)
                    fut_llm = ex.submit(
                        self._run_llm_round,
                        _seed_block_json(seeds_next), n_mut, n_cross, r + 1,
                    )

                # ── Deduplicate ──────────────────────────────────────────────
                candidates = mut_candidates + cross_candidates
                unique_candidates = _get_unique_set(candidates)

                # ── Evaluate via FFO (runs in parallel with next round's LLM) ─
                eval_input = [
                    {"name": c["name"], "expression": c["expression"]}
                    for c in unique_candidates
                    if c.get("expression")
                ]
                self._log(f"  Eval {len(eval_input)} unique candidates …")
                t_eval = time.time()
                results = self.batch_evaluate_fn(eval_input)
                eval_elapsed = time.time() - t_eval
                n_success = sum(1 for res in results if res.get("success"))
                self._log(f"  Eval done {eval_elapsed:.1f}s  —  {n_success}/{len(eval_input)} successful")

                for i, res in enumerate(results):
                    unique_candidates[i]["metrics"] = res.get("metrics", {})

                # ── Log new candidates — top 5 before accept/reject ────────────
                evaluated = [c for c in unique_candidates if c.get("metrics")]
                evaluated.sort(
                    key=lambda x: x.get("metrics", {}).get("rank_ic", -1e9), reverse=True
                )
                self._log(f"  Top 5 Generated ({len(evaluated)} total):")
                for i, f in enumerate(evaluated[:5], 1):
                    m = f.get("metrics", {}) or {}
                    ric  = m.get("rank_ic", float("nan"))
                    ic   = m.get("ic",      float("nan"))
                    icir = m.get("icir",    float("nan"))
                    name = (f.get("name") or "")[:24]
                    expr = (f.get("expression") or "")[:65]
                    self._log(
                        f"    {i}. {name:<24} RankIC={ric:.4f}  IC={ic:.4f}  ICIR={icir:.4f}"
                    )
                    self._log(f"       {expr}")

                # ── Update pool (threshold filter on new candidates) ──────────
                eligible_new = [
                    c for c in unique_candidates
                    if _safe_metric(c, "rank_ic", -1e9) >= self.accept_threshold
                ]
                n_rejected = len(unique_candidates) - len(eligible_new)
                if n_rejected:
                    self._log(
                        f"  Threshold filter: {n_rejected} candidate(s) rejected "
                        f"(RankIC < {self.accept_threshold:.4f})"
                    )
                combined = _get_unique_set(current_pool + eligible_new)
                current_pool = _rank_by_rank_ic(combined)[:pool_size]

                if self.logger and hasattr(self.logger, "pool_status"):
                    self.logger.pool_status(current_pool, label="Pool")

                history.append({
                    "round":                r,
                    "mutation_candidates":  mut_candidates,
                    "crossover_candidates": cross_candidates,
                    "llm_elapsed":          llm_elapsed,
                    "eval_elapsed":         eval_elapsed,
                    "candidates":           candidates,
                    "unique_candidates":    unique_candidates,
                })

        summary = {
            "baseline_rank_ic": baseline_rank_ic,
            "history":          history,
            "final_pool":       current_pool,
            "best":             current_pool[0] if current_pool else {},
        }

        if save_pickle:
            fname = os.path.join(self.save_dir, f"ea_search_{int(time.time())}.pkl")
            with open(fname, "wb") as fh:
                pickle.dump(summary, fh)
            summary["save_path"] = fname

        return summary

    # ------------------------------------------------------------------ #
    # LLM round helper (runs in thread pool)
    # ------------------------------------------------------------------ #

    def _run_llm_round(
        self,
        seed_block: str,
        n_mut: int,
        n_cross: int,
        round_id: int,
    ) -> Tuple[List[Dict], List[Dict], float]:
        """
        Build prompts and run mutation + crossover LLM calls in parallel.
        Returns (mut_candidates, cross_candidates, elapsed_seconds).
        Designed to run inside a thread so it can overlap with batch eval.
        """
        mut_prompt   = self._compose_mutation_prompt(seed_block, n_mut,   round_id)
        cross_prompt = self._compose_crossover_prompt(seed_block, n_cross, round_id)

        t0 = time.time()
        mut_candidates:   List[Dict] = []
        cross_candidates: List[Dict] = []

        def _call_mut(prompt=mut_prompt, n=n_mut):
            return self.search_fn(
                instruction=prompt,
                model=self.model,
                N=n,
                verbose=False,
                temperature=self.temperature,
                enable_reason=self.enable_reason,
                local=self.local,
                local_port=self.local_port,
            )

        def _call_cross(prompt=cross_prompt, n=n_cross):
            return self.search_fn(
                instruction=prompt,
                model=self.model,
                N=n,
                verbose=False,
                temperature=self.temperature,
                enable_reason=self.enable_reason,
                local=self.local,
                local_port=self.local_port,
            )

        if n_mut > 0 and n_cross > 0:
            with ThreadPoolExecutor(max_workers=2) as inner_ex:
                fut_mut   = inner_ex.submit(_call_mut)
                fut_cross = inner_ex.submit(_call_cross)
                try:
                    mut_candidates = self._extract_candidates(fut_mut.result(), "mutation")
                except Exception as e:
                    self._log(f"Mutation LLM failed: {e}", "warning")
                try:
                    cross_candidates = self._extract_candidates(fut_cross.result(), "crossover")
                except Exception as e:
                    self._log(f"Crossover LLM failed: {e}", "warning")
        elif n_mut > 0:
            try:
                mut_candidates = self._extract_candidates(_call_mut(), "mutation")
            except Exception as e:
                self._log(f"Mutation LLM failed: {e}", "warning")
        elif n_cross > 0:
            try:
                cross_candidates = self._extract_candidates(_call_cross(), "crossover")
            except Exception as e:
                self._log(f"Crossover LLM failed: {e}", "warning")

        return mut_candidates, cross_candidates, time.time() - t0

    # ------------------------------------------------------------------ #
    # Prompt builders
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

    # ------------------------------------------------------------------ #
    # Candidate extraction + internal helpers
    # ------------------------------------------------------------------ #

    def _extract_candidates(self, payload: Any, provenance: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        items = []

        if isinstance(payload, dict):
            if payload.get("quality"):
                log_name = provenance + "_" + time.strftime("log_%Y%m%d_%H%M%S.json")
                with open(os.path.join(self.save_log_dir, log_name), "w") as fh:
                    json.dump(payload["quality"], fh)

            if isinstance(payload.get("factors"), list) and payload["factors"]:
                # Primary path: validated factors list from call_qlib_search
                items = payload["factors"]
            elif isinstance(payload.get("results"), dict) and payload["results"]:
                # Fallback: raw {name: expression} dict when all factors failed
                # check_factor_via_api validation.  FFO eval will re-validate.
                for name, expr in payload["results"].items():
                    if name and expr:
                        items.append({"name": str(name), "expression": str(expr)})
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
                "name":       name + "_" + suffix,
                "expression": expr,
                "reason":     it.get("reason", ""),
                "provenance": provenance,
            })
        return out

    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)


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
        rounds         = self.config.get("rounds", 10)
        N              = self.config.get("N", 30)
        mutation_rate  = float(self.config.get("mutation_rate", 0.4))
        crossover_rate = float(self.config.get("crossover_rate", 0.6))
        pool_size      = int(self.config.get("pool_size", max(len(seeds), 30)))
        seeds_top_k    = int(self.config.get("seeds_top_k", 12))
        model             = self.config.get("model", "deepseek-chat")
        temperature       = float(self.config.get("temperature", 1.0))
        enable_reason     = bool(self.config.get("enable_reason", True))
        accept_threshold  = float(self.config.get("accept_threshold", 0.0))

        searcher = EA_Searcher(
            batch_evaluate_fn=self.batch_evaluate_fn,
            search_fn=self.search_fn,
            model=model,
            temperature=temperature,
            enable_reason=enable_reason,
            save_dir=save_dir,
            seeds_top_k=seeds_top_k,
            accept_threshold=accept_threshold,
            logger=self.logger,
        )

        summary = searcher.search_population(
            pool=seeds,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            N=N,
            rounds=rounds,
            pool_size=pool_size,
            save_pickle=True,
        )

        return {
            "best":       summary.get("best", {}),
            "history":    summary.get("history", []),
            "final_pool": summary.get("final_pool", []),
        }
