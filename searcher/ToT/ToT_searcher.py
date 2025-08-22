import os
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple

from agent.generator_qlib_search import call_qlib_search
from api.factor_eval_client import (
    batch_evaluate_factors_via_api,
    evaluate_factor_via_api,
)


class ToTSearcher:
    """
    Chain-of-Thought (ToT) guided factor searcher.

    This class orchestrates an iterative search over alpha-factor expressions using a
    language model (LLM) and an external evaluation backend (e.g., Qlib).

    It assumes you already have three callables available somewhere in your codebase:
      - evaluate_factor_via_api(expr: str) -> Dict[str, float]
      - batch_evaluate_factors_via_api(factors: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]
      - call_qlib_search(instruction: str, **llm_kwargs) -> Dict[str, Any]

    You can inject those at construction time via the corresponding parameters.
    """

    def __init__(
        self,
        evaluate_factor_fn,
        batch_evaluate_factors_fn,
        search_fn,
        *,
        model: str = "deepseek-chat",
        temperature: float = 1.0,
        enable_reason: bool = True,
        local: bool = False,
        local_port: int = 8000,
        save_dir: str = "./runs/tot_search",
    ) -> None:
        # LLM params
        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.local = bool(local)
        self.local_port = int(local_port)

        # External function hooks
        self.evaluate_factor_fn = evaluate_factor_fn
        self.batch_evaluate_factors_fn = batch_evaluate_factors_fn
        self.search_fn = search_fn

        # I/O
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------ Public API ------------------------------ #
    def search_single_factor(
        self,
        seed: Dict[str, str],
        rounds: int,
        N: int,
        *,
        run_name: Optional[str] = None,
        verbose: bool = True,
        avoid_repeat: bool = True,
        max_try: int = 5,
        debug_mode: bool = False,
        save_pickle: bool = True,
        save_dir: Optional[str] = "./runs/tot_search",
    ) -> Dict[str, Any]:
        """
        Run ToT-guided search starting from a seed factor.

        Args:
            seed: {"name": str, "expression": str}
            rounds: number of search rounds
            N: number of candidates to generate per round
            run_name: optional tag for file naming
            verbose: print progress
            avoid_repeat, max_try, debug_mode: forwarded to search_fn
            save_pickle: whether to persist results at the end

        Returns:
            summary dict containing history, best factor, and save path (if any)
        """
        assert (
            isinstance(seed, dict) and "name" in seed and "expression" in seed
        ), "seed must be a dict with keys 'name' and 'expression'"
        rounds = int(rounds)
        N = int(N)
        t0 = time.time()

        # Evaluate seed performance
        if verbose:
            print("Evaluating seed factor …")
        seed_metrics = self._safe_eval_single(seed["expression"])[
            "metrics"
        ]  # Dict[str, float]
        if verbose:
            print(f"Seed metrics: {self._fmt_metrics(seed_metrics)}")

        history: List[Dict[str, Any]] = []
        seen_exprs = {self._normalize_expr(seed["expression"])}
        best_name, best_expr, best_metrics = (
            seed["name"],
            seed["expression"],
            seed_metrics,
        )

        # Round 0 instruction (derive from seed + metrics)
        instruction = self._compose_instruction(
            round_id=0,
            seed=seed,
            best_so_far=(best_name, best_expr, best_metrics),
            prev_round_topk=[],
        )

        for r in range(1, rounds + 1):
            if verbose:
                print("\n" + "=" * 80)
                print(f"Round {r}/{rounds}: Generating {N} candidates …")
                print("Instruction:\n" + instruction)

            # --- LLM search call ---
            start_time = time.time()
            search_payload = self.search_fn(
                instruction=instruction,
                model=self.model,
                N=N,
                max_try=max_try,
                avoid_repeat=avoid_repeat,
                verbose=verbose,
                debug_mode=debug_mode,
                temperature=self.temperature,
                enable_reason=self.enable_reason,
                local=self.local,
                local_port=self.local_port,
            )
            elapsed = time.time() - start_time

            candidates = self._extract_candidates(search_payload, verbose=verbose)
            if verbose:
                print(f"Generated {len(candidates)} raw candidates")

            # Deduplicate by normalized expression
            unique_candidates: List[Dict[str, str]] = []
            for c in candidates:
                key = (
                    self._normalize_expr(c["expression"])
                    if c.get("expression")
                    else None
                )
                if key and key not in seen_exprs:
                    seen_exprs.add(key)
                    unique_candidates.append(
                        {
                            "name": c.get(
                                "name", f"cand_{r}_{len(unique_candidates)+1}"
                            ),
                            "expression": c["expression"],
                            "reason": c.get("reason", ""),
                        }
                    )
            if verbose:
                print(f"Kept {len(unique_candidates)} unique candidates after de-dup")

            if not unique_candidates:
                # Nothing new; record empty round and break
                history.append(
                    {
                        "round": r,
                        "instruction": instruction,
                        "candidates": [],
                        "evaluations": {},
                        "best_of_round": None,
                    }
                )
                break

            # --- Batch evaluation ---
            eval_input = [
                {"name": c["name"], "expr": c["expression"]} for c in unique_candidates
            ]
            if verbose:
                print(f"Batch evaluating {len(eval_input)} candidates …")
            round_results = self._safe_eval_batch(eval_input)

            # Pair up results (ensure stable mapping)
            evaluations: Dict[str, Dict[str, float]] = {}
            for c in unique_candidates:
                name = c["name"]
                metrics = (
                    round_results.get(name)["metrics"]
                    or round_results.get(c["expression"])
                    or {}
                )
                evaluations[name] = metrics

            # Rank by IC primarily, then RankIC, then IR
            ranked = self._rank_by_objective(evaluations)
            best_of_round = None
            if ranked:
                best_of_round = (ranked[0][0], ranked[0][1])  # (name, metrics)
                # Update global best if improved
                if self._is_better(best_metrics, best_of_round[1]):
                    idx = next(
                        (
                            i
                            for i, c in enumerate(unique_candidates)
                            if c["name"] == best_of_round[0]
                        ),
                        None,
                    )
                    if idx is not None:
                        best_name = unique_candidates[idx]["name"]
                        best_expr = unique_candidates[idx]["expression"]
                        best_metrics = best_of_round[1]

            # Record round
            rec = {
                "round": r,
                "instruction": instruction,
                "candidates": unique_candidates,  # list of dicts with name, expression, reason
                "evaluations": evaluations,  # name -> metrics
                "ranking": ranked,  # [(name, metrics), …]
                "best_of_round": best_of_round,  # (name, metrics) or None,
                "elapsed_time": elapsed,
                "best_so_far": {
                    "name": best_name,
                    "expression": best_expr,
                    "metrics": best_metrics,
                },
            }
            history.append(rec)

            if verbose and best_of_round:
                print(
                    f"Best of round: {best_of_round[0]} | "
                    f"{self._fmt_metrics(best_of_round[1])}"
                )
                print(f"Best so far: {best_name} | {self._fmt_metrics(best_metrics)}")

            # Compose next-round instruction using ToT from this round
            instruction = self._compose_instruction(
                round_id=r,
                seed=seed,
                best_so_far=(best_name, best_expr, best_metrics),
                prev_round_topk=ranked[: len(ranked) // 2],
                factors_pool=unique_candidates,
            )

        # Finalize
        elapsed = time.time() - t0
        summary: Dict[str, Any] = {
            "seed": seed,
            "seed_metrics": seed_metrics,
            "history": history,
            "best": {
                "name": best_name,
                "expression": best_expr,
                "metrics": best_metrics,
            },
            "elapsed_sec": elapsed,
        }

        save_path = None
        if save_pickle:
            tag = run_name or self._safe_tag(seed.get("name", "seed"))
            fname = f"tot_search_{tag}.pkl"
            if save_dir:
                save_path = os.path.join(save_dir, fname)
            else:
                save_path = os.path.join(self.save_dir, fname)
            with open(save_path, "wb") as f:
                pickle.dump(summary, f)
            summary["save_path"] = save_path
            if verbose:
                print(f"\nSaved search summary to: {save_path}")

        return summary

    # --------------------------- Instruction Builder ------------------------ #
    def _compose_instruction(
        self,
        *,
        round_id: int,
        seed: Dict[str, str],
        best_so_far: Tuple[str, str, Dict[str, float]],
        prev_round_topk: List[Tuple[str, Dict[str, float]]],
        factors_pool: Dict[str, Any] = None,
    ) -> str:
        """Compose an English instruction for the next LLM search call.

        The prompt references: the seed factor, metrics so far, and (optionally)
        the top results from the previous round to steer the model.
        """
        best_name, best_expr, best_metrics = best_so_far

        def fmt_m(m):
            return (
                f"IC={m.get('ic', float('nan')):.6f}, "
                f"RankIC={m.get('rank_ic', float('nan')):.6f}, "
                f"ICIR={m.get('icir', float('nan')):.6f}"
            )

        header = [
            "You are an expert quantitative researcher generating formulaic alpha factors.",
            "Return exactly N candidate factors as JSON objects with keys: name, expression, and reason.",
            "Focus on improving IC primarily, then RankIC and IR.",
        ]

        context = [
            f"Seed factor: {seed['name']} => {seed['expression']}",
            f"Best so far: {best_name} => {best_expr} with {fmt_m(best_metrics)}",
        ]

        if prev_round_topk:
            bullets = [f"- {nm}: {fmt_m(mx)}" for nm, mx in prev_round_topk]
            context.append("Top signals from last round:\n" + "\n".join(bullets))
            # import pdb;pdb.set_trace()  # Debugging point to inspect the top signals
            context.append(
                "Top signals with their expressions:\n"
                + "\n".join(
                    f"{nm}: {next(d['expression'] for d in factors_pool if d['name'] == nm)}"
                    for nm, _ in prev_round_topk
                )
            )

        # Heuristics for guidance based on metrics
        steer = [
            "Guidelines:",
            "1) Normalize by volatility or range when signals are noisy (e.g., divide by Std).",
            "2) Consider momentum windows (10–60), mean-reversion (2–5), or volume conditioning.",
            "3) Prefer simple, computationally efficient expressions.",
            "4) Avoid reusing identical structures; diversify operators and windows.",
        ]

        if round_id == 0:
            target = (
                "Task: Propose N improved variants of the seed and a few orthogonal baselines. "
                "Explain briefly why each might improve IC."
            )
        else:
            target = (
                "Task: Using the feedback above, generate N refined candidates that either "
                "(a) strengthen the best structure via stability/normalization tweaks or "
                "(b) explore adjacent but distinct operator chains (e.g., volatility-adjusted momentum, "
                "conditional signals with moving-average regimes). Include a one-sentence reason."
            )

        return "\n\n".join(header + [""] + context + [""] + steer + [""] + [target])

    # ----------------------------- Ranking Logic --------------------------- #
    @staticmethod
    def _rank_by_objective(
        evaluations: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Rank candidates primarily by IC, then RankIC, then IR (descending)."""

        def key_fn(item):
            name, m = item
            return (
                float(m.get("ic", float("nan"))),
                float(m.get("rank_ic", float("nan"))),
                float(m.get("ir", float("nan"))),
            )

        # Filter out items without IC
        valid_items = [(k, v) for k, v in evaluations.items() if "ic" in v]
        return sorted(valid_items, key=key_fn, reverse=True)

    @staticmethod
    def _is_better(curr_best: Dict[str, float], candidate: Dict[str, float]) -> bool:
        """Strict improvement on IC; tiebreak by RankIC then IR."""
        c_ic, n_ic = curr_best.get("ic", -1e9), candidate.get("ic", -1e9)
        if n_ic != c_ic:
            return n_ic > c_ic
        c_r, n_r = curr_best.get("rank_ic", -1e9), candidate.get("rank_ic", -1e9)
        if n_r != c_r:
            return n_r > c_r
        return candidate.get("ir", -1e9) > curr_best.get("ir", -1e9)

    # ------------------------------ Utilities ------------------------------ #
    def _extract_candidates(
        self, payload: Dict[str, Any], verbose: bool = False
    ) -> List[Dict[str, str]]:
        """
        Normalize the search_fn return into a list of {name, expression, reason}.
        Accepts either the 'factors' list as shown in the example, or a 'results' mapping.
        """
        factors: List[Dict[str, str]] = []
        if not payload:
            return factors

        if isinstance(payload, dict):
            # Preferred path
            if isinstance(payload.get("factors"), list):
                for item in payload["factors"]:
                    expr = item.get("expression")
                    name = item.get("name")
                    if expr and name:
                        factors.append(
                            {
                                "name": name,
                                "expression": expr,
                                "reason": item.get("reason", ""),
                            }
                        )
            # Fallback path: mapping name -> expr under 'results'
            elif isinstance(payload.get("results"), dict):
                for name, expr in payload["results"].items():
                    if expr:
                        factors.append(
                            {"name": str(name), "expression": str(expr), "reason": ""}
                        )
        if verbose:
            print(f"_extract_candidates: parsed {len(factors)} candidates")
        return factors

    def _safe_eval_single(self, expr: str) -> Dict[str, float]:
        try:
            return dict(self.evaluate_factor_fn(expr) or {})
        except Exception as e:
            return {"error": str(e)}

    def _safe_eval_batch(
        self, items: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, float]]:
        try:
            return dict(self.batch_evaluate_factors_fn(items) or {})
        except Exception as e:
            # Fallback: try evaluating one by one to salvage results
            out: Dict[str, Dict[str, float]] = {}
            for it in items:
                name, expr = it.get("name"), it.get("expr")
                try:
                    out[name] = dict(self.evaluate_factor_fn(expr) or {})
                except Exception as ie:
                    out[name] = {"error": str(ie)}
            return out

    @staticmethod
    def _normalize_expr(expr: str) -> str:
        return "".join(expr.split()).lower()

    @staticmethod
    def _fmt_metrics(m: Dict[str, Any]) -> str:

        keys = ["ic", "rank_ic", "ir", "icir", "rank_icir"]
        parts = []
        for k in keys:
            if k in m and isinstance(m[k], (int, float)):
                parts.append(f"{k}={m[k]:.6f}")
        if not parts and m:
            return str(m)
        return ", ".join(parts) if parts else "<no-metrics>"

    @staticmethod
    def _safe_tag(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:48]


if __name__ == "__main__":
    seed = {"name": "Seed_KLEN", "expression": "Div(Sub($high, $low), $open)"}

    searcher = ToTSearcher(
        evaluate_factor_fn=evaluate_factor_via_api,
        batch_evaluate_factors_fn=batch_evaluate_factors_via_api,
        search_fn=call_qlib_search,
        model="deepseek-chat",
        temperature=1.0,
        enable_reason=True,
        local=False,
        local_port=8000,
    )

    summary = searcher.search_single_factor(
        seed=seed, rounds=15, N=10, verbose=True, save_dir="./runs/tot_search"
    )
    print("Best:", summary["best"])
