"""
Chain-of-Thought (CoT) factor search algorithm.

CoTSearcher: single-path iterative refinement (one factor, one chain).
CoTAlgo:     BaseAlgo wrapper — runs on the best seed from the pool.
"""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional

from .base import BaseAlgo


# ---------------------------------------------------------------------------
# CoTSearcher — core implementation
# ---------------------------------------------------------------------------

class CoTSearcher:
    """
    Chain-of-Thought (CoT) *single-path* factor searcher.

    Each round:
      1) Builds an instruction from the seed, the current best, and the full path.
      2) Calls the LLM search endpoint to generate exactly ONE candidate (N=1).
      3) Evaluates that candidate via the FFO evaluation function.
      4) Compares with current best and updates the chain if improved.

    Required external callables:
      - evaluate_fn(expr: str) -> Dict   (returns dict with "metrics" key)
      - search_fn(instruction, model, N, **kw) -> Dict
    """

    def __init__(
        self,
        *,
        evaluate_fn,
        search_fn,
        batch_evaluate_fn=None,
        model: str = "deepseek-chat",
        temperature: float = 1.75,
        enable_reason: bool = True,
        local: bool = False,
        local_port: int = 8000,
        save_dir: str = "./runs/cot_search",
        accept_threshold: float = 0.0,
        logger=None,
    ) -> None:
        self.evaluate_fn = evaluate_fn        # kept for backward compat; batch path used internally
        self.batch_evaluate_fn = batch_evaluate_fn
        self.search_fn = search_fn
        self.accept_threshold = float(accept_threshold)
        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.local = bool(local)
        self.local_port = int(local_port)
        self.save_dir = save_dir
        self.logger = logger
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_log_dir = os.path.join(self.save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

    def search_single_factor(
        self,
        seed: Dict[str, str],
        rounds: int,
        *,
        run_name: Optional[str] = None,
        save_pickle: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert isinstance(seed, dict) and "name" in seed and "expression" in seed
        rounds = int(rounds)
        save_dir = save_dir or self.save_dir
        self.save_log_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

        self._log(f"  Seed: {seed['name']}  [{seed['expression'][:70]}]")
        self._log("  Evaluating seed …")
        seed_metrics = self._safe_eval_batch(seed["name"], seed["expression"])["metrics"]
        self._log(f"  Seed  {self._fmt_metrics(seed_metrics)}")

        chain: List[Dict[str, Any]] = []
        best_expr    = seed["expression"]
        best_name    = seed["name"]
        best_metrics = seed_metrics

        chain.append({
            "round":       0,
            "name":        best_name,
            "expression":  best_expr,
            "metrics":     best_metrics,
            "instruction": None,
            "generated":   None,
        })

        for r in range(1, rounds + 1):
            if self.logger and hasattr(self.logger, "round_header"):
                self.logger.round_header(r, rounds, "CoT")

            ric = best_metrics.get("rank_ic", float("nan"))
            ic  = best_metrics.get("ic",      float("nan"))
            self._log(f"  Best: {best_name}  RankIC={ric:.4f}  IC={ic:.4f}")

            instruction = self._compose_instruction(
                round_id=r,
                seed_name=seed["name"],
                seed_expr=seed["expression"],
                best_name=best_name,
                best_expr=best_expr,
                best_metrics=best_metrics,
                chain=chain,
            )

            self._log("  LLM generating 1 candidate …")
            t_llm = time.time()
            payload = self.search_fn(
                instruction=instruction,
                model=self.model,
                N=1,
                max_try=5,
                avoid_repeat=False,
                verbose=False,
                debug_mode=False,
                temperature=self.temperature,
                enable_reason=self.enable_reason,
                local=self.local,
                local_port=self.local_port,
            )
            llm_elapsed = time.time() - t_llm
            self._log(f"  LLM done {llm_elapsed:.1f}s")

            cand = self._extract_single_candidate(payload)
            if cand is None:
                self._log("  No candidate returned; skipping round.")
                chain.append({
                    "round":        r,
                    "name":         best_name,
                    "expression":   best_expr,
                    "metrics":      best_metrics,
                    "instruction":  instruction,
                    "elapsed_time": llm_elapsed,
                    "generated":    {},
                })
                continue

            cand_name   = cand.get("name",       f"round{r}_candidate")
            cand_expr   = cand.get("expression", "")
            cand_reason = cand.get("reason",     "")

            self._log(f"  Candidate: {cand_name}  [{cand_expr[:70]}]")
            self._log("  Evaluating …")
            cand_metrics = self._safe_eval_batch(cand_name, cand_expr)["metrics"]
            ric_c = cand_metrics.get("rank_ic", float("nan"))
            ic_c  = cand_metrics.get("ic",      float("nan"))
            self._log(f"  Eval  RankIC={ric_c:.4f}  IC={ic_c:.4f}")

            is_better = self._is_better(best_metrics, cand_metrics)
            meets_threshold = float(cand_metrics.get("rank_ic", float("-inf"))) >= self.accept_threshold
            if is_better and meets_threshold:
                prev_ric = best_metrics.get("rank_ic", float("nan"))
                best_name, best_expr, best_metrics = cand_name, cand_expr, cand_metrics
                self._log(f"  → improved  (RankIC {prev_ric:.4f} → {ric_c:.4f})")
                improved = True
            elif is_better and not meets_threshold:
                self._log(
                    f"  → better but rejected  "
                    f"(RankIC {ric_c:.4f} < threshold {self.accept_threshold:.4f})"
                )
                improved = False
            else:
                self._log("  → not improved, keeping previous best")
                improved = False

            chain.append({
                "round":        r,
                "name":         best_name,
                "expression":   best_expr,
                "metrics":      best_metrics,
                "instruction":  instruction,
                "elapsed_time": llm_elapsed,
                "generated": {
                    "name":       cand_name,
                    "expression": cand_expr,
                    "reason":     cand_reason,
                    "metrics":    cand_metrics,
                    "promoted":   bool(improved),
                },
            })

        summary = {
            "seed":         seed,
            "seed_metrics": seed_metrics,
            "chain":        chain,
            "best": {
                "name":       best_name,
                "expression": best_expr,
                "metrics":    best_metrics,
            },
        }

        if save_pickle:
            tag = self._safe_tag(run_name or seed["name"]) or "cot"
            out_path = os.path.join(save_dir, f"CoT_single_{tag}.pkl")
            with open(out_path, "wb") as fh:
                pickle.dump(summary, fh)
            summary["save_path"] = out_path
            self._log(f"  Saved: {out_path}")

        return summary

    # ------------------------------------------------------------------ #
    # Instruction builder
    # ------------------------------------------------------------------ #

    def _compose_instruction(
        self,
        *,
        round_id: int,
        seed_name: str,
        seed_expr: str,
        best_name: str,
        best_expr: str,
        best_metrics: Dict[str, float],
        chain: List[Dict[str, Any]],
    ) -> str:
        def fmt_m(m: Dict[str, float]) -> str:
            if isinstance(m, dict) and "metrics" in m and isinstance(m["metrics"], dict):
                m = m["metrics"]
            return (
                f"IC={m.get('ic', float('nan')):.6f}, "
                f"RankIC={m.get('rank_ic', float('nan')):.6f}, "
                f"ICIR={m.get('icir', float('nan')):.6f}"
            )

        history_lines: List[str] = []
        for rec in chain:
            if rec.get("round") == 0:
                history_lines.append(
                    f"r0 (seed): {rec['name']} => {rec['expression']} | {fmt_m(rec)}\n"
                )
                continue
            gen = rec.get("generated") or {}
            if gen:
                history_lines.append(
                    f"r{rec['round']} candidate: {gen.get('name','?')} => {gen.get('expression','?')} | \n"
                    f"{fmt_m(gen)} | promoted={gen.get('promoted', False)}\n"
                )
            else:
                history_lines.append(
                    f"r{rec['round']} candidate: no_result => no_result | \n"
                    f"IC=nan, RankIC=nan, ICIR=nan | promoted=False\n"
                )
            history_lines.append(
                f"r{rec['round']} best: {rec['name']} => {rec['expression']} | {fmt_m(rec)}\n"
            )

        history_text = "".join(history_lines) if history_lines else "(no history)"

        header = f"""
        You are an expert quantitative researcher refining a single alpha factor via chain-of-thought steps.,
        Return EXACTLY ONE candidate as a JSON object with keys: name, {"reason," if self.enable_reason else ','} expression.,
        Objective: maximize IC (primary), then RankIC, then IR. Keep expressions computable.,
        """

        context = f"""
            Seed: {seed_name} => {seed_expr},
            Current best: {best_name} => {best_expr} | {fmt_m(best_metrics)},
            Full history (seed → current):,
            {history_text},
        """

        policy = """
        Allowed operations (choose any):,
        - Parameter jumps: freely change window lengths (e.g., 3, 5, 10, 20, 60, 120, 252).,
        - Structural edits: add/remove operators; chain new transforms (e.g., Rank, Corr, Ref, Delta, Power).,
        - Stabilization: normalize by rolling Std/Mean or price range; avoid exploding scales.,
        - Regime gating: condition on moving-average regimes, e.g., If(Gt($close, Mean($close, 20)), A, B).,
        - Volume conditioning: weight by or interact with volume features.,
        - Exploration reset: you may abandon the current structure and propose a brand-new formulation.,
        Constraints: obey variable/operator whitelist; avoid overly long or redundant chains; ensure novelty vs history.,
        Complexity: for each generation, never generate too complex factor or make huge changes.
        """

        if round_id == 1:
            steering = f"""
            Task: Propose ONE improved variant of the current best OR a fresh alternative.,
            {"Provide a one-sentence reason describing why IC (and secondarily RankIC/IR) could improve." if self.enable_reason else ""}
            """
        else:
            steering = f"""
            Task: Given the full history, refine decisively — either stabilize the best structure or explore a new one.,
            If prior attempts plateaued, try bold parameter jumps or structural re-composition (e.g., add Corr/Rank gates).,
            {"Provide a concise reason tied to expected IC/RankIC/IR effects." if self.enable_reason else ""}
            """

        return header + "\n" + context + "\n" + policy + "\n" + steering

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _extract_single_candidate(self, payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return None
        if payload.get("quality"):
            quality_log_name = time.strftime("log_%Y%m%d_%H%M%S.json")
            with open(os.path.join(self.save_log_dir, quality_log_name), "w") as fh:
                json.dump(payload["quality"], fh)
        if isinstance(payload.get("factors"), list) and payload["factors"]:
            first = payload["factors"][0]
            expr  = first.get("expression")
            name  = first.get("name")
            if expr and name:
                return {"name": name, "expression": expr, "reason": first.get("reason", "")}
        if isinstance(payload.get("results"), dict) and payload["results"]:
            name, expr = next(iter(payload["results"].items()))
            if expr:
                return {"name": str(name), "expression": str(expr), "reason": ""}
        return None

    def _safe_eval_batch(self, name: str, expr: str) -> Dict[str, Any]:
        """Evaluate a single factor via the batch path for consistency."""
        try:
            if self.batch_evaluate_fn is not None:
                results = self.batch_evaluate_fn([{"name": name, "expression": expr}])
                if isinstance(results, list) and results:
                    return results[0]
                return {}
            # Fallback to single-factor evaluator
            result = self.evaluate_fn(expr) or {}
            return result if isinstance(result, dict) else {}
        except Exception as e:
            return {"metrics": {}, "error": str(e)}

    @staticmethod
    def _is_better(curr: Dict[str, float], cand: Dict[str, float]) -> bool:
        """RankIC is the primary metric; IC is tiebreaker; then ICIR."""
        c_r = curr.get("rank_ic", float("-inf"))
        n_r = cand.get("rank_ic", float("-inf"))
        if n_r != c_r:
            return n_r > c_r
        c_ic = curr.get("ic", float("-inf"))
        n_ic = cand.get("ic", float("-inf"))
        if n_ic != c_ic:
            return n_ic > c_ic
        return cand.get("icir", float("-inf")) > curr.get("icir", float("-inf"))

    @staticmethod
    def _fmt_metrics(m: Dict[str, Any]) -> str:
        keys  = ["ic", "rank_ic", "ir", "icir", "rank_icir"]
        parts = []
        for k in keys:
            v = m.get(k)
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.4f}")
        return ", ".join(parts)

    @staticmethod
    def _safe_tag(s: Optional[str]) -> str:
        if not s:
            return ""
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:48]

    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)


# ---------------------------------------------------------------------------
# CoTAlgo — BaseAlgo wrapper
# ---------------------------------------------------------------------------

class CoTAlgo(BaseAlgo):
    """
    CoT BaseAlgo adapter.

    Runs CoTSearcher on the top seed (by IC) from the seed pool.
    Config keys (under searching.algo.param):
      rounds:        number of refinement rounds (default: 10)
      model:         LLM model name
      temperature:   sampling temperature (default: 1.75)
      enable_reason: include chain-of-thought reasoning (default: True)
    """

    name = "cot"

    def run(self, seeds: List[Dict[str, Any]], save_dir: str) -> Dict[str, Any]:
        rounds           = self.config.get("rounds", 10)
        model            = self.config.get("model", "deepseek-chat")
        temperature      = float(self.config.get("temperature", 1.75))
        enable_reason    = bool(self.config.get("enable_reason", True))
        accept_threshold = float(self.config.get("accept_threshold", 0.0))

        searcher = CoTSearcher(
            evaluate_fn=self.evaluate_fn,
            search_fn=self.search_fn,
            batch_evaluate_fn=self.batch_evaluate_fn,
            model=model,
            temperature=temperature,
            enable_reason=enable_reason,
            save_dir=save_dir,
            accept_threshold=accept_threshold,
            logger=self.logger,
        )

        # Pick the best seed by IC
        valid_seeds = [s for s in seeds if s.get("metrics")]
        if not valid_seeds:
            valid_seeds = seeds
        top_seed = max(
            valid_seeds,
            key=lambda s: s.get("metrics", {}).get("ic", float("-inf")),
        )

        seed_input = {"name": top_seed["name"], "expression": top_seed["expression"]}
        summary = searcher.search_single_factor(
            seed=seed_input,
            rounds=rounds,
            save_dir=save_dir,
        )

        # Normalise return format
        chain_factors = []
        for rec in summary.get("chain", []):
            if rec.get("expression"):
                chain_factors.append({
                    "name":       rec.get("name", ""),
                    "expression": rec["expression"],
                    "metrics":    rec.get("metrics", {}),
                })

        return {
            "best":       summary.get("best", {}),
            "history":    summary.get("chain", []),
            "final_pool": chain_factors,
        }
