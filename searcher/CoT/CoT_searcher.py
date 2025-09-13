import json
import os
import time
import pickle
from typing import Any, Dict, List, Optional
from agent.generator_qlib_search import call_qlib_search
from api.factor_eval_client import (
    batch_evaluate_factors_via_api,
    evaluate_factor_via_api,
)


class CoTSearcher:
    """
    Chain-of-Thought (CoT) *single-path* factor searcher.

    This class performs round-wise, chain-style refinement for ONE factor at a time.
    Each round:
      1) Builds an instruction from the seed, the current best factor, the full path so far.
      2) Calls your LLM search endpoint to generate exactly ONE candidate (N=1).
      3) Evaluates that candidate via your evaluation API.
      4) Compares with the current best and updates the chain if improved.

    Required external callables (inject via __init__):
      - evaluate_factor_via_api(expr: str) -> Dict[str, float]
      - call_qlib_search(instruction: str, **kwargs) -> Dict[str, Any]
    """

    def __init__(
        self,
        *,
        evaluate_factor_fn,
        search_fn,
        model: str = "deepseek-chat",
        temperature: float = 1.75,
        enable_reason: bool = True,
        local: bool = False,
        local_port: int = 8000,
        save_dir: str = "./runs/cot_search",
    ) -> None:
        self.evaluate_factor_fn = evaluate_factor_fn
        self.search_fn = search_fn

        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.local = bool(local)
        self.local_port = int(local_port)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_log_dir = os.path.join(self.save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

    def search_single_factor(
        self,
        seed: Dict[str, str],
        rounds: int,
        *,
        run_name: Optional[str] = None,
        verbose: bool = True,
        save_pickle: bool = True,
        save_dir: Optional[str] = "./runs/cot_search",
    ) -> Dict[str, Any]:
        assert isinstance(seed, dict) and "name" in seed and "expression" in seed
        rounds = int(rounds)

        self.save_log_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)


        if verbose:
            print("Evaluating seed factor …")
        seed_metrics = self._safe_eval(seed["expression"])["metrics"]
        if verbose:
            print(f"Seed metrics: {self._fmt_metrics(seed_metrics)}")

        chain: List[Dict[str, Any]] = []
        best_expr = seed["expression"]
        best_name = seed["name"]
        best_metrics = seed_metrics

        chain.append(
            {
                "round": 0,
                "name": best_name,
                "expression": best_expr,
                "metrics": best_metrics,
                "instruction": None,
                "generated": None,
            }
        )

        for r in range(1, rounds + 1):
            if verbose:
                print("\n" + "=" * 80)
                print(f"Round {r}/{rounds}: generating a single candidate …")

            instruction = self._compose_instruction(
                round_id=r,
                seed_name=seed["name"],
                seed_expr=seed["expression"],
                best_name=best_name,
                best_expr=best_expr,
                best_metrics=best_metrics,
                chain=chain,
            )
            if verbose:
                print("Instruction:\n" + instruction)

            start_time = time.time()
            payload = self.search_fn(
                instruction=instruction,
                model=self.model,
                N=1,
                max_try=5,
                avoid_repeat=False,
                verbose=verbose,
                debug_mode=False,
                temperature=self.temperature,
                enable_reason=self.enable_reason,
                local=self.local,
                local_port=self.local_port,
            )
            elapsed = time.time() - start_time
            # import pdb;pdb.set_trace()
            cand = self._extract_single_candidate(payload)
            if cand is None:
                if verbose:
                    print(
                        "No candidate returned in this round; continuing to next round."
                    )
                chain.append(
                    {
                        "round": r,
                        "name": best_name,
                        "expression": best_expr,
                        "metrics": best_metrics,
                        "instruction": instruction,
                        "elapsed_time": elapsed,
                        "generated": {},
                    }
                )
                continue

            cand_name = cand.get("name", f"round{r}_candidate")
            cand_expr = cand.get("expression", "")
            cand_reason = cand.get("reason", "")

            if verbose:
                print(f"Evaluating candidate: {cand_name} => {cand_expr}")
            cand_metrics = self._safe_eval(cand_expr)["metrics"]
            if verbose:
                print("Candidate metrics:", self._fmt_metrics(cand_metrics))

            improved = self._is_better(best_metrics, cand_metrics)
            if improved:
                best_name, best_expr, best_metrics = cand_name, cand_expr, cand_metrics
                if verbose:
                    print("✅ Improved — updating chain best.")
            else:
                if verbose:
                    print("⬇️  Not improved — keeping previous best.")
            
            chain.append(
                {
                    "round": r,
                    "name": best_name,
                    "expression": best_expr,
                    "metrics": best_metrics,
                    "instruction": instruction,
                    "elapsed_time": elapsed,
                    "generated": {
                        "name": cand_name,
                        "expression": cand_expr,
                        "reason": cand_reason,
                        "metrics": cand_metrics,
                        "promoted": bool(improved),
                    },
                }
            )

        summary = {
            "seed": seed,
            "seed_metrics": seed_metrics,
            "chain": chain,
            "best": {
                "name": best_name,
                "expression": best_expr,
                "metrics": best_metrics,
            },
        }

        if save_pickle:
            tag = self._safe_tag(run_name or seed["name"]) or "cot"
            if save_dir:
                out_path = os.path.join(save_dir, f"CoT_single_{tag}.pkl")
            else:
                out_path = os.path.join(self.save_dir, f"CoT_single_{tag}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(summary, f)
            summary["save_path"] = out_path
            if verbose:
                print(f"\nSaved search summary to: {out_path}")

        return summary

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
        """Compose a *fresh* instruction each step that *fully* summarizes the path so far.

        The instruction includes every round's new factor and its metric results so the
        LLM has complete context. It also states what kinds of edits are allowed,
        encouraging both parameter jumps and structural changes.
        """

        def fmt_m(m: Dict[str, float]) -> str:
            # Accept either a pure metrics dict or a record containing {'metrics': {...}}
            if (
                isinstance(m, dict)
                and "metrics" in m
                and isinstance(m["metrics"], dict)
            ):
                m = m["metrics"]
            return (
                f"IC={m.get('ic', float('nan')):.6f}, "
                f"RankIC={m.get('rank_ic', float('nan')):.6f}, "
                f"ICIR={m.get('icir', float('nan')):.6f}"
            )

        # Build a *complete* history block from seed -> current
        history_lines: List[str] = []
        for rec in chain:
            # Seed record (round 0)
            if rec.get("round") == 0:
                history_lines.append(
                    f"r0 (seed): {rec['name']} => {rec['expression']} | {fmt_m(rec)}\n"
                )
                continue

            # Later rounds contain a generated candidate and the post-round best
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
            # Snapshot of best after this round
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

        # Strong, explicit editing policy to allow parameter jumps and structural changes
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

    def _extract_single_candidate(self, payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return None
        
        if payload["quality"]:
            quality_log_name = time.strftime("log_%Y%m%d_%H%M%S.json")
            with open(os.path.join(self.save_log_dir, quality_log_name), 'w') as f:
                json.dump(payload["quality"], f)    
                        
                        
        if isinstance(payload.get("factors"), list) and payload["factors"]:
            first = payload["factors"][0]
            expr = first.get("expression")
            name = first.get("name")
            if expr and name:
                return {
                    "name": name,
                    "expression": expr,
                    "reason": first.get("reason", ""),
                }


        if isinstance(payload.get("results"), dict) and payload["results"]:
            name, expr = next(iter(payload["results"].items()))
            if expr:
                return {"name": str(name), "expression": str(expr), "reason": ""}



        return None

    def _safe_eval(self, expr: str) -> Dict[str, float]:
        try:
            return dict(self.evaluate_factor_fn(expr) or {})
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _is_better(curr: Dict[str, float], cand: Dict[str, float]) -> bool:
        c_ic, n_ic = curr.get("ic", float("-inf")), cand.get("ic", float("-inf"))
        if n_ic != c_ic:
            return n_ic > c_ic
        c_r, n_r = (
            curr.get("rank_ic", float("-inf")),
            cand.get("rank_ic", float("-inf")),
        )
        if n_r != c_r:
            return n_r > c_r
        return cand.get("ir", float("-inf")) > curr.get("ir", float("-inf"))

    @staticmethod
    def _fmt_metrics(m: Dict[str, Any]) -> str:
        # import pdb;pdb.set_trace()  # Debugging point to inspect metrics
        keys = ["ic", "rank_ic", "ir", "icir", "rank_icir"]
        parts = []
        for k in keys:
            v = m.get(k)
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.6f}")
        return ", ".join(parts)

    @staticmethod
    def _safe_tag(s: Optional[str]) -> str:
        if not s:
            return ""
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:48]


if __name__ == "__main__":

    searcher = CoTSearcher(
        evaluate_factor_fn=evaluate_factor_via_api,
        search_fn=call_qlib_search,
        model="gpt-5",
        temperature=1.0,
        enable_reason=False,
        local=False,
        local_port=8000,
    )

    seed = {
        "name": "seed_factor",
        "expression": "Mul(Sub($high, Ref(Max($high, 10), 1)), Div($volume, Mean($volume, 10)))",
    }
    summary = searcher.search_single_factor(seed=seed, rounds=10, verbose=True)
    print("\nBest Factor:", summary["best"])
