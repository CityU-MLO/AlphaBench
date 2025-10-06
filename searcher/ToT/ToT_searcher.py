import json
import os
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from agent.generator_qlib_search import call_qlib_search
from api.factor_eval_client import (
    batch_evaluate_factors_via_api,
    evaluate_factor_via_api,
)


class ToTSearcher:
    """
    Tree-of-Thought (ToT) parallel-recursive searcher.

    Per node (round):
      1) Generate N candidates from the current parent (seed for round-1).
      2) Evaluate, then select survivors:
         - Keep only IC > seed_IC; if >= top_k (3), take top_k by (IC, RankIC, IR).
         - If some but < top_k, keep all.
         - If none beat seed_IC, keep the single best-of-round.
      3) For each survivor, run the SAME search in PARALLEL on the next depth.

    External hooks (unchanged):
      - evaluate_factor_via_api(expression: str) -> Dict[str, float]
      - batch_evaluate_factors_via_api(items: List[{"name","expression"}]) -> Dict[str, Dict]
      - call_qlib_search(instruction: str, **kwargs) -> Dict[str, Any]
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
        top_k: int = 3,
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

        # Policy
        self.top_k = int(top_k)

        # I/O
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_log_dir = os.path.join(self.save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

        # Global de-dup across whole run
        self._seen_exprs = set()
        self._seen_lock = threading.Lock()

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
        Top-level entry. Recursively expand in parallel.

        Args:
            seed: {"name": str, "expression": str}
            rounds: max depth (>=1). We treat round-1 as the first expansion from seed.
            N: number of candidates to generate per node (per parent) each round.
        """
        assert (
            isinstance(seed, dict) and "name" in seed and "expression" in seed
        ), "seed must be a dict with keys 'name' and 'expression'"

        rounds = max(1, int(rounds))
        n_expand = max(1, int(N))

        t0 = time.time()

        self.save_log_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

        # Evaluate seed (baseline)
        if verbose:
            print("Evaluating seed factor …")
        seed_eval = self._safe_eval_single(seed["expression"])
        seed_metrics = (
            seed_eval.get("metrics", {}) if isinstance(seed_eval, dict) else {}
        )
        seed_ic = float(seed_metrics.get("ic", float("-inf")))
        if verbose:
            print(f"Seed metrics: {self._fmt_metrics(seed_metrics)}")

        # init global seen with seed
        with self._seen_lock:
            self._seen_exprs.add(self._normalize_expr(seed["expression"]))

        # Run recursive parallel search starting at depth=1 (node = seed)
        history: List[Dict[str, Any]] = []
        best_global = {
            "name": seed["name"],
            "expression": seed["expression"],
            "metrics": seed_metrics,
        }

        def _update_best(
            candidate_metrics: Dict[str, float],
            candidate_name: str,
            candidate_expr: str,
        ):
            nonlocal best_global
            if self._is_better(best_global.get("metrics", {}), candidate_metrics):
                best_global = {
                    "name": candidate_name,
                    "expression": candidate_expr,
                    "metrics": candidate_metrics,
                }

        # Recursive function over a single parent node
        def _search_branch(parent: Dict[str, Any], depth: int) -> Dict[str, Any]:
            """
            parent: {"name","expression","metrics"}
            depth:  current depth (1..rounds)
            Returns: {"history": [...], "best": {...}}
            """
            branch_history: List[Dict[str, Any]] = []

            # Compose prompt (round 1 uses seed-only parent=None; later rounds include current parent)
            instr = self._compose_instruction(
                round_id=depth,
                seed=seed,
                parent=None if depth == 1 else parent,
                size=n_expand,
            )

            # LLM call to expand N candidates from this parent
            if verbose:
                print("\n" + "-" * 80)
                print(
                    f"[Depth {depth}] Expand N={n_expand} from parent: {parent['name'] if depth>1 else seed['name']}"
                )
                print("Instruction:\n" + instr)

            start_time = time.time()
            try:
                payload = self.search_fn(
                    instruction=instr,
                    model=self.model,
                    N=n_expand,
                    max_try=max_try,
                    avoid_repeat=avoid_repeat,
                    verbose=verbose,
                    debug_mode=debug_mode,
                    temperature=self.temperature,
                    enable_reason=self.enable_reason,
                    local=self.local,
                    local_port=self.local_port,
                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] LLM expand failed at depth {depth}: {e}")
                payload = {}
            elapsed_expand = time.time() - start_time

            # Parse candidates & global de-dup (by expression)
            raw_candidates = self._extract_candidates(payload, verbose=verbose)
            unique_candidates: List[Dict[str, Any]] = []
            for c in raw_candidates:
                key = self._normalize_expr(c.get("expression", ""))
                if not key:
                    continue
                with self._seen_lock:
                    if key in self._seen_exprs:
                        continue
                    self._seen_exprs.add(key)
                # ensure name uniqueness within global run
                cname = c.get("name") or f"cand_d{depth}_{len(unique_candidates)+1}"
                unique_candidates.append(
                    {
                        "name": cname,
                        "expression": c.get("expression", ""),
                        "reason": c.get("reason", ""),
                    }
                )

            if verbose:
                print(
                    f"[Depth {depth}] Generated {len(raw_candidates)} raw, kept {len(unique_candidates)} unique"
                )

            if not unique_candidates:
                rec = {
                    "depth": depth,
                    "parent": parent,
                    "instruction": instr,
                    "candidates": [],
                    "evaluations": {},
                    "ranking": [],
                    "survivors": [],
                    "elapsed_expand": elapsed_expand,
                }
                branch_history.append(rec)
                return {"history": branch_history, "best": parent}

            # Batch evaluate candidates at this node
            eval_input = [
                {"name": c["name"], "expression": c["expression"]}
                for c in unique_candidates
            ]
            # import pdb;pdb.set_trace()
            if verbose:
                print(f"[Depth {depth}] Evaluating {len(eval_input)} candidates …")
            round_results = self._safe_eval_batch(eval_input)

            # Bind evaluations
            evaluations: Dict[str, Dict[str, float]] = {}
            for c in unique_candidates:
                name = c["name"]
                m = (
                    (round_results.get(name) or {}).get("metrics", {})
                    or round_results.get(c["expression"], {})
                    or {}
                )
                evaluations[name] = m

            # Ranking
            ranked = self._rank_by_objective(evaluations)  # [(name, metrics), ...]
            best_of_node = ranked[0] if ranked else None
            if best_of_node:
                name2expr = {c["name"]: c["expression"] for c in unique_candidates}
                _update_best(
                    best_of_node[1], best_of_node[0], name2expr.get(best_of_node[0], "")
                )

            # Survivors by seed-IC rule
            def beats_seed(tup) -> bool:
                _, m = tup
                return float(m.get("ic", float("-inf"))) > seed_ic

            qualified = [x for x in ranked if beats_seed(x)]
            if len(qualified) >= self.top_k:
                survivors = qualified[: self.top_k]
            elif len(qualified) > 0:
                survivors = qualified[:]  # keep all
            else:
                survivors = ranked[:1] if ranked else []

            # Build survivor nodes
            name2expr = {c["name"]: c["expression"] for c in unique_candidates}
            survivor_nodes: List[Dict[str, Any]] = []
            for nm, mx in survivors:
                survivor_nodes.append(
                    {"name": nm, "expression": name2expr.get(nm, ""), "metrics": mx}
                )

            # Record this node
            rec = {
                "depth": depth,
                "parent": parent,
                "instruction": instr,
                "candidates": unique_candidates,
                "evaluations": evaluations,
                "ranking": ranked,
                "best_of_node": best_of_node,
                "survivors": survivors,
                "elapsed_expand": elapsed_expand,
            }
            branch_history.append(rec)

            # If reached max depth or no survivors, stop here
            if depth >= rounds or not survivor_nodes:
                # best for this branch is best_of_node (if exists) or parent
                if best_of_node:
                    best_here = {
                        "name": best_of_node[0],
                        "expression": name2expr.get(
                            best_of_node[0], parent["expression"]
                        ),
                        "metrics": best_of_node[1],
                    }
                else:
                    best_here = parent
                return {"history": branch_history, "best": best_here}

            # Else, recursively search deeper for each survivor IN PARALLEL
            results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(survivor_nodes)) as ex:
                futs = [
                    ex.submit(_search_branch, sn, depth + 1) for sn in survivor_nodes
                ]
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] child branch failed at depth {depth+1}: {e}")

            # Merge descendants
            subtree_best = {
                "name": parent["name"],
                "expression": parent["expression"],
                "metrics": parent.get("metrics", {}),
            }
            for res in results:
                if not res:
                    continue
                branch_history.extend(res.get("history", []))
                b = res.get("best")
                if b and self._is_better(
                    subtree_best.get("metrics", {}), b.get("metrics", {})
                ):
                    subtree_best = b
                # also update global best
                if b:
                    _update_best(
                        b.get("metrics", {}), b.get("name", ""), b.get("expression", "")
                    )

            return {"history": branch_history, "best": subtree_best}

        # Kick off from seed as the root parent (depth=1)
        root_parent = {
            "name": seed["name"],
            "expression": seed["expression"],
            "metrics": seed_metrics,
        }
        root_res = _search_branch(root_parent, depth=1)
        history.extend(root_res.get("history", []))
        if root_res.get("best"):
            _update_best(
                root_res["best"].get("metrics", {}),
                root_res["best"]["name"],
                root_res["best"]["expression"],
            )

        summary: Dict[str, Any] = {
            "seed": seed,
            "seed_metrics": seed_metrics,
            "history": history,
            "best": best_global,
            "elapsed_sec": time.time() - t0,
        }

        if save_pickle:
            tag = run_name or self._safe_tag(seed.get("name", "seed"))
            fname = f"tot_parallel_search_{tag}.pkl"
            path = os.path.join(save_dir or self.save_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(summary, f)
            summary["save_path"] = path
            if verbose:
                print(f"\nSaved search summary to: {path}")

        return summary

    # --------------------------- Instruction Builder ------------------------ #
    def _compose_instruction(
        self,
        *,
        round_id: int,
        seed: Dict[str, str],
        parent: Optional[Dict[str, Any]],
        size: int,
    ) -> str:
        """
        Long-text single prompt, NO backslashes.
        - round 1: only seed
        - round >=2: seed + current parent
        """

        def fmt_m(m: Dict[str, float]) -> str:
            ic = m.get("ic", float("nan"))
            ric = m.get("rank_ic", float("nan"))
            ir = m.get("ir", float("nan"))
            icir = m.get("icir", float("nan"))
            return f"IC={ic:.6f}, RankIC={ric:.6f}, IR={ir:.6f}, ICIR={icir:.6f}"

        lines = []
        lines.append(
            f"You are an expert quantitative researcher generating formulaic alpha factors."
        )
        lines.append(
            f"Return exactly {size} candidate factors as a JSON list of objects, each with:"
        )
        lines.append(f'- "name": short CamelCase identifier')
        if self.enable_reason:
            lines.append(
                f'- "reason": one concise sentence (family + why it may improve IC)'
            )
        lines.append(f'- "expression": Qlib-style expression string')
        lines.append("")
        ctx = [f"- Seed: {seed['name']} => {seed['expression']}"]
        if parent is not None:
            pm = parent.get("metrics", {})
            ctx.append(
                f"- Current parent: {parent['name']} => {parent['expression']} with {fmt_m(pm)}"
            )
        lines.append("**Context**")
        lines.extend(ctx)
        lines.append("")
        if round_id == 1:
            lines.append(
                f"You will perform a Tree-of-Thought expand step from the seed. Generate exactly {size} diverse candidates."
            )
        else:
            lines.append(
                f"You will perform a Tree-of-Thought expand step conditioned on the current parent. "
            )
            lines.append(
                f"Generate exactly {size} candidates that either (a) refine this parent for stability or (b) explore adjacent but distinct operator chains."
            )
        lines.append("")
        lines.append("**Strict expression rules**")
        lines.append("1) Allowed variables: $close, $open, $high, $low, $volume")
        lines.append(
            "2) Allowed ops (Qlib style): Mean, Std, Corr, Rank, Ref, Sum, Sub, Add, Mul, Div, Max, Min, Abs, Power, Delta, Slope, Rsquare, Quantile, If, Greater, Gt"
        )
        lines.append(
            "3) Use function calls only (e.g., Div(x, y)); never arithmetic symbols."
        )
        lines.append(
            "4) Windows must be positive integers; keep operator depth within 2–4."
        )
        lines.append("5) Every denominator in Div must be Add(<den>, 1e-12).")
        lines.append(
            "6) Use at most one Rank(...) per expression; avoid Rank(Rank(...))."
        )
        lines.append("7) Keep expressions compact; no triple Power.")
        lines.append("")
        lines.append("**ToT-expand diversity constraints**")
        lines.append(
            "• Branch across families: momentum, mean-reversion, range/breakout, volatility-scaled, volume-conditioned."
        )
        lines.append(
            "• No two candidates share the same operator skeleton (ignoring constants/windows)."
        )
        lines.append(
            "• Ensure >=30% token difference between any pair (operators + window values)."
        )
        lines.append(
            "• Prefer (short, long) pairs: short in {5,10}, long in {20,30,60}."
        )
        lines.append(
            "• Optional gating: If(Gt($close, Mean($close, L)), A, B) with L in {20,60}; keep A/B simple."
        )
        lines.append("")
        lines.append("**Output format (JSON list only)**")
        if self.enable_reason:
            lines.append(
                '[{"name":"Momentum20Adj","reason":"Momentum with volatility scaling.","expression":"Div(Delta($close,20), Add(Std($close,20), 1e-12))"}]'
            )
        else:
            lines.append(
                '[{"name":"Momentum20Adj","expression":"Div(Delta($close,20), Add(Std($close,20), 1e-12))"}]'
            )

        text = "\n".join(lines)
        # Ensure NO backslashes (user constraint)
        if "\\" in text:
            text = text.replace("\\", "")
        return text

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
        Normalize search_fn return into [{name, expression, reason}].
        Accepts either 'factors' list, or 'results' mapping.
        """
        factors: List[Dict[str, str]] = []
        if not payload:
            return factors

        if isinstance(payload, dict):
            if payload["quality"]:
                quality_log_name = time.strftime("log_%Y%m%d_%H%M%S.json")
                with open(os.path.join(self.save_log_dir, quality_log_name), "w") as f:
                    json.dump(payload["quality"], f)

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
                name, expr = it.get("name"), it.get("expression")
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
    seed = {
        "name": "Seed_KLEN",
        "expression": "Div(Sub($high, $low), Add($open, 1e-12))",
    }

    searcher = ToTSearcher(
        evaluate_factor_fn=evaluate_factor_via_api,
        batch_evaluate_factors_fn=batch_evaluate_factors_via_api,
        search_fn=call_qlib_search,
        model="deepseek-chat",
        temperature=1.2,
        enable_reason=False,
        local=False,
        local_port=8000,
        top_k=3,
    )

    # rounds = depth limit; N = candidates per node per round
    summary = searcher.search_single_factor(
        seed=seed, rounds=3, N=6, verbose=True, save_dir="./runs/tot_search"
    )
    print("Best:", summary["best"])
