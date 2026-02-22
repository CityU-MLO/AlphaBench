"""
Tree-of-Thought (ToT) factor search algorithm.

ToTSearcher: parallel-recursive tree expansion from a single seed.
ToTAlgo:     BaseAlgo wrapper — runs on the top seed from the pool.
"""

import json
import os
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .base import BaseAlgo


# ---------------------------------------------------------------------------
# ToTSearcher — core implementation (moved from ToT/ToT_searcher.py)
# ---------------------------------------------------------------------------

class ToTSearcher:
    """
    Tree-of-Thought (ToT) parallel-recursive factor searcher.

    Per node (depth):
      1) Generate N candidates from the current parent.
      2) Evaluate; select survivors:
         - Keep only IC > seed_IC; if >= top_k, take top_k by (IC, RankIC, IR).
         - If some but < top_k, keep all.
         - If none beat seed_IC, keep the single best-of-node.
      3) For each survivor, recurse IN PARALLEL to next depth.

    Required external callables:
      - evaluate_fn(expr: str) -> Dict           (single eval)
      - batch_evaluate_fn(items: List[Dict]) -> Dict[str, Dict]  (keyed by name)
      - search_fn(instruction, model, N, **kw) -> Dict
    """

    def __init__(
        self,
        evaluate_fn,
        batch_evaluate_fn,
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
        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.local = bool(local)
        self.local_port = int(local_port)
        self.evaluate_fn = evaluate_fn
        self.batch_evaluate_fn = batch_evaluate_fn
        self.search_fn = search_fn
        self.top_k = int(top_k)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_log_dir = os.path.join(self.save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)
        self._seen_exprs: set = set()
        self._seen_lock = threading.Lock()

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
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert isinstance(seed, dict) and "name" in seed and "expression" in seed
        rounds = max(1, int(rounds))
        n_expand = max(1, int(N))
        save_dir = save_dir or self.save_dir

        self.save_log_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

        t0 = time.time()

        if verbose:
            print("Evaluating seed factor …")
        seed_eval = self._safe_eval_single(seed["expression"])
        seed_metrics = seed_eval.get("metrics", {}) if isinstance(seed_eval, dict) else {}
        seed_ic = float(seed_metrics.get("ic", float("-inf")))
        if verbose:
            print(f"Seed metrics: {self._fmt_metrics(seed_metrics)}")

        with self._seen_lock:
            self._seen_exprs.add(self._normalize_expr(seed["expression"]))

        history: List[Dict[str, Any]] = []
        best_global = {
            "name": seed["name"],
            "expression": seed["expression"],
            "metrics": seed_metrics,
        }

        def _update_best(metrics: Dict, name: str, expr: str):
            nonlocal best_global
            if self._is_better(best_global.get("metrics", {}), metrics):
                best_global = {"name": name, "expression": expr, "metrics": metrics}

        def _search_branch(parent: Dict[str, Any], depth: int) -> Dict[str, Any]:
            branch_history: List[Dict[str, Any]] = []

            instr = self._compose_instruction(
                round_id=depth,
                seed=seed,
                parent=None if depth == 1 else parent,
                size=n_expand,
            )

            if verbose:
                print("\n" + "-" * 80)
                print(
                    f"[Depth {depth}] Expand N={n_expand} from: "
                    f"{parent['name'] if depth > 1 else seed['name']}"
                )

            t_expand = time.time()
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
            elapsed_expand = time.time() - t_expand

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
                cname = c.get("name") or f"cand_d{depth}_{len(unique_candidates)+1}"
                unique_candidates.append({
                    "name": cname,
                    "expression": c.get("expression", ""),
                    "reason": c.get("reason", ""),
                })

            if not unique_candidates:
                branch_history.append({
                    "depth": depth, "parent": parent, "instruction": instr,
                    "candidates": [], "evaluations": {}, "ranking": [],
                    "survivors": [], "elapsed_expand": elapsed_expand,
                })
                return {"history": branch_history, "best": parent}

            eval_input = [{"name": c["name"], "expression": c["expression"]} for c in unique_candidates]
            if verbose:
                print(f"[Depth {depth}] Evaluating {len(eval_input)} candidates …")
            round_results = self._safe_eval_batch(eval_input)

            evaluations: Dict[str, Dict[str, float]] = {}
            for c in unique_candidates:
                name = c["name"]
                m = (
                    (round_results.get(name) or {}).get("metrics", {})
                    or round_results.get(c["expression"], {})
                    or {}
                )
                evaluations[name] = m

            ranked = self._rank_by_objective(evaluations)
            best_of_node = ranked[0] if ranked else None
            if best_of_node:
                name2expr = {c["name"]: c["expression"] for c in unique_candidates}
                _update_best(best_of_node[1], best_of_node[0], name2expr.get(best_of_node[0], ""))

            def beats_seed(tup) -> bool:
                _, m = tup
                return float(m.get("ic", float("-inf"))) > seed_ic

            qualified = [x for x in ranked if beats_seed(x)]
            if len(qualified) >= self.top_k:
                survivors = qualified[: self.top_k]
            elif len(qualified) > 0:
                survivors = qualified[:]
            else:
                survivors = ranked[:1] if ranked else []

            name2expr = {c["name"]: c["expression"] for c in unique_candidates}
            survivor_nodes = [
                {"name": nm, "expression": name2expr.get(nm, ""), "metrics": mx}
                for nm, mx in survivors
            ]

            branch_history.append({
                "depth": depth, "parent": parent, "instruction": instr,
                "candidates": unique_candidates, "evaluations": evaluations,
                "ranking": ranked, "best_of_node": best_of_node,
                "survivors": survivors, "elapsed_expand": elapsed_expand,
            })

            if depth >= rounds or not survivor_nodes:
                best_here = (
                    {"name": best_of_node[0], "expression": name2expr.get(best_of_node[0], ""), "metrics": best_of_node[1]}
                    if best_of_node else parent
                )
                return {"history": branch_history, "best": best_here}

            results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(survivor_nodes)) as ex:
                futs = [ex.submit(_search_branch, sn, depth + 1) for sn in survivor_nodes]
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] child branch failed at depth {depth+1}: {e}")

            subtree_best = {"name": parent["name"], "expression": parent["expression"], "metrics": parent.get("metrics", {})}
            for res in results:
                if not res:
                    continue
                branch_history.extend(res.get("history", []))
                b = res.get("best")
                if b and self._is_better(subtree_best.get("metrics", {}), b.get("metrics", {})):
                    subtree_best = b
                if b:
                    _update_best(b.get("metrics", {}), b.get("name", ""), b.get("expression", ""))

            return {"history": branch_history, "best": subtree_best}

        root_parent = {"name": seed["name"], "expression": seed["expression"], "metrics": seed_metrics}
        root_res = _search_branch(root_parent, depth=1)
        history.extend(root_res.get("history", []))
        if root_res.get("best"):
            b = root_res["best"]
            _update_best(b.get("metrics", {}), b.get("name", ""), b.get("expression", ""))

        summary: Dict[str, Any] = {
            "seed": seed,
            "seed_metrics": seed_metrics,
            "history": history,
            "best": best_global,
            "elapsed_sec": time.time() - t0,
        }

        if save_pickle:
            tag = run_name or self._safe_tag(seed.get("name", "seed"))
            fname = f"tot_search_{tag}.pkl"
            path = os.path.join(save_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(summary, f)
            summary["save_path"] = path
            if verbose:
                print(f"\nSaved search summary to: {path}")

        return summary

    # ------------------------------------------------------------------ #
    def _compose_instruction(
        self,
        *,
        round_id: int,
        seed: Dict[str, str],
        parent: Optional[Dict[str, Any]],
        size: int,
    ) -> str:
        def fmt_m(m: Dict[str, float]) -> str:
            ic = m.get("ic", float("nan"))
            ric = m.get("rank_ic", float("nan"))
            ir = m.get("ir", float("nan"))
            icir = m.get("icir", float("nan"))
            return f"IC={ic:.6f}, RankIC={ric:.6f}, IR={ir:.6f}, ICIR={icir:.6f}"

        lines = [
            f"You are an expert quantitative researcher generating formulaic alpha factors.",
            f"Return exactly {size} candidate factors as a JSON list of objects, each with:",
            f'- "name": short CamelCase identifier',
        ]
        if self.enable_reason:
            lines.append(f'- "reason": one concise sentence (family + why it may improve IC)')
        lines.append(f'- "expression": Qlib-style expression string')
        lines.append("")

        ctx = [f"- Seed: {seed['name']} => {seed['expression']}"]
        if parent is not None:
            pm = parent.get("metrics", {})
            ctx.append(f"- Current parent: {parent['name']} => {parent['expression']} with {fmt_m(pm)}")
        lines.append("**Context**")
        lines.extend(ctx)
        lines.append("")

        if round_id == 1:
            lines.append(f"Perform a Tree-of-Thought expand step from the seed. Generate exactly {size} diverse candidates.")
        else:
            lines.append(f"Perform a Tree-of-Thought expand step conditioned on the current parent.")
            lines.append(f"Generate exactly {size} candidates that either (a) refine this parent for stability or (b) explore adjacent but distinct operator chains.")
        lines.append("")
        lines.append("**Strict expression rules**")
        lines.append("1) Allowed variables: $close, $open, $high, $low, $volume")
        lines.append("2) Allowed ops (Qlib style): Mean, Std, Corr, Rank, Ref, Sum, Sub, Add, Mul, Div, Max, Min, Abs, Power, Delta, Slope, Rsquare, Quantile, If, Greater, Gt")
        lines.append("3) Use function calls only (e.g., Div(x, y)); never arithmetic symbols.")
        lines.append("4) Windows must be positive integers; keep operator depth within 2-4.")
        lines.append("5) Every denominator in Div must be Add(<den>, 1e-12).")
        lines.append("6) Use at most one Rank(...) per expression; avoid Rank(Rank(...)).")
        lines.append("7) Keep expressions compact; no triple Power.")
        lines.append("")
        lines.append("**ToT-expand diversity constraints**")
        lines.append("- Branch across families: momentum, mean-reversion, range/breakout, volatility-scaled, volume-conditioned.")
        lines.append("- No two candidates share the same operator skeleton (ignoring constants/windows).")
        lines.append("- Prefer (short, long) pairs: short in {5,10}, long in {20,30,60}.")
        lines.append("")
        lines.append("**Output format (JSON list only)**")
        if self.enable_reason:
            lines.append('[{"name":"Momentum20Adj","reason":"Momentum with volatility scaling.","expression":"Div(Delta($close,20), Add(Std($close,20), 1e-12))"}]')
        else:
            lines.append('[{"name":"Momentum20Adj","expression":"Div(Delta($close,20), Add(Std($close,20), 1e-12))"}]')

        return "\n".join(lines)

    @staticmethod
    def _rank_by_objective(evaluations: Dict[str, Dict[str, float]]) -> List[Tuple[str, Dict[str, float]]]:
        def key_fn(item):
            _, m = item
            return (float(m.get("ic", float("nan"))), float(m.get("rank_ic", float("nan"))), float(m.get("ir", float("nan"))))
        valid_items = [(k, v) for k, v in evaluations.items() if "ic" in v]
        return sorted(valid_items, key=key_fn, reverse=True)

    @staticmethod
    def _is_better(curr: Dict[str, float], cand: Dict[str, float]) -> bool:
        c_ic, n_ic = curr.get("ic", -1e9), cand.get("ic", -1e9)
        if n_ic != c_ic:
            return n_ic > c_ic
        c_r, n_r = curr.get("rank_ic", -1e9), cand.get("rank_ic", -1e9)
        if n_r != c_r:
            return n_r > c_r
        return cand.get("ir", -1e9) > curr.get("ir", -1e9)

    def _extract_candidates(self, payload: Dict[str, Any], verbose: bool = False) -> List[Dict[str, str]]:
        factors: List[Dict[str, str]] = []
        if not payload:
            return factors
        if isinstance(payload, dict):
            if payload.get("quality"):
                log_name = time.strftime("log_%Y%m%d_%H%M%S.json")
                with open(os.path.join(self.save_log_dir, log_name), "w") as f:
                    json.dump(payload["quality"], f)
            if isinstance(payload.get("factors"), list):
                for item in payload["factors"]:
                    expr = item.get("expression")
                    name = item.get("name")
                    if expr and name:
                        factors.append({"name": name, "expression": expr, "reason": item.get("reason", "")})
            elif isinstance(payload.get("results"), dict):
                for name, expr in payload["results"].items():
                    if expr:
                        factors.append({"name": str(name), "expression": str(expr), "reason": ""})
        if verbose:
            print(f"_extract_candidates: parsed {len(factors)} candidates")
        return factors

    def _safe_eval_single(self, expr: str) -> Dict[str, Any]:
        try:
            result = self.evaluate_fn(expr) or {}
            return result if isinstance(result, dict) else {}
        except Exception as e:
            return {"error": str(e)}

    def _safe_eval_batch(self, items: List[Dict[str, str]]) -> Dict[str, Dict]:
        try:
            result = self.batch_evaluate_fn(items) or {}
            if isinstance(result, dict):
                return result
            # If the function returned a list, convert to dict by name
            if isinstance(result, list):
                out = {}
                for item, res in zip(items, result):
                    out[item.get("name", item.get("expression", ""))] = res
                return out
            return {}
        except Exception as e:
            out: Dict[str, Dict] = {}
            for it in items:
                name, expr = it.get("name"), it.get("expression")
                try:
                    out[name] = self._safe_eval_single(expr)
                except Exception as ie:
                    out[name] = {"error": str(ie)}
            return out

    @staticmethod
    def _normalize_expr(expr: str) -> str:
        return "".join(expr.split()).lower()

    @staticmethod
    def _fmt_metrics(m: Dict[str, Any]) -> str:
        keys = ["ic", "rank_ic", "ir", "icir", "rank_icir"]
        parts = [f"{k}={m[k]:.6f}" for k in keys if k in m and isinstance(m[k], (int, float))]
        return ", ".join(parts) if parts else "<no-metrics>"

    @staticmethod
    def _safe_tag(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:48]


# ---------------------------------------------------------------------------
# ToTAlgo — BaseAlgo wrapper
# ---------------------------------------------------------------------------

class ToTAlgo(BaseAlgo):
    """
    ToT BaseAlgo adapter.

    Runs ToTSearcher on the top seed (by IC) from the seed pool.
    Config keys (under searching.algo.param):
      rounds:        max tree depth (default: 3)
      N:             candidates per node per depth (default: 6)
      top_k:         max survivors per node (default: 3)
      model:         LLM model name
      temperature:   sampling temperature (default: 1.0)
      enable_reason: include reasoning (default: True)
    """

    name = "tot"

    def run(self, seeds: List[Dict[str, Any]], save_dir: str) -> Dict[str, Any]:
        rounds = int(self.config.get("rounds", 3))
        N = int(self.config.get("N", 6))
        top_k = int(self.config.get("top_k", 3))
        model = self.config.get("model", "deepseek-chat")
        temperature = float(self.config.get("temperature", 1.0))
        enable_reason = bool(self.config.get("enable_reason", True))

        searcher = ToTSearcher(
            evaluate_fn=self.evaluate_fn,
            batch_evaluate_fn=self.batch_evaluate_fn_dict,
            search_fn=self.search_fn,
            model=model,
            temperature=temperature,
            enable_reason=enable_reason,
            save_dir=save_dir,
            top_k=top_k,
        )

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
            N=N,
            save_dir=save_dir,
            verbose=True,
        )

        # Collect all candidates from history into final_pool
        final_pool: List[Dict] = []
        seen = set()
        for rec in summary.get("history", []):
            for c in rec.get("candidates", []):
                expr = c.get("expression", "")
                if expr and expr not in seen:
                    seen.add(expr)
                    final_pool.append({
                        "name": c.get("name", ""),
                        "expression": expr,
                        "metrics": rec.get("evaluations", {}).get(c.get("name", ""), {}),
                    })

        return {
            "best": summary.get("best", {}),
            "history": summary.get("history", []),
            "final_pool": final_pool,
        }
