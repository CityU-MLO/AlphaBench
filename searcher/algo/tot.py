"""
Tree-of-Thought (ToT) factor search algorithm.

ToTSearcher: parallel-recursive tree expansion from a single seed.
ToTAlgo:     BaseAlgo wrapper — runs on the top seed from the pool.
"""

import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAlgo


# ---------------------------------------------------------------------------
# ToTSearcher — core implementation
# ---------------------------------------------------------------------------

class ToTSearcher:
    """
    Tree-of-Thought (ToT) parallel-recursive factor searcher.

    Per node (depth):
      1) Generate N candidates from the current parent via LLM.
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
        accept_threshold: float = 0.0,
        logger=None,
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
        self.accept_threshold = float(accept_threshold)
        self.save_dir = save_dir
        self.logger = logger
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
        avoid_repeat: bool = True,
        max_try: int = 5,
        debug_mode: bool = False,
        save_pickle: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert isinstance(seed, dict) and "name" in seed and "expression" in seed
        rounds   = max(1, int(rounds))
        n_expand = max(1, int(N))
        save_dir = save_dir or self.save_dir

        self.save_log_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_log_dir, exist_ok=True)

        t0 = time.time()

        self._log(f"  Seed: {seed['name']}  [{seed['expression'][:70]}]")
        self._log("  Evaluating seed …")
        seed_batch   = self._safe_eval_batch([{"name": seed["name"], "expression": seed["expression"]}])
        seed_eval    = seed_batch.get(seed["name"]) or {}
        seed_metrics = seed_eval.get("metrics", {}) if isinstance(seed_eval, dict) else {}
        seed_rank_ic = float(seed_metrics.get("rank_ic", float("-inf")))
        self._log(f"  Seed  {self._fmt_metrics(seed_metrics)}")

        with self._seen_lock:
            self._seen_exprs.add(self._normalize_expr(seed["expression"]))

        history: List[Dict[str, Any]] = []
        best_global = {
            "name":       seed["name"],
            "expression": seed["expression"],
            "metrics":    seed_metrics,
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

            self._log(
                f"  [Depth {depth}] Expanding N={n_expand} from: "
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
                    verbose=False,
                    debug_mode=debug_mode,
                    temperature=self.temperature,
                    enable_reason=self.enable_reason,
                    local=self.local,
                    local_port=self.local_port,
                )
            except Exception as e:
                self._log(f"LLM expand failed at depth {depth}: {e}", "warning")
                payload = {}
            elapsed_expand = time.time() - t_expand
            self._log(f"  [Depth {depth}] LLM done {elapsed_expand:.1f}s")

            raw_candidates    = self._extract_candidates(payload)
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
                    "name":       cname,
                    "expression": c.get("expression", ""),
                    "reason":     c.get("reason", ""),
                })

            if not unique_candidates:
                self._log(f"  [Depth {depth}] No unique candidates; pruning branch.")
                branch_history.append({
                    "depth": depth, "parent": parent, "instruction": instr,
                    "candidates": [], "evaluations": {}, "ranking": [],
                    "survivors": [], "elapsed_expand": elapsed_expand,
                })
                return {"history": branch_history, "best": parent}

            eval_input = [{"name": c["name"], "expression": c["expression"]} for c in unique_candidates]
            self._log(f"  [Depth {depth}] Evaluating {len(eval_input)} candidates …")
            t_eval = time.time()
            round_results = self._safe_eval_batch(eval_input)
            eval_elapsed = time.time() - t_eval
            n_ok = sum(1 for v in round_results.values() if v.get("metrics"))
            self._log(f"  [Depth {depth}] Eval done {eval_elapsed:.1f}s  {n_ok}/{len(eval_input)} successful")

            evaluations: Dict[str, Dict[str, float]] = {}
            for c in unique_candidates:
                name = c["name"]
                m = (
                    (round_results.get(name) or {}).get("metrics", {})
                    or round_results.get(c["expression"], {})
                    or {}
                )
                evaluations[name] = m

            ranked       = self._rank_by_objective(evaluations)
            best_of_node = ranked[0] if ranked else None
            if best_of_node:
                name2expr = {c["name"]: c["expression"] for c in unique_candidates}
                _update_best(best_of_node[1], best_of_node[0], name2expr.get(best_of_node[0], ""))

            def beats_seed(tup) -> bool:
                _, m = tup
                cand_ric = float(m.get("rank_ic", float("-inf")))
                return cand_ric > seed_rank_ic and cand_ric >= self.accept_threshold

            qualified = [x for x in ranked if beats_seed(x)]
            if len(qualified) >= self.top_k:
                survivors = qualified[: self.top_k]
            elif len(qualified) > 0:
                survivors = qualified[:]
            else:
                survivors = ranked[:1] if ranked else []

            if qualified:
                self._log(
                    f"  [Depth {depth}] Survivors: {len(survivors)}"
                    f"  (beat seed RankIC={seed_rank_ic:.4f}, threshold={self.accept_threshold:.4f})"
                )
            else:
                self._log(f"  [Depth {depth}] Survivors: {len(survivors)} (best-of-node fallback)")

            name2expr     = {c["name"]: c["expression"] for c in unique_candidates}
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

            self._log(f"  [Depth {depth}] Launching {len(survivor_nodes)} parallel branches …")
            results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(survivor_nodes)) as ex:
                futs = [ex.submit(_search_branch, sn, depth + 1) for sn in survivor_nodes]
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        self._log(f"Child branch failed at depth {depth+1}: {e}", "warning")

            subtree_best = {
                "name":       parent["name"],
                "expression": parent["expression"],
                "metrics":    parent.get("metrics", {}),
            }
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

        root_parent = {
            "name":       seed["name"],
            "expression": seed["expression"],
            "metrics":    seed_metrics,
        }
        root_res = _search_branch(root_parent, depth=1)
        history.extend(root_res.get("history", []))
        if root_res.get("best"):
            b = root_res["best"]
            _update_best(b.get("metrics", {}), b.get("name", ""), b.get("expression", ""))

        summary: Dict[str, Any] = {
            "seed":        seed,
            "seed_metrics": seed_metrics,
            "history":     history,
            "best":        best_global,
            "elapsed_sec": time.time() - t0,
        }

        if save_pickle:
            tag   = run_name or self._safe_tag(seed.get("name", "seed"))
            # Pickle (full)
            pkl_fname = f"tot_search_{tag}.pkl"
            pkl_path  = os.path.join(save_dir, pkl_fname)
            with open(pkl_path, "wb") as fh:
                pickle.dump(summary, fh)
            summary["save_path"] = pkl_path
            self._log(f"  Saved: {pkl_path}")

            # JSON (human-readable: all candidates, expressions, metrics per depth)
            json_summary = {
                "seed": {"name": seed["name"], "expression": seed["expression"]},
                "seed_metrics": seed_metrics,
                "best": {
                    "name": best_global.get("name", ""),
                    "expression": best_global.get("expression", ""),
                    "metrics": best_global.get("metrics", {}),
                },
                "elapsed_sec": summary["elapsed_sec"],
                "depths": [],
            }
            for rec in history:
                depth_rec = {
                    "depth": rec.get("depth"),
                    "parent": {
                        "name": (rec.get("parent") or {}).get("name", ""),
                        "expression": (rec.get("parent") or {}).get("expression", ""),
                    } if rec.get("parent") else None,
                    "elapsed_expand": rec.get("elapsed_expand"),
                    "candidates": [
                        {
                            "name": c.get("name", ""),
                            "expression": c.get("expression", ""),
                            "reason": c.get("reason", ""),
                            "metrics": rec.get("evaluations", {}).get(c.get("name", ""), {}),
                        }
                        for c in rec.get("candidates", [])
                    ],
                    "survivors": [
                        {"name": s[0], "metrics": s[1]} if isinstance(s, (list, tuple))
                        else {"name": s.get("name", ""), "metrics": s.get("metrics", {})}
                        for s in rec.get("survivors", [])
                    ],
                }
                json_summary["depths"].append(depth_rec)

            json_path = os.path.join(save_dir, f"tot_search_{tag}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(json_summary, fh, indent=2, ensure_ascii=False, default=str)

        return summary

    # ------------------------------------------------------------------ #
    # Instruction builder
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
            ic   = m.get("ic",      float("nan"))
            ric  = m.get("rank_ic", float("nan"))
            ir   = m.get("ir",      float("nan"))
            icir = m.get("icir",    float("nan"))
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

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rank_by_objective(
        evaluations: Dict[str, Dict[str, float]],
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Sort by RankIC (primary), IC (secondary), IR (tiebreaker)."""
        def key_fn(item):
            _, m = item
            return (
                float(m.get("rank_ic", float("nan"))),
                float(m.get("ic",      float("nan"))),
                float(m.get("ir",      float("nan"))),
            )
        valid_items = [(k, v) for k, v in evaluations.items() if "rank_ic" in v or "ic" in v]
        return sorted(valid_items, key=key_fn, reverse=True)

    @staticmethod
    def _is_better(curr: Dict[str, float], cand: Dict[str, float]) -> bool:
        """RankIC is the primary metric; IC is tiebreaker; then IR."""
        c_r, n_r = curr.get("rank_ic", -1e9), cand.get("rank_ic", -1e9)
        if n_r != c_r:
            return n_r > c_r
        c_ic, n_ic = curr.get("ic", -1e9), cand.get("ic", -1e9)
        if n_ic != c_ic:
            return n_ic > c_ic
        return cand.get("icir", -1e9) > curr.get("icir", -1e9)

    def _extract_candidates(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        factors: List[Dict[str, str]] = []
        if not payload:
            return factors
        if isinstance(payload, dict):
            if payload.get("quality"):
                log_name = time.strftime("log_%Y%m%d_%H%M%S.json")
                with open(os.path.join(self.save_log_dir, log_name), "w") as fh:
                    json.dump(payload["quality"], fh)
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
        keys  = ["ic", "rank_ic", "ir", "icir", "rank_icir"]
        parts = [f"{k}={m[k]:.4f}" for k in keys if k in m and isinstance(m[k], (int, float))]
        return ", ".join(parts) if parts else "<no-metrics>"

    @staticmethod
    def _safe_tag(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:48]

    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)


# ---------------------------------------------------------------------------
# ToTAlgo — BaseAlgo wrapper
# ---------------------------------------------------------------------------

class ToTAlgo(BaseAlgo):
    """
    ToT BaseAlgo adapter.

    Runs ToTSearcher on seeds from the pool. When workers > 1, runs
    multiple independent tree searches in parallel, each starting from
    a different seed. Results are merged at the end.

    Config keys (under searching.algo.param):
      rounds:        max tree depth (default: 3)
      N:             candidates per node per depth (default: 6)
      top_k:         max survivors per node (default: 3)
      workers:       number of parallel search trees (default: 1)
      model:         LLM model name
      temperature:   sampling temperature (default: 1.0)
      enable_reason: include reasoning (default: True)
    """

    name = "tot"

    def run(self, seeds: List[Dict[str, Any]], save_dir: str) -> Dict[str, Any]:
        rounds           = int(self.config.get("rounds", 3))
        N                = int(self.config.get("N", 6))
        top_k            = int(self.config.get("top_k", 3))
        workers          = int(self.config.get("workers", 1))
        model            = self.config.get("model", "deepseek-chat")
        temperature      = float(self.config.get("temperature", 1.0))
        enable_reason    = bool(self.config.get("enable_reason", True))
        accept_threshold = float(self.config.get("accept_threshold", 0.0))

        # Select top-W seeds (one per worker)
        valid_seeds = [s for s in seeds if s.get("metrics")]
        if not valid_seeds:
            valid_seeds = seeds
        ranked_seeds = sorted(
            valid_seeds,
            key=lambda s: s.get("metrics", {}).get("ic", float("-inf")),
            reverse=True,
        )
        selected_seeds = ranked_seeds[:max(1, workers)]

        def _run_tree(seed_item, worker_id):
            worker_dir = os.path.join(save_dir, f"worker_{worker_id}")
            os.makedirs(worker_dir, exist_ok=True)
            searcher = ToTSearcher(
                evaluate_fn=self.evaluate_fn,
                batch_evaluate_fn=self.batch_evaluate_fn_dict,
                search_fn=self.search_fn,
                model=model,
                temperature=temperature,
                enable_reason=enable_reason,
                save_dir=worker_dir,
                top_k=top_k,
                accept_threshold=accept_threshold,
                logger=self.logger,
            )
            seed_input = {"name": seed_item["name"], "expression": seed_item["expression"]}
            return searcher.search_single_factor(
                seed=seed_input,
                rounds=rounds,
                N=N,
                run_name=f"worker{worker_id}_{seed_item['name']}",
                save_dir=worker_dir,
            )

        if len(selected_seeds) == 1:
            summaries = [_run_tree(selected_seeds[0], 0)]
        else:
            self._log(f"Running {len(selected_seeds)} ToT workers in parallel …")
            summaries = [None] * len(selected_seeds)
            with ThreadPoolExecutor(max_workers=len(selected_seeds)) as ex:
                futs = {
                    ex.submit(_run_tree, seed, i): i
                    for i, seed in enumerate(selected_seeds)
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        summaries[idx] = fut.result()
                    except Exception as e:
                        self._log(f"Worker {idx} failed: {e}", "warning")
            summaries = [s for s in summaries if s is not None]

        # Merge results from all workers
        all_history = []
        final_pool: List[Dict] = []
        seen = set()
        best_global = {}

        for summary in summaries:
            all_history.extend(summary.get("history", []))
            for rec in summary.get("history", []):
                for c in rec.get("candidates", []):
                    expr = c.get("expression", "")
                    if expr and expr not in seen:
                        seen.add(expr)
                        final_pool.append({
                            "name":       c.get("name", ""),
                            "expression": expr,
                            "metrics":    rec.get("evaluations", {}).get(c.get("name", ""), {}),
                        })
            candidate = summary.get("best", {})
            if not best_global or self._is_better_static(best_global, candidate):
                best_global = candidate

        return {
            "best":       best_global,
            "history":    all_history,
            "final_pool": final_pool,
        }

    @staticmethod
    def _is_better_static(curr, cand):
        c_m = curr.get("metrics", {})
        n_m = cand.get("metrics", {})
        c_r = c_m.get("rank_ic", float("-inf"))
        n_r = n_m.get("rank_ic", float("-inf"))
        if n_r != c_r:
            return n_r > c_r
        return n_m.get("ic", float("-inf")) > c_m.get("ic", float("-inf"))

    def _log(self, msg, level="info"):
        if self.logger:
            getattr(self.logger, level)(msg)
