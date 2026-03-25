"""
SearchPipeline — end-to-end factor discovery pipeline.

Pipeline flow
─────────────
1. Cold start  (LLM generates initial factors randomly)
   or Warm start  (load from text/JSON/JSONL file  or  built-in alpha158)
   or Resume  (reload a previously saved pool checkpoint)

2. Run an initial backtest of the seed pool via FFO to collect baseline metrics.

3. Call the configured search algorithm (CoT / EA / ToT) which internally
   evaluates every new candidate through the same FFO Backtester.
   If verification is enabled, each evaluated factor is also evaluated on
   the validation period (val metrics are saved but NOT used for search
   decisions — no data leakage).

4. Collect the final best factors and the full search trajectory, then save
   results to disk (JSONL + pickle).

Output directory structure
──────────────────────────
<savedir>/
├── config.yaml              # copy of run configuration
├── llm_logs/                # raw LLM call logs (instruction, response, token usage)
│   └── call_<timestamp>.json
├── backtest_records/        # per-factor backtest results (search + val metrics)
│   └── round_<N>.jsonl
├── initial_pool.jsonl
├── final_pool.jsonl         # includes both search_metrics and val_metrics
├── best_factor.json
└── <algo>/
    └── ...

Usage
─────
    from searcher.config.config import load_config_from_yaml
    from searcher.pipeline import SearchPipeline

    config = load_config_from_yaml("search_config.yaml")
    pipeline = SearchPipeline(config)
    results = pipeline.run(seed_file="seeds.txt")   # warm start
    # or
    results = pipeline.run()                         # cold start
    # or
    results = pipeline.run(alpha158=True)            # warm start from alpha158
    # or
    results = pipeline.run(resume="results/final_pool.jsonl")
"""

import json
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backtester import Backtester
from .config.config import FullConfig
from .algo import create_algo
from .utils.logger import SearchLogger


# ---------------------------------------------------------------------------
# Seed loading helpers
# ---------------------------------------------------------------------------

def load_seeds_from_file(filepath: str) -> List[Dict[str, str]]:
    """
    Load seed factors from a file.

    Supports:
      - Text  (.txt): one Qlib expression per line (# comment lines ignored)
      - JSON  (.json): list of {"name", "expression"} dicts, or name→expr dict
      - JSONL (.jsonl): one JSON object per line

    Returns:
        List of {"name": str, "expression": str}
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {filepath}")

    factors = []

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    factors.append({
                        "name": item.get("name", f"seed_{i}"),
                        "expression": item.get("expression", item.get("qlib_expression_default", "")),
                    })
                elif isinstance(item, str):
                    factors.append({"name": f"seed_{i}", "expression": item})
        elif isinstance(data, dict):
            for name, expr in data.items():
                if isinstance(expr, str):
                    factors.append({"name": name, "expression": expr})

    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                factors.append({
                    "name": obj.get("name", f"seed_{i}"),
                    "expression": obj.get("expression", ""),
                })

    else:  # plain text
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                factors.append({"name": f"seed_{i}", "expression": line})

    return [f for f in factors if f.get("expression")]


def load_seeds_from_alpha158() -> List[Dict[str, str]]:
    """Load built-in alpha158 factor library as seed pool."""
    # Add AlphaBench root to sys.path if needed
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from factors.lib.alpha158 import load_factors_alpha158

    _, compile_factors = load_factors_alpha158(exclude_var="vwap", collection=["kbar", "rolling"])
    factors = []
    for name, item in compile_factors.items():
        expr = item.get("qlib_expression_default", "")
        if expr:
            factors.append({"name": name, "expression": expr})
    return factors


def cold_start_generate(config: FullConfig, logger) -> List[Dict[str, str]]:
    """Generate an initial random pool via LLM (cold start)."""
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from agent.generator_qlib_search import call_qlib_search

    logger.info("Cold start: generating initial factors via LLM …")

    instruction = (
        "Generate diverse alpha factors for stock ranking. "
        "Cover different ideas: momentum, mean reversion, volatility, "
        "volume dynamics, cross-variable relations. "
        "Each factor should be a valid Qlib expression using only: "
        "$close, $open, $high, $low, $volume."
    )

    result = call_qlib_search(
        instruction=instruction,
        model=config.searching.model.name,
        N=30,
        verbose=True,
        temperature=config.searching.model.temperature,
        enable_reason=True,
    )

    factors = [
        {"name": f.get("name", ""), "expression": f.get("expression", "")}
        for f in result.get("factors", [])
        if f.get("expression")
    ]
    logger.info(f"Cold start: generated {len(factors)} initial factors")
    return factors


# ---------------------------------------------------------------------------
# LLM Call Logger
# ---------------------------------------------------------------------------

class LLMCallLogger:
    """
    Thread-safe logger that saves every LLM search_fn call to disk.

    Each call is saved as a JSON file in <save_dir>/llm_logs/ containing:
      - timestamp, instruction (input prompt), model, N
      - response summary (n_factors, quality dict with token usage)
      - full factors list
    """

    def __init__(self, save_dir: str):
        self.log_dir = Path(save_dir) / "llm_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._call_count = 0

    def log_call(
        self,
        instruction: str,
        kwargs: Dict[str, Any],
        result: Dict[str, Any],
        elapsed: float,
    ) -> None:
        """Save one LLM call record to disk."""
        with self._lock:
            self._call_count += 1
            call_id = self._call_count

        entry = {
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 2),
            "model": kwargs.get("model", ""),
            "N": kwargs.get("N", 0),
            "temperature": kwargs.get("temperature", 0),
            "instruction": instruction,
            "n_factors_returned": len(result.get("factors", [])),
            "quality": result.get("quality", {}),
            "factors": result.get("factors", []),
        }

        fname = f"call_{call_id:04d}_{int(time.time())}.json"
        with open(self.log_dir / fname, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Val Evaluation Tracker
# ---------------------------------------------------------------------------

class ValEvalTracker:
    """
    Thread-safe tracker that evaluates factors on the validation period
    as a side effect of search-period evaluation.

    The algo only sees search-period results (no data leakage).
    Val-period results are stored internally and saved to disk.
    """

    def __init__(self, val_backtester: Backtester, save_dir: str):
        self.val_backtester = val_backtester
        self.records_dir = Path(save_dir) / "backtest_records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._val_metrics: Dict[str, Dict] = {}  # expression -> val metrics
        self._round_records: List[Dict] = []
        self._eval_count = 0

    def wrap_batch_evaluate_fn(self, search_batch_fn):
        """
        Wrap a search-period batch_evaluate_fn to also evaluate on val period.

        Returns a function with the same signature that:
        1. Calls search_batch_fn (results returned to caller/algo)
        2. Also evaluates on val period (results stored, not returned)
        """
        def wrapped(factors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
            # Search period eval (returned to algo)
            search_results = search_batch_fn(factors)

            # Val period eval (stored, not returned to algo)
            try:
                val_results = self.val_backtester.evaluate_batch(factors)
                with self._lock:
                    self._eval_count += 1
                    batch_record = []
                    for f, s_res, v_res in zip(factors, search_results, val_results):
                        expr = f.get("expression", "")
                        name = f.get("name", "")
                        if v_res.get("success") and v_res.get("metrics"):
                            self._val_metrics[expr] = v_res["metrics"]
                        batch_record.append({
                            "name": name,
                            "expression": expr,
                            "search_metrics": s_res.get("metrics", {}),
                            "val_metrics": v_res.get("metrics", {}),
                            "search_success": s_res.get("success", False),
                            "val_success": v_res.get("success", False),
                        })
                    self._round_records.append(batch_record)

                    # Save per-batch record
                    fname = f"batch_{self._eval_count:04d}.jsonl"
                    with open(self.records_dir / fname, "w", encoding="utf-8") as fh:
                        for rec in batch_record:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass  # val eval failure should not block search

            return search_results

        return wrapped

    def wrap_batch_evaluate_fn_dict(self, search_batch_dict_fn):
        """Wrap a dict-returning batch evaluate fn for val evaluation."""
        def wrapped(factors: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
            search_results = search_batch_dict_fn(factors)

            try:
                val_results = self.val_backtester.evaluate_batch(factors)
                with self._lock:
                    self._eval_count += 1
                    batch_record = []
                    for f, v_res in zip(factors, val_results):
                        expr = f.get("expression", "")
                        name = f.get("name", "")
                        s_res = search_results.get(name, {})
                        if v_res.get("success") and v_res.get("metrics"):
                            self._val_metrics[expr] = v_res["metrics"]
                        batch_record.append({
                            "name": name,
                            "expression": expr,
                            "search_metrics": s_res.get("metrics", {}),
                            "val_metrics": v_res.get("metrics", {}),
                        })
                    self._round_records.append(batch_record)

                    fname = f"batch_{self._eval_count:04d}.jsonl"
                    with open(self.records_dir / fname, "w", encoding="utf-8") as fh:
                        for rec in batch_record:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

            return search_results

        return wrapped

    def wrap_evaluate_fn(self, search_eval_fn):
        """Wrap a single-factor evaluate fn for val evaluation."""
        def wrapped(expression: str) -> Dict[str, Any]:
            search_result = search_eval_fn(expression)

            try:
                val_result = self.val_backtester.evaluate_single(expression)
                if val_result.get("success") and val_result.get("metrics"):
                    with self._lock:
                        self._val_metrics[expression] = val_result["metrics"]
            except Exception:
                pass

            return search_result

        return wrapped

    def get_val_metrics(self, expression: str) -> Dict:
        """Get cached val metrics for an expression."""
        with self._lock:
            return self._val_metrics.get(expression, {})

    def get_all_val_metrics(self) -> Dict[str, Dict]:
        """Get all cached val metrics."""
        with self._lock:
            return dict(self._val_metrics)


# ---------------------------------------------------------------------------
# SearchPipeline
# ---------------------------------------------------------------------------

class SearchPipeline:
    """
    End-to-end factor discovery pipeline.

    Parameters
    ----------
    config : FullConfig
        Loaded from YAML via load_config_from_yaml().
    logger : optional
        SearchLogger-compatible logger. Created automatically if None.
    """

    def __init__(self, config: FullConfig, logger=None):
        self.config = config
        self.logger = logger or SearchLogger("SearchPipeline")
        self.save_dir = Path(config.savedir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of the config for reproducibility
        self._save_config_copy()

        # Build the FFO Backtester for search period
        self.search_backtester = Backtester.from_config(config.backtesting, logger=self.logger)

        # Build val backtester if verification is enabled
        self.val_tracker: Optional[ValEvalTracker] = None
        if config.verification.enabled:
            val_backtester = Backtester.for_validation(
                config.backtesting, config.verification, logger=self.logger
            )
            self.val_tracker = ValEvalTracker(val_backtester, str(self.save_dir))
            self.logger.info(
                f"Verification enabled: val period {config.verification.val_start} ~ "
                f"{config.verification.val_end}"
            )

        # LLM call logger
        self.llm_logger = LLMCallLogger(str(self.save_dir))

        # Resolve LLM search_fn from model config (wrapped with logging)
        self._search_fn = self._build_search_fn()

    def run(
        self,
        seed_file: Optional[str] = None,
        alpha158: bool = False,
        resume: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Args:
            seed_file: Path to seed factors file (warm start).
            alpha158:  Use built-in alpha158 as seeds (warm start).
            resume:    Path to a JSONL checkpoint to resume from.

        Returns:
            Summary dict with "best", "final_pool", "history", "stats".
        """
        t_start = time.time()
        self.logger.info("=" * 70)
        self.logger.info("AlphaBench Search Pipeline")
        self.logger.info("=" * 70)

        # ── Step 0: Health check ────────────────────────────────────────
        if not self.search_backtester.check_health():
            self.logger.warning(
                "FFO backend may be unavailable. Continuing anyway — "
                "individual evaluations will fail gracefully."
            )

        # ── Step 1: Load / generate seeds ──────────────────────────────
        seeds = self._load_seeds(seed_file=seed_file, alpha158=alpha158, resume=resume)
        if not seeds:
            raise RuntimeError("No seed factors available — cannot start search.")

        # ── Step 2: Baseline evaluation via FFO ─────────────────────────
        if resume:
            self.logger.info(f"Resumed {len(seeds)} factors from checkpoint (skipping re-evaluation).")
            evaluated_seeds = seeds  # assume metrics already attached
        else:
            evaluated_seeds = self._evaluate_baseline(seeds)

        if not evaluated_seeds:
            raise RuntimeError("All baseline evaluations failed — check FFO server.")

        baseline_ic = self._mean_rank_ic(evaluated_seeds)
        self.logger.info(f"Baseline pool: {len(evaluated_seeds)} factors, mean RankIC={baseline_ic:.4f}")
        self._save_jsonl(evaluated_seeds, "initial_pool.jsonl")

        # ── Step 3: Run search algorithm ────────────────────────────────
        self.logger.info("")
        self.logger.info(f"Starting search: algo={self.config.searching.algo.name}")
        self.logger.info(f"  FFO server       : {self.config.backtesting.ffo_server}")
        self.logger.info(f"  Market           : {self.config.backtesting.market}")
        self.logger.info(f"  Search period    : {self.config.backtesting.search_start} ~ {self.config.backtesting.search_end}")
        if self.config.verification.enabled:
            self.logger.info(f"  Val period       : {self.config.verification.val_start} ~ {self.config.verification.val_end}")
        self.logger.info(f"  Fast mode        : {self.config.backtesting.fast}")
        self.logger.info(f"  Accept threshold : RankIC >= {self.config.backtesting.accept_threshold}")
        self.logger.info("")

        algo_result = self._run_algo(evaluated_seeds)

        # ── Step 4: Attach val metrics to final pool ─────────────────────
        final_pool = algo_result.get("final_pool", [])
        if self.val_tracker:
            all_val = self.val_tracker.get_all_val_metrics()
            for factor in final_pool:
                expr = factor.get("expression", "")
                val_m = all_val.get(expr, {})
                if val_m:
                    factor["val_metrics"] = val_m
                # Rename existing metrics to search_metrics for clarity
                if "metrics" in factor:
                    factor["search_metrics"] = factor["metrics"]

        # ── Step 5: Save results ────────────────────────────────────────
        self._save_results(algo_result)

        # ── Step 6: Mining summary ───────────────────────────────────────
        if hasattr(self.logger, "mining_summary"):
            self.logger.mining_summary(final_pool, max_show=15)

        elapsed = time.time() - t_start
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Search Complete!")
        self.logger.info("=" * 70)
        best = algo_result.get("best", {})
        self.logger.info(f"  Best factor : {best.get('name', 'N/A')}")
        self.logger.info(f"  Best RankIC : {best.get('metrics', {}).get('rank_ic', 0.0):.4f}")
        self.logger.info(f"  Best IC     : {best.get('metrics', {}).get('ic', 0.0):.4f}")
        if self.val_tracker and best.get("expression"):
            val_m = self.val_tracker.get_val_metrics(best["expression"])
            if val_m:
                self.logger.info(f"  Val RankIC  : {val_m.get('rank_ic', 0.0):.4f}")
                self.logger.info(f"  Val IC      : {val_m.get('ic', 0.0):.4f}")
        self.logger.info(f"  Final pool  : {len(final_pool)} factors")
        self.logger.info(f"  Total time  : {elapsed:.1f}s")
        self.logger.info(f"  Results dir : {self.save_dir}")

        return {
            "best": best,
            "final_pool": final_pool,
            "history": algo_result.get("history", []),
            "stats": {
                "baseline_rank_ic": baseline_ic,
                "best_rank_ic": best.get("metrics", {}).get("rank_ic", 0.0),
                "best_ic":      best.get("metrics", {}).get("ic", 0.0),
                "elapsed_sec":  elapsed,
            },
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _load_seeds(
        self,
        seed_file: Optional[str],
        alpha158: bool,
        resume: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Load seeds based on start mode."""

        if resume:
            self.logger.info(f"Resuming from checkpoint: {resume}")
            factors = []
            with open(resume, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        factors.append(json.loads(line))
            self.logger.info(f"Loaded {len(factors)} factors from checkpoint")
            return factors

        # Warm / cold start
        file_path = seed_file or self.config.searching.algo.seed_file

        if file_path:
            self.logger.info(f"Warm start: loading seeds from {file_path}")
            factors = load_seeds_from_file(file_path)
            self.logger.info(f"Loaded {len(factors)} seed factors from file")
            return factors

        if alpha158:
            self.logger.info("Warm start: loading alpha158 …")
            factors = load_seeds_from_alpha158()
            self.logger.info(f"Loaded {len(factors)} alpha158 factors")
            return factors

        # Cold start
        return cold_start_generate(self.config, self.logger)

    def _evaluate_baseline(self, factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate the initial seed pool via FFO and attach metrics."""
        self.logger.info(f"Evaluating {len(factors)} seed factors via FFO …")
        t0 = time.time()

        results = self.search_backtester.evaluate_batch(factors)

        evaluated = []
        for original, result in zip(factors, results):
            if result.get("success") and result.get("metrics"):
                merged = dict(original)
                merged["metrics"] = result["metrics"]
                evaluated.append(merged)
            else:
                err = result.get("error", "unknown error")
                self.logger.warning(f"Baseline eval failed for '{original.get('name', '?')}': {err}")

        elapsed = time.time() - t0
        self.logger.info(
            f"Baseline evaluation: {len(evaluated)}/{len(factors)} successful "
            f"in {elapsed:.1f}s"
        )

        # Also evaluate baseline on val period if verification enabled
        if self.val_tracker and evaluated:
            self.logger.info("Evaluating baseline on validation period …")
            val_results = self.val_tracker.val_backtester.evaluate_batch(evaluated)
            for factor, v_res in zip(evaluated, val_results):
                if v_res.get("success") and v_res.get("metrics"):
                    expr = factor.get("expression", "")
                    with self.val_tracker._lock:
                        self.val_tracker._val_metrics[expr] = v_res["metrics"]

        return evaluated

    def _run_algo(self, seeds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Instantiate the configured algo and run it."""
        algo_cfg = self.config.searching.algo
        model_cfg = self.config.searching.model

        # Merge model params + backtesting params into algo config
        algo_params = dict(algo_cfg.param or {})
        algo_params.setdefault("model", model_cfg.name)
        algo_params.setdefault("temperature", model_cfg.temperature)
        algo_params.setdefault("accept_threshold", self.config.backtesting.accept_threshold)

        # Build evaluate callables — wrap with val tracker if enabled
        evaluate_fn = self.search_backtester.as_evaluate_fn()
        batch_evaluate_fn = self.search_backtester.as_batch_evaluate_fn()
        batch_evaluate_fn_dict = self.search_backtester.as_batch_evaluate_fn_dict()

        if self.val_tracker:
            evaluate_fn = self.val_tracker.wrap_evaluate_fn(evaluate_fn)
            batch_evaluate_fn = self.val_tracker.wrap_batch_evaluate_fn(batch_evaluate_fn)
            batch_evaluate_fn_dict = self.val_tracker.wrap_batch_evaluate_fn_dict(batch_evaluate_fn_dict)

        algo = create_algo(
            name=algo_cfg.name,
            config=algo_params,
            evaluate_fn=evaluate_fn,
            batch_evaluate_fn=batch_evaluate_fn,
            search_fn=self._search_fn,
            batch_evaluate_fn_dict=batch_evaluate_fn_dict,
            logger=self.logger,
        )

        save_dir = str(self.save_dir / algo_cfg.name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        return algo.run(seeds=seeds, save_dir=save_dir)

    def _save_results(self, algo_result: Dict[str, Any]) -> None:
        """Persist final pool and best factor to disk."""
        final_pool = algo_result.get("final_pool", [])
        if final_pool:
            self._save_jsonl(final_pool, "final_pool.jsonl")

        best = algo_result.get("best", {})
        if best:
            # Attach val metrics to best factor
            if self.val_tracker and best.get("expression"):
                val_m = self.val_tracker.get_val_metrics(best["expression"])
                if val_m:
                    best["val_metrics"] = val_m
                if "metrics" in best:
                    best["search_metrics"] = best["metrics"]

            best_path = self.save_dir / "best_factor.json"
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Best factor saved to: {best_path}")

        pool_path = self.save_dir / "final_pool.jsonl"
        self.logger.info(f"Final pool saved to: {pool_path}")

    def _save_jsonl(self, factors: List[Dict], filename: str) -> Path:
        path = self.save_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for factor in factors:
                f.write(json.dumps(factor, ensure_ascii=False) + "\n")
        return path

    def _save_config_copy(self) -> None:
        """Save a YAML copy of the run config for reproducibility."""
        import yaml
        config_path = self.save_dir / "config.yaml"
        config_dict = {
            "savedir": self.config.savedir,
            "searching": {
                "algo": {
                    "name": self.config.searching.algo.name,
                    "seed_file": self.config.searching.algo.seed_file,
                    "param": self.config.searching.algo.param,
                },
                "model": {
                    "name": self.config.searching.model.name,
                    "temperature": self.config.searching.model.temperature,
                },
            },
            "backtesting": {
                "ffo_server": self.config.backtesting.ffo_server,
                "market": self.config.backtesting.market,
                "benchmark": self.config.backtesting.benchmark,
                "search_start": self.config.backtesting.search_start,
                "search_end": self.config.backtesting.search_end,
                "top_k": self.config.backtesting.top_k,
                "n_drop": self.config.backtesting.n_drop,
                "fast": self.config.backtesting.fast,
                "n_jobs": self.config.backtesting.n_jobs,
                "accept_threshold": self.config.backtesting.accept_threshold,
            },
            "verification": {
                "enabled": self.config.verification.enabled,
                "auto_verify": self.config.verification.auto_verify,
                "search_start": self.config.verification.search_start,
                "search_end": self.config.verification.search_end,
                "val_start": self.config.verification.val_start,
                "val_end": self.config.verification.val_end,
                "test_start": self.config.verification.test_start,
                "test_end": self.config.verification.test_end,
                "verification_forward_n": self.config.verification.verification_forward_n,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def _build_search_fn(self):
        """Build the LLM search callable from model config, wrapped with logging."""
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from agent.generator_qlib_search import call_qlib_search

        model_cfg = self.config.searching.model
        api_key = model_cfg.resolve_key() if model_cfg.key else ""
        llm_logger = self.llm_logger

        def _search_fn(instruction, model=None, N=10, **kwargs):
            t0 = time.time()
            result = call_qlib_search(
                instruction=instruction,
                model=model or model_cfg.name,
                N=N,
                temperature=kwargs.pop("temperature", model_cfg.temperature),
                **kwargs,
            )
            elapsed = time.time() - t0

            # Log the full LLM call
            llm_logger.log_call(
                instruction=instruction,
                kwargs={"model": model or model_cfg.name, "N": N, **kwargs},
                result=result,
                elapsed=elapsed,
            )

            return result

        return _search_fn

    @staticmethod
    def _mean_rank_ic(factors: List[Dict[str, Any]]) -> float:
        rics = [f.get("metrics", {}).get("rank_ic", 0.0) for f in factors if f.get("metrics")]
        return sum(rics) / max(len(rics), 1)
