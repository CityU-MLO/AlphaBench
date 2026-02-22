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

4. Collect the final best factors and the full search trajectory, then save
   results to disk (JSONL + pickle).

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

        # Build the FFO Backtester from config
        self.backtester = Backtester.from_config(config.backtesting, logger=self.logger)

        # Resolve LLM search_fn from model config
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
        if not self.backtester.check_health():
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

        baseline_ic = self._mean_ic(evaluated_seeds)
        self.logger.info(f"Baseline pool: {len(evaluated_seeds)} factors, mean IC={baseline_ic:.4f}")
        self._save_jsonl(evaluated_seeds, "initial_pool.jsonl")

        # ── Step 3: Run search algorithm ────────────────────────────────
        self.logger.info("")
        self.logger.info(f"Starting search: algo={self.config.searching.algo.name}")
        self.logger.info(f"  FFO server : {self.config.backtesting.ffo_server}")
        self.logger.info(f"  Market     : {self.config.backtesting.market}")
        self.logger.info(f"  Period     : {self.config.backtesting.period_start} ~ {self.config.backtesting.period_end}")
        self.logger.info(f"  Fast mode  : {self.config.backtesting.fast}")
        self.logger.info("")

        algo_result = self._run_algo(evaluated_seeds)

        # ── Step 4: Save results ────────────────────────────────────────
        self._save_results(algo_result)

        elapsed = time.time() - t_start
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Search Complete!")
        self.logger.info("=" * 70)
        best = algo_result.get("best", {})
        self.logger.info(f"  Best factor : {best.get('name', 'N/A')}")
        self.logger.info(f"  Best IC     : {best.get('metrics', {}).get('ic', 0.0):.4f}")
        self.logger.info(f"  Best RankIC : {best.get('metrics', {}).get('rank_ic', 0.0):.4f}")
        self.logger.info(f"  Final pool  : {len(algo_result.get('final_pool', []))} factors")
        self.logger.info(f"  Total time  : {elapsed:.1f}s")

        return {
            "best": best,
            "final_pool": algo_result.get("final_pool", []),
            "history": algo_result.get("history", []),
            "stats": {
                "baseline_ic": baseline_ic,
                "best_ic": best.get("metrics", {}).get("ic", 0.0),
                "elapsed_sec": elapsed,
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

        results = self.backtester.evaluate_batch(factors)

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
        return evaluated

    def _run_algo(self, seeds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Instantiate the configured algo and run it."""
        algo_cfg = self.config.searching.algo
        model_cfg = self.config.searching.model

        # Merge model params into algo config so algos can use the right LLM
        algo_params = dict(algo_cfg.param or {})
        algo_params.setdefault("model", model_cfg.name)
        algo_params.setdefault("temperature", model_cfg.temperature)

        algo = create_algo(
            name=algo_cfg.name,
            config=algo_params,
            evaluate_fn=self.backtester.as_evaluate_fn(),
            batch_evaluate_fn=self.backtester.as_batch_evaluate_fn(),
            search_fn=self._search_fn,
            batch_evaluate_fn_dict=self.backtester.as_batch_evaluate_fn_dict(),
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

    def _build_search_fn(self):
        """Build the LLM search callable from model config."""
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from agent.generator_qlib_search import call_qlib_search

        model_cfg = self.config.searching.model
        api_key = model_cfg.resolve_key() if model_cfg.key else ""

        def _search_fn(instruction, model=None, N=10, **kwargs):
            return call_qlib_search(
                instruction=instruction,
                model=model or model_cfg.name,
                N=N,
                temperature=kwargs.pop("temperature", model_cfg.temperature),
                **kwargs,
            )

        return _search_fn

    @staticmethod
    def _mean_ic(factors: List[Dict[str, Any]]) -> float:
        ics = [f.get("metrics", {}).get("ic", 0.0) for f in factors if f.get("metrics")]
        return sum(ics) / max(len(ics), 1)
