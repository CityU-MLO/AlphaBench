"""
Base class for all factor search algorithms.

Every algo in searcher/algo/ must inherit from BaseAlgo and implement run().
The algo receives pre-built callables for LLM generation and FFO evaluation,
keeping it fully decoupled from config/infra concerns.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class BaseAlgo(ABC):
    """
    Abstract base class for factor search algorithms.

    Design principles:
    - Algos receive injectable callables → no hard-coded URLs or imports
    - Config is a plain dict keyed by algo-specific param names
    - run() accepts an already-evaluated seed pool and returns a summary dict

    Attributes:
        name: Short identifier used in YAML config (e.g. "ea", "cot", "tot")
        evaluate_fn:       fn(expression: str) -> Dict  (single factor eval)
        batch_evaluate_fn: fn(factors: List[Dict]) -> List[Dict]  (list result)
        batch_evaluate_fn_dict: fn(factors: List[Dict]) -> Dict[str, Dict]  (dict by name)
        search_fn:         fn(instruction, model, N, **kw) -> Dict  (LLM call)
        config:            algo-specific parameters dict
    """

    name: str = "base"

    def __init__(
        self,
        evaluate_fn: Callable,
        batch_evaluate_fn: Callable,
        search_fn: Callable,
        config: Dict[str, Any],
        batch_evaluate_fn_dict: Callable = None,
        logger=None,
    ):
        self.evaluate_fn = evaluate_fn
        self.batch_evaluate_fn = batch_evaluate_fn
        self.batch_evaluate_fn_dict = batch_evaluate_fn_dict or self._list_to_dict_wrapper
        self.search_fn = search_fn
        self.config = config
        self.logger = logger

    def _list_to_dict_wrapper(self, factors: List[Dict]) -> Dict[str, Dict]:
        """Fallback: convert list batch result to dict keyed by name."""
        results = self.batch_evaluate_fn(factors)
        out = {}
        for f, r in zip(factors, results):
            key = f.get("name", f.get("expression", ""))
            out[key] = r
        return out

    @abstractmethod
    def run(self, seeds: List[Dict[str, Any]], save_dir: str) -> Dict[str, Any]:
        """
        Run the search algorithm.

        Args:
            seeds: Initial factors with evaluated metrics.
                   Each item: {"name": str, "expression": str, "metrics": {...}}
                   Seeds should already be evaluated (have non-empty metrics).
            save_dir: Directory where the algo may persist intermediate outputs.

        Returns:
            Summary dict containing at minimum:
              - "best":       {"name": str, "expression": str, "metrics": {...}}
              - "history":    List of per-round records (search trajectory)
              - "final_pool": List of all discovered factors (where applicable)
        """
        raise NotImplementedError
