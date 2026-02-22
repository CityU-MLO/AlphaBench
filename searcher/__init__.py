# AlphaBench Searcher — general-purpose factor discovery platform
#
# Public API
# ──────────
#   SearchPipeline      end-to-end pipeline (seeds → baseline → algo → results)
#   Backtester          FFO-backed factor evaluator
#   create_algo         instantiate any search algo by name
#   register_algo       register a custom algo
#   list_algos          list all registered algo names
#   load_config_from_yaml / load_config_from_dict
#
# Algo classes (searcher/algo/)
# ──────────────────────────────
#   BaseAlgo, CoTAlgo, EAAlgo, ToTAlgo
#   CoTSearcher, EA_Searcher, ToTSearcher

from .pipeline import SearchPipeline
from .backtester import Backtester
from .algo import (
    BaseAlgo,
    CoTAlgo,
    CoTSearcher,
    EAAlgo,
    EA_Searcher,
    ToTAlgo,
    ToTSearcher,
    create_algo,
    register_algo,
    list_algos,
)
from .config.config import (
    FullConfig,
    SearchingConfig,
    BacktestConfig,
    AlgoConfig,
    ModelConfig,
    load_config_from_yaml,
    load_config_from_dict,
)

__all__ = [
    # Pipeline
    "SearchPipeline",
    "Backtester",
    # Algo registry
    "create_algo",
    "register_algo",
    "list_algos",
    # Algo classes
    "BaseAlgo",
    "CoTAlgo",
    "CoTSearcher",
    "EAAlgo",
    "EA_Searcher",
    "ToTAlgo",
    "ToTSearcher",
    # Config
    "FullConfig",
    "SearchingConfig",
    "BacktestConfig",
    "AlgoConfig",
    "ModelConfig",
    "load_config_from_yaml",
    "load_config_from_dict",
]
