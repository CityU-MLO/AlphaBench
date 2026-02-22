"""
Configuration dataclasses for the AlphaBench searcher platform.

YAML format (search_config.yaml)
─────────────────────────────────
searching:
  algo:
    name: ea          # "ea" | "cot" | "tot"
    param:
      # algo-specific params (passed directly to the algo)
      rounds: 10
      N: 30
      mutation_rate: 0.4
      crossover_rate: 0.6
      pool_size: 30

  model:
    name: deepseek-chat
    base_url: https://api.deepseek.com/v1
    key: ${DEEPSEEK_API_KEY}   # or literal key
    temperature: 0.7

backtesting:
  ffo_server: "127.0.0.1:19777"
  market: csi300
  benchmark: SH000300
  period_start: "2022-01-01"
  period_end: "2023-01-01"
  top_k: 30
  n_drop: 5
  fast: true

savedir: "./results"
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class AlgoConfig:
    """
    Algorithm selection and parameters.

    Attributes:
        name:       Algorithm name — "ea", "cot", or "tot".
        param:      Algorithm-specific parameter dict (forwarded verbatim to the algo).
        seed_file:  Optional path to a seed factor file (warm start).
        seed_top_k: Max seeds to pass to LLM context per round (for controller-based algos).
    """
    name: str = "ea"
    param: Dict[str, Any] = field(default_factory=dict)
    seed_file: str = ""
    seed_top_k: int = 50


@dataclass
class ModelConfig:
    """
    LLM model configuration.

    Attributes:
        name:        Model identifier (e.g. "deepseek-chat", "gpt-4o").
        base_url:    API base URL.  Leave empty for default OpenAI-compatible endpoint.
        key:         API key.  Use "${ENV_VAR}" syntax to read from environment.
        temperature: Sampling temperature (0.0–2.0).
    """
    name: str = "deepseek-chat"
    base_url: str = ""
    key: str = ""
    temperature: float = 0.7

    def resolve_key(self) -> str:
        """Resolve the API key, expanding environment variable references."""
        key = self.key
        if key.startswith("${") and key.endswith("}"):
            env_var = key[2:-1]
            resolved = os.getenv(env_var, "")
            if not resolved:
                raise ValueError(
                    f"Environment variable '{env_var}' is not set. "
                    f"Set it with: export {env_var}=<your-api-key>"
                )
            return resolved
        return key


@dataclass
class BacktestConfig:
    """
    Backtesting configuration — all evaluation goes through the FFO server.

    Attributes:
        ffo_server:    FFO API server address ("host:port").
        market:        Market universe identifier (e.g. "csi300", "csi500").
        benchmark:     Benchmark index for portfolio comparison (e.g. "SH000300").
        period_start:  Backtest start date ("YYYY-MM-DD").
        period_end:    Backtest end date ("YYYY-MM-DD").
        top_k:         Long-only portfolio size (top-K factors/stocks selected).
        n_drop:        Number of positions dropped per rebalance.
        fast:          Fast mode — compute IC metrics only (True) or full portfolio
                       backtest (False).  Fast mode is ~5-10x quicker.
        n_jobs:        Parallel evaluation workers.
        timeout:       Per-factor evaluation timeout in seconds.
    """
    ffo_server: str = "127.0.0.1:19777"
    market: str = "csi300"
    benchmark: str = "SH000300"
    period_start: str = "2022-01-01"
    period_end: str = "2023-01-01"
    top_k: int = 30
    n_drop: int = 5
    fast: bool = True
    n_jobs: int = 4
    timeout: int = 120

    def get_api_url(self) -> str:
        """Return the full HTTP URL for the FFO API server."""
        server = self.ffo_server
        if not server.startswith("http"):
            server = f"http://{server}"
        return server


@dataclass
class SearchingConfig:
    """
    Search process configuration.

    Attributes:
        algo:  Algorithm selection and parameters.
        model: LLM model to use for factor generation.
    """
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class FullConfig:
    """
    Complete configuration for a search run.

    Attributes:
        searching:   Search algorithm and model settings.
        backtesting: FFO backtesting settings.
        savedir:     Directory to save results.
    """
    searching: SearchingConfig = field(default_factory=SearchingConfig)
    backtesting: BacktestConfig = field(default_factory=BacktestConfig)
    savedir: str = "./results"


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_config_from_yaml(yaml_path: str) -> FullConfig:
    """
    Load a FullConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed FullConfig object.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return load_config_from_dict(data or {})


def load_config_from_dict(data: Dict[str, Any]) -> FullConfig:
    """
    Build a FullConfig from a raw dictionary (e.g. parsed YAML).

    Args:
        data: Configuration dictionary.

    Returns:
        FullConfig object.
    """
    # ── searching ─────────────────────────────────────────────────────
    search_data = data.get("searching", {})

    algo_data = search_data.get("algo", {})
    algo_config = AlgoConfig(
        name=algo_data.get("name", "ea"),
        param=algo_data.get("param", {}),
        seed_file=algo_data.get("seed_file", ""),
        seed_top_k=algo_data.get("seed_top_k", 50),
    )

    model_data = search_data.get("model", {})
    model_config = ModelConfig(
        name=model_data.get("name", "deepseek-chat"),
        base_url=model_data.get("base_url", ""),
        key=model_data.get("key", ""),
        temperature=float(model_data.get("temperature", 0.7)),
    )

    searching_config = SearchingConfig(algo=algo_config, model=model_config)

    # ── backtesting ───────────────────────────────────────────────────
    bt_data = data.get("backtesting", {})
    backtest_config = BacktestConfig(
        ffo_server=bt_data.get("ffo_server", "127.0.0.1:19777"),
        market=bt_data.get("market", "csi300"),
        benchmark=bt_data.get("benchmark", "SH000300"),
        period_start=bt_data.get("period_start", "2022-01-01"),
        period_end=bt_data.get("period_end", "2023-01-01"),
        top_k=int(bt_data.get("top_k", 30)),
        n_drop=int(bt_data.get("n_drop", 5)),
        fast=bool(bt_data.get("fast", True)),
        n_jobs=int(bt_data.get("n_jobs", 4)),
        timeout=int(bt_data.get("timeout", 120)),
    )

    savedir = data.get("savedir", "./results")

    return FullConfig(
        searching=searching_config,
        backtesting=backtest_config,
        savedir=savedir,
    )
