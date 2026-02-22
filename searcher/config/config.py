"""
Configuration dataclasses for AlphaBench search system.

Follows EvoAlpha's config pattern but simplified:
- No DatabaseConfig (uses file-based FactorPool)
- No task management fields
- YAML format matches the AlphaBench open-source plan
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class AlgoConfig:
    """
    Algorithm configuration.

    Attributes:
        name: Algorithm name (e.g., "ea")
        param: Algorithm parameters dictionary
        seed_file: Path to seed factors file (one expression per line, or JSON)
        seed_top_k: Number of top seeds to use
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
        name: Model name (e.g., "deepseek-chat")
        base_url: API base URL
        key: API key (can use ${ENV_VAR} for environment variables)
        temperature: Sampling temperature
    """
    name: str = "deepseek-chat"
    base_url: str = ""
    key: str = ""
    temperature: float = 0.7

    def resolve_key(self) -> str:
        """Resolve API key from environment variable if needed."""
        key = self.key
        if key.startswith("${") and key.endswith("}"):
            env_var = key[2:-1]
            key = os.getenv(env_var, "")
            if not key:
                raise ValueError(f"Environment variable {env_var} not set")
        return key


@dataclass
class SearchingConfig:
    """
    Search algorithm configuration.

    Attributes:
        algo: Algorithm configuration
        model: LLM model configuration
        num_rounds: Number of search rounds
        mutation_rate: Fraction of mutation operations
        crossover_rate: Fraction of crossover operations
        window_size: Seeds fed to LLM per batch
        factors_per_batch: Factors requested per batch
        num_workers: Parallel batches per round
        batch_max_retries: Max retries if batch fails
        batch_failure_threshold: Retry if failure rate > threshold
        min_ic: Minimum IC threshold
        min_rank_ic: Minimum RankIC threshold
        adaptive_threshold: Use adaptive thresholds
        threshold_mode: "and" or "or" for threshold logic
        adaptive_threshold_ratio: Dynamic threshold ratio
        start_rounds: Initial rounds accepting all factors
        diversity_rounds: Rounds accepting by abs(IC)
    """
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    num_rounds: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    # Batch settings
    window_size: int = 5
    factors_per_batch: int = 10
    num_workers: int = 3
    batch_max_retries: int = 3
    batch_failure_threshold: float = 0.7
    # Threshold settings
    min_ic: float = 0.005
    min_rank_ic: float = 0.01
    adaptive_threshold: bool = True
    threshold_mode: str = "or"
    adaptive_threshold_ratio: float = 0.8
    start_rounds: int = 2
    diversity_rounds: int = 2


@dataclass
class BacktestConfig:
    """
    Backtesting configuration (via FFO server).

    Attributes:
        ffo_server: FFO API server address (host:port)
        market: Market identifier
        benchmark: Benchmark identifier
        period_start: Start date
        period_end: End date
        top_k: Number of top stocks to select
        n_drop: Number of stocks to drop
        fast: Use fast mode (IC only) or full mode
        n_jobs: Number of parallel backtest jobs
    """
    ffo_server: str = "127.0.0.1:19350"
    market: str = "csi300"
    benchmark: str = "SH000300"
    period_start: str = "2022-01-01"
    period_end: str = "2023-01-01"
    top_k: int = 30
    n_drop: int = 5
    fast: bool = True
    n_jobs: int = 4

    def get_api_url(self) -> str:
        """Get full API URL for factor evaluation."""
        return f"http://{self.ffo_server}"


@dataclass
class FullConfig:
    """
    Complete configuration for an AlphaBench search.

    Attributes:
        searching: Search algorithm configuration
        backtesting: Backtesting configuration
        savedir: Directory to save results
    """
    searching: SearchingConfig
    backtesting: BacktestConfig
    savedir: str = "./results"


def load_config_from_yaml(yaml_path: str) -> FullConfig:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        FullConfig object
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    return load_config_from_dict(data)


def load_config_from_dict(data: Dict[str, Any]) -> FullConfig:
    """
    Load configuration from dictionary.

    Args:
        data: Configuration dictionary

    Returns:
        FullConfig object
    """
    # Parse searching config
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
        temperature=model_data.get("temperature", 0.7),
    )

    searching_config = SearchingConfig(
        algo=algo_config,
        model=model_config,
        num_rounds=search_data.get("num_rounds", 10),
        mutation_rate=search_data.get("mutation_rate", 0.3),
        crossover_rate=search_data.get("crossover_rate", 0.7),
        window_size=search_data.get("window_size", 5),
        factors_per_batch=search_data.get("factors_per_batch", 10),
        num_workers=search_data.get("num_workers", 3),
        batch_max_retries=search_data.get("batch_max_retries", 3),
        batch_failure_threshold=search_data.get("batch_failure_threshold", 0.7),
        min_ic=search_data.get("min_ic", 0.005),
        min_rank_ic=search_data.get("min_rank_ic", 0.01),
        adaptive_threshold=search_data.get("adaptive_threshold", True),
        threshold_mode=search_data.get("threshold_mode", "or"),
        adaptive_threshold_ratio=search_data.get("adaptive_threshold_ratio", 0.8),
        start_rounds=search_data.get("start_rounds", 2),
        diversity_rounds=search_data.get("diversity_rounds", 2),
    )

    # Parse backtesting config
    bt_data = data.get("backtesting", {})
    backtest_config = BacktestConfig(
        ffo_server=bt_data.get("ffo_server", "127.0.0.1:19350"),
        market=bt_data.get("market", "csi300"),
        benchmark=bt_data.get("benchmark", "SH000300"),
        period_start=bt_data.get("period_start", "2022-01-01"),
        period_end=bt_data.get("period_end", "2023-01-01"),
        top_k=bt_data.get("top_k", 30),
        n_drop=bt_data.get("n_drop", 5),
        fast=bt_data.get("fast", True),
        n_jobs=bt_data.get("n_jobs", 4),
    )

    savedir = data.get("savedir", "./results")

    return FullConfig(
        searching=searching_config,
        backtesting=backtest_config,
        savedir=savedir,
    )
