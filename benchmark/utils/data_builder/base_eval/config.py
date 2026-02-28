"""
Configuration for the base evaluation dataset builders (T2: Ranking & Scoring).

YAML format
-----------
meta:
  market: CSI300          # filled by user
  start_date: "2021-01-01"
  end_date:   "2025-06-30"

# Paths to pre-computed backtesting tables
ic_table_path:     "benchmark/data/evaluate/raw/ic_table_csi300.csv"
rankic_table_path: "benchmark/data/evaluate/raw/rankic_table_csi300.csv"

# Factor pool: {factor_name: expression}
factor_pool_path:  "benchmark/data/evaluate/pool/factor_pool.json"

# Market regimes to evaluate (paper settings)
regimes:
  - name:  "2021-2025 Overall"
    start: "2021-01-01"
    end:   "2025-06-30"
  - name:  "CSI300 Bear (2023-2024)"
    start: "2023-01-01"
    end:   "2023-12-31"
  - name:  "CSI300 Bull (2024.10-2025.06)"
    start: "2024-10-01"
    end:   "2025-06-30"

# ---- Ranking task (paper §B.2.2) ----
ranking:
  # "Good" factor gate: |IC| > ic_threshold OR |RankIC| > rankic_threshold
  good_ic_threshold:     0.025
  good_rankic_threshold: 0.03
  # Difficulty settings
  settings:
    - n: 10   k: 3    # small
    - n: 20   k: 5    # medium
    - n: 40   k: 10   # large
  instances_per_setting: 50   # test cases per (regime, setting)
  seed: 42

# ---- Scoring task (paper §B.2.3) ----
scoring:
  # Signal classification: Noise if avg |IC| < noise_ic_threshold
  noise_ic_threshold: 0.01
  # Balanced counts per regime
  n_positive: 100
  n_negative: 100
  n_noise:     50
  seed: 42
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class RegimeConfig:
    name: str
    start: str
    end: str

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeConfig":
        return cls(
            name=d["name"],
            start=d["start"],
            end=d["end"],
        )


@dataclass
class RankingSettingConfig:
    n: int    # total factors shown
    k: int    # how many to pick

    @classmethod
    def from_dict(cls, d: dict) -> "RankingSettingConfig":
        return cls(n=int(d["n"]), k=int(d["k"]))


@dataclass
class RankingConfig:
    good_ic_threshold: float = 0.025
    good_rankic_threshold: float = 0.03
    settings: List[RankingSettingConfig] = field(default_factory=lambda: [
        RankingSettingConfig(10, 3),
        RankingSettingConfig(20, 5),
        RankingSettingConfig(40, 10),
    ])
    instances_per_setting: int = 50
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "RankingConfig":
        settings = [RankingSettingConfig.from_dict(s) for s in d.get("settings", [])]
        if not settings:
            settings = [
                RankingSettingConfig(10, 3),
                RankingSettingConfig(20, 5),
                RankingSettingConfig(40, 10),
            ]
        return cls(
            good_ic_threshold=float(d.get("good_ic_threshold", 0.025)),
            good_rankic_threshold=float(d.get("good_rankic_threshold", 0.03)),
            settings=settings,
            instances_per_setting=int(d.get("instances_per_setting", 50)),
            seed=int(d.get("seed", 42)),
        )


@dataclass
class ScoringConfig:
    noise_ic_threshold: float = 0.01
    n_positive: int = 100
    n_negative: int = 100
    n_noise: int = 50
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringConfig":
        return cls(
            noise_ic_threshold=float(d.get("noise_ic_threshold", 0.01)),
            n_positive=int(d.get("n_positive", 100)),
            n_negative=int(d.get("n_negative", 100)),
            n_noise=int(d.get("n_noise", 50)),
            seed=int(d.get("seed", 42)),
        )


@dataclass
class BaseEvalConfig:
    # Meta
    market: str = "CSI300"
    start_date: str = "2021-01-01"
    end_date: str = "2025-06-30"
    # Paths
    ic_table_path: str = "benchmark/data/evaluate/raw/ic_table_csi300.csv"
    rankic_table_path: str = "benchmark/data/evaluate/raw/rankic_table_csi300.csv"
    factor_pool_path: str = "benchmark/data/evaluate/pool/factor_pool.json"
    # Regimes (paper: Overall, Bear, Bull)
    regimes: List[RegimeConfig] = field(default_factory=lambda: [
        RegimeConfig("2021-2025 Overall", "2021-01-01", "2025-06-30"),
        RegimeConfig("CSI300 Bear (2023-2024)", "2023-01-01", "2023-12-31"),
        RegimeConfig("CSI300 Bull (2024.10-2025.06)", "2024-10-01", "2025-06-30"),
    ])
    ranking: RankingConfig = field(default_factory=RankingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "BaseEvalConfig":
        meta = d.get("meta", {})
        regimes_raw = d.get("regimes", [])
        regimes = [RegimeConfig.from_dict(r) for r in regimes_raw] if regimes_raw else [
            RegimeConfig("2021-2025 Overall", "2021-01-01", "2025-06-30"),
            RegimeConfig("CSI300 Bear (2023-2024)", "2023-01-01", "2023-12-31"),
            RegimeConfig("CSI300 Bull (2024.10-2025.06)", "2024-10-01", "2025-06-30"),
        ]
        return cls(
            market=meta.get("market", "CSI300"),
            start_date=meta.get("start_date", "2021-01-01"),
            end_date=meta.get("end_date", "2025-06-30"),
            ic_table_path=d.get("ic_table_path", "benchmark/data/evaluate/raw/ic_table_csi300.csv"),
            rankic_table_path=d.get("rankic_table_path", "benchmark/data/evaluate/raw/rankic_table_csi300.csv"),
            factor_pool_path=d.get("factor_pool_path", "benchmark/data/evaluate/pool/factor_pool.json"),
            regimes=regimes,
            ranking=RankingConfig.from_dict(d.get("ranking", {})),
            scoring=ScoringConfig.from_dict(d.get("scoring", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "BaseEvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
