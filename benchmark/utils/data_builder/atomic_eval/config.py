"""
Configuration classes for atomic evaluation dataset builders.

Usage (from dict):
    cfg = AtomicEvalConfig.from_dict({
        "meta": {"market": "CSI300", "start_date": "2024-01-01", "end_date": "2025-01-01"},
        "noise_threshold": {"ic": 0.005, "rankic": 0.025, "abs": True, "condition": "and"},
        "split": {"sample_n": 300, "train_ratio": 0.6, "val_ratio": 0.1, "test_ratio": 0.3},
    })

Usage (from yaml):
    cfg = AtomicEvalConfig.from_yaml("path/to/config.yaml")

YAML format:
    meta:
      market: CSI300
      start_date: "2024-01-01"
      end_date: "2025-01-01"

    noise_threshold:           # only needed for Task 2 (binary noise)
      ic: 0.005
      rankic: 0.025
      abs: true                # true => use |ic|; false => accept negative IC as valid
      condition: "and"         # "and" => must pass BOTH ic & rankic gates; "or" => either

    split:                     # option A: build a new split
      sample_n: 300            # sample up to N from pool (capped at pool size)
      train_ratio: 0.6
      val_ratio: 0.1
      test_ratio: 0.3

    # OR option B: copy an existing split (cross-market transfer)
    refer: "/path/to/existing/split/dir"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import yaml


@dataclass
class NoiseThreshold:
    """Thresholds that define a factor as 'noise' for Task 2."""
    ic: float = 0.005
    rankic: float = 0.025
    # If True, compare |ic| and |rankic| against thresholds (negative IC also valid).
    # If False, only positive IC/RankIC counts as signal; negatives are treated as noise.
    abs: bool = True
    # Whether both gates must fire ('and') or either one ('or') to label as noise.
    condition: Literal["and", "or"] = "and"

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseThreshold":
        return cls(
            ic=float(d.get("ic", 0.005)),
            rankic=float(d.get("rankic", 0.025)),
            abs=bool(d.get("abs", True)),
            condition=d.get("condition", "and"),
        )


@dataclass
class SplitConfig:
    """Configuration for building a fresh train/val/test split."""
    # If sample_n > available pool size, pool size is used.
    sample_n: int = 300
    train_ratio: float = 0.6
    val_ratio: float = 0.0
    test_ratio: float = 0.4

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "SplitConfig":
        return cls(
            sample_n=int(d.get("sample_n", 300)),
            train_ratio=float(d.get("train_ratio", 0.6)),
            val_ratio=float(d.get("val_ratio", 0.0)),
            test_ratio=float(d.get("test_ratio", 0.4)),
        )


@dataclass
class MetaConfig:
    """Market / date metadata for the dataset."""
    market: str = "CSI300"
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"

    @classmethod
    def from_dict(cls, d: dict) -> "MetaConfig":
        return cls(
            market=d.get("market", "CSI300"),
            start_date=d.get("start_date", "2024-01-01"),
            end_date=d.get("end_date", "2025-01-01"),
        )


@dataclass
class AtomicEvalConfig:
    """
    Top-level config for building atomic evaluation datasets.

    Exactly one of `split` or `refer` must be set:
    - split  => build a new train/val/test split from scratch.
    - refer  => copy factor indices from an existing split directory
                (for cross-market transfer experiments).
    """
    meta: MetaConfig = field(default_factory=MetaConfig)
    noise_threshold: Optional[NoiseThreshold] = None
    split: Optional[SplitConfig] = None
    refer: Optional[str] = None   # path to existing split dir
    seed: int = 42

    def __post_init__(self):
        if self.split is None and self.refer is None:
            raise ValueError("Either 'split' or 'refer' must be specified.")
        if self.split is not None and self.refer is not None:
            raise ValueError("Only one of 'split' or 'refer' may be specified, not both.")

    @classmethod
    def from_dict(cls, d: dict) -> "AtomicEvalConfig":
        meta = MetaConfig.from_dict(d.get("meta", {}))

        noise_threshold = None
        if "noise_threshold" in d:
            noise_threshold = NoiseThreshold.from_dict(d["noise_threshold"])

        split = None
        if "split" in d:
            split = SplitConfig.from_dict(d["split"])

        refer = d.get("refer", None)
        seed = int(d.get("seed", 42))

        return cls(
            meta=meta,
            noise_threshold=noise_threshold,
            split=split,
            refer=refer,
            seed=seed,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AtomicEvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
