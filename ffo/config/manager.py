"""
FFO/PPO Configuration Manager

Loads and merges configuration from multiple sources (priority high→low):
  1. Environment variables  (FFO_<SECTION>_<KEY>=value)
  2. ~/.ppo/config.yaml     (user-level overrides)
  3. ./config/ffo.yaml      (project-level config)
  4. Built-in defaults
"""

from __future__ import annotations

import copy
import os
import pathlib
from typing import Any

import yaml

# ── Built-in defaults ─────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "server": {
        "backend": {
            "host": "0.0.0.0",
            "port": 19777,
            "workers": 2,
            "threads": 4,
            "timeout": 900,
            "debug": False,
            "log_level": "INFO",
        },
        "web": {
            "host": "0.0.0.0",
            "port": 19787,
            "debug": False,
            "log_level": "INFO",
        },
        "mcp": {
            "host": "0.0.0.0",
            "port": 8765,
            "transport": "stdio",
        },
    },
    "evaluation": {
        "market": "csi300",
        "start": "2023-01-01",
        "end": "2024-01-01",
        "label": "close_return",
        "fast": True,
        "topk": 50,
        "n_drop": 5,
        "use_cache": True,
        "timeout_eval": 180,
        "timeout_check": 120,
        "timeout_batch": 600,
        "n_jobs_backtest": 4,
    },
    "cache": {
        "path": "~/.ppo/factor_cache.sqlite",
        "max_entries": 50000,
    },
    "qlib": {
        "data_path": "~/.qlib/qlib_data/cn_data",
        "region": "cn",
        "instruments": "csi300",
    },
    "markets": {
        "csi300": {
            "data_path": "~/.qlib/qlib_data/cn_data",
            "region": "cn",
            "benchmark": "SH000300",
            "limit_threshold": 0.095,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
        "csi500": {
            "data_path": "~/.qlib/qlib_data/cn_data",
            "region": "cn",
            "benchmark": "SH000905",
            "limit_threshold": 0.095,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
        "csi1000": {
            "data_path": "~/.qlib/qlib_data/cn_data",
            "region": "cn",
            "benchmark": "SH000852",
            "limit_threshold": 0.095,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
        "sp500": {
            "data_path": "~/.qlib/qlib_data/us_data_ours",
            "region": "us",
            "benchmark": "^gspc",
            "limit_threshold": None,   # no daily price limit in US
            "open_cost": 0.0001,
            "close_cost": 0.0001,
            "min_cost": 0,
        },
        "nasdaq100": {
            "data_path": "~/.qlib/qlib_data/us_data_ours",
            "region": "us",
            "benchmark": "^ixic",
            "limit_threshold": None,
            "open_cost": 0.0001,
            "close_cost": 0.0001,
            "min_cost": 0,
        },
    },
    "runtime": {
        "pid_dir": "~/.ppo/pids",
        "log_dir": "~/.ppo/logs",
    },
}

# ── ENV variable prefix ───────────────────────────────────────────────────────
_ENV_PREFIX = "FFO_"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data


def _apply_env_overrides(cfg: dict) -> dict:
    """
    Apply environment variables of the form FFO_<SECTION>_<KEY>=value.
    Keys are case-insensitive. Only one level of nesting is supported.

    Examples:
        FFO_BACKEND_PORT=19321  → cfg["server"]["backend"]["port"] = 19321
        FFO_EVALUATION_MARKET=csi500
        FFO_CACHE_MAX_ENTRIES=100000
    """
    result = copy.deepcopy(cfg)

    # Flat mapping of env aliases → (section, subsection_or_none, key)
    _aliases: list[tuple[str, list[str]]] = [
        # server.backend
        ("FFO_BACKEND_HOST",    ["server", "backend", "host"]),
        ("FFO_BACKEND_PORT",    ["server", "backend", "port"]),
        ("FFO_BACKEND_WORKERS", ["server", "backend", "workers"]),
        ("FFO_BACKEND_THREADS", ["server", "backend", "threads"]),
        ("FFO_BACKEND_TIMEOUT", ["server", "backend", "timeout"]),
        ("FFO_BACKEND_DEBUG",   ["server", "backend", "debug"]),
        # server.web
        ("FFO_WEB_HOST",  ["server", "web", "host"]),
        ("FFO_WEB_PORT",  ["server", "web", "port"]),
        ("FFO_WEB_DEBUG", ["server", "web", "debug"]),
        # server.mcp
        ("FFO_MCP_PORT",      ["server", "mcp", "port"]),
        ("FFO_MCP_TRANSPORT", ["server", "mcp", "transport"]),
        # evaluation
        ("FFO_MARKET",          ["evaluation", "market"]),
        ("FFO_START",           ["evaluation", "start"]),
        ("FFO_END",             ["evaluation", "end"]),
        ("FFO_FAST",            ["evaluation", "fast"]),
        ("FFO_TOPK",            ["evaluation", "topk"]),
        ("FFO_N_DROP",          ["evaluation", "n_drop"]),
        ("FFO_TIMEOUT_EVAL",    ["evaluation", "timeout_eval"]),
        ("FFO_TIMEOUT_CHECK",   ["evaluation", "timeout_check"]),
        ("FFO_TIMEOUT_BATCH",   ["evaluation", "timeout_batch"]),
        # cache
        ("FFO_CACHE_PATH",        ["cache", "path"]),
        ("FFO_CACHE_MAX_ENTRIES", ["cache", "max_entries"]),
        # qlib
        ("FFO_QLIB_DATA_PATH", ["qlib", "data_path"]),
        ("FFO_QLIB_REGION",    ["qlib", "region"]),
        # legacy aliases still supported
        ("QLIB_DATA_PATH",     ["qlib", "data_path"]),
        ("DEFAULT_MARKET",     ["evaluation", "market"]),
        ("DEFAULT_START",      ["evaluation", "start"]),
        ("DEFAULT_END",        ["evaluation", "end"]),
        ("TIMEOUT_EVAL_SEC",   ["evaluation", "timeout_eval"]),
        ("TIMEOUT_CHECK_SEC",  ["evaluation", "timeout_check"]),
        ("TIMEOUT_BATCH_SEC",  ["evaluation", "timeout_batch"]),
        ("PORT",               ["server", "backend", "port"]),
        ("DEBUG",              ["server", "backend", "debug"]),
    ]

    for env_key, path in _aliases:
        val = os.environ.get(env_key)
        if val is None:
            continue
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        leaf_key = path[-1]
        # Type coercion based on current default type
        current = node.get(leaf_key)
        if isinstance(current, bool):
            node[leaf_key] = val.lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            node[leaf_key] = int(val)
        elif isinstance(current, float):
            node[leaf_key] = float(val)
        else:
            node[leaf_key] = val

    return result


class ConfigManager:
    """
    Central configuration object for PPO/FFO.

    Usage::

        from ffo.config import get_config
        cfg = get_config()
        port = cfg.get("server.backend.port")   # dot-path access
        port = cfg["server"]["backend"]["port"]  # dict access

    The singleton is loaded once and can be reloaded with :func:`reload_config`.
    """

    def __init__(self, project_config_path: str | pathlib.Path | None = None):
        self._cfg: dict = {}
        self._project_config_path = (
            pathlib.Path(project_config_path)
            if project_config_path
            else self._find_project_config()
        )
        self._load()

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_project_config() -> pathlib.Path:
        """
        Locate ffo.yaml using the following priority:
          1. ffo/config/ffo.yaml  — bundled inside the package (canonical)
          2. config/ffo.yaml      — legacy project-root location (backward compat)
        """
        # 1. Bundled with the ffo package (ffo/config/ffo.yaml)
        pkg_config = pathlib.Path(__file__).parent / "ffo.yaml"
        if pkg_config.exists():
            return pkg_config

        # 2. Legacy: walk up from CWD looking for config/ffo.yaml
        cwd = pathlib.Path.cwd()
        for candidate in [cwd, *cwd.parents]:
            p = candidate / "config" / "ffo.yaml"
            if p.exists():
                return p

        # Return the canonical path even if it doesn't exist yet
        return pkg_config

    def _load(self) -> None:
        # 1. Start with built-in defaults
        cfg = copy.deepcopy(_DEFAULTS)

        # 2. Project-level config
        project_cfg = _load_yaml(self._project_config_path)
        cfg = _deep_merge(cfg, project_cfg)

        # 3. User-level config (~/.ppo/config.yaml)
        user_cfg_path = pathlib.Path.home() / ".ppo" / "config.yaml"
        user_cfg = _load_yaml(user_cfg_path)
        cfg = _deep_merge(cfg, user_cfg)

        # 4. Environment variables
        cfg = _apply_env_overrides(cfg)

        self._cfg = cfg

    # ── public ────────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Reload configuration from all sources."""
        self._load()

    def get(self, dot_path: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot-notation path.

        Example::

            cfg.get("server.backend.port")        # → 19777
            cfg.get("evaluation.market")           # → "csi300"
            cfg.get("nonexistent.key", "fallback") # → "fallback"
        """
        parts = dot_path.split(".")
        node: Any = self._cfg
        for part in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(part, None)
            if node is None:
                return default
        return node

    def set(self, dot_path: str, value: Any) -> None:
        """
        Set a value using dot-notation path (in-memory only).

        To persist, call :meth:`save_user_config`.
        """
        parts = dot_path.split(".")
        node = self._cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def save_user_config(self) -> pathlib.Path:
        """
        Persist the current in-memory config to ~/.ppo/config.yaml.
        Returns the path that was written.
        """
        user_cfg_path = pathlib.Path.home() / ".ppo" / "config.yaml"
        user_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with user_cfg_path.open("w") as f:
            yaml.dump(self._cfg, f, default_flow_style=False, sort_keys=False)
        return user_cfg_path

    def as_dict(self) -> dict:
        """Return a deep copy of the full config dict."""
        return copy.deepcopy(self._cfg)

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def backend_url(self) -> str:
        host = self.get("server.backend.host", "127.0.0.1")
        port = self.get("server.backend.port", 19777)
        # Use localhost for client connections regardless of bind host
        h = "127.0.0.1" if host in ("0.0.0.0", "") else host
        return f"http://{h}:{port}"

    @property
    def web_url(self) -> str:
        host = self.get("server.web.host", "127.0.0.1")
        port = self.get("server.web.port", 19787)
        h = "127.0.0.1" if host in ("0.0.0.0", "") else host
        return f"http://{h}:{port}"

    @property
    def pid_dir(self) -> pathlib.Path:
        return pathlib.Path(self.get("runtime.pid_dir", "~/.ppo/pids")).expanduser()

    @property
    def log_dir(self) -> pathlib.Path:
        return pathlib.Path(self.get("runtime.log_dir", "~/.ppo/logs")).expanduser()

    @property
    def cache_path(self) -> pathlib.Path:
        return pathlib.Path(self.get("cache.path", "~/.ppo/factor_cache.sqlite")).expanduser()

    def get_market_config(self, market: str) -> dict:
        """
        Look up (data_path, region, benchmark) for a market name.

        Falls back to qlib defaults if the market is not in the markets table.

        Returns:
            {"data_path": str, "region": str, "benchmark": str, "instruments": str}
        """
        market_lower = market.lower()
        markets = self.get("markets", {})
        if market_lower in markets:
            cfg = dict(markets[market_lower])
            cfg.setdefault("instruments", market_lower)
            cfg["data_path"] = os.path.expanduser(cfg["data_path"])
            return cfg
        # Fallback: use top-level qlib defaults
        return {
            "data_path": os.path.expanduser(
                self.get("qlib.data_path", "~/.qlib/qlib_data/cn_data")
            ),
            "region": self.get("qlib.region", "cn"),
            "benchmark": "SH000300",
            "instruments": market_lower,
        }

    # ── dict-like access ──────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def __repr__(self) -> str:
        return f"ConfigManager(project_config={self._project_config_path})"


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Return the global :class:`ConfigManager` singleton (lazy-loaded)."""
    global _instance
    if _instance is None:
        _instance = ConfigManager()
    return _instance


def reload_config() -> ConfigManager:
    """Force reload of the global config singleton."""
    global _instance
    _instance = ConfigManager()
    return _instance
