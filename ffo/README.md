# FFO — Factor Feature Oracle

High-performance quantitative factor evaluation platform with multi-market support (China A-share & US equities).
Exposes a REST API, a typed Python API, a CLI (`ppo`), an MCP server for LLM agents, and a web UI with portfolio backtest visualization.

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Cache & Storage System](#cache--storage-system)
- [Multi-Market Worker Pool](#multi-market-worker-pool)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Configuration](#configuration)
- [MCP Server (LLM Agents)](#mcp-server-llm-agents)
- [REST API Endpoints](#rest-api-endpoints)
- [Web UI](#web-ui)
- [Package Structure](#package-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# From the project root — installs the ppo CLI and ffo package
pip install -e .

# Verify
ppo --version
```

After installation the `ppo` command is available system-wide.

---

## Quick Start

### 1. Start the backend

```bash
ppo start backend        # starts on http://127.0.0.1:19777
ppo status               # confirm it's running
```

### 2. Evaluate a factor

**CLI:**
```bash
ppo eval "Rank($close, 20)"
ppo eval "Mean($volume, 5) / Std($volume, 20)" --market csi500
ppo eval "Rank($close, 20)" --market sp500    # US market
```

**Python:**
```python
from ffo.api import evaluate_factor

result = evaluate_factor("Rank($close, 20)", market="csi300")
if result.success:
    print(f"IC={result.metrics.ic:.4f}  Rank_IC={result.metrics.rank_ic:.4f}")
```

### 3. Open the web UI

```bash
ppo start web   # http://127.0.0.1:19787
```

The web UI provides interactive factor evaluation, daily IC charts, portfolio backtest with
holdings/actions detail, and factor comparison.

### 4. Stop services

```bash
ppo stop all
```

---

## Architecture

```
                  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                  │ Python API   │    │ CLI (ppo)    │    │ MCP Server   │
                  │ ffo.api.*    │    │ ffo.cli      │    │ ffo.mcp      │
                  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                         │                   │                   │
                         └───────────┬───────┘───────────────────┘
                                     │  HTTP
                              ┌──────▼──────┐
                              │ Backend API  │ :19777
                              │ Flask        │
                              │ factors.py   │
                              └──┬───────┬──┘
                                 │       │
                    ┌────────────▼─┐   ┌─▼────────────┐
                    │ Factor Eval  │   │ Worker Pool   │
                    │ Subprocesses │   │ (persistent)  │
                    │ (IC/RankIC)  │   │               │
                    └──────┬───────┘   │  ┌─────────┐  │
                           │           │  │CN Worker │  │  qlib.init(cn_data)
                    ┌──────▼───────┐   │  └─────────┘  │
                    │ FactorStore  │   │  ┌─────────┐  │
                    │ (SQLite +    │   │  │US Worker │  │  qlib.init(us_data)
                    │  pickle)     │   │  └─────────┘  │
                    └──────────────┘   └───────────────┘
                                              │
                              ┌───────────────▼──────────┐
                              │ Web UI (Flask)            │ :19787
                              │ Plotly charts, Holdings,  │
                              │ Actions, Portfolio detail │
                              └──────────────────────────┘
```

**Key components:**

| Component | Purpose |
|---|---|
| **Backend API** (`backend_app.py`) | Flask server exposing `/factors/eval`, `/factors/check`, `/combination/*` |
| **Factor Eval Subprocesses** (`utils/utils.py`) | Isolated `multiprocessing.Process` workers with hard timeout + kill. Each calls `qlib.init()` to compute IC/RankIC |
| **FactorStore** (`utils/factor_store.py`) | Incremental daily IC cache in SQLite + factor scores on disk as pickle files |
| **Worker Pool** (`utils/qlib_worker_pool.py`) | Two persistent processes (CN, US) for portfolio backtests. Each initialises qlib once and preserves the in-memory data cache across requests |
| **Web UI** (`web_app.py`) | Interactive frontend with factor evaluation, daily IC charts, portfolio backtest visualisation (value chart, holdings table, trading actions) |

---

## Cache & Storage System

FFO uses a two-layer caching system that enables incremental evaluation and avoids redundant computation.

### Layer 1: Daily IC Cache (SQLite)

Stored in `cache_data/factor_perf.sqlite`. Each row records the IC and Rank IC for a single
factor on a single trading day:

```
PRIMARY KEY (expr_hash, market, label, date)
```

| Column | Type | Description |
|---|---|---|
| `expr_hash` | TEXT | SHA-256 of the normalised expression |
| `market` | TEXT | Market universe (csi300, sp500, ...) |
| `label` | TEXT | Return label used (e.g. close_return) |
| `date` | TEXT | Trading date (YYYY-MM-DD) |
| `ic` | REAL | Information Coefficient for that day |
| `rank_ic` | REAL | Rank IC for that day |

**Incremental updates:** When you evaluate a factor for `2023-01-01` to `2024-01-01`, the system
checks which dates are already cached. If `2023-01-01` to `2023-06-30` exists, only
`2023-07-01` to `2024-01-01` is computed. The results are merged automatically.

```python
# How it works internally
missing = store.get_missing_ranges(expr_hash, market, label, start, end)
# Returns: [("2023-07-01", "2024-01-01")]  — only the gap
# If fully cached: []
# If nothing cached: [("2023-01-01", "2024-01-01")]
```

**Summary metrics** (IC, ICIR, Rank IC, Rank ICIR) are computed on-the-fly from the daily
values, so extending the date range seamlessly incorporates old + new data.

### Layer 2: Factor Scores (Pickle on Disk)

Raw factor scores (the per-stock, per-day signal values) are stored as pickle files:

```
cache_data/
  factor_scores/
    <expr_hash>/
      csi300.pkl      # pd.DataFrame, MultiIndex (datetime, instrument), col: "score"
      sp500.pkl
```

These are used by portfolio backtests (`backtest_by_scores`) to avoid recomputing the
factor expression. Scores are merged incrementally with atomic writes (temp file + `os.replace`)
to prevent corruption.

### Cache Lifecycle

```
Request: evaluate "Rank($close, 20)" on csi300, 2023-01-01 → 2024-01-01

1. Hash expression → expr_hash = "a1b2c3..."
2. Query FactorStore: get_missing_ranges(expr_hash, "csi300", "close_return", "2023-01-01", "2024-01-01")
3. If fully cached → return stored daily IC, compute summary → done (instant)
4. If partially cached → compute only missing date ranges in subprocess
5. Subprocess: qlib.init() → QlibDataLoader → compute IC/RankIC per day → save scores to disk
6. Store new daily IC rows in SQLite, merge scores pickle
7. Return combined results (old + new)
```

### Cache Management

```bash
ppo cache stats    # entries, score files, DB size
ppo cache clear    # wipe everything
```

```python
from ffo.api import get_cache_stats, clear_cache
stats = get_cache_stats()
clear_cache()
```

---

## Multi-Market Worker Pool

FFO supports both China A-share and US equity markets. Since qlib can only initialise one
data provider at a time (global singleton), FFO runs **persistent worker processes** — one
per region — so both markets are always warm and ready.

### Supported Markets

| Market | Region | Data Path | Benchmark | Instruments |
|---|---|---|---|---|
| `csi300` | cn | `~/.qlib/qlib_data/cn_data` | SH000300 | CSI 300 |
| `csi500` | cn | `~/.qlib/qlib_data/cn_data` | SH000905 | CSI 500 |
| `csi1000` | cn | `~/.qlib/qlib_data/cn_data` | SH000852 | CSI 1000 |
| `sp500` | us | `~/.qlib/qlib_data/us_data` | ^gspc | S&P 500 |
| `nasdaq100` | us | `~/.qlib/qlib_data/us_data` | ^ixic | NASDAQ 100 |

### How the Worker Pool Works

```
Server starts
  └── QlibWorkerPool spawns:
        ├── CN Worker (Process) → qlib.init(cn_data) once
        └── US Worker (Process) → qlib.init(us_data) once

Request: backtest csi300 factor
  └── Router → CN Worker (already warm)
        └── backtest_by_scores() runs with cached qlib data
        └── D.features() hits in-memory cache → ~2-5s (vs ~30s cold)

Request: backtest sp500 factor
  └── Router → US Worker (already warm, no interference with CN)
```

**Benefits:**
- First backtest per region: ~30s (one-time data loading)
- Subsequent backtests: ~2-5s (qlib's in-memory `H` cache is preserved)
- Switching CN ↔ US: no penalty (separate processes)
- Auto-restart on worker crash
- Hard timeout with kill + restart if stuck

### Adding a New Market

Add an entry to `ffo/config/ffo.yaml`:

```yaml
markets:
  # ... existing markets ...
  my_market:
    data_path: "~/.qlib/qlib_data/my_data"
    region: "cn"          # or "us"
    benchmark: "SH000001"
```

The worker pool picks up new regions automatically on restart. Markets sharing the
same region reuse the same worker process.

---

## CLI Reference

### Service control

```bash
# Start
ppo start backend              # Backend API   → http://127.0.0.1:19777
ppo start web                  # Web UI        → http://127.0.0.1:19787
ppo start mcp                  # MCP server    (stdio transport, for Claude Desktop)
ppo start mcp --transport sse  # MCP server    (SSE transport, persistent)
ppo start all                  # Backend + web

# Stop / restart
ppo stop  backend|web|mcp|all
ppo restart backend|web|mcp|all

# Status (with health check)
ppo status
```

### Factor operations

```bash
ppo eval "Rank($close, 20)"                 # Fast mode (IC only)
ppo eval "Rank($close, 20)" --no-fast       # Full mode (IC + portfolio backtest)
ppo eval "Rank($close, 20)" --market csi500
ppo eval "Rank($close, 20)" --json          # Raw JSON output

ppo check "Rank($close, 20)"                # Syntax validation
```

### Cache management

```bash
ppo cache stats    # Show cache size and hit rate
ppo cache clear    # Clear all cached evaluations
```

### Logs

```bash
ppo logs backend           # Last 50 lines
ppo logs backend -n 200    # Last 200 lines
ppo logs backend -f        # Follow (tail -f)
ppo logs web -f
```

### Configuration

```bash
ppo config show                        # Print effective config (YAML)
ppo config show --section evaluation   # Show one section only
ppo config set evaluation.market csi500
ppo config set server.backend.port 19778
ppo config init                        # Write default ~/.ppo/config.yaml
ppo config init --force                # Overwrite existing
```

---

## Python API

### Installation (import path)

```python
# Top-level convenience imports
from ffo.api import (
    evaluate_factor,         # Evaluate one factor
    batch_evaluate_factors,  # Evaluate many factors (parallel)
    check_factor,            # Validate syntax
    get_cache_stats,         # Cache statistics
    clear_cache,             # Clear cache
    server_health,           # Health check
)

# Or via the package root
import ffo
result = ffo.evaluate_factor("Rank($close, 20)")
```

### Single factor evaluation

```python
from ffo.api import evaluate_factor

result = evaluate_factor(
    expression="Rank($close, 20)",
    market="csi300",       # csi300 | csi500 | csi1000 | sp500 | nasdaq100
    start="2023-01-01",
    end="2024-01-01",
    fast=True,             # IC only (default). False = IC + portfolio backtest
    use_cache=True,        # Uses incremental daily IC cache
)

if result.success:
    m = result.metrics
    print(f"IC       = {m.ic:.4f}")
    print(f"Rank IC  = {m.rank_ic:.4f}")
    print(f"ICIR     = {m.icir:.4f}")
    print(f"Turnover = {m.turnover:.4f}")
    print(f"N Dates  = {m.n_dates}")
    print(f"Cached   = {result.cached}")
else:
    print(f"Error: {result.error}")
```

### Batch evaluation (parallel — 6-10× faster)

```python
from ffo.api import batch_evaluate_factors

factors = [
    "Rank($close, 5)",
    "Rank($close, 20)",
    "Mean($volume, 5) / Mean($volume, 20)",
    "Corr($close, $volume, 10)",
    "$close / Delay($close, 1) - 1",
]

results = batch_evaluate_factors(
    expressions=factors,
    market="csi300",
    start="2023-01-01",
    end="2024-01-01",
    fast=True,
    parallel=True,     # thread-pool parallelism
    max_workers=8,
    progress=True,     # print progress to stdout
)

# Sort by Rank IC
for r in sorted(results, key=lambda x: -x.metrics.rank_ic):
    status = f"IC={r.metrics.ic:+.4f}" if r.success else r.error
    print(f"{r.expression[:45]:45s}  {status}")
```

### Syntax check

```python
from ffo.api import check_factor

chk = check_factor("Rank($close, 20)")
print(chk.is_valid)   # True

chk = check_factor("BadOp($close)")
print(chk.is_valid)   # False
print(chk.error)      # "Unknown operator: BadOp"
```

### Health check & cache

```python
from ffo.api import server_health, get_cache_stats, clear_cache

health = server_health()
print(health.is_healthy, health.latency_ms)

stats = get_cache_stats()
print(f"Cache: {stats.cache_size}/{stats.max_cache_size}")

clear_cache()
```

### Low-level client (advanced)

```python
from ffo.client import FactorEvalClient

with FactorEvalClient(base_url="http://127.0.0.1:19777") as client:
    if client.health_check():
        results = client.evaluate_batch_parallel(
            ["Rank($close, 20)", "Mean($volume, 5)"],
            max_workers=8,
            fast=True,
        )
```

### Return types

| Class | Fields |
|---|---|
| `FactorResult` | `success`, `expression`, `metrics`, `error`, `cached`, `market`, `start`, `end` |
| `FactorMetrics` | `ic`, `rank_ic`, `icir`, `rank_icir`, `turnover`, `n_dates` |
| `SyntaxCheckResult` | `is_valid`, `expression`, `error`, `name` |
| `CacheStats` | `cache_size`, `max_cache_size`, `hit_rate` |
| `ServerHealth` | `is_healthy`, `status`, `latency_ms`, `cache`, `error` |

All return types have a `.to_dict()` method for JSON serialisation.

---

## Configuration

### Config file: `ffo/config/ffo.yaml`

The canonical config is bundled with the package at `ffo/config/ffo.yaml`.
User overrides go in `~/.ppo/config.yaml` (created with `ppo config init`).

**Priority (highest → lowest):**
1. Environment variables (`FFO_BACKEND_PORT=19778`)
2. `~/.ppo/config.yaml` — per-user overrides
3. `ffo/config/ffo.yaml` — package defaults ← this file
4. Built-in hardcoded defaults

### Key defaults

```yaml
server:
  backend:
    port: 19777      # REST API
  web:
    port: 19787      # Web UI

evaluation:
  market: csi300
  start:  2023-01-01
  end:    2024-01-01
  fast:   true
  topk:   50
  n_drop: 5

cache:
  path:        ~/.ppo/factor_cache.sqlite
  max_entries: 50000

qlib:
  data_path: ~/.qlib/qlib_data/cn_data
  region:    cn

markets:
  csi300:
    data_path: "~/.qlib/qlib_data/cn_data"
    region: "cn"
    benchmark: "SH000300"
  csi500:
    data_path: "~/.qlib/qlib_data/cn_data"
    region: "cn"
    benchmark: "SH000905"
  sp500:
    data_path: "~/.qlib/qlib_data/us_data"
    region: "us"
    benchmark: "^gspc"
  nasdaq100:
    data_path: "~/.qlib/qlib_data/us_data"
    region: "us"
    benchmark: "^ixic"
```

### Override via environment variables

```bash
FFO_BACKEND_PORT=19778      # backend port
FFO_WEB_PORT=19788          # web UI port
FFO_EVALUATION_MARKET=csi500
FFO_EVALUATION_FAST=false
FFO_CACHE_MAX_ENTRIES=100000
FFO_QLIB_DATA_PATH=/data/qlib
```

### Override via CLI

```bash
ppo config set evaluation.market csi500
ppo config set server.backend.port 19778
ppo config show
```

### Access in code

```python
from ffo.config import get_config

cfg = get_config()
cfg.get("server.backend.port")     # 19777
cfg.get("evaluation.market")       # "csi300"
cfg.backend_url                    # "http://127.0.0.1:19777"
cfg.web_url                        # "http://127.0.0.1:19787"
```

---

## MCP Server (LLM Agents)

FFO exposes all evaluation tools as [Model Context Protocol](https://modelcontextprotocol.io) tools,
making them directly callable by Claude, GPT-4o, and other MCP-compatible agents.

### Claude Desktop integration

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ffo": {
      "command": "ppo",
      "args": ["start", "mcp"]
    }
  }
}
```

### SSE transport (persistent server)

```bash
ppo start mcp --transport sse --port 8765
# Connect at: http://127.0.0.1:8765/sse
```

### Available MCP tools

| Tool | Description |
|---|---|
| `evaluate_factor` | Evaluate a single alpha factor (IC, Rank IC, ICIR) |
| `batch_evaluate_factors` | Evaluate many factors in parallel |
| `check_factor_syntax` | Validate expression syntax without evaluation |
| `get_server_health` | Check backend status and latency |
| `get_cache_stats` | Show cache occupancy and hit rate |
| `clear_cache` | Clear all cached evaluations |

### Available MCP resources

| URI | Description |
|---|---|
| `ffo://operators` | List of all supported Qlib operators |
| `ffo://markets` | Supported market universes (csi300, csi500, csi1000, sp500, nasdaq100) |

### Direct MCP usage (Python SDK)

```python
# SSE transport
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8765/sse") as streams:
    async with ClientSession(*streams) as session:
        await session.initialize()
        result = await session.call_tool(
            "evaluate_factor",
            arguments={"expression": "Rank($close, 20)"}
        )
        print(result.content)
```

---

## REST API Endpoints

Base URL: `http://127.0.0.1:19777`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + cache stats |
| `POST` | `/factors/check` | Validate factor syntax |
| `POST` | `/factors/eval` | Evaluate one or many factors |
| `POST` | `/combination/train` | Train factor combination (LASSO / IC opt) |
| `POST` | `/combination/backtest` | Backtest a trained model |
| `GET` | `/cache_stats` | Cache statistics |
| `POST` | `/clear_cache` | Clear cache |

### Example: evaluate via curl

```bash
# Health check
curl http://127.0.0.1:19777/health

# Evaluate a factor
curl -s -X POST http://127.0.0.1:19777/factors/eval \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "Rank($close, 20)",
    "market": "csi300",
    "start": "2023-01-01",
    "end":   "2024-01-01",
    "fast": true
  }' | python -m json.tool
```

For full REST API documentation see [README_API.md](README_API.md).

---

## Web UI

The web UI (`ppo start web`, port 19787) provides:

- **Single Factor Analysis**: Enter an expression, select market/date range, evaluate. See IC, Rank IC, ICIR, daily IC chart.
- **Portfolio Backtest**: One-click backtest with TopkDropout strategy. View:
  - Portfolio value chart (total value / stock value / cash — switchable tabs)
  - Daily holdings table (date picker with calendar input)
  - Trading actions log (buy/sell events detected from position diffs)
- **Factor Comparison**: Evaluate multiple factors side-by-side, ranked by Rank IC.
- **Factor Combination**: Optimise factor weights via LASSO or IC optimisation across multiple periods.

---

## Package Structure

```
ffo/
├── __init__.py              # Top-level re-exports (ffo.evaluate_factor etc.)
│
├── api/                     # Agent-friendly typed API
│   ├── __init__.py
│   └── functions.py         # evaluate_factor, batch_evaluate_factors, ...
│
├── cli/                     # ppo CLI (Click)
│   ├── __init__.py
│   └── main.py              # ppo start|stop|status|eval|check|cache|config
│
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── manager.py           # ConfigManager, get_config()
│   └── ffo.yaml             # Bundled default configuration ← edit here
│
├── mcp/                     # MCP server (FastMCP)
│   ├── __init__.py
│   └── server.py            # MCP tools wrapping ffo.api
│
├── client/                  # Low-level HTTP client
│   ├── __init__.py
│   └── factor_eval_client.py
│
├── demos/                   # Runnable demo scripts
│   ├── 01_basic_usage.py
│   ├── 02_batch_eval.py
│   ├── 03_mcp_client.py
│   └── 04_config_and_cli.py
│
├── routes/                  # Flask route handlers
│   ├── factors.py           # /factors/check, /factors/eval + worker pool
│   └── combinations.py      # /combination/...
│
├── utils/                   # Internal utilities
│   ├── utils.py             # Subprocess workers, timeouts, IC computation
│   ├── factor_store.py      # Incremental daily IC cache (SQLite + pickle)
│   ├── qlib_worker_pool.py  # Persistent multi-region worker pool
│   └── qlib_extend_ops.py   # Custom Qlib operators
│
├── backtest/                # Backtesting (Qlib integration)
│   └── qlib/
│       └── single_alpha_backtest.py  # backtest_by_scores, backtest_by_single_alpha
│
├── data/                    # Optimization pipelines
├── tests/                   # Test suite
├── templates/               # Web UI (Jinja2 + Plotly + Bootstrap)
│
├── cache_data/              # Generated at runtime
│   ├── factor_perf.sqlite   # Daily IC/RankIC cache
│   └── factor_scores/       # Pickle files per (expr_hash, market)
│
├── backend_app.py           # Flask app entry point (gunicorn target)
├── web_app.py               # Web UI Flask app
└── start.sh                 # Legacy gunicorn launch script
```

---

## Performance

### Factor Evaluation

| Scenario | Method | Typical time (100 factors) |
|---|---|---|
| Single call | `evaluate_factor()` | ~5s each |
| Sequential batch | `batch_evaluate_factors(parallel=False)` | ~500s |
| Parallel (8 workers) | `batch_evaluate_factors(max_workers=8)` | ~70s |
| Cached (any) | automatic | <100ms |

### Portfolio Backtest

| Scenario | Time |
|---|---|
| First backtest (cold start) | ~30s (Exchange + data loading) |
| Subsequent backtest (warm worker) | ~2-5s (in-memory cache) |
| Switch market (CN ↔ US) | No penalty (separate workers) |

### Tips

- Use `fast=True` (default) to skip portfolio backtest — 5-10x faster
- Results are cached incrementally in SQLite; repeated calls with the same or overlapping date ranges are instant
- Extending a date range only computes the missing dates, not the full range
- The worker pool preserves qlib's in-memory cache, so repeated backtests on the same market are much faster
- Set `use_cache=False` only when you need a fresh evaluation

---

## Demos

```bash
# Make sure backend is running first
ppo start backend

python ffo/demos/01_basic_usage.py      # single factor + health check
python ffo/demos/02_batch_eval.py       # parallel batch + ranking table
python ffo/demos/03_mcp_client.py       # MCP tool schema + agent simulation
python ffo/demos/04_config_and_cli.py   # full config system + CLI reference
```

---

## Troubleshooting

### Connection refused

```bash
# Check if backend is running
ppo status

# Start it
ppo start backend

# Verify directly
curl http://127.0.0.1:19777/health
```

### Port conflict

```bash
# Change port in config
ppo config set server.backend.port 19778

# Or via environment variable (no restart of ppo needed)
FFO_BACKEND_PORT=19778 ppo start backend
```

### Slow evaluation

1. Enable fast mode: `fast=True` (already the default)
2. Use parallel batch: `batch_evaluate_factors(parallel=True, max_workers=8)`
3. Check cache: `ppo cache stats` — high hit rate means the cache is working

### Out of memory

```python
# Process in chunks
all_results = []
for chunk in [all_factors[i:i+100] for i in range(0, len(all_factors), 100)]:
    all_results.extend(batch_evaluate_factors(chunk, parallel=True, fast=True))
```

### View logs

```bash
ppo logs backend -f    # follow live output
ppo logs web           # last 50 lines of web UI log
```

---

## Documentation

| File | Contents |
|---|---|
| `README.md` (this file) | Overview, architecture, cache system, CLI, Python API, configuration |
| [`README_API.md`](README_API.md) | Full REST API reference |
| [`README_DESIGN.md`](README_DESIGN.md) | Architecture and design decisions |
| [`ffo/demos/`](demos/) | Runnable end-to-end examples |
