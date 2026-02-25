# FFO — Factor Feature Oracle

High-performance quantitative factor evaluation platform for A-share markets.
Exposes a REST API, a typed Python API, a CLI (`ppo`), and an MCP server for LLM agents.

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Configuration](#configuration)
- [MCP Server (LLM Agents)](#mcp-server-llm-agents)
- [REST API Endpoints](#rest-api-endpoints)
- [Package Structure](#package-structure)
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
```

**Python:**
```python
from ffo.api import evaluate_factor

result = evaluate_factor("Rank($close, 20)")
if result.success:
    print(f"IC={result.metrics.ic:.4f}  Rank_IC={result.metrics.rank_ic:.4f}")
```

### 3. Stop the backend

```bash
ppo stop backend
```

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
    market="csi300",       # csi300 | csi500 | csi1000
    start="2023-01-01",
    end="2024-01-01",
    fast=True,             # IC only (default). False = full backtest
    use_cache=True,
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
| `ffo://markets` | Supported market universes (csi300, csi500, csi1000) |

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
│   ├── factors.py           # /factors/check, /factors/eval
│   └── combinations.py      # /combination/...
│
├── utils/                   # Internal utilities
│   ├── utils.py             # Cache, timeouts, parsing
│   ├── execution_engine.py
│   └── qlib_extend_ops.py
│
├── backtest/                # Backtesting (Qlib integration)
├── data/                    # Optimization pipelines
├── tests/                   # Test suite
├── examples/                # Legacy usage examples
├── templates/               # Web UI templates
│
├── backend_app.py           # Flask app entry point (gunicorn target)
├── web_app.py               # Web UI Flask app
└── start.sh                 # Legacy gunicorn launch script
```

---

## Performance

| Scenario | Method | Typical time (100 factors) |
|---|---|---|
| Single call | `evaluate_factor()` | ~5s each |
| Sequential batch | `batch_evaluate_factors(parallel=False)` | ~500s |
| Parallel (4 workers) | `batch_evaluate_factors(max_workers=4)` | ~130s |
| Parallel (8 workers) | `batch_evaluate_factors(max_workers=8)` | ~70s |
| Parallel (16 workers) | `batch_evaluate_factors(max_workers=16)` | ~50s |

Tips:
- Use `fast=True` (default) to skip portfolio backtest — 5-10× faster
- Results are cached in SQLite; repeated calls are instant
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
| `README.md` (this file) | Overview, CLI, Python API, configuration |
| [`README_API.md`](README_API.md) | Full REST API reference |
| [`README_DESIGN.md`](README_DESIGN.md) | Architecture and design decisions |
| [`ffo/demos/`](demos/) | Runnable end-to-end examples |
