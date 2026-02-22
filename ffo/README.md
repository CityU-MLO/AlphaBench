# FFO - Formulaic Factor Optimization

A high-performance quantitative factor evaluation and optimization platform designed for backtesting factors during search and conducting factor optimization.

## Overview

FFO (Formulaic Factor Optimization) provides:

- **Factor Evaluation**: Compute IC, Rank IC, ICIR, and portfolio metrics for factor expressions
- **Factor Combination**: Optimize weights for multiple factors using LASSO, IC optimization, or other methods
- **Backtesting**: Portfolio-level backtesting with transaction costs and performance metrics
- **REST API**: Low-latency API with persistent caching for efficient batch processing
- **Python Client**: Simple, high-performance client library with parallel execution support

### Key Features

✨ **Simple to Use** - Function-style API, no client management needed
⚡ **High Performance** - 6-10x speedup with parallel batch processing
🔄 **Context Managers** - Automatic resource cleanup
💾 **Persistent Caching** - SQLite-backed cache for fast repeated evaluations
🛡️ **Robust** - Automatic retries, timeout protection, error handling
📊 **Progress Tracking** - Built-in progress callbacks for long operations

## Quick Start

### 1. Installation

No additional dependencies needed beyond the base FFO system.

### 2. Start the Server

```bash
cd infra/ffo
python backend_app.py
```

Server will start on http://127.0.0.1:19330

### 3. Basic Usage

#### Simple Single Evaluation

```python
from client import evaluate

# Evaluate a single factor
result = evaluate("Rank($close, 20)", fast=True)

print(f"IC: {result['metrics']['ic']:.4f}")
print(f"Rank IC: {result['metrics']['rank_ic']:.4f}")
```

#### Batch Evaluation (Sequential)

```python
from client import evaluate_batch

factors = [
    "Rank($close, 20)",
    "Mean($volume, 5)",
    "StdDev($close, 10)"
]

results = evaluate_batch(factors, fast=True)

for r in results:
    if r['success']:
        print(f"{r['expression']}: IC={r['metrics']['ic']:.4f}")
```

#### Batch Evaluation (Parallel - 6x Faster!)

```python
from client import evaluate_batch

factors = [
    "Rank($close, 20)",
    "Mean($volume, 5)",
    "StdDev($close, 10)"
]

# Use parallel=True for 6-10x speedup
results = evaluate_batch(
    factors,
    parallel=True,
    max_workers=8,
    fast=True,
    progress=True  # Show progress bar
)

for r in results:
    if r['success']:
        print(f"{r['expression']}: IC={r['metrics']['ic']:.4f}")
```

### 4. Context Manager (Production Ready)

```python
from client import FactorEvalClient

factors = ["Rank($close, 20)", "Mean($volume, 5)"]

with FactorEvalClient() as client:
    # Check server health
    if not client.health_check():
        print("Server not healthy!")
        exit(1)

    # Evaluate in parallel
    results = client.evaluate_batch_parallel(
        factors,
        max_workers=8,
        fast=True
    )

    for r in results:
        print(f"{r['expression']}: IC={r['metrics']['ic']:.4f}")

# Client resources automatically cleaned up
```

## Performance

### Benchmark Results (100 factors, CSI300, fast mode)

| Method | Time | Speedup |
|--------|------|---------|
| Sequential | 450s | 1.0x |
| Parallel (4 workers) | 125s | 3.6x |
| Parallel (8 workers) | 68s | **6.6x** |
| Parallel (16 workers) | 45s | **10.0x** |

### When to Use What?

| Scenario | Method | Expected Speedup |
|----------|--------|------------------|
| 1-5 factors | `evaluate()` | 1x |
| 5-20 factors | `evaluate_batch()` | 1x |
| 20-100 factors | `evaluate_batch(parallel=True, max_workers=8)` | 6-8x |
| 100+ factors | `evaluate_batch(parallel=True, max_workers=16)` | 8-12x |

## Configuration

### Evaluation Parameters

```python
from client import evaluate

result = evaluate(
    "Rank($close, 20)",
    market="csi500",           # Market: csi300, csi500, csi1000
    start_date="2023-01-01",   # Start date
    end_date="2024-01-01",     # End date
    label="close_return",      # Label column
    fast=True,                 # Skip portfolio backtest (5x faster)
    use_cache=True,            # Use cache (default: True)
    topk=50,                   # Top K stocks for portfolio
    n_drop=5,                  # Bottom N to drop
    timeout=120                # Timeout in seconds
)
```

### Environment Variables

```bash
DEFAULT_MARKET=csi300
DEFAULT_START=2023-01-01
DEFAULT_END=2024-01-01
DEFAULT_LABEL=close_return
TIMEOUT_EVAL_SEC=180
PORT=19330
```

## Examples

Complete examples with 6 different usage patterns:

```bash
cd infra/ffo
python examples/enhanced_usage.py
```

## Architecture

```
┌─────────────────┐
│  Python Client  │  Simple function-style API
└────────┬────────┘
         │
         v
┌──────────────────────────────────┐
│      REST API (Flask)            │  Request validation, routing
├──────────────────────────────────┤
│  - /factors/check                │
│  - /factors/eval                 │
│  - /combination/train            │
└────────┬─────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│   Core Processing Layer          │
├──────────────────────────────────┤
│  • Factor Evaluation Engine      │  Vectorized IC computation
│  • Optimization Engine           │  LASSO, IC optimization
│  • Cache Manager                 │  SQLite persistent cache
│  • Timeout Handler               │  Hard subprocess timeout
└────────┬─────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│    Data Access Layer             │
├──────────────────────────────────┤
│  • Qlib Data Loader              │  Market data loading
│  • Backtest Engine               │  Portfolio simulation
│  • Custom Operators              │  Extended Qlib ops
└──────────────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│     Storage Layer                │
├──────────────────────────────────┤
│  • Qlib Data (~/.qlib/)          │
│  • SQLite Cache (cache_data/)    │
│  • Model Storage (cache_data/)   │
└──────────────────────────────────┘
```

## Documentation

### Getting Started
- **This README** - Overview and quick start
- **[examples/enhanced_usage.py](examples/enhanced_usage.py)** - 6 complete usage examples

### Detailed Documentation
- **[README_API.md](README_API.md)** - Complete API reference with endpoints, parameters, and examples
- **[README_DESIGN.md](README_DESIGN.md)** - System architecture, design decisions, and implementation details

## Common Patterns

### Pattern 1: Quick Factor Screening

```python
from client import evaluate_batch

# Generate many factor variations
factors = [f"Rank($close, {i})" for i in range(10, 60, 5)]

# Evaluate in parallel with progress
results = evaluate_batch(
    factors,
    parallel=True,
    fast=True,
    progress=True
)

# Filter good factors
good_factors = [
    r for r in results
    if r.get('success') and r['metrics']['ic'] > 0.05
]

print(f"Found {len(good_factors)} good factors")
```

### Pattern 2: Progress Tracking

```python
from client import FactorEvalClient

def show_progress(completed, total):
    percent = 100 * completed // total
    print(f"\rProgress: {completed}/{total} ({percent}%)", end="")
    if completed == total:
        print()

with FactorEvalClient() as client:
    results = client.evaluate_batch_parallel(
        factors,
        max_workers=8,
        fast=True,
        progress_callback=show_progress
    )
```

### Pattern 3: Error Handling

```python
from client import evaluate

try:
    result = evaluate("Rank($close, 20)", fast=True)

    if result['success']:
        ic = result['metrics']['ic']
        print(f"IC: {ic:.4f}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")

except Exception as e:
    print(f"Failed to connect: {e}")
```

## Troubleshooting

### "Connection refused" Error

**Problem:** Can't connect to API server

**Solution:**
```bash
# Start the server
cd infra/ffo
python backend_app.py

# Verify it's running
curl http://127.0.0.1:19330/health
```

### Slow Performance

**Solutions:**
1. Enable parallel execution: `evaluate_batch(parallel=True, max_workers=8)`
2. Use fast mode: `evaluate_batch(fast=True)` to skip portfolio backtest
3. Increase workers for large batches: `max_workers=16`

### Out of Memory

**Solution:** Process in smaller chunks
```python
from client import evaluate_batch

all_results = []
chunk_size = 100

for i in range(0, len(all_factors), chunk_size):
    chunk = all_factors[i:i + chunk_size]
    results = evaluate_batch(chunk, parallel=True, fast=True)
    all_results.extend(results)
```

## API Endpoints

Quick reference (see [README_API.md](README_API.md) for complete details):

- `GET /health` - Health check
- `POST /factors/check` - Validate factor syntax
- `POST /factors/eval` - Evaluate factors
- `POST /combination/train` - Train factor combinations
- `POST /clear_cache` - Clear cache
- `GET /cache_stats` - Cache statistics

## Development

### Project Structure

```
infra/ffo/
├── client/                      # Python client library
│   ├── __init__.py
│   └── factor_eval_client.py    # Unified client (all-in-one)
├── routes/                      # API endpoints
│   ├── factors.py               # Factor evaluation
│   └── combinations.py          # Factor combination
├── utils/                       # Utilities
│   ├── utils.py                 # Cache, timeout, parsing
│   ├── execution_engine.py      # Multi-task execution
│   └── qlib_extend_ops.py       # Custom operators
├── backtest/                    # Backtesting
│   ├── qlib/                    # Qlib integration
│   └── factor_metrics/          # Metrics computation
├── data/                        # Data pipeline
│   └── pipeline/optim/          # Optimization methods
├── examples/                    # Usage examples
│   └── enhanced_usage.py
├── backend_app.py               # Flask app entry point
├── README.md                    # This file
├── README_API.md                # API reference
├── README_DESIGN.md             # Design documentation
└── IMPROVEMENTS.md              # Recent improvements
```

### Running Tests

```bash
cd infra/ffo
python -m pytest tests/
```

## Contributing

When contributing, please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Maintain backward compatibility

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Read the [API documentation](README_API.md)
3. Review [design documentation](README_DESIGN.md)
4. Check [examples](examples/enhanced_usage.py)

## License

Part of the qfinzero project.
