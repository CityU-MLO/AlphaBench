# FFO - Formulaic Factor Optimization

## REST API Documentation

FFO (Formulaic Factor Optimization) provides a comprehensive REST API for evaluating quantitative trading factors and optimizing factor combinations.

---

## Table of Contents

- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Factor Evaluation](#factor-evaluation)
  - [Factor Combination](#factor-combination)
- [Python Client](#python-client)
- [Request/Response Examples](#requestresponse-examples)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

---

## Quick Start

### Start the API Server

```bash
# Option 1: Using Python directly
python backend_app.py

# Option 2: Using web app (with UI)
python web_app.py

# Default ports:
# - Backend API: http://localhost:19320
# - Web App: http://localhost:5002
```

### Test the API

```bash
# Health check
curl http://localhost:19320/health

# Simple factor evaluation
curl -X POST http://localhost:19320/factors/eval \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "Rank($close, 20)",
    "start": "2023-01-01",
    "end": "2024-01-01",
    "market": "csi300"
  }'
```

---

## API Endpoints

### Health Check

**GET** `/health`

Check API server status and cache statistics.

**Response:**
```json
{
  "status": "healthy",
  "service": "Factor Evaluation API",
  "timestamp": "2024-01-01T12:00:00Z",
  "cache": {
    "size": 150,
    "max_size": 10000
  }
}
```

---

### Factor Evaluation

#### Check Factor Syntax

**POST** `/factors/check`

Validate factor expression syntax without full evaluation.

**Request:**
```json
{
  "expression": "Mean($close, 20)",
  "instruments": "csi300",
  "start": "2020-01-01",
  "end": "2020-01-15"
}
```

**Response:**
```json
[
  {
    "success": true,
    "name": "",
    "expression": "Mean($close, 20)",
    "is_valid": true
  }
]
```

#### Evaluate Single Factor

**POST** `/factors/eval`

Evaluate factor performance with IC metrics and optional portfolio backtest.

**Request:**
```json
{
  "expression": "Rank(Corr($close, $volume, 10), 252)",
  "start": "2023-01-01",
  "end": "2024-01-01",
  "market": "csi300",
  "label": "close_return",
  "use_cache": true,
  "topk": 50,
  "n_drop": 5,
  "fast": false,
  "n_jobs_backtest": 4
}
```

**Parameters:**
- `expression` (string|dict|list): Factor expression(s) to evaluate
- `start` (string): Start date (YYYY-MM-DD)
- `end` (string): End date (YYYY-MM-DD)
- `market` (string): Market identifier (csi300, csi500, csi1000)
- `label` (string): Label column for IC calculation (default: close_return)
- `use_cache` (boolean): Enable caching (default: true)
- `topk` (integer): Top K stocks to select (default: 50)
- `n_drop` (integer): Stocks to drop (default: 5)
- `fast` (boolean): Skip portfolio backtest, only compute IC (default: false)
- `n_jobs_backtest` (integer): Parallel threads for backtest (default: 4)

**Response:**
```json
[
  {
    "success": true,
    "name": "",
    "expression": "Rank(Corr($close, $volume, 10), 252)",
    "market": "csi300",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "metrics": {
      "ic": 0.0234,
      "rank_ic": 0.0312,
      "ir": 0.4521,
      "icir": 0.5123,
      "rank_icir": 0.5234,
      "turnover": 0.7234,
      "n_dates": 245
    },
    "portfolio_metrics": {
      "excess_return_without_cost": {
        "annualized_return": 12.5,
        "information_ratio": 1.8,
        "max_drawdown": -8.3
      }
    },
    "timestamp": "2024-01-01T12:00:00Z"
  }
]
```

#### Batch Evaluate Multiple Factors

**POST** `/factors/eval` (with multiple expressions)

Evaluate multiple factors in a single request.

**Request:**
```json
{
  "expression": {
    "momentum": "Rank($close, 20)",
    "volume": "Mean($volume, 10)",
    "volatility": "Std($close, 20)"
  },
  "start": "2023-01-01",
  "end": "2024-01-01",
  "market": "csi300",
  "fast": true,
  "use_cache": true
}
```

**Alternative array format:**
```json
{
  "expression": [
    "Rank($close, 20)",
    "Mean($volume, 10)",
    {"volatility": "Std($close, 20)"}
  ],
  "start": "2023-01-01",
  "end": "2024-01-01",
  "market": "csi300"
}
```

**Response:**
```json
[
  {
    "success": true,
    "name": "momentum",
    "expression": "Rank($close, 20)",
    "metrics": {...}
  },
  {
    "success": true,
    "name": "volume",
    "expression": "Mean($volume, 10)",
    "metrics": {...}
  },
  {
    "success": true,
    "name": "volatility",
    "expression": "Std($close, 20)",
    "metrics": {...}
  }
]
```

---

### Factor Combination

#### Train Factor Combination

**POST** `/combination/train`

Train factor combination models using LASSO or IC optimization.

**Request:**
```json
{
  "factor_expressions": [
    "Rank($close, 20)",
    "Mean($volume, 10)",
    "Std($close, 20)"
  ],
  "method": "lasso",
  "start_date": "2020-01-01",
  "end_date": "2021-12-31",
  "market": "csi300",
  "method_config": {
    "alpha": 0.01,
    "rolling_window": 60,
    "max_iter": 1000
  },
  "n_jobs": 4
}
```

**Methods:**
- `lasso`: LASSO regression with L1 regularization
- `ic_optimization`: Direct IC maximization with risk constraints

**Response:**
```json
{
  "success": true,
  "method": "lasso",
  "result": {
    "weights": [0.35, 0.42, 0.23],
    "factor_names": ["Rank($close, 20)", "Mean($volume, 10)", "Std($close, 20)"],
    "combined_expression": "(0.35)*(Rank($close, 20)) + (0.42)*(Mean($volume, 10)) + (0.23)*(Std($close, 20))",
    "training_ic": 0.045,
    "non_zero_count": 3
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Backtest Factor Combination

**POST** `/combination/backtest`

Run portfolio backtest using trained factor combination.

**Request:**
```json
{
  "trained_model": {
    "weights": [0.35, 0.42, 0.23],
    "factor_names": ["Rank($close, 20)", "Mean($volume, 10)", "Std($close, 20)"]
  },
  "start_date": "2022-01-01",
  "end_date": "2023-12-31",
  "market": "csi300",
  "topk": 50,
  "n_drop": 5
}
```

**Response:**
```json
{
  "success": true,
  "backtest_result": {
    "analysis": {
      "annualized_return": 15.3,
      "sharpe_ratio": 1.85,
      "max_drawdown": -12.5,
      "information_ratio": 2.1
    },
    "report_normal": {...},
    "positions_normal": {...}
  },
  "method": "lasso",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Compare Optimization Methods

**POST** `/combination/compare_methods`

Compare LASSO vs IC optimization approaches.

**Request:**
```json
{
  "factor_expressions": ["expr1", "expr2", "expr3"],
  "train_start": "2020-01-01",
  "train_end": "2021-12-31",
  "test_start": "2022-01-01",
  "test_end": "2023-12-31",
  "market": "csi300",
  "lasso_config": {"alpha": 0.01},
  "ic_config": {"lambda_risk": 1.0, "alpha_l1": 0.0}
}
```

**Response:**
```json
{
  "success": true,
  "comparison": {
    "lasso": {
      "train_ic": 0.045,
      "test_ic": 0.038,
      "weights": [0.3, 0.5, 0.2],
      "sparsity": 1.0
    },
    "ic_optimization": {
      "train_ic": 0.048,
      "test_ic": 0.041,
      "weights": [0.35, 0.45, 0.20],
      "sparsity": 1.0
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Python Client

### Installation

```python
from ffo.client import FactorEvalClient

# Initialize client
client = FactorEvalClient(base_url="http://localhost:19320")
```

### Basic Usage

```python
# Evaluate single factor
result = client.evaluate_factor(
    expr="Rank($close, 20)",
    market="csi300",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

if result['success']:
    print(f"Rank IC: {result['metrics']['rank_ic']:.4f}")
    print(f"ICIR: {result['metrics']['icir']:.4f}")
```

### Batch Evaluation

```python
# Evaluate multiple factors
factors = [
    {"name": "momentum", "expression": "Rank($close, 20)"},
    {"name": "volume", "expression": "Mean($volume, 10)"}
]

results = client.batch_evaluate_factors(
    factors=factors,
    market="csi300",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

for result in results:
    if result['success']:
        print(f"{result['name']}: IC={result['metrics']['ic']:.4f}")
```

### Fast Evaluation (IC only, skip backtest)

```python
# Fast mode - only compute IC metrics
result = client.evaluate_factor(
    expr="Rank($close, 20)",
    market="csi300",
    start_date="2023-01-01",
    end_date="2024-01-01",
    fast=True  # Skip portfolio backtest
)
```

---

## Request/Response Examples

### Example 1: Simple Factor Check

```bash
curl -X POST http://localhost:19320/factors/check \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "Rank($close, 20)",
    "start": "2020-01-01",
    "end": "2020-01-15"
  }'
```

### Example 2: Fast Evaluation (Multiple Factors)

```bash
curl -X POST http://localhost:19320/factors/eval \
  -H "Content-Type: application/json" \
  -d '{
    "expression": ["Rank($close, 20)", "Mean($volume, 10)"],
    "start": "2023-01-01",
    "end": "2024-01-01",
    "market": "csi300",
    "fast": true
  }'
```

### Example 3: Full Evaluation with Portfolio Backtest

```bash
curl -X POST http://localhost:19320/factors/eval \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "Rank(Corr($close, $volume, 10), 252)",
    "start": "2023-01-01",
    "end": "2024-01-01",
    "market": "csi300",
    "fast": false,
    "topk": 50,
    "n_drop": 5
  }'
```

---

## Configuration

### Environment Variables

```bash
# Server configuration
export PORT=19320
export DEBUG=false

# Default parameters
export DEFAULT_MARKET=csi300
export DEFAULT_START=2023-01-01
export DEFAULT_END=2024-01-01
export DEFAULT_LABEL=close_return

# Timeout settings (seconds)
export TIMEOUT_EVAL_SEC=180
export TIMEOUT_CHECK_SEC=120
export TIMEOUT_BATCH_SEC=600

# Cache settings
export CACHE_DIR=./cache_data
export MAX_CACHE_SIZE=10000
```

### Cache Management

The API uses persistent SQLite cache for factor evaluations.

**Cache Key Format:**
```
hash(expression + market + start_date + end_date + label + topk + n_drop)
```

**Cache Statistics:**
```bash
# Available via health endpoint
curl http://localhost:19320/health
```

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message",
  "error_type": "ExpressionError",
  "expression": "Invalid($close)",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Types

- `ExpressionError`: Invalid factor expression syntax
- `TimeoutError`: Evaluation exceeded timeout limit
- `DataError`: Data loading or processing error
- `ValidationError`: Invalid request parameters

### HTTP Status Codes

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error
- `504`: Gateway timeout

---

## Performance Optimization

### Use Fast Mode for Screening

```python
# Fast screening of hundreds of factors
factors = [f"Rank($close, {i})" for i in range(5, 100, 5)]

results = []
for expr in factors:
    result = client.evaluate_factor(
        expr=expr,
        market="csi300",
        start_date="2023-01-01",
        end_date="2024-01-01",
        fast=True,  # Only IC, no backtest
        use_cache=True
    )
    results.append(result)
```

### Parallel Backtest Execution

```python
# Enable parallel backtest threads
result = client.evaluate_factor(
    expr="Rank($close, 20)",
    market="csi300",
    start_date="2023-01-01",
    end_date="2024-01-01",
    fast=False,
    n_jobs_backtest=8  # 8 parallel threads
)
```

### Cache Strategy

1. Enable caching for repeated evaluations
2. Cache is persistent across server restarts
3. Cache is shared between all clients
4. Cache invalidation is automatic based on parameters

---

## Best Practices

1. **Use `fast=true` for initial screening** of large factor pools
2. **Enable caching** for iterative research workflows
3. **Batch evaluate** multiple factors in single request when possible
4. **Set appropriate timeouts** based on factor complexity
5. **Use parallel backtest** for compute-intensive evaluations
6. **Monitor cache statistics** to optimize memory usage

---

## Support

For issues and questions:
- GitHub Issues: [qfinzero/issues](https://github.com/chester1uo/qfinzero/issues)
- Documentation: See [README_DESIGN.md](README_DESIGN.md) for system architecture

---

## License

Part of the qfinzero project.
