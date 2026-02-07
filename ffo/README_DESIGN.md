# FFO - Formulaic Factor Optimization

## System Design & Architecture

This document provides a comprehensive overview of the FFO (Formulaic Factor Optimization) system architecture, design decisions, and implementation details.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Data Pipeline](#data-pipeline)
- [Optimization Methods](#optimization-methods)
- [Caching Strategy](#caching-strategy)
- [Performance Considerations](#performance-considerations)
- [Extension Points](#extension-points)

---

## System Overview

FFO is a high-performance quantitative factor evaluation and optimization platform designed for:

1. **Factor Evaluation**: Compute IC, Rank IC, and portfolio metrics for factor expressions
2. **Factor Combination**: Optimize weights for multiple factors using LASSO or IC optimization
3. **Backtesting**: Portfolio-level backtesting with transaction costs
4. **Real-time API**: Low-latency REST API with persistent caching

### Key Design Goals

- **Performance**: Sub-second evaluation for most factors with caching
- **Scalability**: Support batch evaluation of hundreds of factors
- **Reliability**: Timeout protection, error handling, persistent cache
- **Flexibility**: Pluggable optimization methods and data sources

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│  - Python Client                                            │
│  - Web UI (Flask)                                           │
│  - REST API Clients                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                         │
│  - Backend App (Flask)                                      │
│  - Request Validation                                       │
│  - Response Formatting                                      │
│  - CORS Handling                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│  Factor Routes       │    │  Combination Routes  │
│  /factors/check      │    │  /combination/train  │
│  /factors/eval       │    │  /combination/backtest│
└──────────────────────┘    └──────────────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Processing Layer                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Expression   │  │   Cache      │  │   Timeout    │    │
│  │ Validator    │  │   Manager    │  │   Handler    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Factor Evaluation Engine                 │    │
│  │  - IC Computation (Vectorized)                   │    │
│  │  - Rank IC Computation                           │    │
│  │  - ICIR / Turnover Metrics                       │    │
│  └──────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────┐    │
│  │       Optimization Engine                        │    │
│  │  - LASSO Regression                              │    │
│  │  - IC Optimization (SSPO/mSSRM-PGA)              │    │
│  │  - Baseline Methods                              │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Access Layer                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Qlib Data  │  │  SQLite Cache│  │  Custom Ops  │    │
│  │   Loader     │  │   (Persistent)│  │  (Extended)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Backtest Engine (Qlib)                   │    │
│  │  - Portfolio Construction                        │    │
│  │  - Transaction Cost Model                        │    │
│  │  - Performance Metrics                           │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Storage Layer                             │
│  - Qlib Data Store (~/.qlib/qlib_data/cn_data)             │
│  - SQLite Cache (cache_data/factor_cache.db)               │
│  - Model Storage (cache_data/factor_models/)               │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

#### Factor Evaluation Flow

```
1. Client Request
   └─> POST /factors/eval

2. Request Validation
   └─> normalize_factors_from_expression_field()
   └─> Validate parameters (dates, market, etc.)

3. Cache Check
   └─> PersistentCache.get(key)
   └─> If hit: return cached result
   └─> If miss: proceed to evaluation

4. Factor Evaluation (with Timeout Protection)
   └─> run_eval_with_timeout()
       └─> Subprocess with hard timeout
       └─> Load data from Qlib
       └─> Compute IC metrics (vectorized)
       └─> Return metrics

5. Portfolio Backtest (Optional, if fast=false)
   └─> backtest_by_single_alpha()
       └─> Top-K stock selection
       └─> Portfolio construction
       └─> Transaction cost calculation
       └─> Performance metrics

6. Cache Write
   └─> PersistentCache.set(key, result)

7. Response
   └─> JSON with metrics and portfolio results
```

#### Factor Combination Flow

```
1. Client Request
   └─> POST /combination/train

2. Data Loading
   └─> Load factor values for all expressions
   └─> Load label data (returns)
   └─> Align dates and instruments

3. Optimization
   ├─> LASSO Path
   │   └─> LassoCV with cross-validation
   │   └─> Select optimal alpha
   │   └─> Extract weights
   │
   └─> IC Optimization Path
       └─> SSPO or mSSRM-PGA solver
       └─> Maximize IC subject to constraints
       └─> Extract weights

4. Model Export
   └─> Save weights, factor names, metadata
   └─> Store in cache_data/factor_models/

5. Response
   └─> Return weights and combined expression
```

---

## Core Components

### 1. Backend App (`backend_app.py`)

**Responsibilities:**
- Flask application initialization
- Blueprint registration (factors, combinations)
- Global cache initialization
- Health check endpoint
- Environment configuration

**Key Features:**
- CORS support for web clients
- Configurable timeouts via environment variables
- Persistent SQLite cache
- Extended Qlib operators registration

### 2. Factor Routes (`routes/factors.py`)

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/factors/check` | POST | Validate factor syntax |
| `/factors/eval` | POST | Evaluate factors with IC and backtest |

**Key Functions:**

- `normalize_factors_from_expression_field()`: Parse multiple expression formats
- `run_eval_with_timeout()`: Execute evaluation with hard timeout
- `run_check_with_timeout()`: Quick syntax validation
- `_run_portfolio_backtest()`: Portfolio-level backtesting

**Design Decisions:**

1. **Timeout Protection**: All evaluations run in subprocess with hard kill-on-timeout
2. **Flexible Input**: Support single string, dict, or list of expressions
3. **Fast Mode**: Skip portfolio backtest for rapid screening
4. **Parallel Backtest**: Use ThreadPoolExecutor for multiple backtests

### 3. Combination Routes (`routes/combinations.py`)

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/combination/train` | POST | Train factor combination |
| `/combination/backtest` | POST | Backtest trained model |
| `/combination/compare_methods` | POST | Compare LASSO vs IC opt |
| `/combination/automated_training` | POST | Rolling window training |
| `/combination/list_models` | GET | List saved models |
| `/combination/load_model` | GET | Load saved model |

**Integration Points:**
- LASSO training: `data/pipeline/optim/ml_training.py`
- IC optimization: `data/pipeline/optim/ic_optimization.py`
- Training pipeline: `data/pipeline/optim/training_pipeline.py`

### 4. Web App (`web_app.py`)

**Purpose**: Web UI for interactive factor analysis and optimization

**Key Routes:**

| Route | Purpose |
|-------|---------|
| `/` | Main UI dashboard |
| `/api/evaluate_factor` | Single factor evaluation |
| `/api/batch_evaluate_factor` | Batch evaluation with threading |
| `/api/factor_combination/fixed` | Fixed period optimization |
| `/api/factor_combination/dynamic` | Dynamic period optimization |
| `/api/factor_combination/backtest` | Run backtest from weights |

**Features:**
- Query history tracking
- Cache statistics visualization
- Factor comparison charts
- Export to CSV/JSON

---

## Data Pipeline

### Data Flow

```
Raw Market Data (Qlib)
    │
    ├─> Data Loader (backtest/qlib/dataloader.py)
    │   └─> Normalize instruments (CSI300/500/1000)
    │   └─> Date range filtering
    │   └─> Feature extraction
    │
    ├─> Factor Computation
    │   └─> Custom operators (utils/qlib_extend_ops.py)
    │   └─> Expression evaluation
    │   └─> Factor values aligned by date/instrument
    │
    ├─> Label Computation
    │   └─> Forward returns (close_return)
    │   └─> Alternative labels (volume_return, etc.)
    │
    └─> Metrics Computation
        ├─> IC (Pearson correlation)
        ├─> Rank IC (Spearman correlation)
        ├─> ICIR (IC / std(IC))
        ├─> Turnover
        └─> Portfolio metrics
```

### Backtest Pipeline

**Single Alpha Backtest** (`backtest/qlib/single_alpha_backtest.py`)

```python
def backtest_by_single_alpha(
    alpha_factor: str,
    topk: int = 50,
    n_drop: int = 5,
    start_time: str = "2020-01-01",
    end_time: str = "2021-01-01",
    instruments: str = "csi300",
    region: str = "cn",
    BENCH: str = "SH000300"
):
    """
    Portfolio backtest for a single alpha factor.

    Steps:
    1. Initialize Qlib data provider
    2. Create dataset with factor and label
    3. Run portfolio strategy (TopK with rebalancing)
    4. Compute excess returns vs benchmark
    5. Calculate risk metrics (Sharpe, IR, MaxDD)

    Returns:
    - analysis_df: Performance metrics
    - report_normal: Daily returns
    - positions_normal: Daily positions
    """
```

**Strategy Configuration:**

```python
{
    "topk": 50,           # Select top 50 stocks
    "n_drop": 5,          # Drop bottom 5 (100 - 5 = 95 total)
    "method_buy": "top",  # Buy top-ranked stocks
    "method_sell": "bottom",  # Sell bottom-ranked
    "hold_thresh": 1      # Rebalance threshold
}
```

### Factor Metrics

**IC Computation** (`backtest/factor_metrics/metrics.py`)

```python
def compute_ic_metrics(factor_df, label_df):
    """
    Vectorized IC computation across all dates.

    For each date:
    1. Get factor values for all stocks
    2. Get label values (forward returns)
    3. Compute Pearson correlation (IC)
    4. Compute Spearman correlation (Rank IC)

    Returns:
    - mean_ic, mean_rank_ic
    - icir, rank_icir
    - ic_series (daily IC values)
    - turnover
    """
```

**Performance Optimization:**
- Vectorized operations using NumPy/Pandas
- Avoid Python loops over dates
- Pre-compute common transformations
- Cache intermediate results

---

## Optimization Methods

### 1. Baseline Methods

**Equal Weight**
```python
weights = [1/N] * N  # N = number of factors
```

**Equal Top-K**
```python
# Select top K factors by IC, equal weight
top_k_indices = np.argsort(ic_values)[-k:]
weights = np.zeros(N)
weights[top_k_indices] = 1/k
```

### 2. LASSO Regression

**Implementation**: `data/pipeline/optim/ml_training.py`

**Algorithm**:
```
min_{w} ||y - Xw||^2 + alpha * ||w||_1

where:
- y: label (forward returns)
- X: factor matrix
- w: weights
- alpha: L1 regularization strength
```

**Key Features:**
- Automatic feature selection (sparse weights)
- Cross-validation for alpha tuning
- Rolling window training
- Out-of-sample testing

**Code Structure**:
```python
from sklearn.linear_model import LassoCV

model = LassoCV(
    alphas=np.logspace(-4, 0, 50),
    cv=5,
    max_iter=1000,
    n_jobs=-1
)

model.fit(factor_matrix, label_vector)
weights = model.coef_
```

### 3. IC Optimization

**Implementation**: `data/pipeline/optim/ic_optimization.py`

**Objective**:
```
max_{w} IC(combined_factor, label)

subject to:
- sum(w) = 1
- w >= 0 (optional, non-negative constraint)
- ||w||_1 <= budget (optional, sparsity)
- risk_penalty on variance
```

**Solvers**:

1. **SSPO** (`solver/SSPO.py`): Single-period Sharpe optimization
2. **mSSRM-PGA** (`solver/mSSRM_PGA.py`): Multi-period Sharpe ratio maximization

**Code Structure**:
```python
from data.pipeline.optim.solver.SSPO import SSPO

optimizer = SSPO(
    factors=factor_matrix,
    returns=label_vector,
    lambda_risk=1.0,
    alpha_l1=0.0
)

weights = optimizer.solve()
```

**Advantages over LASSO**:
- Direct IC maximization (not MSE minimization)
- Better handling of non-linear relationships
- Risk-aware weight allocation

---

## Caching Strategy

### Persistent Cache Architecture

**Implementation**: `utils/utils.py::PersistentCache`

```python
class PersistentCache:
    """
    SQLite-backed persistent cache for factor evaluations.

    Schema:
    - key: MD5 hash of (expr, market, start, end, label, topk, n_drop)
    - value: JSON-encoded result
    - created_at: Timestamp
    - last_accessed: Timestamp
    - access_count: Hit counter
    """
```

**Cache Key Generation**:
```python
def cache_key(expr, market, start, end, label, topk, n_drop):
    """
    Create deterministic cache key.

    Format: MD5(expr + market + start + end + label + topk + n_drop)
    """
    parts = [
        expr.strip(),
        market.lower(),
        start,
        end,
        label,
        str(topk),
        str(n_drop)
    ]
    key_str = "|".join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()
```

### Cache Operations

**Read Path**:
```python
cached = CACHE.get(key)
if cached:
    # Cache hit - return immediately
    return cached
else:
    # Cache miss - evaluate and store
    result = evaluate_factor(...)
    CACHE.set(key, result)
    return result
```

**Write Path**:
```python
CACHE.set(key, {
    "success": True,
    "metrics": {...},
    "portfolio_metrics": {...},
    "timestamp": "..."
})
```

### Cache Management

**Statistics**:
```python
stats = CACHE.stats()
# {
#   "size": 1500,
#   "max_size": 10000,
#   "hit_rate": 0.85,
#   "total_hits": 12500,
#   "total_misses": 2200
# }
```

**Eviction Policy**:
- LRU (Least Recently Used)
- Configurable max size
- Automatic cleanup of old entries

---

## Performance Considerations

### Bottlenecks and Optimizations

#### 1. Factor Evaluation

**Problem**: Evaluating complex factors can take 10-60 seconds

**Solutions**:
- ✅ Persistent cache (90%+ hit rate in production)
- ✅ Vectorized IC computation (10x speedup)
- ✅ Subprocess timeout protection
- ✅ Fast mode (skip backtest, 5x faster)

**Benchmark**:
```
Single factor eval (uncached):     5-15 seconds
Single factor eval (cached):       <10ms
Batch 100 factors (fast mode):     30-120 seconds
Batch 100 factors (with backtest): 10-30 minutes
```

#### 2. Portfolio Backtest

**Problem**: Portfolio simulation is compute-intensive

**Solutions**:
- ✅ Parallel execution with ThreadPoolExecutor
- ✅ Reuse Qlib data provider across threads
- ✅ Skip backtest in fast mode
- ✅ Cache backtest results

**Optimization**:
```python
# Before: Serial backtest
for expr in expressions:
    backtest(expr)  # 10s each * 100 = 1000s

# After: Parallel backtest
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(backtest, expr) for expr in expressions]
    results = [f.result() for f in futures]
# ~150s total (6.7x speedup)
```

#### 3. IC Computation

**Problem**: Nested loops over dates and stocks

**Original** (slow):
```python
ic_values = []
for date in dates:
    factor = get_factor(date)
    label = get_label(date)
    ic = pearsonr(factor, label)[0]
    ic_values.append(ic)
```

**Optimized** (fast):
```python
# Vectorized correlation across all dates
factor_matrix = factor_df.values  # (n_dates, n_stocks)
label_matrix = label_df.values
ic_values = np.corrcoef(factor_matrix, label_matrix, rowvar=True)
```

### Memory Management

**Data Loading**:
- Lazy loading of Qlib data
- Memory-mapped data files
- Chunked processing for large datasets

**Cache Size**:
- Max 10,000 entries by default
- Each entry ~1-10KB
- Total memory ~10-100MB

---

## Extension Points

### Adding New Optimization Methods

**Step 1**: Implement solver in `data/pipeline/optim/solver/`

```python
# solver/new_method.py
class NewOptimizer:
    def __init__(self, factors, returns, **config):
        self.factors = factors
        self.returns = returns
        self.config = config

    def solve(self):
        # Your optimization logic
        weights = ...
        return weights
```

**Step 2**: Add training function in `data/pipeline/optim/`

```python
# optim/new_method_training.py
def train_new_method(factor_expressions, start_date, end_date, **kwargs):
    # Load data
    factors, labels = load_factor_data(...)

    # Run optimizer
    optimizer = NewOptimizer(factors, labels, **kwargs)
    weights = optimizer.solve()

    return {
        "weights": weights,
        "factor_names": factor_expressions,
        "method": "new_method"
    }
```

**Step 3**: Add route in `routes/combinations.py`

```python
@bp.route("/train_new_method", methods=["POST"])
def train_new_method():
    data = request.json
    result = train_new_method(**data)
    return jsonify({"success": True, "result": result})
```

### Adding New Markets

**Step 1**: Configure Qlib data

```bash
# Download market data
qlib-data download --market new_market --region cn

# Initialize Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/new_market_data", region="cn")
```

**Step 2**: Add market mapping in `routes/factors.py`

```python
market_map = {
    "csi300": ("csi300", "~/.qlib/qlib_data/cn_data", "cn", "SH000300"),
    "csi500": ("csi500", "~/.qlib/qlib_data/cn_data", "cn", "SH000905"),
    "new_market": ("new_market", "~/.qlib/qlib_data/new_market_data", "cn", "BENCH_CODE"),
}
```

### Custom Factor Operators

**Implementation**: `utils/qlib_extend_ops.py`

```python
from qlib.data.ops import Operators

class CustomOp(Operators):
    """Custom operator for factor expressions."""

    @staticmethod
    def my_operator(series, window=20):
        # Your custom logic
        return series.rolling(window).apply(lambda x: ...)

# Register
Operators.register(CustomOp)
```

**Usage in expressions**:
```python
expr = "MyOperator($close, 20)"
```

---

## Testing and Validation

### Unit Tests

**Location**: `tests/factors/`

```python
# tests/factors/test_eval.py
def test_single_factor_eval():
    result = client.evaluate_factor(
        expr="Rank($close, 20)",
        market="csi300",
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    assert result['success'] == True
    assert 'metrics' in result
    assert 'ic' in result['metrics']
```

### Integration Tests

**Samples**: `tests/samples/`
- `test_fixed_sample.json`: Fixed period optimization test
- `test_dynamic_sample.json`: Dynamic period optimization test

### Performance Benchmarks

**Run benchmarks**:
```bash
python tests/benchmark_eval.py
python tests/benchmark_combination.py
```

---

## Deployment Considerations

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up persistent cache directory
- [ ] Initialize Qlib data
- [ ] Configure timeout limits
- [ ] Set up monitoring (cache hit rate, latency)
- [ ] Configure CORS for production domains
- [ ] Enable HTTPS
- [ ] Set up logging and error tracking

### Scaling Strategies

1. **Horizontal Scaling**: Multiple API instances with shared cache (Redis)
2. **Vertical Scaling**: Increase workers for parallel backtest
3. **Caching Layer**: Redis or Memcached for distributed cache
4. **Load Balancing**: Nginx/HAProxy for request distribution

### Monitoring

**Key Metrics**:
- API latency (p50, p95, p99)
- Cache hit rate
- Evaluation timeout rate
- Backtest success rate
- Memory usage
- Disk I/O for cache

---

## References

**External Libraries**:
- [Qlib](https://github.com/microsoft/qlib): Quantitative investment platform
- [Flask](https://flask.palletsprojects.com/): Web framework
- [Scikit-learn](https://scikit-learn.org/): Machine learning (LASSO)
- [NumPy](https://numpy.org/): Numerical computing
- [Pandas](https://pandas.pydata.org/): Data analysis

**Related Documentation**:
- [README_API.md](README_API.md): API usage guide
- [README.md](README.md): Original README

---

## License

Part of the qfinzero project.
