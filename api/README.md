# Factor Evaluation REST API

This REST API provides efficient factor evaluation services for the SwiftAlpha EA search system.

## Features

- **Single Factor Evaluation**: Evaluate individual factor expressions
- **Batch Evaluation**: Evaluate multiple factors in a single request
- **Expression Fixing**: Automatically fixes common expression issues
- **Caching**: In-memory caching for repeated evaluations
- **Health Monitoring**: Health check and cache statistics endpoints

## Quick Start

### 1. Start the API Server

**Windows:**
```bash
api\start_api_server.bat
```

**Linux/Mac:**
```bash
bash api/start_api_server.sh
```

**Direct Python:**
```bash
python api/factor_eval_api.py
```

The server will start on `http://localhost:8080` by default.

### 2. Test the API

Check if the server is running:
```bash
curl http://localhost:8080/health
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Factor Evaluation API",
  "timestamp": "2024-01-01T12:00:00",
  "cache_size": 10
}
```

### GET/POST /eval
Evaluate a single factor expression.

**GET Example:**
```
http://localhost:8080/eval?expr="Rank(Corr($close,$volume,10),252)"&start='2023-01-01'&end='2024-01-01'&market='csi300'
```

**POST Example:**
```json
{
  "expr": "Rank(Corr($close, $volume, 10), 252)",
  "start": "2023-01-01",
  "end": "2024-01-01",
  "market": "csi300",
  "label": "close_return",
  "use_cache": true
}
```

**Response:**
```json
{
  "success": true,
  "expression": "Rank(Corr($close, $volume, 10), 252)",
  "fixed_expression": "Rank(Corr($close, $volume, 10), 252)",
  "market": "csi300",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "metrics": {
    "ic": 0.0234,
    "rank_ic": 0.0312,
    "ir": 0.4521,
    "icir": 0.5123,
    "rank_icir": 0.5234,
    "turnover": 0.7234
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /batch_eval
Evaluate multiple factors in batch.

**Request:**
```json
{
  "factors": [
    {"name": "factor1", "expr": "Rank($close, 20)"},
    {"name": "factor2", "expr": "Mean($volume, 10)"}
  ],
  "start": "2023-01-01",
  "end": "2024-01-01",
  "market": "csi300",
  "label": "close_return",
  "use_cache": true
}
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "results": [
    {
      "name": "factor1",
      "success": true,
      "metrics": {...}
    },
    {
      "name": "factor2",
      "success": true,
      "metrics": {...}
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /clear_cache
Clear the evaluation cache.

**Response:**
```json
{
  "success": true,
  "message": "Cache cleared",
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /cache_stats
Get cache statistics.

**Response:**
```json
{
  "cache_size": 150,
  "max_cache_size": 1000,
  "cache_keys": ["expr1_csi300_2023-01-01_2024-01-01_close_return", ...],
  "timestamp": "2024-01-01T12:00:00"
}
```

## Python Client Usage

### Basic Usage

```python
from api.factor_eval_client import evaluate_factor_via_api

# Evaluate a single factor
result = evaluate_factor_via_api(
    expr="Rank(Corr($close, $volume, 10), 252)",
    market="csi300",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

if result['success']:
    print(f"Rank IC: {result['metrics']['rank_ic']:.4f}")
```

### Batch Evaluation

```python
from api.factor_eval_client import batch_evaluate_factors_via_api

factors = [
    {'name': 'mom_factor', 'expression': 'Rank($close, 20)'},
    {'name': 'vol_factor', 'expression': 'Mean($volume, 10)'}
]

results = batch_evaluate_factors_via_api(
    factors=factors,
    market="csi300",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

for result in results:
    if result['success']:
        print(f"{result['name']}: IC={result['metrics']['ic']:.4f}")
```

### Using with EA Search

```python
from searcher.EAFactorSearcher import one_shot_ea_search

# Run EA search with API evaluation
results = one_shot_ea_search(
    population_size=10,
    offspring_size=20,
    market="CSI300",
    start_date="2023-01-01",
    end_date="2024-01-01",
    use_api=True,  # Enable API evaluation
    api_url="http://localhost:8080"
)
```

## Examples

### Run EA with API

```bash
# Start the API server first
python api/factor_eval_api.py

# In another terminal, run EA with API
python api/example_ea_with_api.py --mode api
```

### Compare API vs Direct Evaluation

```bash
python api/example_ea_with_api.py --mode compare
```

## Configuration

### Environment Variables

- `PORT`: API server port (default: 8080)
- `DEBUG`: Enable debug mode (default: False)

### API Client Configuration

```python
from api.factor_eval_client import FactorEvalClient

# Custom configuration
client = FactorEvalClient(
    base_url="http://localhost:8080",
    timeout=30  # Request timeout in seconds
)
```

## Performance Tips

1. **Use Caching**: Keep `use_cache=true` for repeated evaluations
2. **Batch Requests**: Use `/batch_eval` for multiple factors
3. **Optimize Expressions**: The API automatically fixes common issues
4. **Monitor Cache**: Use `/cache_stats` to monitor cache usage

## Troubleshooting

### API Server Not Starting

1. Check if port 8080 is already in use
2. Install required packages: `pip install flask flask-cors requests`
3. Check Python version (requires Python 3.7+)

### Connection Errors

1. Verify the API server is running: `curl http://localhost:8080/health`
2. Check firewall settings
3. Verify the API URL in your client code

### Evaluation Errors

1. Check factor expression syntax
2. Verify date format (YYYY-MM-DD)
3. Ensure market identifier is valid (e.g., 'csi300', 'csi500')

## Architecture

```
┌─────────────┐     HTTP      ┌──────────────┐
│  EA Search  │──────────────>│   API Server │
└─────────────┘                └──────────────┘
                                      │
                                      v
                              ┌──────────────┐
                              │  Expression  │
                              │    Fixer     │
                              └──────────────┘
                                      │
                                      v
                              ┌──────────────┐
                              │   Qlib Data  │
                              │   Loader     │
                              └──────────────┘
                                      │
                                      v
                              ┌──────────────┐
                              │  Performance │
                              │   Metrics    │
                              └──────────────┘
```

## License

Part of the SwiftAlpha project.