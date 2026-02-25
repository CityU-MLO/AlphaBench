# AlphaBench Searcher

A general-purpose platform for LLM-guided quantitative factor discovery.

All backtesting is delegated to the **FFO server** — no local backtest logic lives here.

---

## Directory Structure

```
searcher/
├── README.md               ← you are here
├── start_search.py         ← CLI entry point
├── pipeline.py             ← SearchPipeline (end-to-end orchestrator)
├── backtester.py           ← Backtester (FFO client wrapper)
├── config.yaml             ← default YAML configuration
│
├── algo/                   ← search algorithms (pluggable)
│   ├── base.py             ← BaseAlgo abstract interface
│   ├── cot.py              ← Chain-of-Thought  (CoTSearcher + CoTAlgo)
│   ├── ea.py               ← Evolutionary Algorithm (EA_Searcher + EAAlgo)
│   ├── tot.py              ← Tree-of-Thought    (ToTSearcher + ToTAlgo)
│   └── __init__.py         ← registry: create_algo(), register_algo(), list_algos()
│
├── config/
│   └── config.py           ← dataclasses + YAML loader
│
└── utils/
    └── logger.py           ← SearchLogger
```

---

## Quick Start

```bash
# Cold start — LLM generates initial factors
python searcher/start_search.py --config searcher/config.yaml

# Warm start — load seed factors from a file (one expression per line)
python searcher/start_search.py --config searcher/config.yaml --seed-file seeds.txt

# Warm start — use built-in alpha158 library
python searcher/start_search.py --config searcher/config.yaml --alpha158

# Resume — continue from a previous run's checkpoint
python searcher/start_search.py --config searcher/config.yaml --resume results/final_pool.jsonl
```

---

## Configuration

```yaml
searching:
  algo:
    name: ea            # "ea" | "cot" | "tot"
    seed_file: ""       # optional warm-start file path

    param:              # algo-specific params (forwarded verbatim)
      rounds: 10
      N: 30
      mutation_rate: 0.4
      crossover_rate: 0.6
      pool_size: 30

  model:
    name: "deepseek-chat"
    base_url: "https://api.deepseek.com"
    key: "${DEEPSEEK_API_KEY}"
    temperature: 0.7

backtesting:
  ffo_server: "127.0.0.1:19777"
  market: "csi300"
  benchmark: "SH000300"
  period_start: "2022-01-01"
  period_end: "2023-01-01"
  top_k: 30
  n_drop: 5
  fast: true      # IC-only mode (~1-2 s/factor); false = full portfolio backtest
  n_jobs: 4

savedir: "./results/search_001"
```

---

## Pipeline Flow

```
Cold / Warm / Resume start
         │
         ▼
  Baseline evaluation        ← FFO server evaluates every seed factor
         │
         ▼
  Search algorithm runs      ← algo calls Backtester for every candidate
  (CoT / EA / ToT)           ← algo calls LLM for mutation / crossover
         │
         ▼
  Save results               ← final_pool.jsonl + best_factor.json
```

---

## Algo Modules (`searcher/algo/`)

All algorithms inherit from `BaseAlgo` and receive injected callables — no hard-coded URLs or global state.

### EA — Evolutionary Algorithm (`algo/ea.py`)

Population-based search. Each generation:
1. Select top seeds from pool.
2. Call LLM for **mutation** (tweak parameters/operators) and **crossover** (combine sub-expressions from multiple seeds).
3. Evaluate all offspring via FFO.
4. Keep best `pool_size` factors.

| Param | Default | Description |
|---|---|---|
| `rounds` | 10 | Number of generations |
| `N` | 30 | New candidates per round |
| `mutation_rate` | 0.4 | Fraction of N from mutation |
| `crossover_rate` | 0.6 | Fraction of N from crossover |
| `pool_size` | 30 | Max pool size (kept constant) |
| `seeds_top_k` | 12 | Seeds shown to LLM per round |

### CoT — Chain-of-Thought (`algo/cot.py`)

Single-path iterative refinement. Each round:
1. Build a prompt with the **full chain history** (seed → all prior results).
2. Call LLM for **exactly 1 candidate**.
3. Evaluate; keep if improved.

Runs on the **top seed** (by IC) from the pool.

| Param | Default | Description |
|---|---|---|
| `rounds` | 10 | Refinement rounds |
| `temperature` | 1.75 | LLM sampling temperature |

### ToT — Tree-of-Thought (`algo/tot.py`)

Parallel recursive tree expansion. At each node:
1. Call LLM to generate **N candidates** from the current parent.
2. Evaluate all candidates; select **survivors** (IC > seed IC, up to `top_k`).
3. Recursively expand each survivor **in parallel** (next depth).

Runs on the **top seed** (by IC) from the pool.

| Param | Default | Description |
|---|---|---|
| `rounds` | 3 | Max tree depth |
| `N` | 6 | Candidates per node |
| `top_k` | 3 | Max survivors per node |

---

## Backtester (`backtester.py`)

Routes all factor evaluation through the FFO server. Exposes three callable factories for algos:

```python
bt = Backtester.from_config(config.backtesting)

bt.as_evaluate_fn()           # fn(expr: str) -> Dict         (for CoT, ToT single eval)
bt.as_batch_evaluate_fn()     # fn(factors: List[Dict]) -> List[Dict]  (for EA)
bt.as_batch_evaluate_fn_dict()# fn(factors: List[Dict]) -> Dict[str, Dict]  (for ToT batch)

bt.check_health()             # bool — pings FFO /health
```

---

## Extending with a Custom Algorithm

```python
from searcher.algo.base import BaseAlgo
from searcher.algo import register_algo

class MyAlgo(BaseAlgo):
    name = "my_algo"

    def run(self, seeds, save_dir):
        # seeds: [{"name": str, "expression": str, "metrics": {...}}, ...]
        # self.evaluate_fn(expr) -> Dict
        # self.batch_evaluate_fn(factors) -> List[Dict]
        # self.search_fn(instruction, model, N, **kw) -> Dict
        ...
        return {"best": ..., "history": [...], "final_pool": [...]}

register_algo("my_algo", MyAlgo)
```

Then in your YAML:
```yaml
searching:
  algo:
    name: my_algo
    param:
      my_param: 42
```

---

## Python API

```python
from searcher import SearchPipeline, load_config_from_yaml

config = load_config_from_yaml("searcher/config.yaml")
pipeline = SearchPipeline(config)

# Warm start
results = pipeline.run(seed_file="seeds.txt")

# Cold start
results = pipeline.run()

print(results["best"])           # best factor found
print(len(results["final_pool"])) # all factors in final pool
```
