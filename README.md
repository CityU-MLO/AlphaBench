# AlphaBench

**The first systematic benchmark to rigorously evaluate LLMs across the entire formulaic alpha mining workflow.**

AlphaBench assesses how well large language models can generate, evaluate, and refine mathematical expressions (alpha factors) that predict stock returns — covering Chinese A-share markets (CSI300/CSI500/CSI1000) and US markets (SP500).

> Paper: [AlphaBench — ICLR 2026](https://alphabench.cc/) | Website: [alphabench.cc](https://alphabench.cc/)

---

## What Is AlphaBench?

Formulaic alpha mining is the process of discovering mathematical formulas derived from market data (price, volume) that predict future stock performance. Traditionally this requires expert quant knowledge and expensive backtesting loops. AlphaBench evaluates whether LLMs can automate and accelerate this process.

The benchmark covers four tasks:

| Task | Name | Description |
|------|------|-------------|
| **T1** | Factor Generation | Translate natural-language instructions into executable Qlib factor expressions |
| **T2** | Factor Evaluation | Predict factor quality metrics (IC, RankIC, win rate, skewness) without backtesting |
| **T3** | Iterative Searching | Use LLM-guided search (CoT, ToT, EA) to discover novel alpha factors |
| **T4** | Atomic Evaluation | Binary noise classification and pairwise factor comparison |

---

## Repository Structure

```
AlphaBench/
│
├── ffo/                      # Factor Feature Oracle — evaluation engine
│   ├── api/                  # Typed Python API (evaluate_factor, batch_evaluate_factors)
│   ├── cli/                  # `ppo` CLI (start/stop/eval/check/config/cache)
│   ├── client/               # Low-level HTTP client (FactorEvalClient)
│   ├── mcp/                  # MCP server for LLM agents (Claude Desktop, etc.)
│   ├── routes/               # Flask REST endpoints (/factors/eval, /combination/train)
│   ├── config/               # ffo.yaml — default configuration
│   ├── backtest/             # Qlib portfolio backtesting integration
│   ├── demos/                # 5 runnable demo scripts
│   └── tests/                # Test suite
│
├── searcher/                 # LLM-guided factor search platform (T3)
│   ├── algo/                 # Pluggable algorithms: EA, CoT, ToT
│   ├── pipeline.py           # SearchPipeline — end-to-end orchestrator
│   ├── backtester.py         # FFO client wrapper
│   └── config/               # Config dataclasses + YAML loader
│
├── agent/                    # LLM integration & factor generation
│   ├── compiler.py           # Natural language → Qlib expression compiler
│   ├── generator_qlib.py     # LLM-based factor generator (with retries)
│   ├── llm_client.py         # LLM API client (DeepSeek, OpenAI, local vLLM)
│   ├── prompts_bank/         # Prompt templates
│   └── qlib_contrib/         # Qlib operator extensions & validators
│
├── benchmark/                # Benchmark orchestration (T1–T4)
│   ├── run_benchmark.py      # Main entry point
│   ├── engine/               # Task-specific runners (generate/, evaluate/, searching/)
│   └── data/                 # Benchmark datasets (raw IC tables, factor pool)
│
├── example/                  # Ready-to-run example scripts
│   ├── evaluation/           # T2/T4: build datasets + run evaluation
│   ├── generate/             # T1: generate factors + evaluate fitness
│   └── search/               # T3: run LLM-guided search
│
├── factors/                  # Factor libraries
│   ├── lib/alpha158/         # 158 pre-built baseline factors
│   └── registry/             # Factor taxonomy & metadata
│
├── backtest/                 # Backtesting utilities (Qlib integration)
├── config/                   # Top-level YAML configs (benchmark.yaml, api_keys.yaml)
└── pyproject.toml            # Package metadata (installs the `ppo` CLI)
```

---

## Prerequisites

### 1. Python environment

Python ≥ 3.9 required.

### 2. Qlib data

AlphaBench uses [Qlib](https://github.com/microsoft/qlib) for backtesting. Download the Chinese market data:

```bash
pip install pyqlib
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 3. LLM API keys

Set your API key(s) as environment variables:

```bash
export DEEPSEEK_API_KEY="sk-..."      # DeepSeek (default model)
export OPENAI_API_KEY="sk-..."        # OpenAI (optional)
```

Or configure them in `config/api_keys.yaml`.

---

## Quick Start

### Step 1 — Install FFO

FFO (Factor Feature Oracle) is the evaluation engine that all other components depend on. Install it first:

```bash
# From the repo root
pip install -e .

# Verify
ppo --version
```

### Step 2 — Start the FFO backend

```bash
ppo start backend        # starts REST API at http://127.0.0.1:19777
ppo status               # confirm it's running
```

### Step 3 — Evaluate your first factor

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
    print(f"IC={result.metrics.ic:.4f}  RankIC={result.metrics.rank_ic:.4f}  ICIR={result.metrics.icir:.4f}")
```

**REST API:**
```bash
curl -s -X POST http://127.0.0.1:19777/factors/eval \
  -H "Content-Type: application/json" \
  -d '{"expression": "Rank($close, 20)", "market": "csi300", "fast": true}' \
  | python -m json.tool
```

---

## Benchmark Tasks

### T1 — Factor Generation

Tests whether an LLM can translate natural-language descriptions into valid Qlib factor expressions, covering two sub-tasks (Text2Alpha, Directional Mining) and a stability probe.

```bash
# Generate factors with deepseek-chat
python example/generate/run_t1_generate.py

# Generate with GPT-4.1 and chain-of-thought
python example/generate/run_t1_generate.py --model gpt-4.1 --cot

# Evaluate generated factors (fitness + diversity)
python example/generate/eval_t1_fitness.py --run_dir runs/T1/deepseek-chat_False
```

**Metrics:** Fitness (accuracy vs. LLM judge), AST mean distance (structural diversity), IC mean correlation (factor diversity), Stability consistency.

---

### T2 — Factor Evaluation

Measures zero-shot LLM ability to rank and score alpha factors by predicted IC/RankIC/win rate/skewness.

```bash
# Step 1: Build evaluation datasets
python example/evaluation/01_build_datasets/build_all.py

# Step 2: Run ranking evaluation
python example/evaluation/02_run_evaluation/run_t2_ranking.py \
    --model gpt-4.1 \
    --data_dir benchmark/data/evaluate/built/ranking_csi300

# Step 2: Run scoring evaluation
python example/evaluation/02_run_evaluation/run_t2_scoring.py \
    --model gpt-4.1 \
    --data_dir benchmark/data/evaluate/built/scoring_csi300

# Step 3: Generate comparison report across models
python example/evaluation/02_run_evaluation/generate_report.py \
    --compare \
    --run_dirs "GPT-4.1=runs/T2/gpt-4.1_False" \
               "DeepSeek=runs/T2/deepseek-chat_False"
```

**Metrics:** Precision@K, NDCG@K (ranking); MAE, Acc_Signal, per-class F1 (scoring).

---

### T3 — Iterative Searching

Evaluates LLMs as search agents that iteratively discover better-performing alpha factors using three algorithms.

```bash
# Copy and edit the config
cp example/search/configs/search_csi300.yaml my_search.yaml

# Run EA search (default)
python example/search/run_t3_search.py --config my_search.yaml

# Override model at runtime
python example/search/run_t3_search.py --config my_search.yaml --model_name gpt-4.1

# Use a local vLLM server
python example/search/run_t3_search.py \
    --config my_search.yaml \
    --model_name Qwen2.5-72B-Instruct --model_local true --local_port 8000
```

**Algorithms:**

| Algorithm | Description |
|-----------|-------------|
| **EA** (Evolutionary) | Population-based; LLM performs mutation & crossover each generation |
| **CoT** (Chain-of-Thought) | Single-path iterative refinement using full history as context |
| **ToT** (Tree-of-Thought) | Parallel recursive tree expansion; only survivors branch further |

**Metrics:** IC, RankIC, ICIR per discovered factor; best-of-generation IC for EA.

---

### T4 — Atomic Evaluation

Probes LLM understanding of factor properties through two focused tasks.

```bash
# Build atomic datasets
python example/evaluation/01_build_datasets/build_atomic_noise_dataset.py
python example/evaluation/01_build_datasets/build_atomic_pairwise_dataset.py

# Run both atomic tasks
python example/evaluation/02_run_evaluation/run_t4_atomic.py \
    --model gpt-4.1 \
    --tasks all \
    --splits test
```

**Sub-tasks:**
- **Noise Classification** — binary: is this factor signal or noise?
- **Pairwise Selection** — which of two factors has higher IC?

**Metrics:** Accuracy, Precision, Recall, F1 (noise); Accuracy (pairwise).

---

## FFO Demos

Five ready-to-run scripts that showcase FFO capabilities (make sure the backend is running first):

```bash
ppo start backend

python ffo/demos/01_basic_usage.py      # single factor evaluation + health check
python ffo/demos/02_batch_eval.py       # parallel batch eval + ranking table
python ffo/demos/03_mcp_client.py       # MCP tool schema + agent simulation
python ffo/demos/04_config_and_cli.py   # config system + CLI reference
python ffo/demos/05_beta_batch_eval.py  # advanced batch options
```

---

## FFO — Key Features

### Batch evaluation (parallel)

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
    fast=True,          # IC-only; set False for full portfolio backtest
    parallel=True,
    max_workers=8,
    progress=True,
)

for r in sorted(results, key=lambda x: -x.metrics.rank_ic):
    print(f"{r.expression[:50]:50s}  RankIC={r.metrics.rank_ic:+.4f}")
```

**Throughput:** 8 workers → ~70s for 100 factors; cached results are instant.

### MCP server for LLM agents

FFO exposes all evaluation tools as [Model Context Protocol](https://modelcontextprotocol.io) tools, making them directly callable by Claude and other MCP-compatible agents.

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

Available MCP tools: `evaluate_factor`, `batch_evaluate_factors`, `check_factor_syntax`, `get_server_health`, `get_cache_stats`, `clear_cache`.

### Custom search algorithm

```python
from searcher.algo.base import BaseAlgo
from searcher.algo import register_algo

class MyAlgo(BaseAlgo):
    name = "my_algo"

    def run(self, seeds, save_dir):
        # self.evaluate_fn(expr) -> Dict
        # self.batch_evaluate_fn(factors) -> List[Dict]
        # self.search_fn(instruction, model, N) -> Dict
        ...
        return {"best": ..., "history": [...], "final_pool": [...]}

register_algo("my_algo", MyAlgo)
```

Then in your config YAML:
```yaml
searching:
  algo:
    name: my_algo
    param:
      my_param: 42
```

---

## Factor Expression Format

Factors are Qlib-style mathematical expressions combining:

| Category | Symbols / Operators |
|----------|-------------------|
| **Variables** | `$open`, `$high`, `$low`, `$close`, `$volume` |
| **Math** | `Add`, `Sub`, `Mul`, `Div`, `Abs`, `Log`, `Sqrt` |
| **Time-series** | `Ref`, `Delay`, `Delta`, `Mean`, `Std`, `Sum`, `Max`, `Min` |
| **Rank / Corr** | `Rank`, `Slope`, `Corr` |
| **Logic** | `And`, `Or`, `Lt`, `Gt`, `Eq` |

**Examples:**
```
Rank($close, 20)                             # 20-day price rank
Mean($volume, 5) / Mean($volume, 20)         # short/long volume ratio
Std(Delta(Log($close), 1), 60)               # 60-day realized volatility
Corr($close, $volume, 10)                    # price-volume correlation
```

---

## Supported Markets & Metrics

**Markets:** CSI300, CSI500, CSI1000 (Chinese A-shares), SP500 (US)

**Evaluation metrics:**

| Metric | Description |
|--------|-------------|
| IC | Information Coefficient (Pearson correlation with next-day returns) |
| Rank IC | Spearman rank correlation with next-day returns |
| ICIR | IC / std(IC) — information ratio |
| Turnover | Daily portfolio rebalancing rate |
| Win Rate | Fraction of days with positive IC |

---

## Configuration

FFO is configured via `~/.ppo/config.yaml` (user overrides) layered over the bundled `ffo/config/ffo.yaml`:

```bash
ppo config init                          # create user config file
ppo config show                          # print effective config
ppo config set evaluation.market csi500  # override a single key
```

Environment variables are also supported:
```bash
FFO_BACKEND_PORT=19778 ppo start backend
FFO_EVALUATION_MARKET=csi500 ppo eval "Rank($close, 20)"
```

---

## Further Reading

| Document | Contents |
|----------|----------|
| [ffo/README.md](ffo/README.md) | FFO CLI, Python API, REST endpoints, configuration |
| [ffo/README_API.md](ffo/README_API.md) | Full REST API reference |
| [ffo/README_DESIGN.md](ffo/README_DESIGN.md) | Architecture and design decisions |
| [searcher/README.md](searcher/README.md) | Search algorithms, pipeline API, custom algo guide |
| [example/README.md](example/README.md) | Step-by-step evaluation workflow |
| [example/generate/README.md](example/generate/README.md) | T1 generation + fitness evaluation |
| [example/search/README.md](example/search/README.md) | T3 search config reference |

---

## License

MIT License. See `LICENSE` for details.
