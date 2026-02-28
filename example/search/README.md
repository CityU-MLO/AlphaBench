# AlphaBench — T3 Search Examples

This folder provides ready-to-run scripts for the **T3 Factor Searching** task:
using LLM-driven search algorithms (CoT, ToT, EA) to discover novel alpha
factors that outperform a set of seed factors.

---

## Prerequisites

Run all scripts from the **repo root**:
```bash
cd /path/to/AlphaBench
```

A running **FFO (Factor Fitness Oracle) server** is required for online factor
evaluation.  Configure its address in `ffo/config/ffo.yaml` before running.

Seed factors are loaded automatically from the built-in Alpha158 factor
library — no additional data preparation is needed.

---

## Quick Start

### 1. Edit the config

Copy and customize the provided example config:
```bash
cp example/search/configs/search_csi300.yaml my_search.yaml
```

Set your model, market, and which algorithms to enable, then run:

### 2. Run searching

```bash
# CSI300, EA only (default config)
python example/search/run_t3_search.py \
    --config example/search/configs/search_csi300.yaml

# Override model from CLI
python example/search/run_t3_search.py \
    --config example/search/configs/search_csi300.yaml \
    --model_name gpt-4.1

# Local vLLM server
python example/search/run_t3_search.py \
    --config example/search/configs/search_csi300.yaml \
    --model_name Qwen2.5-72B-Instruct --model_local true --local_port 8000

# Target SP500 market, custom output dir
python example/search/run_t3_search.py \
    --config example/search/configs/search_csi300.yaml \
    --market sp500 --save_dir runs/T3/sp500_ea
```

---

## Output Layout

```
runs/T3/<model>_<market>/
  factor_seed_metrics.json         ← baseline performance of seed factors
  CoT/                             ← (if cot.enable: true)
    <factor_name>/
      round_0.json
      round_1.json
      ...
  ToT/                             ← (if tot.enable: true)
    <factor_name>/
      ...
  EA/                              ← (if ea.enable: true)
    population_<gen>.pkl           ← population snapshot per generation
    summary.json                   ← best factors found
```

---

## Config Reference

YAML configs in `configs/` control all search parameters.

### Model

| Key | Description |
|-----|-------------|
| `model.name` | LLM model name (e.g. `deepseek-chat`, `gpt-4.1`) |
| `model.local` | `true` to use a local vLLM server |
| `model.local_port` | Port for local server |
| `model.temperature` | Sampling temperature (higher → more exploration) |

### Market

| Key | Description |
|-----|-------------|
| `market` | Target market: `csi300` or `sp500` |

### CoT Search

| Key | Default | Description |
|-----|---------|-------------|
| `cot.enable` | `false` | Enable CoT search |
| `cot.rounds` | `10` | Refinement rounds per seed factor |

**Seed factors:** kbar + price factors from Alpha158.

### ToT Search

| Key | Default | Description |
|-----|---------|-------------|
| `tot.enable` | `false` | Enable ToT search |
| `tot.rounds` | `3` | Tree depth (branching search rounds) |
| `tot.size` | `6` | Candidates per node per round |

**Seed factors:** kbar + price factors from Alpha158.

### EA Search

| Key | Default | Description |
|-----|---------|-------------|
| `ea.enable` | `true` | Enable EA search |
| `ea.mutation_rate` | `0.4` | Mutation probability |
| `ea.crossover_rate` | `0.6` | Crossover probability |
| `ea.generations` | `10` | Number of evolution generations |
| `ea.generate_size` | `30` | New candidates per generation |
| `ea.population_size` | `30` | Population size |

**Seed factors:** rolling factors from Alpha158.

---

## CLI Reference — `run_t3_search.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `example/search/configs/search_csi300.yaml` | YAML config path |
| `--model_name` | *(from config)* | Override model name |
| `--model_local` | *(from config)* | Override local flag (`true`/`false`) |
| `--local_port` | *(from config)* | Override local server port |
| `--market` | *(from config)* | Override target market |
| `--save_dir` | *(auto)* | Override output directory |

---

## Key Metrics

| Algorithm | Metric | Description |
|-----------|--------|-------------|
| CoT / ToT / EA | IC | Information Coefficient vs. next-day returns |
| CoT / ToT / EA | RankIC | Rank Information Coefficient |
| CoT / ToT / EA | ICIR | IC Information Ratio (IC / std) |
| CoT / ToT / EA | IR | Return / risk-adjusted return |
| EA | Best-of-generation IC | Best IC achieved each generation |
