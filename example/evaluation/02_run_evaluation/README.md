# 02 — Run Evaluation

Scripts for running LLM inference on evaluation datasets and generating performance reports.

## Prerequisites

Run scripts in `01_build_datasets/` first to create the dataset files.

---

## Scripts

### `run_t2_ranking.py` — T2 Factor Ranking
Presents the LLM with N factor expressions and asks it to identify the top-K.
Measures **Precision@K** and **NDCG@K** per market regime and difficulty setting.

```bash
# Default: GPT-4.1, no CoT, test split
python example/02_run_evaluation/run_t2_ranking.py

# Chain-of-thought
python example/02_run_evaluation/run_t2_ranking.py --model gpt-4.1 --cot

# Local vLLM server (e.g. Qwen2.5-72B)
python example/02_run_evaluation/run_t2_ranking.py \
    --model Qwen2.5-72B-Instruct --local --local_port 8000

# Only re-compute metrics (results already exist)
python example/02_run_evaluation/run_t2_ranking.py \
    --run_dir runs/T2/gpt-4.1_False --eval_only
```

**Expected data layout:**
```
benchmark/data/evaluate/built/ranking_csi300/
    all_env_scenarios.json
benchmark/data/evaluate/pool/
    factor_pool.json
```

**Outputs** (`runs/T2/<model>_<cot>/`):
```
ranking_results.json      ← raw LLM responses
report/
    ranking_summary.csv   ← Precision@K / NDCG@K per setting × regime
    report.md
```

---

### `run_t2_scoring.py` — T2 Factor Scoring
Asks the LLM to score each factor on four dimensions (1–5) and classify its signal direction.
Measures **MAE** per dimension and **Precision/Recall/F1** for Positive/Negative/Noise.

```bash
# Default: GPT-4.1, no CoT
python example/02_run_evaluation/run_t2_scoring.py

# With CoT
python example/02_run_evaluation/run_t2_scoring.py --cot

# Different model
python example/02_run_evaluation/run_t2_scoring.py --model deepseek-chat

# Skip inference, re-run report only
python example/02_run_evaluation/run_t2_scoring.py --eval_only
```

**Expected data layout:**
```
benchmark/data/evaluate/built/scoring_csi300/
    alphabench_testset.json
benchmark/data/evaluate/pool/
    factor_pool.json
```

**Outputs** (`runs/T2/<model>_<cot>/`):
```
scoring_results.json
report/
    scoring_mae_summary.csv      ← MAE per score dimension × regime
    scoring_classification.csv   ← Precision/Recall/F1 per class × regime
    scoring_per_case.csv
    report.md
```

---

### `run_t4_atomic.py` — T4 Atomic Evaluation
Runs LLM inference for binary noise classification and/or pairwise factor selection.

```bash
# Both tasks, test split (default)
python example/02_run_evaluation/run_t4_atomic.py

# Noise only
python example/02_run_evaluation/run_t4_atomic.py --tasks noise

# Pairwise only with CoT
python example/02_run_evaluation/run_t4_atomic.py --tasks pairwise --cot

# All splits
python example/02_run_evaluation/run_t4_atomic.py --splits train val test

# Local server
python example/02_run_evaluation/run_t4_atomic.py \
    --model Qwen2.5-72B-Instruct --local --local_port 8000

# Chinese market prompt context
python example/02_run_evaluation/run_t4_atomic.py --market_prompt cn
```

**Expected data layout:**
```
benchmark/data/evaluate/atomic/
    noise_csi300/    train.jsonl  val.jsonl  test.jsonl
    pairwise_csi300/ train.jsonl  val.jsonl  test.jsonl
```

**Outputs** (`runs/T4/<model>_<cot>/<task>/<split>/`):
```
results.jsonl       ← predicted label per record
metrics.json        ← accuracy, precision, recall, F1
report.txt          ← human-readable summary
infer.log
llm_output.json     ← raw LLM responses
```

---

### `generate_report.py` — Performance Report Generator
Loads existing result files and produces structured CSV / Markdown reports.
Can also compare multiple model runs side-by-side.

```bash
# Single run report
python example/02_run_evaluation/generate_report.py \
    --run_dir runs/T2/gpt-4.1_False

# Point to custom test-case paths
python example/02_run_evaluation/generate_report.py \
    --run_dir runs/T2/gpt-4.1_False \
    --ranking_cases benchmark/data/evaluate/built/ranking_csi300/all_env_scenarios.json \
    --scoring_cases benchmark/data/evaluate/built/scoring_csi300/alphabench_testset.json

# Multi-model comparison table
python example/02_run_evaluation/generate_report.py \
    --compare \
    --run_dirs "GPT-4.1=runs/T2/gpt-4.1_False" \
               "GPT-4.1 CoT=runs/T2/gpt-4.1_True" \
               "Deepseek=runs/T2/deepseek-chat_False" \
    --output_dir runs/comparison
```

**Outputs** (single run, `<run_dir>/report/`):
```
ranking_summary.csv
scoring_mae_summary.csv
scoring_classification.csv
scoring_per_case.csv
report.md
final_evaluation_results.pkl    ← compatible with collect_results.py
```

**Outputs** (comparison, `runs/comparison/`):
```
ranking_comparison.csv
scoring_comparison.csv
<ModelName>/report/...          ← per-model sub-reports
```

---

## Typical End-to-End Flow

```bash
# 1. Build datasets (once)
python example/01_build_datasets/build_all.py

# 2. Run T2 evaluation
python example/02_run_evaluation/run_t2_ranking.py --model gpt-4.1
python example/02_run_evaluation/run_t2_scoring.py --model gpt-4.1

# 3. Run T4 evaluation
python example/02_run_evaluation/run_t4_atomic.py --model gpt-4.1

# 4. Generate consolidated report
python example/02_run_evaluation/generate_report.py \
    --run_dir runs/T2/gpt-4.1_False
```
