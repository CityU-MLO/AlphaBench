# AlphaBench — Evaluation Examples

This folder provides ready-to-run scripts for building evaluation datasets and
running the full benchmark pipeline.

---

## Prerequisites

Run all scripts from the **repo root**:
```bash
cd /path/to/AlphaBench
```

Required input files (already present in this repo):
```
benchmark/data/evaluate/
├── raw/
│   ├── ic_table_csi300.csv        ← daily IC table, CSI300
│   ├── rankic_table_csi300.csv    ← daily RankIC table, CSI300
│   ├── ic_table_sp500.csv         ← daily IC table, SP500
│   └── rankic_table_sp500.csv     ← daily RankIC table, SP500
└── pool/
    └── factor_pool.json           ← {factor_name: expression}
```

---

## Step 1 — Build Datasets

### Option A: Build everything at once
```bash
python example/01_build_datasets/build_all.py
```
Builds all four datasets for CSI300. Add `--also_sp500` to also build SP500
datasets (using CSI300 as the atomic split reference).

### Option B: Build individually

**T2 Base Eval (Ranking + Scoring):**
```bash
python example/01_build_datasets/build_base_eval_datasets.py \
    --tasks all \
    --output_dir benchmark/data/evaluate/built
```

**T4 Binary Noise Classification:**
```bash
python example/01_build_datasets/build_atomic_noise_dataset.py \
    --output_dir benchmark/data/evaluate/atomic/noise_csi300
```

**T4 Pairwise Selection:**
```bash
python example/01_build_datasets/build_atomic_pairwise_dataset.py \
    --output_dir benchmark/data/evaluate/atomic/pairwise_csi300
```

### Output layout after Step 1:
```
benchmark/data/evaluate/
├── built/
│   ├── ranking_csi300/
│   │   ├── all_env_scenarios.json       ← combined ranking test cases
│   │   └── ranking_<regime>.json        ← per-regime
│   └── scoring_csi300/
│       ├── alphabench_testset.json      ← combined scoring test cases
│       └── scoring_<regime>.json        ← per-regime
└── atomic/
    ├── noise_csi300/
    │   ├── train.jsonl  val.jsonl  test.jsonl
    │   └── manifest.json
    └── pairwise_csi300/
        ├── train.jsonl  val.jsonl  test.jsonl
        └── manifest.json
```

---

## Step 2 — Run Evaluation

### T2 Ranking
```bash
python example/02_run_evaluation/run_t2_ranking.py \
    --model gpt-4.1 \
    --data_dir benchmark/data/evaluate/built/ranking_csi300
```

### T2 Scoring
```bash
python example/02_run_evaluation/run_t2_scoring.py \
    --model gpt-4.1 \
    --data_dir benchmark/data/evaluate/built/scoring_csi300
```

### T4 Atomic (both tasks)
```bash
python example/02_run_evaluation/run_t4_atomic.py \
    --model gpt-4.1 \
    --tasks all \
    --splits test
```

### Generate Report
```bash
# Single run
python example/02_run_evaluation/generate_report.py \
    --run_dir runs/T2/gpt-4.1_False

# Compare multiple models
python example/02_run_evaluation/generate_report.py \
    --compare \
    --run_dirs "GPT-4.1=runs/T2/gpt-4.1_False" \
               "GPT-4.1 CoT=runs/T2/gpt-4.1_True" \
               "Deepseek=runs/T2/deepseek-chat_False" \
    --output_dir runs/comparison
```

---

## Config Files

YAML configs in `01_build_datasets/configs/` let you customize:
- Market regimes (date ranges)
- Noise thresholds (IC / RankIC gates)
- Train/val/test split ratios
- Ranking difficulty settings (N-pick-K)
- Scoring class counts (positive / negative / noise)

```bash
# Build with custom config
python example/01_build_datasets/build_base_eval_datasets.py \
    --config example/01_build_datasets/configs/base_eval_csi300.yaml
```

---

## Key Metrics

| Task | Metric |
|------|--------|
| T2 Ranking | Precision@K, NDCG@K |
| T2 Scoring | MAE (Performance/Stability/WinRate/Skewness), Acc_Signal, per-class F1 |
| T4 Noise | Accuracy, Precision, Recall, F1 |
| T4 Pairwise | Accuracy |
