# 01 — Build Datasets

Scripts for constructing all evaluation datasets from raw IC / RankIC tables.

## Input files

| File | Path |
|------|------|
| IC table (CSI300) | `benchmark/data/evaluate/raw/ic_table_csi300.csv` |
| RankIC table (CSI300) | `benchmark/data/evaluate/raw/rankic_table_csi300.csv` |
| IC table (SP500) | `benchmark/data/evaluate/raw/ic_table_sp500.csv` |
| RankIC table (SP500) | `benchmark/data/evaluate/raw/rankic_table_sp500.csv` |
| Factor pool | `benchmark/data/evaluate/pool/factor_pool.json` |

The IC / RankIC CSVs have shape `(n_days, n_factors)` with a `datetime` index column.
The factor pool is `{factor_name: expression}`.

---

## Scripts

### `build_all.py` — Build everything at once
```bash
# CSI300 only
python example/01_build_datasets/build_all.py

# CSI300 + SP500 (SP500 reuses CSI300 split indices)
python example/01_build_datasets/build_all.py --also_sp500
```

Produces:
- `benchmark/data/evaluate/built/ranking_csi300/`
- `benchmark/data/evaluate/built/scoring_csi300/`
- `benchmark/data/evaluate/atomic/noise_csi300/`
- `benchmark/data/evaluate/atomic/pairwise_csi300/`

---

### `build_base_eval_datasets.py` — T2 Ranking + Scoring
```bash
# Default (paper settings, CSI300)
python example/01_build_datasets/build_base_eval_datasets.py

# Custom config
python example/01_build_datasets/build_base_eval_datasets.py \
    --config example/01_build_datasets/configs/base_eval_csi300.yaml

# SP500 with inline overrides
python example/01_build_datasets/build_base_eval_datasets.py \
    --ic_table     benchmark/data/evaluate/raw/ic_table_sp500.csv \
    --rankic_table benchmark/data/evaluate/raw/rankic_table_sp500.csv \
    --market SP500 \
    --output_dir benchmark/data/evaluate/built_sp500

# Only ranking or only scoring
python example/01_build_datasets/build_base_eval_datasets.py --tasks ranking
python example/01_build_datasets/build_base_eval_datasets.py --tasks scoring
```

Output (`--output_dir` default: `benchmark/data/evaluate/built`):
```
built/
└── ranking_csi300/
    ├── all_env_scenarios.json     ← combined (all regimes), used by run_t2_ranking.py
    ├── ranking_2021-2025_Overall.json
    ├── ranking_CSI300_Bear_.json
    └── ranking_CSI300_Bull_.json
└── scoring_csi300/
    ├── alphabench_testset.json    ← combined (all regimes), used by run_t2_scoring.py
    ├── scoring_2021-2025_Overall.json
    └── ...
```

---

### `build_atomic_noise_dataset.py` — T4 Binary Noise Classification
```bash
# Default (CSI300, fresh split)
python example/01_build_datasets/build_atomic_noise_dataset.py

# Custom output dir
python example/01_build_datasets/build_atomic_noise_dataset.py \
    --output_dir benchmark/data/evaluate/atomic/noise_csi300

# Cross-market: SP500 reusing CSI300 factor split
python example/01_build_datasets/build_atomic_noise_dataset.py \
    --ic_table     benchmark/data/evaluate/raw/ic_table_sp500.csv \
    --rankic_table benchmark/data/evaluate/raw/rankic_table_sp500.csv \
    --market SP500 \
    --refer  benchmark/data/evaluate/atomic/noise_csi300 \
    --output_dir benchmark/data/evaluate/atomic/noise_sp500
```

Output JSONL record:
```json
{
  "expression":   "Div($close, Add(Mean($close, 15), 1e-12))",
  "ground_truth": "signal",
  "factor_name":  "close_to_rolling_mean_...",
  "market":       "CSI300",
  "window":       {"start": "2021-01-01", "end": "2025-06-30"},
  "meta":         {"mean_ic": 0.031, "mean_rankic": 0.028, "icir": 0.41, ...}
}
```

---

### `build_atomic_pairwise_dataset.py` — T4 Pairwise Selection
```bash
# Default (CSI300, fresh split)
python example/01_build_datasets/build_atomic_pairwise_dataset.py

# Cross-market: SP500 reusing CSI300 pairs
python example/01_build_datasets/build_atomic_pairwise_dataset.py \
    --ic_table     benchmark/data/evaluate/raw/ic_table_sp500.csv \
    --rankic_table benchmark/data/evaluate/raw/rankic_table_sp500.csv \
    --market SP500 \
    --refer  benchmark/data/evaluate/atomic/pairwise_csi300 \
    --output_dir benchmark/data/evaluate/atomic/pairwise_sp500
```

Output JSONL record:
```json
{
  "id":            "CSI300_TEST_0000",
  "A":             "Div($close, Add(Mean($close, 15), 1e-12))",
  "B":             "Add($close, $volume)",
  "ground_truth":  "A",
  "factor_name_A": "close_to_rolling_mean_...",
  "factor_name_B": "sum_open_close_diff_...",
  "market":        "CSI300",
  "window":        {"start": "2021-01-01", "end": "2025-06-30"},
  "meta_A":        {"mean_ic": 0.031, ...},
  "meta_B":        {"mean_ic": 0.002, ...}
}
```

---

## Configs

YAML files in `configs/` let you customize all dataset parameters without editing code:

| Config | Purpose |
|--------|---------|
| `base_eval_csi300.yaml` | T2 ranking + scoring (regimes, thresholds, split sizes) |
| `atomic_noise_csi300.yaml` | T4 noise classification (noise gate, split ratios) |
| `atomic_pairwise_csi300.yaml` | T4 pairwise selection (noise gate, split ratios) |

Pass any config with `--config <path>`.
