# Dataset Build Configs

YAML configuration files for the dataset builders. Pass any of these with `--config <path>`.

## Files

| Config | Used by | Description |
|--------|---------|-------------|
| `base_eval_csi300.yaml` | `build_base_eval_datasets.py` | T2 ranking + scoring, CSI300, paper settings |
| `atomic_noise_csi300.yaml` | `build_atomic_noise_dataset.py` | T4 binary noise classification, CSI300 |
| `atomic_pairwise_csi300.yaml` | `build_atomic_pairwise_dataset.py` | T4 pairwise selection, CSI300 |

## Key parameters

### `base_eval_csi300.yaml`

```yaml
regimes:            # Market regimes (time windows for IC computation)
  - name / start / end

ranking:
  good_ic_threshold:      0.025   # |IC| > this → "good" factor
  good_rankic_threshold:  0.03    # |RankIC| > this → "good" factor
  settings:               [10/3, 20/5, 40/10]   # N-pick-K difficulty levels
  instances_per_setting:  50      # test cases per (regime, setting)

scoring:
  noise_ic_threshold: 0.01   # |IC| < this → "Noise" class
  n_positive: 100            # Positive factors per regime
  n_negative: 100            # Negative factors per regime
  n_noise:     50            # Noise factors per regime
```

### `atomic_noise_csi300.yaml` / `atomic_pairwise_csi300.yaml`

```yaml
noise_threshold:
  ic:        0.005    # |IC| threshold
  rankic:    0.025    # |RankIC| threshold
  abs:       true     # compare absolute values
  condition: "and"    # "and": must fail BOTH gates to be noise
                      # "or":  fail EITHER gate to be noise

split:
  sample_n:    300    # total factors (or pairs) to sample
  train_ratio: 0.6
  val_ratio:   0.1
  test_ratio:  0.3

# OR (cross-market transfer):
refer: "/path/to/existing/split/dir"
```
