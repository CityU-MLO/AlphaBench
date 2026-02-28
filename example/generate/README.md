# AlphaBench — T1 Generation Examples

This folder provides ready-to-run scripts for the **T1 Factor Generation** task:
given natural-language instructions (or theme tags), an LLM writes alpha factor
expressions in Qlib syntax.

---

## Prerequisites

Run all scripts from the **repo root**:
```bash
cd /path/to/AlphaBench
```

Required input files (already present in this repo):
```
benchmark/data/generate/
  T1_Text2Alpha.json                ← Text2Alpha instructions per difficulty level
  T1_DirectionalMining.json         ← Directional-Mining instructions per level
  T1_Text2Alpha_stability.json      ← Stability synonym groups
  T1_DirectionalMining_creativity.json ← Creativity prompts for directional mining
  tags_system.json                  ← Tag descriptions used in directional mining
```

---

## Step 1 — Generate Factors

### Basic usage (deepseek-chat, no CoT)
```bash
python example/generate/run_t1_generate.py
```

### GPT-4.1 with chain-of-thought
```bash
python example/generate/run_t1_generate.py --model gpt-4.1 --cot
```

### Local vLLM server
```bash
python example/generate/run_t1_generate.py \
    --model Qwen2.5-72B-Instruct --local --local_port 8000
```

### Only build instruction files (skip LLM calls)
```bash
python example/generate/run_t1_generate.py --build_only
```

### Only run generation (instructions already exist)
```bash
python example/generate/run_t1_generate.py --skip_build
```

### Output layout after Step 1
```
runs/T1/<model>_<cot>/
  instructions/
    T1_1_easy_instruction.pkl       ← Text2Alpha instructions (easy level)
    T1_1_medium_instruction.pkl
    T1_1_hard_instruction.pkl
    T1_2_easy_instruction.pkl       ← Directional-Mining instructions
    T1_2_medium_instruction.pkl
    T1_2_hard_instruction.pkl
    T1_stability_0.pkl              ← Stability synonym groups
    T1_stability_1.pkl
    ...
    T1_DirectionalMining_creativity.json
    T1_Text2Alpha_stability.json
  outputs/
    T1_1_easy_results.pkl           ← Generated factor results (per level)
    T1_1_medium_results.pkl
    T1_1_hard_results.pkl
    T1_2_easy_results.pkl
    T1_2_medium_results.pkl
    T1_2_hard_results.pkl
    T1_stability_0_results.pkl
    ...
```

---

## Step 2 — Evaluate Generated Factors

### Fitness + Diversity (default judge: deepseek-chat)
```bash
python example/generate/eval_t1_fitness.py \
    --run_dir runs/T1/deepseek-chat_False
```

### Use a different judge model
```bash
python example/generate/eval_t1_fitness.py \
    --run_dir runs/T1/gpt-4.1_True \
    --judge_model gpt-4.1
```

### Only fitness evaluation (skip diversity)
```bash
python example/generate/eval_t1_fitness.py \
    --run_dir runs/T1/deepseek-chat_False \
    --skip_diversity
```

### Only diversity evaluation (skip fitness)
```bash
python example/generate/eval_t1_fitness.py \
    --run_dir runs/T1/deepseek-chat_False \
    --skip_fitness
```

### Output layout after Step 2
```
runs/T1/<model>_<cot>/outputs/scores/
  eval_fitness_results_<judge_model>.pkl    ← per-prefix correctness labels
  eval_fitness_responses_<judge_model>.pkl  ← raw LLM judge responses
  creativity_results.json                   ← AST distance + IC correlation stats
```

---

## Key Metrics

| Sub-task | Metric | Description |
|----------|--------|-------------|
| Text2Alpha | Fitness (Accuracy) | % of generated expressions judged correct by LLM judge |
| Directional Mining | Fitness (Accuracy) | % of expressions consistent with the target theme |
| Directional Mining | AST mean_dist | Average tree-edit distance across generated factors (structural diversity) |
| Directional Mining | IC mean_abs_corr | Average pairwise IC correlation (lower → more diverse) |
| Stability | Consistency | Agreement rate of expressions generated from synonym prompts |

---

## CLI Reference

### `run_t1_generate.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `deepseek-chat` | LLM model name |
| `--cot` | `False` | Enable chain-of-thought |
| `--local` | `False` | Use a local vLLM server |
| `--local_port` | `8000` | Port for local server |
| `--num_workers` | `4` | Parallel LLM workers |
| `--data_dir` | `benchmark/data/generate` | Input data directory |
| `--output_root` | `runs/T1` | Root output directory |
| `--skip_build` | `False` | Skip instruction building |
| `--build_only` | `False` | Only build instructions, skip LLM |

### `eval_t1_fitness.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_dir` | *(required)* | Run directory from Step 1 |
| `--judge_model` | `deepseek-chat` | Judge LLM for fitness evaluation |
| `--skip_fitness` | `False` | Skip correctness evaluation |
| `--skip_diversity` | `False` | Skip diversity/creativity evaluation |
| `--creativity_n` | `30` | Number of creativity groups to evaluate |
| `--num_workers` | `8` | Parallel workers for judge calls |
