#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Path to your benchmark main
main_script="/home/hluo/workdir/AlphaBench/benchmark/engine/searching/benchmark_searching.py"

# Choose python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

# Models to run (顺序执行)
model_name_list=(
  # "gpt-4.1-mini"
  # "deepseek-chat"
  # "gemini-1.5-flash-8b"
  # "gemini-2.5-flash"
  # "llama-3.1-70b-instruct"
  # "gemini-2.5-pro"
  "gpt-5"
)

# Slugify model name for safe folder names
slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  echo "$s"
}

for model in "${model_name_list[@]}"; do
  safe_model="$(slugify "$model")"
  # base_dir="./runs/T3_official_latest_temp75/${safe_model}"
  base_dir="/home/hluo/workdir/AlphaBench/runs/T3_official_ea_30/${safe_model}"
  mkdir -p "$base_dir"

  echo
  echo "=== Running model='${model}' ==="
  ts="$(date +'%Y%m%d_%H%M%S')"
  log_file="${base_dir}/run_${ts}.log"
  ln -sf "$(basename "$log_file")" "${base_dir}/latest.log"

  # Build args
  args=(
    "$main_script"
    --config_path ./config/search.yaml
    --model_name "$model"
    --save_dir "$base_dir"
  )

  # 顺序执行并实时输出 + 写日志
  "$PY" "${args[@]}" 2>&1 | tee "$log_file"

  echo "=== Finished model='${model}' (logs: ${log_file}) ==="
done

echo
echo "All models finished."
