#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Path to your benchmark main
main_script="/home/hluo/workdir/AlphaBench/benchmark/engine/evaluate/benchmark_main.py"

# Choose python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

# Models to run
model_name_list=(
  # "gemini-2.5-pro"
  # "gemini-2.5-flash"
  # "gpt-4.1-mini"
  # "gpt-5"
  # "deepseek-ai/DeepSeek-V3"
  # "deepseek-ai/DeepSeek-R1"
  # "llama-3.1-70b-instruct"
  # "llama-3.3-70b-instruct-fp8-fast"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "Qwen/Qwen2.5-14B-Instruct"
  # "llama-3.1-8b-instruct"
  # "gemini-1.5-flash-8b"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# COT options (lowercase true/false; adjust if your script expects True/False)
cot_options=(true false)

# Slugify model name for safe folder names (replace '/' and spaces)
slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  echo "$s"
}

pids=()

for model in "${model_name_list[@]}"; do
  safe_model="$(slugify "$model")"
  for cot in "${cot_options[@]}"; do
    base_dir="./runs/T2_evaluate_official/${safe_model}_cot_${cot}"
    mkdir -p "$base_dir"

    echo "Launching model='${model}' cot='${cot}' -> ${base_dir}"
    ts="$(date +'%Y%m%d_%H%M%S')"
    log_file="${base_dir}/run_${ts}.log"
    ln -sf "$(basename "$log_file")" "${base_dir}/latest.log"

    # Build the args list dynamically
    args=(
      "$main_script"
      --model "$model"
      --base_dir "$base_dir"
    )
    if [[ "$cot" == "true" ]]; then
      args+=(--enable_cot)
    fi

    nohup "$PY" "${args[@]}" >"$log_file" 2>&1 &

    echo "  PID $! (logs: ${log_file})"
  done
done

echo
echo "Launched ${#pids[@]} jobs:"
printf '  %s\n' "${pids[@]}"
echo "Use:  tail -f runs/T2_evaluate_official/*/latest.log  to watch logs."
