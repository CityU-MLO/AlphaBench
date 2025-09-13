#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Path to your benchmark main
main_script="/home/hluo/workdir/AlphaBench/benchmark/engine/generate/benchmark_main.py"

# Choose python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

# Models to run (sequentially)
model_name_list=(
  # "gemini-2.5-pro"
  # "gemini-2.5-flash"
  # "codellama-70b-instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "gpt-4.1-mini"
  # "gpt-5"
  # "deepseek-ai/DeepSeek-V3"
  # "deepseek-ai/DeepSeek-R1"
  # "llama-3.1-70b-instruct"
  # "llama-3.3-70b-instruct-fp8-fast"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  # "Qwen/Qwen2.5-14B-Instruct"
  # "llama-3.1-8b-instruct"
  # "gemini-1.5-flash-8b"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# COT options (lowercase true/false)
cot_options=(false)

# Slugify model name for safe folder names (replace '/' and spaces)
slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  echo "$s"
}

run_one() {
  local model="$1"
  local cot="$2"

  local safe_model
  safe_model="$(slugify "$model")"

  local base_dir="./offcial_exp_results/T1_official/${safe_model}_cot_${cot}"
  mkdir -p "$base_dir"

  local ts
  ts="$(date +'%Y%m%d_%H%M%S')"
  local log_file="${base_dir}/run_${ts}.log"
  local status_file="${base_dir}/status_${ts}.txt"

  # update latest symlink
  ln -sf "$(basename "$log_file")" "${base_dir}/latest.log"

  echo "=== Launching ==="
  echo "Model: ${model}"
  echo "COT:   ${cot}"
  echo "Dir:   ${base_dir}"
  echo "Log:   ${log_file}"
  echo "Time:  $(date -Is)"
  echo "================="

  # Build args
  args=(
    "$main_script"
    --model "$model"
    --base_dir "$base_dir"
  )
  if [[ "$cot" == "true" ]]; then
    args+=(--enable_cot)
  fi

  # Run sequentially (no background). Redirect output to both console and log.
  # Remove "| tee" if you want only log file without console output.
  start_sec=$(date +%s)
  if command -v tee >/dev/null 2>&1; then
    "$PY" "${args[@]}" 2>&1 | tee -a "$log_file"
    exit_code="${PIPESTATUS[0]}"
  else
    # Fallback: only to log
    "$PY" "${args[@]}" >>"$log_file" 2>&1
    exit_code="$?"
  fi
  end_sec=$(date +%s)
  duration=$(( end_sec - start_sec ))

  echo "-----------------"
  echo "Finished: $(date -Is)"
  echo "Exit code: ${exit_code}"
  echo "Duration:  ${duration}s"
  echo "Log saved: ${log_file}"
  echo "================="

  {
    echo "model=${model}"
    echo "cot=${cot}"
    echo "start_ts=${ts}"
    echo "end_iso=$(date -Is)"
    echo "exit_code=${exit_code}"
    echo "duration_sec=${duration}"
    echo "log_file=${log_file}"
  } >"$status_file"

  return "$exit_code"
}

main() {
  local total=0 ok=0 fail=0
  for model in "${model_name_list[@]}"; do
    for cot in "${cot_options[@]}"; do
      total=$((total+1))
      if run_one "$model" "$cot"; then
        ok=$((ok+1))
      else
        fail=$((fail+1))
        echo "!! Job failed for model='${model}' cot='${cot}'. See latest.log for details." >&2
        # If you want to stop on first failure, uncomment the next line:
        # exit 1
      fi
      echo
    done
  done
  echo "All jobs done. total=${total} ok=${ok} fail=${fail}"
}

main "$@"
