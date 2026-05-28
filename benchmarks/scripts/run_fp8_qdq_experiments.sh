#!/usr/bin/env bash
set -euo pipefail

# Batch runner for FP8 e4m3 weight QDQ experiments.
#
# Default behavior:
#   - create QDQ checkpoints for the selected experiments
#   - do not run benchmark unless RUN_EVAL=1
#
# Example:
#   bash scripts/run_fp8_qdq_experiments.sh
#
# Run benchmark after each checkpoint is created:
#   RUN_EVAL=1 bash scripts/run_fp8_qdq_experiments.sh
#
# Run a subset:
#   EXPERIMENTS="gate_only up_only" bash scripts/run_fp8_qdq_experiments.sh
#
# Run benchmark on the cached ad 1000-sample subset:
#   RUN_EVAL=1 SAMPLE_SIZE=1000 EXPERIMENTS="gate_only up_only" bash scripts/run_fp8_qdq_experiments.sh

MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots}"
FP8_FORMAT="${FP8_FORMAT:-e4m3}"
SCALE_GRANULARITY="${SCALE_GRANULARITY:-per_row}"
DEVICE="${DEVICE:-cpu}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
if [[ -z "${RESULT_SUFFIX:-}" ]]; then
  if [[ -n "${SAMPLE_SIZE}" && "${SAMPLE_SIZE}" != "full" ]]; then
    RESULT_SUFFIX="ad_sample_${SAMPLE_SIZE}"
  else
    RESULT_SUFFIX="ad_full"
  fi
fi
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_script.sh}"

DEFAULT_EXPERIMENTS=(
  "gate_only"
  "up_only"
  "mlp_down"
  "mlp_all"
  "attn_o"
  "attn_qvo"
  "attn_all"
  "block_linears"
)

if [[ -n "${EXPERIMENTS:-}" ]]; then
  # shellcheck disable=SC2206
  SELECTED_EXPERIMENTS=(${EXPERIMENTS})
else
  SELECTED_EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

experiment_regex() {
  local name="$1"
  case "$name" in
    gate_only)
      printf '%s\n' 'model\.layers\.\d+\.mlp\.gate_proj\.weight$'
      ;;
    up_only)
      printf '%s\n' 'model\.layers\.\d+\.mlp\.up_proj\.weight$'
      ;;
    mlp_down)
      printf '%s\n' 'model\.layers\.\d+\.mlp\.down_proj\.weight$'
      ;;
    mlp_all)
      printf '%s\n' 'model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$'
      ;;
    attn_o)
      printf '%s\n' 'model\.layers\.\d+\.self_attn\.o_proj\.weight$'
      ;;
    attn_qvo)
      printf '%s\n' 'model\.layers\.\d+\.self_attn\.(q_proj|v_proj|o_proj)\.weight$'
      ;;
    attn_all)
      printf '%s\n' 'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$'
      ;;
    block_linears)
      printf '%s\n' 'model\.layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))\.weight$'
      ;;
    *)
      echo "Unknown experiment: ${name}" >&2
      echo "Valid experiments: ${DEFAULT_EXPERIMENTS[*]}" >&2
      return 1
      ;;
  esac
}

experiment_slug() {
  local name="$1"
  case "$name" in
    gate_only)
      printf '%s\n' 'gate-only'
      ;;
    up_only)
      printf '%s\n' 'up-only'
      ;;
    mlp_down)
      printf '%s\n' 'mlp-down'
      ;;
    mlp_all)
      printf '%s\n' 'mlp-all'
      ;;
    attn_o)
      printf '%s\n' 'attn-o'
      ;;
    attn_qvo)
      printf '%s\n' 'attn-qvo'
      ;;
    attn_all)
      printf '%s\n' 'attn-all'
      ;;
    block_linears)
      printf '%s\n' 'block-linears'
      ;;
    *)
      echo "Unknown experiment: ${name}" >&2
      return 1
      ;;
  esac
}

run_quantize() {
  local name="$1"
  local regex="$2"
  local slug
  slug="$(experiment_slug "${name}")"
  local artifact_name="OneRec-1.7B-fp8${FP8_FORMAT}-${slug}"
  local output_path="${OUTPUT_ROOT}/${artifact_name}"

  echo "========== QDQ experiment: ${name} =========="
  echo "model_path: ${MODEL_PATH}"
  echo "output_path: ${output_path}"
  echo "target_regex: ${regex}"

  local cmd=(
    python scripts/quantize_qdq_fp8.py
    --model_path "${MODEL_PATH}"
    --output_path "${output_path}"
    --target_regex "${regex}"
    --fp8_format "${FP8_FORMAT}"
    --scale_granularity "${SCALE_GRANULARITY}"
    --device "${DEVICE}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry_run)
    "${cmd[@]}"
  elif [[ -e "${output_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "Checkpoint already exists; skipping quantization. Set OVERWRITE=1 to rebuild."
  else
    if [[ "${OVERWRITE}" == "1" ]]; then
      cmd+=(--overwrite)
    fi
    "${cmd[@]}"
  fi

  if [[ "${RUN_EVAL}" == "1" && "${DRY_RUN}" != "1" ]]; then
    echo "========== Benchmark: ${name} =========="
    SAMPLE_SIZE="${SAMPLE_SIZE}" bash "${EVAL_SCRIPT}" "${output_path}" "${artifact_name}_${RESULT_SUFFIX}" "${ENABLE_THINKING}"
  fi
}

echo "Selected experiments: ${SELECTED_EXPERIMENTS[*]}"
echo "RUN_EVAL=${RUN_EVAL}, DRY_RUN=${DRY_RUN}, OVERWRITE=${OVERWRITE}, SAMPLE_SIZE=${SAMPLE_SIZE:-<none>}, RESULT_SUFFIX=${RESULT_SUFFIX}"

for experiment in "${SELECTED_EXPERIMENTS[@]}"; do
  regex="$(experiment_regex "${experiment}")"
  run_quantize "${experiment}" "${regex}"
done

echo "All selected FP8 QDQ experiments completed."
