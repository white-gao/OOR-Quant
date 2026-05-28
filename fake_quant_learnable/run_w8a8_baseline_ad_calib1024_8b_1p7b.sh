#!/usr/bin/env bash
set -euo pipefail

# Run W8A8 fake-quant baselines with the fake_quant_learnable runner.
# The two models run in parallel by default: 8B on cuda:0, 1.7B on cuda:1.

MODEL_8B_PATH="${MODEL_8B_PATH:-/home/guowei/OneRec-8B/}"
MODEL_1P7B_PATH="${MODEL_1P7B_PATH:-/home/guowei/OneRec-1.7B/}"
DEVICE_8B="${DEVICE_8B:-cuda:0}"
DEVICE_1P7B="${DEVICE_1P7B:-cuda:1}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data-calib1024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-fake_quant_learnable/results}"
LAYERS="${LAYERS:-all}"
ACT_QUANT="${ACT_QUANT:-per_token}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-shared_input}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
EVAL_OFFSET="${EVAL_OFFSET:-0}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
SEED="${SEED:-42}"
EVALUATE="${EVALUATE:-1}"
OVERWRITE="${OVERWRITE:-1}"
RUN_8B="${RUN_8B:-1}"
RUN_1P7B="${RUN_1P7B:-1}"

run_w8a8() {
  local model_path="$1"
  local model_name="$2"
  local run_name="$3"
  local device="$4"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${model_name} W8A8 on ${device}"
  echo "  model_path=${model_path}"
  echo "  data_dir=${DATA_DIR} eval_sample_size=${EVAL_SAMPLE_SIZE}"
  echo "  run_name=${run_name}"

  env \
    MODE=baseline_w8a8 \
    MODEL_PATH="${model_path}" \
    MODEL_NAME="${model_name}" \
    DATA_DIR="${DATA_DIR}" \
    RUN_NAME="${run_name}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}" \
    LAYERS="${LAYERS}" \
    ACT_QUANT="${ACT_QUANT}" \
    ACT_QUANT_MODE="${ACT_QUANT_MODE}" \
    DTYPE="${DTYPE}" \
    EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" \
    EVAL_OFFSET="${EVAL_OFFSET}" \
    NUM_BEAMS="${NUM_BEAMS}" \
    NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    DEVICE="${device}" \
    SEED="${SEED}" \
    EVALUATE="${EVALUATE}" \
    OVERWRITE="${OVERWRITE}" \
    SAVE_QUANT_PARAMS=0 \
    bash fake_quant_learnable/run_learnable_quant_ad.sh

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${model_name} W8A8 on ${device}"
}

pids=()
labels=()

if [[ "${RUN_8B}" == "1" ]]; then
  run_w8a8 \
    "${MODEL_8B_PATH}" \
    "OneRec-8B-w8a8-baseline" \
    "baseline_w8a8_8b_ad_calib1024" \
    "${DEVICE_8B}" &
  pids+=("$!")
  labels+=("OneRec-8B")
fi

if [[ "${RUN_1P7B}" == "1" ]]; then
  run_w8a8 \
    "${MODEL_1P7B_PATH}" \
    "OneRec-1.7B-w8a8-baseline" \
    "baseline_w8a8_1p7b_ad_calib1024" \
    "${DEVICE_1P7B}" &
  pids+=("$!")
  labels+=("OneRec-1.7B")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No W8A8 jobs requested. Set RUN_8B=1 and/or RUN_1P7B=1."
  exit 0
fi

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  label="${labels[$idx]}"
  if wait "${pid}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} completed successfully."
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} failed." >&2
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All requested W8A8 baselines finished."
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Some W8A8 baseline jobs failed." >&2
fi
exit "${status}"
