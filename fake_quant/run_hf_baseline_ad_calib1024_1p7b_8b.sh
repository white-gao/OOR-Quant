#!/usr/bin/env bash
set -euo pipefail

# Run no-quant BF16 HF baselines on the AD calib1024 split.
# The two models run in parallel by default: 1.7B on cuda:0, 8B on cuda:1.

MODEL_1P7B_PATH="${MODEL_1P7B_PATH:-/home/guowei/OneRec-1.7B/}"
MODEL_8B_PATH="${MODEL_8B_PATH:-/home/guowei/OneRec-8B/}"
DEVICE_1P7B="${DEVICE_1P7B:-cuda:0}"
DEVICE_8B="${DEVICE_8B:-cuda:1}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data-calib1024}"
SAMPLE_SIZE="${SAMPLE_SIZE:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-fake_quant/results/v1.0}"
DEVICE_MAP="${DEVICE_MAP:-}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
SEED="${SEED:-42}"
EVALUATE="${EVALUATE:-1}"
OVERWRITE="${OVERWRITE:-1}"
RUN_1P7B="${RUN_1P7B:-1}"
RUN_8B="${RUN_8B:-1}"

run_baseline() {
  local model_path="$1"
  local model_name="$2"
  local output_dir="$3"
  local device="$4"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${model_name} on ${device}"
  echo "  model_path=${model_path}"
  echo "  data_dir=${DATA_DIR} sample_size=${SAMPLE_SIZE}"
  echo "  output_dir=${output_dir}"

  local args=(
    --model_path "${model_path}"
    --data_dir "${DATA_DIR}"
    --sample_size "${SAMPLE_SIZE}"
    --quant_scheme none
    --dtype "${DTYPE}"
    --device "${device}"
    --num_beams "${NUM_BEAMS}"
    --num_return_sequences "${NUM_RETURN_SEQUENCES}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --output_dir "${output_dir}"
    --model_name "${model_name}"
  )

  if [[ -n "${DEVICE_MAP}" ]]; then
    args+=(--device_map "${DEVICE_MAP}")
  fi
  if [[ "${EVALUATE}" == "1" ]]; then
    args+=(--evaluate)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi

  python fake_quant/run_ad_sid.py "${args[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${model_name} on ${device}"
}

pids=()
labels=()

if [[ "${RUN_1P7B}" == "1" ]]; then
  run_baseline     "${MODEL_1P7B_PATH}"     "OneRec-1.7B-hf-baseline"     "${OUTPUT_ROOT}/results_OneRec-1.7B-hf-baseline-ad-calib1024"     "${DEVICE_1P7B}" &
  pids+=("$!")
  labels+=("OneRec-1.7B")
fi

if [[ "${RUN_8B}" == "1" ]]; then
  run_baseline     "${MODEL_8B_PATH}"     "OneRec-8B-hf-baseline"     "${OUTPUT_ROOT}/results_OneRec-8B-hf-baseline-ad-calib1024"     "${DEVICE_8B}" &
  pids+=("$!")
  labels+=("OneRec-8B")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No baseline jobs requested. Set RUN_1P7B=1 and/or RUN_8B=1."
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All requested baselines finished."
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Some baseline jobs failed." >&2
fi
exit "${status}"
