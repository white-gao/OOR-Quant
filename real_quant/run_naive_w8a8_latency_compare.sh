#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B/}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data-calib1024}"
TASK="${TASK:-ad}"
SPLIT="${SPLIT:-test}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
OVERWRITE="${OVERWRITE:-1}"
EVALUATE="${EVALUATE:-0}"

MODEL_NAME="${MODEL_PATH%/}"
MODEL_NAME="${MODEL_NAME##*/}"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-real_quant/full_precision/results/${TASK}_${MODEL_NAME}_bf16_latency}"
NAIVE_OUTPUT_DIR="${NAIVE_OUTPUT_DIR:-real_quant/naive_w8a8/results/${TASK}_${MODEL_NAME}_latency}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-real_quant/naive_w8a8/results/${TASK}_${MODEL_NAME}_latency_compare.md}"

export PYTHONPATH="${PYTHONPATH:-.}"

COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --task "${TASK}"
  --split "${SPLIT}"
  --sample_size "${SAMPLE_SIZE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --batch_size "${BATCH_SIZE}"
  --num_beams "${NUM_BEAMS}"
  --num_return_sequences "${NUM_RETURN_SEQUENCES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS+=(--overwrite)
fi
if [[ "${EVALUATE}" == "1" ]]; then
  WRITE_ARGS+=(--evaluate)
fi

echo "[latency_compare] preflight real naive W8A8 runtime"
"${PYTHON_BIN}" -m real_quant.naive_w8a8.preflight --device "${DEVICE}"

echo "[latency_compare] running full precision baseline"
"${PYTHON_BIN}" -m real_quant.full_precision.run_hf_baseline \
  "${COMMON_ARGS[@]}" \
  --output_dir "${FULL_OUTPUT_DIR}" \
  "${WRITE_ARGS[@]}"

echo "[latency_compare] running real naive W8A8"
"${PYTHON_BIN}" -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  "${COMMON_ARGS[@]}" \
  --output_dir "${NAIVE_OUTPUT_DIR}" \
  "${WRITE_ARGS[@]}"

BASELINE_JSON="${FULL_OUTPUT_DIR}/${MODEL_NAME}/${TASK}/${SPLIT}_generated.json"
CANDIDATE_JSON="${NAIVE_OUTPUT_DIR}/${MODEL_NAME}-real-naive-w8a8/${TASK}/${SPLIT}_generated.json"

mkdir -p "$(dirname "${COMPARE_OUTPUT}")"
echo "[latency_compare] comparing latency"
"${PYTHON_BIN}" -m real_quant.compare_latency \
  --baseline "${BASELINE_JSON}" \
  --candidate "${CANDIDATE_JSON}" \
  > "${COMPARE_OUTPUT}"

cat "${COMPARE_OUTPUT}"
echo "[latency_compare] comparison saved to ${COMPARE_OUTPUT}"
