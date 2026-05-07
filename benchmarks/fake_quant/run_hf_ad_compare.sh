#!/usr/bin/env bash
set -euo pipefail

# Compare HF baseline vs FP8 fake-quant on AD SID prediction.
#
# Common overrides:
#   MODEL_PATH=/home/guowei/OneRec-1.7B
#   DATA_DIR=../data/onerec_data/benchmark_data
#   SAMPLE_SIZE=1000
#   CUDA_VISIBLE_DEVICES=0
#   RUN_BASELINE=true RUN_QUANT=true bash fake_quant/run_hf_ad_compare.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data/ad/}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda:4}"
SEED="${SEED:-42}"
RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_QUANT="${RUN_QUANT:-true}"
EVALUATE="${EVALUATE:-true}"
OVERWRITE="${OVERWRITE:-true}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-fake_quant/results/${VERSION}}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-baseline-ad-sample-${SAMPLE_SIZE}}"
QUANT_OUTPUT_DIR="${QUANT_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-fake-fp8-weight-channel-ad-sample-${SAMPLE_SIZE}}"
BASELINE_MODEL_NAME="${BASELINE_MODEL_NAME:-OneRec-1.7B-hf-baseline}"
QUANT_MODEL_NAME="${QUANT_MODEL_NAME:-OneRec-1.7B-hf-fake-fp8-weight-channel}"
ACT_QUANT="${ACT_QUANT:-none}"
TARGET_REGEX="${TARGET_REGEX:-}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --data_dir "$DATA_DIR"
  --sample_size "$SAMPLE_SIZE"
  --num_beams "$NUM_BEAMS"
  --num_return_sequences "$NUM_RETURN_SEQUENCES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ "$EVALUATE" == "true" ]]; then
  COMMON_ARGS+=(--evaluate)
fi

if [[ "$OVERWRITE" == "true" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "NUM_BEAMS=${NUM_BEAMS}"
echo "NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES}"
echo "SEED=${SEED}"

if [[ "$RUN_BASELINE" == "true" ]]; then
  echo "Running HF baseline..."
  python fake_quant/run_ad_sid.py \
    "${COMMON_ARGS[@]}" \
    --quant_scheme none \
    --output_dir "$BASELINE_OUTPUT_DIR" \
    --model_name "$BASELINE_MODEL_NAME"
fi

if [[ "$RUN_QUANT" == "true" ]]; then
  echo "Running HF FP8 fake quant..."
  QUANT_ARGS=(
    "${COMMON_ARGS[@]}"
    --quant_scheme fp8_weight_channel
    --act_quant "$ACT_QUANT"
    --output_dir "$QUANT_OUTPUT_DIR"
    --model_name "$QUANT_MODEL_NAME"
  )

  if [[ -n "$TARGET_REGEX" ]]; then
    QUANT_ARGS+=(--target_regex "$TARGET_REGEX")
  fi

  python fake_quant/run_ad_sid.py "${QUANT_ARGS[@]}"
fi
