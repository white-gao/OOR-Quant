#!/usr/bin/env bash
set -euo pipefail

# Run SmoothQuant-style FP8 fake quant on AD SID prediction.
#
# Expected cwd:
#   cd /zssd/home/yhhuang/Projects/OOR-Quant
#   bash fake_quant/smoothquant/run_smoothquant_ad.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
CALIB_SAMPLE_OFFSET="${CALIB_SAMPLE_OFFSET:-1000}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda:6}"
SEED="${SEED:-42}"
EVALUATE="${EVALUATE:-true}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_CALIB="${SKIP_CALIB:-false}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.5}"
SMOOTH_LAYER_MIN="${SMOOTH_LAYER_MIN:-}"
SMOOTH_LAYER_CUTOFF="${SMOOTH_LAYER_CUTOFF:-}"
TARGET_REGEX="${TARGET_REGEX:-}"
SKIP_REGEX="${SKIP_REGEX:-}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-fake_quant/results/${VERSION}}"
SCALE_PATH="${SCALE_PATH:-fake_quant/smoothquant/scales/onerec_ad_smoothquant_absmax_sample${CALIB_SAMPLE_SIZE}_offset${CALIB_SAMPLE_OFFSET}.pt}"
SMOOTH_CUTOFF_TAG=""
if [[ -n "$SMOOTH_LAYER_MIN" ]]; then
  SMOOTH_CUTOFF_TAG="${SMOOTH_CUTOFF_TAG}-smoothmin${SMOOTH_LAYER_MIN}"
fi
if [[ -n "$SMOOTH_LAYER_CUTOFF" ]]; then
  SMOOTH_CUTOFF_TAG="${SMOOTH_CUTOFF_TAG}-smoothcutoff${SMOOTH_LAYER_CUTOFF}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-fake-fp8-smoothquant-a${SMOOTH_ALPHA}${SMOOTH_CUTOFF_TAG}-calib${CALIB_SAMPLE_SIZE}-offset${CALIB_SAMPLE_OFFSET}-ad-${SAMPLE_SIZE}}"
MODEL_NAME="${MODEL_NAME:-OneRec-1.7B-hf-fake-fp8-smoothquant-a${SMOOTH_ALPHA}${SMOOTH_CUTOFF_TAG}-calib${CALIB_SAMPLE_SIZE}-offset${CALIB_SAMPLE_OFFSET}}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --data_dir "$DATA_DIR"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ -n "$TARGET_REGEX" ]]; then
  COMMON_ARGS+=(--target_regex "$TARGET_REGEX")
fi

if [[ -n "$SKIP_REGEX" ]]; then
  COMMON_ARGS+=(--skip_regex "$SKIP_REGEX")
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "CALIB_SAMPLE_SIZE=${CALIB_SAMPLE_SIZE}"
echo "CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE}"
echo "CALIB_SAMPLE_OFFSET=${CALIB_SAMPLE_OFFSET}"
echo "SMOOTH_ALPHA=${SMOOTH_ALPHA}"
echo "SMOOTH_LAYER_MIN=${SMOOTH_LAYER_MIN}"
echo "SMOOTH_LAYER_CUTOFF=${SMOOTH_LAYER_CUTOFF}"
echo "act_quant_mode=shared_input"
echo "SCALE_PATH=${SCALE_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

if [[ "$SKIP_CALIB" != "true" || ! -f "$SCALE_PATH" ]]; then
  echo "Collecting SmoothQuant calibration stats..."
  python fake_quant/smoothquant/collect_smooth_scales.py \
    "${COMMON_ARGS[@]}" \
    --sample_size "$CALIB_SAMPLE_SIZE" \
    --batch_size "$CALIB_BATCH_SIZE" \
    --sample_offset "$CALIB_SAMPLE_OFFSET" \
    --output_path "$SCALE_PATH"
else
  echo "Skipping calibration; using existing SCALE_PATH."
fi

EVAL_ARGS=(
  "${COMMON_ARGS[@]}"
  --sample_size "$SAMPLE_SIZE"
  --batch_size "$BATCH_SIZE"
  --num_beams "$NUM_BEAMS"
  --num_return_sequences "$NUM_RETURN_SEQUENCES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --quant_scheme fp8_smoothquant
  --act_quant per_token
  --act_quant_mode shared_input
  --smooth_scales_path "$SCALE_PATH"
  --smooth_alpha "$SMOOTH_ALPHA"
  --output_dir "$OUTPUT_DIR"
  --model_name "$MODEL_NAME"
)

if [[ -n "$SMOOTH_LAYER_MIN" ]]; then
  EVAL_ARGS+=(--smooth_layer_min "$SMOOTH_LAYER_MIN")
fi

if [[ -n "$SMOOTH_LAYER_CUTOFF" ]]; then
  EVAL_ARGS+=(--smooth_layer_cutoff "$SMOOTH_LAYER_CUTOFF")
fi

if [[ "$EVALUATE" == "true" ]]; then
  EVAL_ARGS+=(--evaluate)
fi

if [[ "$OVERWRITE" == "true" ]]; then
  EVAL_ARGS+=(--overwrite)
fi

echo "Running SmoothQuant-style FP8 fake quant AD evaluation..."
python fake_quant/run_ad_sid.py "${EVAL_ARGS[@]}"
