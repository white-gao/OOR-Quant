#!/usr/bin/env bash
set -euo pipefail

# Run full AD evaluation for HF baseline and fake-quant variants:
#   1. BF16/FP16 HF baseline
#   2. FP8 weight-only, per-output-channel Linear weights
#   3. FP8 weight + dynamic activation, shared-input per-token activation QDQ
#   4. FP8 weight + static activation, shared-input per-tensor activation QDQ
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0 bash fake_quant/run_hf_ad_full_quant.sh
#   RUN_BASELINE=false bash fake_quant/run_hf_ad_full_quant.sh
#   RUN_WEIGHT_ONLY=false RUN_WEIGHT_ACT=true bash fake_quant/run_hf_ad_full_quant.sh
#   SKIP_DONE=false bash fake_quant/run_hf_ad_full_quant.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda:6}"
SEED="${SEED:-42}"
EVALUATE="${EVALUATE:-true}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_DONE="${SKIP_DONE:-true}"
RUN_BASELINE="${RUN_BASELINE:-false}"
RUN_WEIGHT_ONLY="${RUN_WEIGHT_ONLY:-true}"
RUN_WEIGHT_ACT="${RUN_WEIGHT_ACT:-false}"
RUN_WEIGHT_ACT_STATIC="${RUN_WEIGHT_ACT_STATIC:-false}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
CALIB_SAMPLE_OFFSET="${CALIB_SAMPLE_OFFSET:-1000}"
SKIP_STATIC_CALIB="${SKIP_STATIC_CALIB:-false}"
TARGET_REGEX="${TARGET_REGEX:-}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-fake_quant/results/${VERSION}}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-baseline-ad-${SAMPLE_SIZE}}"
WEIGHT_ONLY_OUTPUT_DIR="${WEIGHT_ONLY_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-fake-fp8-weight-channel-ad-${SAMPLE_SIZE}}"
WEIGHT_ACT_OUTPUT_DIR="${WEIGHT_ACT_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-fake-fp8-weight-act-per-token-shared_input-ad-${SAMPLE_SIZE}}"
WEIGHT_ACT_STATIC_OUTPUT_DIR="${WEIGHT_ACT_STATIC_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/results_OneRec-1.7B-hf-fake-fp8-weight-act-static-tensor-shared_input-calib${CALIB_SAMPLE_SIZE}-offset${CALIB_SAMPLE_OFFSET}-ad-${SAMPLE_SIZE}}"
STATIC_ACT_SCALE_PATH="${STATIC_ACT_SCALE_PATH:-fake_quant/static_activation/scales/onerec_ad_static_tensor_absmax_sample${CALIB_SAMPLE_SIZE}_offset${CALIB_SAMPLE_OFFSET}.pt}"
BASELINE_MODEL_NAME="${BASELINE_MODEL_NAME:-OneRec-1.7B-hf-baseline}"
WEIGHT_ONLY_MODEL_NAME="${WEIGHT_ONLY_MODEL_NAME:-OneRec-1.7B-hf-fake-fp8-weight-channel}"
WEIGHT_ACT_MODEL_NAME="${WEIGHT_ACT_MODEL_NAME:-OneRec-1.7B-hf-fake-fp8-weight-act-per-token-shared_input}"
WEIGHT_ACT_STATIC_MODEL_NAME="${WEIGHT_ACT_STATIC_MODEL_NAME:-OneRec-1.7B-hf-fake-fp8-weight-act-static-tensor-shared_input-calib${CALIB_SAMPLE_SIZE}-offset${CALIB_SAMPLE_OFFSET}}"

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

if [[ -n "$TARGET_REGEX" ]]; then
  COMMON_ARGS+=(--target_regex "$TARGET_REGEX")
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "NUM_BEAMS=${NUM_BEAMS}"
echo "NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "DTYPE=${DTYPE}"
echo "DEVICE=${DEVICE}"
echo "SEED=${SEED}"
echo "SKIP_DONE=${SKIP_DONE}"
echo "CALIB_SAMPLE_SIZE=${CALIB_SAMPLE_SIZE}"
echo "CALIB_SAMPLE_OFFSET=${CALIB_SAMPLE_OFFSET}"
echo "act_quant_mode=shared_input"
echo "BASELINE_OUTPUT_DIR=${BASELINE_OUTPUT_DIR}"
echo "WEIGHT_ONLY_OUTPUT_DIR=${WEIGHT_ONLY_OUTPUT_DIR}"
echo "WEIGHT_ACT_OUTPUT_DIR=${WEIGHT_ACT_OUTPUT_DIR}"
echo "WEIGHT_ACT_STATIC_OUTPUT_DIR=${WEIGHT_ACT_STATIC_OUTPUT_DIR}"
echo "STATIC_ACT_SCALE_PATH=${STATIC_ACT_SCALE_PATH}"

should_skip_done() {
  local output_dir="$1"
  if [[ "$SKIP_DONE" == "true" && "$EVALUATE" == "true" && -f "${output_dir}/eval_results.json" ]]; then
    return 0
  fi
  return 1
}

if [[ "$RUN_BASELINE" == "true" ]]; then
  if should_skip_done "$BASELINE_OUTPUT_DIR"; then
    echo "Skipping HF baseline full AD; eval_results.json already exists."
  else
    echo "Running HF baseline full AD..."
    python fake_quant/run_ad_sid.py \
      "${COMMON_ARGS[@]}" \
      --quant_scheme none \
      --output_dir "$BASELINE_OUTPUT_DIR" \
      --model_name "$BASELINE_MODEL_NAME"
  fi
fi

if [[ "$RUN_WEIGHT_ONLY" == "true" ]]; then
  if should_skip_done "$WEIGHT_ONLY_OUTPUT_DIR"; then
    echo "Skipping HF FP8 weight-only full AD; eval_results.json already exists."
  else
    echo "Running HF FP8 weight-only full AD..."
    python fake_quant/run_ad_sid.py \
      "${COMMON_ARGS[@]}" \
      --quant_scheme fp8_weight_channel \
      --act_quant none \
      --output_dir "$WEIGHT_ONLY_OUTPUT_DIR" \
      --model_name "$WEIGHT_ONLY_MODEL_NAME"
  fi
fi

if [[ "$RUN_WEIGHT_ACT" == "true" ]]; then
  if should_skip_done "$WEIGHT_ACT_OUTPUT_DIR"; then
    echo "Skipping HF FP8 weight + activation full AD; eval_results.json already exists."
  else
    echo "Running HF FP8 weight + activation full AD..."
    python fake_quant/run_ad_sid.py \
      "${COMMON_ARGS[@]}" \
      --quant_scheme fp8_weight_channel \
      --act_quant per_token \
      --act_quant_mode shared_input \
      --output_dir "$WEIGHT_ACT_OUTPUT_DIR" \
      --model_name "$WEIGHT_ACT_MODEL_NAME"
  fi
fi

if [[ "$RUN_WEIGHT_ACT_STATIC" == "true" ]]; then
  if [[ "$SKIP_STATIC_CALIB" != "true" || ! -f "$STATIC_ACT_SCALE_PATH" ]]; then
    echo "Collecting static activation calibration stats..."
    python fake_quant/smoothquant/collect_smooth_scales.py \
      --model_path "$MODEL_PATH" \
      --data_dir "$DATA_DIR" \
      --dtype "$DTYPE" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --sample_size "$CALIB_SAMPLE_SIZE" \
      --sample_offset "$CALIB_SAMPLE_OFFSET" \
      --output_path "$STATIC_ACT_SCALE_PATH"
  else
    echo "Skipping static activation calibration; using existing STATIC_ACT_SCALE_PATH."
  fi

  if should_skip_done "$WEIGHT_ACT_STATIC_OUTPUT_DIR"; then
    echo "Skipping HF FP8 weight + static activation full AD; eval_results.json already exists."
  else
    echo "Running HF FP8 weight + static activation full AD..."
    python fake_quant/run_ad_sid.py \
      "${COMMON_ARGS[@]}" \
      --quant_scheme fp8_weight_channel \
      --act_quant static_tensor \
      --act_quant_mode shared_input \
      --static_act_scales_path "$STATIC_ACT_SCALE_PATH" \
      --output_dir "$WEIGHT_ACT_STATIC_OUTPUT_DIR" \
      --model_name "$WEIGHT_ACT_STATIC_MODEL_NAME"
  fi
fi
