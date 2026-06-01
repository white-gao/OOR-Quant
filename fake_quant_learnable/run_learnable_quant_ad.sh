#!/usr/bin/env bash
set -euo pipefail

# Run OneRec learnable FP8 W+A fake quantization on the AD benchmark.
#
# Default run:
#   bash fake_quant_learnable/run_learnable_quant_ad.sh
#
# Common overrides:
#   MODE=m1_lwt LAYERS=last:8 bash fake_quant_learnable/run_learnable_quant_ad.sh
#   CALIB_ONLY=1 EPOCHS=1 bash fake_quant_learnable/run_learnable_quant_ad.sh
#   MODEL_PATH=/path/to/OneRec-1.7B DATA_DIR=/path/to/benchmark-data bash fake_quant_learnable/run_learnable_quant_ad.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-m2_lwt_let}"  # baseline_w8a8, smoothquant_w8a8, m1_lwt, m2_let, or m2_lwt_let
MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B/}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data-calib1024}"
CALIB_DATA_DIR="${CALIB_DATA_DIR:-}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-}"
CALIB_SPLIT="${CALIB_SPLIT:-}"
MODEL_NAME="${MODEL_NAME:-}"

LAYERS="${LAYERS:-all}"
ACT_QUANT="${ACT_QUANT:-per_token}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-shared_input}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-}"

CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-1024}"
CALIB_OFFSET="${CALIB_OFFSET:-0}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
EVAL_OFFSET="${EVAL_OFFSET:-0}"
EPOCHS="${EPOCHS:-2}"
LWT_LR="${LWT_LR:-3e-4}"
LET_LR="${LET_LR:-6e-4}"
LET_INIT="${LET_INIT:-ones}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.5}"
SMOOTH_SCOPE="${SMOOTH_SCOPE:-omni}"
SMOOTH_FOLD="${SMOOTH_FOLD:-1}"
SMOOTHQUANT_MIN_SCALE="${SMOOTHQUANT_MIN_SCALE:-}"
SMOOTHQUANT_MAX_SCALE="${SMOOTHQUANT_MAX_SCALE:-}"
INIT_CLIP_MULTIPLIER="${INIT_CLIP_MULTIPLIER:-1.0}"

NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
SEED="${SEED:-42}"

OVERWRITE="${OVERWRITE:-1}"
EVALUATE="${EVALUATE:-1}"
CALIB_ONLY="${CALIB_ONLY:-0}"
SAVE_MODEL_STATE="${SAVE_MODEL_STATE:-0}"
SAVE_QUANT_PARAMS="${SAVE_QUANT_PARAMS:-1}"
LOAD_QUANT_PARAMS="${LOAD_QUANT_PARAMS:-}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"

RUN_NAME="${RUN_NAME:-${MODE}_ad_calib${CALIB_SAMPLE_SIZE}_epochs${EPOCHS}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant_learnable/results/${RUN_NAME}}"

args=(
  --mode "${MODE}"
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --layers "${LAYERS}"
  --calib_sample_size "${CALIB_SAMPLE_SIZE}"
  --calib_offset "${CALIB_OFFSET}"
  --eval_sample_size "${EVAL_SAMPLE_SIZE}"
  --eval_offset "${EVAL_OFFSET}"
  --epochs "${EPOCHS}"
  --init_clip_multiplier "${INIT_CLIP_MULTIPLIER}"
  --let_init "${LET_INIT}"
  --smoothquant_alpha "${SMOOTHQUANT_ALPHA}"
  --smooth_scope "${SMOOTH_SCOPE}"
  --smooth_fold "${SMOOTH_FOLD}"
  --act_quant "${ACT_QUANT}"
  --act_quant_mode "${ACT_QUANT_MODE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --num_beams "${NUM_BEAMS}"
  --num_return_sequences "${NUM_RETURN_SEQUENCES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --seed "${SEED}"
)

if [[ -n "${MODEL_NAME}" ]]; then
  args+=(--model_name "${MODEL_NAME}")
fi

if [[ -n "${LWT_LR}" ]]; then
  args+=(--lwt_lr "${LWT_LR}")
fi

if [[ -n "${LET_LR}" ]]; then
  args+=(--let_lr "${LET_LR}")
fi

if [[ -n "${SMOOTHQUANT_MIN_SCALE}" ]]; then
  args+=(--smoothquant_min_scale "${SMOOTHQUANT_MIN_SCALE}")
fi

if [[ -n "${SMOOTHQUANT_MAX_SCALE}" ]]; then
  args+=(--smoothquant_max_scale "${SMOOTHQUANT_MAX_SCALE}")
fi

if [[ -n "${CALIB_DATA_DIR}" ]]; then
  args+=(--calib_data_dir "${CALIB_DATA_DIR}")
fi

if [[ -n "${CALIB_SPLIT}" ]]; then
  args+=(--calib_split "${CALIB_SPLIT}")
fi

if [[ -n "${EVAL_DATA_DIR}" ]]; then
  args+=(--eval_data_dir "${EVAL_DATA_DIR}")
fi

if [[ -n "${DEVICE_MAP}" ]]; then
  args+=(--device_map "${DEVICE_MAP}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "${CALIB_ONLY}" == "1" ]]; then
  args+=(--calib_only)
elif [[ "${EVALUATE}" == "1" ]]; then
  args+=(--evaluate)
fi

if [[ "${SAVE_MODEL_STATE}" == "1" ]]; then
  args+=(--save_model_state)
fi

if [[ "${SAVE_QUANT_PARAMS}" == "1" ]]; then
  args+=(--save_quant_params)
else
  args+=(--no_save_quant_params)
fi

if [[ -n "${LOAD_QUANT_PARAMS}" ]]; then
  args+=(--load_quant_params "${LOAD_QUANT_PARAMS}")
fi

if [[ "${SKIP_CALIBRATION}" == "1" ]]; then
  args+=(--skip_calibration)
fi

echo "Running learnable quantization:"
printf '  MODE=%s\n' "${MODE}"
printf '  MODEL_PATH=%s\n' "${MODEL_PATH}"
printf '  DATA_DIR=%s\n' "${DATA_DIR}"
printf '  CALIB_DATA_DIR=%s EVAL_DATA_DIR=%s CALIB_SPLIT=%s\n' "${CALIB_DATA_DIR}" "${EVAL_DATA_DIR}" "${CALIB_SPLIT}"
printf '  OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
printf '  LAYERS=%s\n' "${LAYERS}"
printf '  CALIB_SAMPLE_SIZE=%s CALIB_OFFSET=%s\n' "${CALIB_SAMPLE_SIZE}" "${CALIB_OFFSET}"
printf '  EVAL_SAMPLE_SIZE=%s EVAL_OFFSET=%s\n' "${EVAL_SAMPLE_SIZE}" "${EVAL_OFFSET}"
printf '  EPOCHS=%s LWT_LR=%s LET_LR=%s LET_INIT=%s\n' "${EPOCHS}" "${LWT_LR}" "${LET_LR}" "${LET_INIT}"
printf '  SMOOTHQUANT_ALPHA=%s SMOOTH_SCOPE=%s SMOOTH_FOLD=%s\n' "${SMOOTHQUANT_ALPHA}" "${SMOOTH_SCOPE}" "${SMOOTH_FOLD}"
printf '  SMOOTHQUANT_MIN_SCALE=%s SMOOTHQUANT_MAX_SCALE=%s\n' "${SMOOTHQUANT_MIN_SCALE}" "${SMOOTHQUANT_MAX_SCALE}"
printf '  SAVE_QUANT_PARAMS=%s LOAD_QUANT_PARAMS=%s\n' "${SAVE_QUANT_PARAMS}" "${LOAD_QUANT_PARAMS}"

python3 -m fake_quant_learnable.run_m1_onerec_ad "${args[@]}" "$@"
