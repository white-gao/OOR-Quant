#!/usr/bin/env bash
set -euo pipefail

# Run OneRec learnable FP8 W+A fake quantization on the AD benchmark.
#
# Default run:
#   bash fake_quant_learnable/run_learnable_quant_ad.sh
#
# Common overrides:
#   MODE=m1_lwt LAYERS=last:8 bash fake_quant_learnable/run_learnable_quant_ad.sh
#   CALIB_ONLY=1 STEPS=50 bash fake_quant_learnable/run_learnable_quant_ad.sh
#   MODEL_PATH=/path/to/OneRec-1.7B DATA_DIR=/path/to/benchmark-data bash fake_quant_learnable/run_learnable_quant_ad.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-m2_lwt_let}"  # m1_lwt or m2_lwt_let
MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
MODEL_NAME="${MODEL_NAME:-}"

LAYERS="${LAYERS:-last:1}"
ACT_QUANT="${ACT_QUANT:-per_token}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-}"

CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
CALIB_OFFSET="${CALIB_OFFSET:-1000}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"

EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-1000}"
EVAL_OFFSET="${EVAL_OFFSET:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

STEPS="${STEPS:-200}"
LR="${LR:-1e-3}"
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

RUN_NAME="${RUN_NAME:-${MODE}_ad1000_calib_offset_${CALIB_OFFSET}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant_learnable/results/${RUN_NAME}}"

args=(
  --mode "${MODE}"
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --layers "${LAYERS}"
  --calib_sample_size "${CALIB_SAMPLE_SIZE}"
  --calib_offset "${CALIB_OFFSET}"
  --calib_batch_size "${CALIB_BATCH_SIZE}"
  --eval_sample_size "${EVAL_SAMPLE_SIZE}"
  --eval_offset "${EVAL_OFFSET}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --steps "${STEPS}"
  --lr "${LR}"
  --init_clip_multiplier "${INIT_CLIP_MULTIPLIER}"
  --act_quant "${ACT_QUANT}"
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
printf '  OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
printf '  LAYERS=%s\n' "${LAYERS}"
printf '  CALIB_SAMPLE_SIZE=%s CALIB_OFFSET=%s\n' "${CALIB_SAMPLE_SIZE}" "${CALIB_OFFSET}"
printf '  EVAL_SAMPLE_SIZE=%s EVAL_OFFSET=%s\n' "${EVAL_SAMPLE_SIZE}" "${EVAL_OFFSET}"
printf '  STEPS=%s LR=%s\n' "${STEPS}" "${LR}"
printf '  SAVE_QUANT_PARAMS=%s LOAD_QUANT_PARAMS=%s\n' "${SAVE_QUANT_PARAMS}" "${LOAD_QUANT_PARAMS}"

python3 -m fake_quant_learnable.run_m1_onerec_ad "${args[@]}" "$@"
