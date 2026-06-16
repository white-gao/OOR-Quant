#!/usr/bin/env bash
set -euo pipefail

# Run OneRec learnable FP8 W+A fake quantization on the AD benchmark.
# Fixed defaults live in fake_quant_learnable/run_m1_onerec_ad.py:
# calib split auto-detects calib1024, eval split is test, shared-input W8A8,
# SmoothQuant uses omni scope with fold enabled, and generation uses 32 beams.
#
# Examples:
#   bash fake_quant_learnable/run_learnable_quant_ad.sh
#   MODE=baseline_w8a8 DEVICE=cuda:1 bash fake_quant_learnable/run_learnable_quant_ad.sh
#   MODE=m2_lwt_let LET_INIT=smoothquant EPOCHS=2 DEVICE=cuda:6 bash fake_quant_learnable/run_learnable_quant_ad.sh
#   CALIB_ONLY=1 LAYERS=0 EPOCHS=1 bash fake_quant_learnable/run_learnable_quant_ad.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-m2_lwt_let}"  # baseline_w8a8, smoothquant_w8a8, m1_lwt, m2_let, or m2_lwt_let
MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B/}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data-calib1024}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"

LAYERS="${LAYERS:-all}"
DEVICE="${DEVICE:-cuda}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-1024}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
EPOCHS="${EPOCHS:-2}"
LWT_LR="${LWT_LR:-3e-4}"
LET_LR="${LET_LR:-6e-4}"
LET_INIT="${LET_INIT:-ones}"
LOAD_QUANT_PARAMS="${LOAD_QUANT_PARAMS:-}"

OVERWRITE="${OVERWRITE:-1}"
EVALUATE="${EVALUATE:-1}"
CALIB_ONLY="${CALIB_ONLY:-0}"
COMPUTE_SID_PPL="${COMPUTE_SID_PPL:-0}"
SID_PPL_MAX_ITEMS="${SID_PPL_MAX_ITEMS:-1}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="${MODE}_ad_calib${CALIB_SAMPLE_SIZE}_epochs${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  fi
  OUTPUT_DIR="fake_quant_learnable/results/${RUN_NAME}"
fi

args=(
  --mode "${MODE}"
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --layers "${LAYERS}"
  --device "${DEVICE}"
  --calib_sample_size "${CALIB_SAMPLE_SIZE}"
  --eval_sample_size "${EVAL_SAMPLE_SIZE}"
  --epochs "${EPOCHS}"
  --lwt_lr "${LWT_LR}"
  --let_lr "${LET_LR}"
  --let_init "${LET_INIT}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "${CALIB_ONLY}" == "1" ]]; then
  args+=(--calib_only)
elif [[ "${EVALUATE}" == "1" ]]; then
  args+=(--evaluate)
fi

if [[ -n "${LOAD_QUANT_PARAMS}" ]]; then
  args+=(--load_quant_params "${LOAD_QUANT_PARAMS}")
fi

if [[ "${COMPUTE_SID_PPL}" == "1" ]]; then
  args+=(--compute_sid_ppl --sid_ppl_max_items "${SID_PPL_MAX_ITEMS}")
fi

echo "Running learnable quantization:"
printf '  MODE=%s DEVICE=%s LAYERS=%s\n' "${MODE}" "${DEVICE}" "${LAYERS}"
printf '  MODEL_PATH=%s\n' "${MODEL_PATH}"
printf '  DATA_DIR=%s\n' "${DATA_DIR}"
printf '  OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
printf '  CALIB_SAMPLE_SIZE=%s EVAL_SAMPLE_SIZE=%s\n' "${CALIB_SAMPLE_SIZE}" "${EVAL_SAMPLE_SIZE}"
printf '  EPOCHS=%s LWT_LR=%s LET_LR=%s LET_INIT=%s\n' "${EPOCHS}" "${LWT_LR}" "${LET_LR}" "${LET_INIT}"
printf '  LOAD_QUANT_PARAMS=%s\n' "${LOAD_QUANT_PARAMS}"
printf '  COMPUTE_SID_PPL=%s SID_PPL_MAX_ITEMS=%s\n' "${COMPUTE_SID_PPL}" "${SID_PPL_MAX_ITEMS}"

python3 -m fake_quant_learnable.run_m1_onerec_ad "${args[@]}" "$@"
