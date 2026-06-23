#!/usr/bin/env bash
set -euo pipefail

# Runs product/video experiments serially on a single visible GPU:
#   1. full precision
#   2. naive W8A8 fake quant
#   3. GPTQ weight_v3: layer-wise gradient token group weights + per-token act + tail1
#
# Usage examples:
#   GPU_ID=0 bash fake_quant_learnable/run_product_video_quant_suite.sh
#   MODEL_PATH=/home/guowei/OneRec-8B/ MODEL_TAG=8b GPU_ID=3 bash fake_quant_learnable/run_product_video_quant_suite.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B/}"
MODEL_TAG="${MODEL_TAG:-1p7b}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data-calib1024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-fake_quant_learnable/results/product_video_quant_suite_${MODEL_TAG}_calib1024}"
DEVICE="${DEVICE:-cuda:0}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-1024}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_one() {
  local task="$1"
  local mode="$2"
  local label="$3"
  local output_dir="${OUTPUT_ROOT}/${label}_${task}_${MODEL_TAG}_calib${CALIB_SAMPLE_SIZE}"
  local log_file="${output_dir}/run.log"

  mkdir -p "${output_dir}"
  echo "[run] task=${task} mode=${mode} output_dir=${output_dir} device=${DEVICE}"

  PYTHONPATH=. "${PYTHON_BIN}" -m fake_quant_learnable.run_m1_onerec_ad \
    --task "${task}" \
    --mode "${mode}" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${output_dir}" \
    --device "${DEVICE}" \
    --calib_sample_size "${CALIB_SAMPLE_SIZE}" \
    --eval_sample_size "${EVAL_SAMPLE_SIZE}" \
    --overwrite \
    --evaluate 2>&1 | tee "${log_file}"
}

run_one "product" "full_precision" "full_precision"
run_one "product" "baseline_w8a8" "naive_w8a8"
run_one "product" "grad_weighted_gptq_fp8_w8a8_tail1" "gptq_weight_v3_grad_group_tail1"

run_one "video" "full_precision" "full_precision"
run_one "video" "baseline_w8a8" "naive_w8a8"
run_one "video" "grad_weighted_gptq_fp8_w8a8_tail1" "gptq_weight_v3_grad_group_tail1"
