#!/usr/bin/env bash
set -euo pipefail

# Profile fixed-position OneRec AD activations with teacher-forced SID decoding.
# Run from repository root:
#   bash fake_quant/probes/activation_probe/run_teacher_forced_decode_profile.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-128}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-0}"
OUTLIER_THRESHOLDS="${OUTLIER_THRESHOLDS:-6,10,20}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/probes/activation_probe/activation_profiles/${VERSION}/OneRec-1.7B-ad-teacher-forced-sample-${SAMPLE_SIZE}}"

if [[ ! -e "$MODEL_PATH" && -e "/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B" ]]; then
  MODEL_PATH="/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B"
fi

if [[ ! -f "${DATA_DIR}/ad/ad_test.parquet" && -f "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data/ad/ad_test.parquet" ]]; then
  DATA_DIR="/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data"
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python fake_quant/probes/activation_probe/profile_teacher_forced_decode_steps.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --max_tokens "$MAX_TOKENS" \
  --outlier_thresholds "$OUTLIER_THRESHOLDS"
