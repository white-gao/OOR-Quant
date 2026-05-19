#!/usr/bin/env bash
set -euo pipefail

# Analyze cross-sample channel stability for fixed teacher-forced SID positions.
# Run from repository root:
#   bash fake_quant/probes/activation_probe/run_teacher_forced_channel_stability.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-128}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-0}"
LAYERS="${LAYERS:-0,7,14,21,27,28}"
NODES="${NODES:-attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input,residual_block_output,final_norm}"
TOPK_COUNTS="${TOPK_COUNTS:-32}"
TOPK_FRACTIONS="${TOPK_FRACTIONS:-0.01,0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/probes/activation_probe/activation_profiles/${VERSION}/OneRec-1.7B-ad-teacher-forced-channel-stability-sample-${SAMPLE_SIZE}}"

if [[ ! -e "$MODEL_PATH" && -e "/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B" ]]; then
  MODEL_PATH="/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B"
fi

if [[ ! -f "${DATA_DIR}/ad/ad_test.parquet" && -f "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data/ad/ad_test.parquet" ]]; then
  DATA_DIR="/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data"
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "LAYERS=${LAYERS}"
echo "NODES=${NODES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python fake_quant/probes/activation_probe/analyze_teacher_forced_channel_stability.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --max_tokens "$MAX_TOKENS" \
  --layers "$LAYERS" \
  --nodes "$NODES" \
  --topk_counts "$TOPK_COUNTS" \
  --topk_fractions "$TOPK_FRACTIONS"
