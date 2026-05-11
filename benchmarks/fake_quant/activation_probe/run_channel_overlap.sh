#!/usr/bin/env bash
set -euo pipefail

# Analyze whether activation outlier channels are fixed across layers.
# Run from benchmark repo root:
#   bash fake_quant/activation_probe/run_channel_overlap.sh

MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-../data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
LAYERS="${LAYERS:-all}"
NODES="${NODES:-attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TOPK="${TOPK:-32}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/activation_profiles/${VERSION}/channel_overlap_sample_${SAMPLE_INDEX}}"

if [[ ! -e "$MODEL_PATH" && -e "/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B" ]]; then
  MODEL_PATH="/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B"
fi

if [[ ! -f "${DATA_DIR}/ad/ad_test.parquet" && -f "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data/ad/ad_test.parquet" ]]; then
  DATA_DIR="/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data"
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_INDEX=${SAMPLE_INDEX}"
echo "LAYERS=${LAYERS}"
echo "NODES=${NODES}"
echo "TOPK=${TOPK}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python fake_quant/activation_probe/analyze_channel_overlap.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --sample_index "$SAMPLE_INDEX" \
  --layers "$LAYERS" \
  --nodes "$NODES" \
  --max_tokens "$MAX_TOKENS" \
  --topk "$TOPK" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --seed "$SEED"
