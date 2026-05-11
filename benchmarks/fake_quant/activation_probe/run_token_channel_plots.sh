#!/usr/bin/env bash
set -euo pipefail

# Plot token-channel activation maps for selected OneRec layers/nodes.
# Run from benchmark repo root:
#   bash fake_quant/activation_probe/run_token_channel_plots.sh

MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-../data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
LAYERS="${LAYERS:-0,4,8,12,16,20,24,27}"
NODES="${NODES:-attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CHANNEL_STRIDE="${CHANNEL_STRIDE:-4}"
SURFACE_MAX_TOKENS="${SURFACE_MAX_TOKENS:-128}"
SURFACE_MAX_CHANNELS="${SURFACE_MAX_CHANNELS:-512}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/activation_profiles/${VERSION}/token_channel_sample_${SAMPLE_INDEX}}"

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
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python fake_quant/activation_probe/plot_token_channel_activations.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --sample_index "$SAMPLE_INDEX" \
  --layers "$LAYERS" \
  --nodes "$NODES" \
  --max_tokens "$MAX_TOKENS" \
  --channel_stride "$CHANNEL_STRIDE" \
  --surface_max_tokens "$SURFACE_MAX_TOKENS" \
  --surface_max_channels "$SURFACE_MAX_CHANNELS" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --seed "$SEED"
