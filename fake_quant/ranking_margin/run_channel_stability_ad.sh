#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B/}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
DEVICE="${DEVICE:-cuda:7}"
DTYPE="${DTYPE:-bfloat16}"
SAMPLE_SIZE="${SAMPLE_SIZE:-128}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-1000}"
LAYERS="${LAYERS:-0,7,14,21,27}"
TOPK_FRACTIONS="${TOPK_FRACTIONS:-0.01,0.05,0.10}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/ranking_margin/channel_stability/ad_sample${SAMPLE_SIZE}_offset${SAMPLE_OFFSET}}"

python3 -m fake_quant.ranking_margin.analyze_channel_stability \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --sample_size "${SAMPLE_SIZE}" \
  --sample_offset "${SAMPLE_OFFSET}" \
  --layers "${LAYERS}" \
  --topk_fractions "${TOPK_FRACTIONS}" \
  --output_dir "${OUTPUT_DIR}"
