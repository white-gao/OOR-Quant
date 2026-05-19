#!/usr/bin/env bash
set -euo pipefail

# Profile OneRec-1.7B AD activation distributions with HF inference.
# Run from repository root:
#   bash fake_quant/probes/activation_probe/run_activation_profile.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-32}"
NUM_DECODE_STEPS="${NUM_DECODE_STEPS:-3}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-0}"
OUTLIER_THRESHOLDS="${OUTLIER_THRESHOLDS:-6,10,20}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/probes/activation_probe/activation_profiles/${VERSION}/OneRec-1.7B-ad-sample-${SAMPLE_SIZE}}"
SAVE_HISTOGRAMS="${SAVE_HISTOGRAMS:-true}"
HIST_MODULES="${HIST_MODULES:-residual_block_output,mlp.down_proj,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj}"
HIST_STAGES="${HIST_STAGES:-prefill,decode_step_1,decode_step_2,decode_step_3}"
HIST_BINS="${HIST_BINS:-120}"
HIST_LOG2_MIN="${HIST_LOG2_MIN:--12}"
HIST_LOG2_MAX="${HIST_LOG2_MAX:-14}"

if [[ ! -e "$MODEL_PATH" && -e "/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B" ]]; then
  MODEL_PATH="/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B"
fi

if [[ ! -f "${DATA_DIR}/ad/ad_test.parquet" && -f "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data/ad/ad_test.parquet" ]]; then
  DATA_DIR="/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data"
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "NUM_DECODE_STEPS=${NUM_DECODE_STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SAVE_HISTOGRAMS=${SAVE_HISTOGRAMS}"

EXTRA_ARGS=()
if [[ "$SAVE_HISTOGRAMS" == "true" ]]; then
  EXTRA_ARGS+=(
    --save_histograms
    --hist_modules "$HIST_MODULES"
    --hist_stages "$HIST_STAGES"
    --hist_bins "$HIST_BINS"
    --hist_log2_min "$HIST_LOG2_MIN"
    --hist_log2_max "$HIST_LOG2_MAX"
  )
fi

python fake_quant/probes/activation_probe/profile_ad_activations.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --num_decode_steps "$NUM_DECODE_STEPS" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --max_tokens "$MAX_TOKENS" \
  --outlier_thresholds "$OUTLIER_THRESHOLDS" \
  "${EXTRA_ARGS[@]}"
