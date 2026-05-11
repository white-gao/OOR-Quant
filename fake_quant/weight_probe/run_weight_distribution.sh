#!/usr/bin/env bash
set -euo pipefail

# Plot OneRec weight distributions. Run from repository root.

MODEL_PATH="${MODEL_PATH:-/home/guowei/OneRec-1.7B}"
VERSION="${VERSION:-v1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-fake_quant/weight_probe/results/${VERSION}/OneRec-1.7B}"
SAMPLE_PER_TENSOR="${SAMPLE_PER_TENSOR:-2000000}"
CHUNK_ELEMENTS="${CHUNK_ELEMENTS:-8000000}"
HIST_BINS="${HIST_BINS:-400}"
SEED="${SEED:-42}"
TARGETS="${TARGETS:-default}"
INCLUDE_EMBEDDING="${INCLUDE_EMBEDDING:-false}"

ARGS=(
  --model_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --sample_per_tensor "$SAMPLE_PER_TENSOR"
  --chunk_elements "$CHUNK_ELEMENTS"
  --hist_bins "$HIST_BINS"
  --targets "$TARGETS"
  --seed "$SEED"
)

if [[ "$INCLUDE_EMBEDDING" == "true" ]]; then
  ARGS+=(--include_embedding)
fi

python3 fake_quant/weight_probe/plot_weight_distributions.py "${ARGS[@]}"
