#!/usr/bin/env bash
set -euo pipefail

# Run AD layer leave-one-out sensitivity with HF fake quant.
#
# Experiments:
#   1. HF baseline
#   2. All Linear FP8 weight + activation per-token
#   3. Restore one transformer layer to BF16 while quantizing all other Linear layers
#
# Run from repository root:
#   bash fake_quant/run_layer_sensitivity.sh
#
# Resume examples:
#   RUN_BASELINE=false RUN_ALL_QUANT=false START_LAYER=12 bash fake_quant/run_layer_sensitivity.sh
#   START_LAYER=0 END_LAYER=0 SAMPLE_SIZE=1 NUM_BEAMS=2 NUM_RETURN_SEQUENCES=2 MAX_NEW_TOKENS=1 bash fake_quant/run_layer_sensitivity.sh

MODEL_PATH="${MODEL_PATH:-/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-data/onerec_data/benchmark-data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${DATA_DIR}/ad}"
VERSION="${VERSION:-v1.0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
NUM_LAYERS="${NUM_LAYERS:-28}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-$((NUM_LAYERS - 1))}"
NUM_BEAMS="${NUM_BEAMS:-32}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda:4}"
SEED="${SEED:-42}"
EVALUATE="${EVALUATE:-true}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_DONE="${SKIP_DONE:-true}"
RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_ALL_QUANT="${RUN_ALL_QUANT:-true}"
RUN_RESTORE="${RUN_RESTORE:-true}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-shared_input}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-fake_quant/results/${VERSION}/layer_sensitivity_ad_sample_${SAMPLE_SIZE}}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/baseline}"
ALL_QUANT_OUTPUT_DIR="${ALL_QUANT_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/all_quant}"
BASELINE_MODEL_NAME="${BASELINE_MODEL_NAME:-OneRec-1.7B-hf-baseline-layer-sensitivity}"
ALL_QUANT_MODEL_NAME="${ALL_QUANT_MODEL_NAME:-OneRec-1.7B-hf-fake-fp8-all-quant-layer-sensitivity}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --data_dir "$DATA_DIR"
  --eval_data_dir "$EVAL_DATA_DIR"
  --sample_size "$SAMPLE_SIZE"
  --num_beams "$NUM_BEAMS"
  --num_return_sequences "$NUM_RETURN_SEQUENCES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ "$EVALUATE" == "true" ]]; then
  COMMON_ARGS+=(--evaluate)
fi

if [[ "$OVERWRITE" == "true" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

should_skip_done() {
  local output_dir="$1"
  if [[ "$SKIP_DONE" == "true" && "$EVALUATE" == "true" && -f "${output_dir}/eval_results.json" ]]; then
    return 0
  fi
  return 1
}

run_case() {
  local label="$1"
  local output_dir="$2"
  shift 2

  if should_skip_done "$output_dir"; then
    echo "Skipping ${label}; eval_results.json already exists."
    return
  fi

  echo "Running ${label}..."
  python fake_quant/run_ad_sid.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$output_dir" \
    "$@"
}

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "NUM_LAYERS=${NUM_LAYERS}"
echo "START_LAYER=${START_LAYER}"
echo "END_LAYER=${END_LAYER}"
echo "NUM_BEAMS=${NUM_BEAMS}"
echo "NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "DTYPE=${DTYPE}"
echo "DEVICE=${DEVICE}"
echo "SEED=${SEED}"
echo "SKIP_DONE=${SKIP_DONE}"
echo "ACT_QUANT_MODE=${ACT_QUANT_MODE}"
echo "BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT}"

if [[ "$RUN_BASELINE" == "true" ]]; then
  run_case \
    "HF baseline" \
    "$BASELINE_OUTPUT_DIR" \
    --quant_scheme none \
    --model_name "$BASELINE_MODEL_NAME"
fi

if [[ "$RUN_ALL_QUANT" == "true" ]]; then
  run_case \
    "FP8 all-quant" \
    "$ALL_QUANT_OUTPUT_DIR" \
    --quant_scheme fp8_weight_channel \
    --act_quant per_token \
    --act_quant_mode "$ACT_QUANT_MODE" \
    --model_name "$ALL_QUANT_MODEL_NAME"
fi

if [[ "$RUN_RESTORE" == "true" ]]; then
  for layer in $(seq "$START_LAYER" "$END_LAYER"); do
    layer_padded="$(printf "%02d" "$layer")"
    output_dir="${BASE_OUTPUT_ROOT}/restore_layer_${layer_padded}"
    model_name="OneRec-1.7B-hf-fake-fp8-restore-layer-${layer_padded}"
    skip_regex="^model\\.layers\\.${layer}\\."

    run_case \
      "FP8 restore layer ${layer_padded}" \
      "$output_dir" \
      --quant_scheme fp8_weight_channel \
      --act_quant per_token \
      --act_quant_mode "$ACT_QUANT_MODE" \
      --skip_regex "$skip_regex" \
      --model_name "$model_name"
  done
fi
