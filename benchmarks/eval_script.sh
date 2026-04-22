#!/bin/bash
set -u

export CUDA_VISIBLE_DEVICES=2
export BENCHMARK_BASE_DIR="."
export BENCHMARK_DATA_DIR="../data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# Set common variables
MODEL_PATH=$1
VERSION="${VERSION:-v1.0}"
BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}"
BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/$2"
ENABLE_THINKING="${3:-false}"
SAMPLE_SIZE="${SAMPLE_SIZE:-${4:-}}"

# Read configuration from environment variables (set by eval_script.py)
# Fallback to hardcoded paths if not set
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-/home/user/benchmark}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${BENCHMARK_BASE_DIR}/data_${DATA_VERSION}}"
DATA_DIR="$BENCHMARK_DATA_DIR"
# Optional sample size. When set to 1000 for ad, the loader prefers:
# ../data/onerec_data/benchmark_data/ad/ad_test_sample_1000.parquet
SAMPLE_ARGS=()
if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLE_ARGS=(--sample_size "$SAMPLE_SIZE")
fi

# Create output directory and log directory
mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "$BASE_OUTPUT_DIR"

# Write debug info to log file
{
    echo "========== Task Configuration =========="
    echo "DATA_DIR: $DATA_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "Sample Size: ${SAMPLE_SIZE:-full/default}"
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

echo "Thinking args: $THINKING_ARGS"
echo "Sample args: ${SAMPLE_ARGS[*]:-<none>}"

echo "Running all tasks"

run_eval_task() {
    local task_name="$1"
    shift

    if ! python3 -u scripts/ray-vllm/evaluate.py "$@" >> "${BASE_LOG_NAME}.log" 2>&1; then
        echo "[ERROR] Task ${task_name} failed. Forcing Ray cleanup before exit." | tee -a "${BASE_LOG_NAME}.log"
        ray stop --force >> "${BASE_LOG_NAME}.log" 2>&1 || true
        exit 1
    fi

    # Defensive cleanup to avoid leaked local Ray/vLLM processes between tasks.
    ray stop --force >> "${BASE_LOG_NAME}.log" 2>&1 || true
}

# echo "Task 1/8: rec_reason"
# Task: rec_reason
# Need api calls!! plz set_proxy
# python3 -u scripts/ray-vllm/evaluate.py \
#     --task_types rec_reason \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 5 \
#     --overwrite \
#     --sample_size $SAMPLA_SIZE \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

# echo "Task 2/8: item_understand"
# Task: item_understand
# Need api calls!! plz set_proxy
# python3 -u scripts/ray-vllm/evaluate.py \
#     --task_types item_understand \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 250 \
#     --overwrite \
#     --sample_size $SAMPLA_SIZE \
#     $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 3/8: ad"
# Task: ad
run_eval_task ad \
    --task_types ad \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    "${SAMPLE_ARGS[@]}" \
    $THINKING_ARGS

# echo "Task 4/8: product"
# Task: product
# run_eval_task product \
#     --task_types product \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS

# echo "Task 5/8: label_cond"
# Task: label_cond
# run_eval_task label_cond \
#     --task_types label_cond \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     --sample_size $SAMPLA_SIZE \
#     $THINKING_ARGS

# echo "Task 6/8: video"  
# Task: video
# run_eval_task video \
#     --task_types video \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 1875 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     $THINKING_ARGS

# echo "Task 7/8: interactive"
# Task: interactive
# run_eval_task interactive \
#     --task_types interactive \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 250 \
#     --overwrite \
#     --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
#     --sample_size $SAMPLA_SIZE \
#     $THINKING_ARGS

# echo "Task 8/8: seq_pred"
# Task: label_pred
# run_eval_task label_pred \
#     --task_types label_pred \
#     --gpu_memory_utilization 0.8 \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "${BASE_OUTPUT_DIR}" \
#     --dtype bfloat16 \
#     --worker_batch_size 3200 \
#     --max_logprobs 10000 \
#     --overwrite \
#     --sample_size $SAMPLA_SIZE \
#     $THINKING_ARGS

echo "All tasks completed successfully"
