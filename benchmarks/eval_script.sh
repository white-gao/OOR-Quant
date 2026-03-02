#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
export BENCHMARK_BASE_DIR="."
export BENCHMARK_DATA_DIR="../data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# Set common variables
MODEL_PATH=$1
VERSION="${VERSION:-v1.0}"
BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}"
BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/$2"
ENABLE_THINKING=$3

# Read configuration from environment variables (set by eval_script.py)
# Fallback to hardcoded paths if not set
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-/home/user/benchmark}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${BENCHMARK_BASE_DIR}/data_${DATA_VERSION}}"
DATA_DIR="$BENCHMARK_DATA_DIR"
SAMPLA_SIZE=5000

# Create output directory and log directory
mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "$BASE_OUTPUT_DIR"

# Write debug info to log file
{
    echo "========== Task Configuration =========="
    echo "DATA_DIR: $DATA_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

echo "Thinking args: $THINKING_ARGS"

echo "Running all tasks"

echo "Task 1/8: rec_reason"
# Task: rec_reason
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types rec_reason \
    --gpu_memory_utilization 0.9 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 5 \
    --overwrite \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 2/8: item_understand"
# Task: item_understand
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types item_understand \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 250 \
    --overwrite \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 3/8: ad"
# Task: ad
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types ad \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 4/8: product"
# Task: product
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types product \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 5/8: label_cond"
# Task: label_cond
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types label_cond \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 6/8: video"  
# Task: video
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types video \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 1875 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 7/8: interactive"
# Task: interactive
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types interactive \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 250 \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "Task 8/8: seq_pred"
# Task: label_pred
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types label_pred \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size 3200 \
    --max_logprobs 10000 \
    --overwrite \
    --sample_size $SAMPLA_SIZE \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "All tasks completed successfully"
