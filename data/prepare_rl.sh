#!/bin/bash
# RL 数据切分脚本：将多个 RL 任务的数据集合并后切分为训练集和测试集

set -e

# 配置
# onerec 数据集产出路径，rl使用sft开头的数据集
REC_DATA_PATH="data/onerec_data"

# RL 依赖的任务
VIDEO_REC=${REC_DATA_PATH}/sft_video_rec.parquet
AD_REC=${REC_DATA_PATH}/sft_ad_rec.parquet
PRODUCT_REC=${REC_DATA_PATH}/sft_product_rec.parquet
INTERACTIVE_REC=${REC_DATA_PATH}/sft_interactive_rec.parquet
LABEL_COND_REC=${REC_DATA_PATH}/sft_label_cond_rec.parquet

# 输出配置
OUTPUT_DIR="./output/rl_data"
TEST_SIZE=1000
SEED=42
ENGINE="pyarrow"

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 定义所有需要处理的任务文件
declare -a TASK_FILES=(
    "${VIDEO_REC}"
    "${AD_REC}"
    "${PRODUCT_REC}"
    "${INTERACTIVE_REC}"
    "${LABEL_COND_REC}"
)

# 检查输入文件是否存在
echo "检查输入文件..."
MISSING_FILES=0
for file in "${TASK_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "警告: 文件不存在: ${file}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ ${MISSING_FILES} -eq ${#TASK_FILES[@]} ]; then
    echo "错误: 所有输入文件都不存在"
    exit 1
fi

# 执行 train_test_split，将所有文件合并后统一处理
echo ""
echo "开始处理 RL 数据切分..."
echo "=========================================="
echo "输入文件:"
for file in "${TASK_FILES[@]}"; do
    if [ -f "${file}" ]; then
        echo "  - ${file}"
    fi
done
echo "输出目录: ${OUTPUT_DIR}"
echo "测试集大小: ${TEST_SIZE}"
echo "=========================================="

python3 "${SCRIPT_DIR}/scripts/train_test_split.py" \
    --input_files "${TASK_FILES[@]}" \
    --test_size "${TEST_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --engine "${ENGINE}" \
    --test_filename "test.parquet" \
    --train_filename "train.parquet"

echo ""
echo "=========================================="
echo "RL 数据处理完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "  - train.parquet (训练集)"
echo "  - test.parquet (测试集)"
echo "=========================================="
