#!/bin/bash
# 数据采样脚本：从通用数据集中采样指定数量的样本用于 on-policy distillation

set -e

# 配置
INPUT_PATH="data/general_text"
OUTPUT_FILE="./output/onpolicy_distillation.parquet"
TEMP_FILE="./output/onpolicy_distillation_temp.parquet"
NUM_SAMPLES=200000
SEED=42
ENGINE="pyarrow"

# 检查路径是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${INPUT_PATH}" ]; then
    echo "错误: 输入路径不存在: ${INPUT_PATH}"
    exit 1
fi

# 步骤 1: 采样数据
echo "步骤 1: 采样数据..."
python3 "${SCRIPT_DIR}/scripts/sample_data.py" \
    --input "${INPUT_PATH}" \
    --output "${TEMP_FILE}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --engine "${ENGINE}"

# 步骤 2: 修复 unicode 编码
echo ""
echo "步骤 2: 修复 unicode 编码..."
python3 "${SCRIPT_DIR}/scripts/parquet_unicode_fix.py" \
    --input "${TEMP_FILE}" \
    --output "${OUTPUT_FILE}" \
    --engine "${ENGINE}"

# 清理临时文件
if [ -f "${TEMP_FILE}" ]; then
    rm "${TEMP_FILE}"
    echo "已清理临时文件"
fi

echo ""
echo "处理完成！输出文件: ${OUTPUT_FILE}"

