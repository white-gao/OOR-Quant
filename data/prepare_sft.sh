#!/bin/bash
# 数据切割脚本：将 general text 和推荐数据合并后按每 1000 条样本切割

set -e

# 配置
# general和onerec都使用sft开头的数据集
GENERAL_TEXT_PATH="data/general_text/sft"
REC_DATA_PATH="data/onerec_data/sft"
OUTPUT_DIR="./output/split_data"
MAX_ROWS=1000
ENGINE="pyarrow"

# 检查路径是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${GENERAL_TEXT_PATH}" ]; then
    echo "错误: General text 路径不存在: ${GENERAL_TEXT_PATH}"
    exit 1
fi

if [ ! -e "${REC_DATA_PATH}" ]; then
    echo "错误: 推荐数据路径不存在: ${REC_DATA_PATH}"
    exit 1
fi

# 执行
python3 "${SCRIPT_DIR}/scripts/split_data.py" \
    --general_text_path "${GENERAL_TEXT_PATH}" \
    --rec_data_path "${REC_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_rows "${MAX_ROWS}" \
    --engine "${ENGINE}"

