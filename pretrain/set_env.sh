#!/bin/bash

# 检查当前的 shell 是否为 bash
if [ -z "$BASH_VERSION" ]; then
    echo "此脚本必须使用 bash 启动，请使用 'bash script.bash' 来运行它。" >&2
    exit 1
fi

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# 检查 .deepspeed_env 文件是否存在
if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: ${ENV_FILE} not found" >&2
    exit 1
fi

# 加载环境变量
set -a  # 自动导出所有变量
source "${ENV_FILE}"
set +a  # 禁用自动导出

# 打印已加载的环境变量
echo "Loaded environment variables from ${ENV_FILE}:"
cat "${ENV_FILE}"

# 安装系统依赖
PIP_CMD='pip'
PROXY="http://oversea-squid1.jp.txyun:11080"
HOSTFILE="/etc/mpi/hostfile"

# 在所有节点安装 numactl
mpirun --allow-run-as-root \
    --hostfile "${HOSTFILE}" \
    -x http_proxy="${PROXY}" \
    -x https_proxy="${PROXY}" \
    --pernode \
    bash -c "apt-get install -y numactl"

# 在所有节点安装 Python 依赖
mpirun --allow-run-as-root \
    --hostfile "${HOSTFILE}" \
    --pernode \
    bash -c "${PIP_CMD} install transformers==4.53 && \
             ${PIP_CMD} install easydict && \
             ${PIP_CMD} install torchao==0.10 && \
             ${PIP_CMD} install sortedcontainers"
