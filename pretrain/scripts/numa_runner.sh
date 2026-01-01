#!/bin/bash 

# 简单获取本地 NUMA 节点数量
num_numa=$(numactl -H | grep "node [0-9] cpus" | wc -l)
if [ "$num_numa" -lt 1 ]; then
  num_numa=1
fi

# 默认使用 NUMA 0
numa_id=0

echo "Bind to NUMA node $numa_id"

# 运行命令时绑定内存和 CPU 到 NUMA 节点 0
numactl --membind=$numa_id --cpunodebind=$numa_id "$@"