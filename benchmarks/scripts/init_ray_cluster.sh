script_dir=$(cd $(dirname $0); pwd)

parent_dir=$(dirname ${script_dir})
head_node_addr=$(awk 'NR==1 {print $1}' /etc/mpi/hostfile)
port=6381
echo "script dir: $script_dir"
echo "parent dir: $parent_dir"
echo "head_node_addr: $head_node_addr"

# We get unique nodes from the hostfile. `sort -u` changes order, so we use `awk`
# to get unique nodes while preserving the original order from the file.
ALL_NODES=$(awk '!a[$1]++ {print $1}' /etc/mpi/hostfile)

log_dir="./tmp_ray"
mkdir -p "${log_dir}"

echo "Will start Ray on the following nodes:"
echo "${ALL_NODES}"
echo ""

rank=0
for node in ${ALL_NODES}; do
  echo "--> Sending init command to ${node} (rank ${rank})..."
  if [[ $rank -eq 0 ]]; then
    ssh -n ${node} "bash ${parent_dir}/scripts/init_ray.sh ${head_node_addr} ${port} ${rank}" > "${log_dir}/ray_init_log.${node}.txt" 2>&1
  else
    ssh -n ${node} "bash ${parent_dir}/scripts/init_ray.sh ${head_node_addr} ${port} ${rank}" > "${log_dir}/ray_init_log.${node}.txt" 2>&1 &
  fi
  rank=$((rank + 1))
done

echo ""
echo "Waiting for all nodes to complete initialization..."
wait

echo ""
echo "Ray initialization commands have been sent to all nodes."
echo "Logs have been saved to the '${log_dir}' directory."
echo "To check status, see the log files, e.g.:"
echo "cat ${log_dir}/ray_init_log.${head_node_addr}.txt"

ray status