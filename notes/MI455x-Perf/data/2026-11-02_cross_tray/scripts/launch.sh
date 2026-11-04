#!/bin/bash
# usage: launch.sh NODE_RANK VARIANT   (run on each node; node 0 = e15-01)
set -u
NR=$1 V=$2
D=/home/xiaompen/minimax-poc/.scratch/multinode
case $V in
    roce) ENVS="NCCL_IB_HCA=ionic_0" ;;
    socket) ENVS="NCCL_IB_DISABLE=1" ;;
    *) echo "unknown variant $V"; exit 2 ;;
esac
mkdir -p "$D/logs" "$D/out"
docker exec xiaompen_ep_overlap env $ENVS NCCL_SOCKET_IFNAME=enp4s0np0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
    timeout "${TIMEOUT:-600}" torchrun --nnodes 2 --node_rank "$NR" --nproc_per_node 4 \
    --master_addr 10.190.224.68 --master_port "${PORT:-29531}" "$D/a2a_probe.py" "$D/out/$V.json" \
    > "$D/logs/${V}_node$NR.log" 2>&1
echo "node$NR $V exit=$?"
