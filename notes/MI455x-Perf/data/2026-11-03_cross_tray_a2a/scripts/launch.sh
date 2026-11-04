#!/bin/bash
# usage: launch.sh NODE_RANK VARIANT CONTAINER   (node 0 = e15-01, master 10.190.224.68)
set -u
NR=$1 V=$2 C=$3
D=/home/xiaompen/minimax-poc/.scratch/multinode
case $V in
    roce) ENVS="NCCL_IB_HCA=ionic_0 NCCL_IB_GID_INDEX=1" ;;
    socket) ENVS="NCCL_IB_DISABLE=1" ;;
    *) echo "unknown variant $V"; exit 2 ;;
esac
mkdir -p "$D/logs" "$D/out"
docker exec "$C" bash -c 'ulimit -n $(ulimit -Hn); exec "$@"' _ env $ENVS HIP_VISIBLE_DEVICES="${GPUS:-2,3}" NCCL_SOCKET_IFNAME=enp4s0np0 \
    SIZES="${SIZES:-12,48,96,192,384}" ITERS="${ITERS:-20}" EPS="${EPS:-10}" \
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH ${EXTRA:-} \
    timeout "${TIMEOUT:-600}" torchrun --nnodes 2 --node_rank "$NR" --nproc_per_node "${NPROC:-2}" \
    --master_addr 10.190.224.68 --master_port "${PORT:-29531}" "$D/a2a_xnode.py" "$D/out/${V}_${TAG:-2x2}.json" \
    > "$D/logs/${V}_${TAG:-2x2}_node$NR.log" 2>&1
echo "node$NR $V exit=$?"
