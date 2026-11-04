#!/bin/bash
# usage: pair.sh VARIANT TAG   (on e15-01; env SIZES / ITERS / EPS / EXTRA / TIMEOUT go to both nodes)
# Starts node 1 on e15-02 over the multiplexed ssh, node 0 here; aborts if GPU 2 or 3 on either node is busy.
set -u
V=$1 TAG=$2
D=/home/xiaompen/minimax-poc/.scratch/multinode
S2="ssh -o ControlPath=/tmp/xmp-e15-02.sock x"
busy() { grep -A2 -E "^GPU: (2|3)$" | grep -q PID; }
if amd-smi process 2>/dev/null | busy || $S2 'amd-smi process 2>/dev/null' | busy; then
    echo "GPU 2/3 busy, not starting"; exit 3
fi
ENVS="TAG=$TAG TIMEOUT=${TIMEOUT:-300} SIZES=${SIZES:-12,48,96,192,384} ITERS=${ITERS:-20} EPS=${EPS:-10}"
ENVS="$ENVS NPROC=${NPROC:-2} GPUS=${GPUS:-2,3}"
export NPROC=${NPROC:-2} GPUS=${GPUS:-2,3}
X="EXTRA='${EXTRA:-}'"
$S2 "cd $D && env $ENVS $X bash launch.sh 1 $V xiaompen_a2a" > /tmp/xmp_n1.txt 2>&1 &
env TAG=$TAG TIMEOUT=${TIMEOUT:-300} EXTRA="${EXTRA:-}" bash "$D/launch.sh" 0 "$V" xiaompen_ep_overlap
wait
cat /tmp/xmp_n1.txt
grep -E '^\{|^hosts' "$D/logs/${V}_${TAG}_node0.log"
grep -E "Last error|ncclSystemError|ncclInternalError" "$D/logs/${V}_${TAG}_node0.log" | head -3
