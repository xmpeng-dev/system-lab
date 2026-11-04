#!/bin/bash
# In the container (slab is not mounted there; copy this directory's ag_trace.py / ag_trace.sh to /tmp/ag_diag with
# docker cp, run, docker cp trace/ and tuning.log back):
#   1. NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING: RCCL's path / algorithm / channels per allgather size -> tuning.log
#      (rank 0's allgather lines kept in ../results/rccl_tuning_allgather.log)
#   2. rocprofv3 kernel + memory copy trace of ag_trace.py -> trace/r<rank>/ (../trace, not in git)
# then on the host: python3 scripts/ag_trace_sum.py trace
cd "$(dirname "$0")"
ulimit -n 524288
export TENSILE_PERSISTENT_FIXED_GRID=${TENSILE_PERSISTENT_FIXED_GRID:-1000000}
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING timeout 300 torchrun --nproc_per_node=4 ag_trace.py --kernels gemm \
    --pair-mib 64 --calls 2 > tuning.log 2>&1
rm -rf trace
timeout 300 torchrun --nproc_per_node=4 --no-python rocprofv3 --kernel-trace --memory-copy-trace --output-format csv \
    -d "$PWD/trace/r%env{LOCAL_RANK}%" -o k -- python3 ag_trace.py 2>&1 | grep -E "comm end|Error"
