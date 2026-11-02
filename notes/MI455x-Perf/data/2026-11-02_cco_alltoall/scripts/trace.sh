#!/bin/bash
# rocprofv3 kernel traces of compute + alltoall (a2a_trace.py), one dir per case under ./trace; then python3 scripts/trace_sum.py
cd "$(dirname "$0")/.."
ulimit -n 524288
for c in "gmm 768 0" "gmm 64 0" "gemm 768 0" "attn_fwd 768 0" "attn_bwd 768 0" "gmm 768 100" "gemm 768 100" "attn_fwd 768 100"; do
  set -- $c
  d=$PWD/trace/cf$3_$1_$2
  rm -rf "$d"
  timeout 300 torchrun --nproc_per_node=4 --no-python rocprofv3 --kernel-trace --output-format csv -d "$d/r%env{LOCAL_RANK}%" -o k -- \
    python3 scripts/a2a_trace.py --kernel $1 --mib $2 --comm-first-us $3 --episodes 6 2>&1 | grep -E "comm end|Error"
done
