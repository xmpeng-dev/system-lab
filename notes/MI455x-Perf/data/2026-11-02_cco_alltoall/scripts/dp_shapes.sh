#!/bin/bash
# GEMM shapes + alltoall, default hipBLASLt grid vs data-parallel grid (TENSILE_PERSISTENT_FIXED_GRID >= tiles).
#   dp_shapes.sh OUT_DIR [cco_bench args...]       writes <variant>_<shape>.json / .log, then python3 dp_table.py OUT_DIR
set -u
out=$1; shift
mkdir -p "$out"
ulimit -n 524288
declare -A SHAPES=(
  [gmm-up912]="--compute gmm --gmm 72,912,6144,2048"
  [gmm-down912]="--compute gmm --gmm 72,912,1024,6144"
  [gmm-up456]="--compute gmm --gmm 72,456,6144,2048"
  [gmm-down456]="--compute gmm --gmm 72,456,1024,6144"
  [gemm-up8192]="--compute gemm --gemm 8192,6144,24576"
  [gemm-up4096]="--compute gemm --gemm 4096,6144,24576"
  [gemm-down8192]="--compute gemm --gemm 8192,24576,6144"
)
declare -A ENVS=([default]="" [dp]="TENSILE_PERSISTENT_FIXED_GRID=1000000")
for shape in ${SHAPE_LIST:-gmm-up912 gmm-down912 gmm-up456 gmm-down456 gemm-up8192 gemm-up4096 gemm-down8192}; do
  for v in ${VARIANTS:-default dp}; do
    name=${v}_$shape
    for attempt in 1 2; do
      log=$out/$name.log; [ $attempt -gt 1 ] && log=$out/$name.retry.log
      env ${ENVS[$v]} timeout 1800 torchrun --nproc_per_node=4 "${CCO_COMMON:-/home/xiaompen/minimax-poc/examples-phase2/training/common}/cco_bench.py" --no-telemetry \
        ${SHAPES[$shape]} -o "$out/$name.json" "$@" > "$log" 2>&1
      rc=$?
      echo "$name attempt $attempt exit $rc $(date -u +%FT%TZ)" | tee -a "$out/status.txt"
      [ $rc -eq 0 ] && break
    done
  done
done
