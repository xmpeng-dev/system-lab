#!/bin/bash
# Leave R CUs to RCCL by capping the persistent hipBLASLt grid at 256 - R (TENSILE_PERSISTENT_MAX_CUS), then cco_bench.
#   cu_split.sh OUT_DIR [cco_bench args...]       RESERVE="0 16 32 64 96 128" by default
set -u
out=$1; shift
mkdir -p "$out"
ulimit -n 524288
for r in ${RESERVE:-0 16 32 64 96 128}; do
  env=()
  [ "$r" -gt 0 ] && env=(TENSILE_PERSISTENT_MAX_CUS=$((256 - r)))
  name=r$r
  env "${env[@]}" timeout 1800 torchrun --nproc_per_node=4 "${CCO_COMMON:-/home/xiaompen/minimax-poc/examples-phase2/training/common}/cco_bench.py" --no-telemetry -o "$out/$name.json" "$@" \
    > "$out/$name.log" 2>&1
  echo "$name exit $? $(date -u +%FT%TZ)" | tee -a "$out/status.txt"
done
