#!/bin/bash
# per size, up to 3 attempts; args: kernel tag [env...]
k=$1; tag=$2; shift 2
cd /home/xiaompen/minimax-poc; ulimit -n 524288
for s in 768 384 256 128 64; do
  for t in 1 2 3; do
    o=.scratch/cco-analysis/flydsl/${tag}_${k}_${s}
    env "$@" timeout 900 torchrun --nproc_per_node=4 examples-phase2/training/common/cco_bench.py --no-telemetry --pg-timeout-s 120 \
      --comm alltoall --compute $k --sizes-mib $s -o $o.json > $o.log 2>&1
    r=$?; echo "$tag $k $s try $t exit $r $(date +%T)" >> .scratch/cco-analysis/flydsl/progress.txt
    [ $r = 0 ] && break
  done
done
