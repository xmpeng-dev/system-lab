#!/bin/bash
cd /home/xiaompen/minimax-poc; ulimit -n 524288
for i in 1 2 3 4; do for k in gmm gmm_flydsl; do
  o=.scratch/nogate/${k}_$i
  TENSILE_PERSISTENT_FIXED_GRID=1000000 timeout 1200 torchrun --nproc_per_node=4 examples-phase2/training/common/cco_bench.py \
    --no-telemetry --pg-timeout-s 120 --comm alltoall --compute $k --sizes-mib 768,384,256,128,64 -o $o.json > $o.log 2>&1
  echo "$k run $i exit $? $(date +%T)" >> .scratch/nogate/progress.txt
done; done
