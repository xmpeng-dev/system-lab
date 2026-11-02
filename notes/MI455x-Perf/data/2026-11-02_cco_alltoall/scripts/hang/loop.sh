#!/bin/bash
# repeat cco_bench (data-parallel grid) until a run hangs; watchdog 1 h so the hung state stays for inspection
cd /home/xiaompen/minimax-poc
ulimit -n 524288
for i in $(seq 3 30); do
  k=$([ $((i % 2)) = 0 ] && echo attn_bwd || echo gmm)
  echo "run $i $k $(date +%T)" >> .scratch/hang/loop.txt
  TENSILE_PERSISTENT_FIXED_GRID=1000000 torchrun --nproc_per_node=4 .scratch/hang/cco_bench_dbg.py --no-telemetry \
    --pg-timeout-s 3600 --comm alltoall --compute $k --sizes-mib 768,384,256,128,64 -o .scratch/hang/h$i.json > .scratch/hang/h$i.log 2>&1
done
