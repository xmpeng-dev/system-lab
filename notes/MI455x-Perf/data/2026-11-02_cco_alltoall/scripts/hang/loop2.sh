#!/bin/bash
# hang rate with a given environment: RUNS runs of cco_bench_dbg (gmm / attn_bwd alternating), pg watchdog 150 s
cd /home/xiaompen/minimax-poc
ulimit -n 524288
tag=$1; shift
for i in $(seq 1 ${RUNS:-6}); do
  k=$([ $((i % 2)) = 0 ] && echo attn_bwd || echo gmm)
  log=.scratch/hang/${tag}_$i.log
  env TENSILE_PERSISTENT_FIXED_GRID=1000000 "$@" timeout 900 torchrun --nproc_per_node=4 .scratch/hang/cco_bench_dbg.py --no-telemetry \
    --pg-timeout-s 150 --comm alltoall --compute $k --sizes-mib 768,384,256,128,64 -o .scratch/hang/${tag}_$i.json > $log 2>&1
  echo "$tag run $i $k exit $? hangdbg $(grep -c HANGDBG $log) $(date +%T)" | tee -a .scratch/hang/loop2.txt
done
