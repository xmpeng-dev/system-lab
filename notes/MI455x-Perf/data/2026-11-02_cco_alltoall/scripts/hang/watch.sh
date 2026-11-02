#!/bin/bash
# poll a cco_bench run; exit when its log stops growing for STALL seconds (hang) or the run ends
log_glob=$1; STALL=${STALL:-120}
last=0; same=0
while docker exec xiaompen_ep_overlap pgrep -f cco_bench.py > /dev/null; do
  log=$(ls -t $log_glob 2>/dev/null | head -1); n="$log:$(grep -cE " m1 | m2 " "$log")"
  act=$(docker exec xiaompen_ep_overlap amd-smi metric -u | grep GFX_ACT | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' ')
  echo "$(date +%T) cases $n gfx% $act"
  if [ "$n" = "$last" ]; then same=$((same + 20)); else same=0; last=$n; fi
  [ $same -ge $STALL ] && { echo HANG; exit 0; }
  sleep 20
done
echo ENDED
