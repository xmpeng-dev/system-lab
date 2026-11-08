#!/bin/bash
# final code (turbo_fix): Qwen3-30B-A3B CP4 64K / 128K — overlap bench (n=4) then E2E base / fused. Progress: e2e/long_progress.txt
E=/home/xiaompen/uly_reg/e2e; P=$E/long_progress.txt
echo "$(date +%T) start bench" >> $P
bash /home/xiaompen/uly_reg/run_bench.sh fix q30b_overlap_long 4 overlap --model qwen3-30b-a3b --seq 65536 131072
echo "$(date +%T) end   bench $(tail -1 /home/xiaompen/uly_reg/logs/q30b_overlap_long.log)" >> $P
for seq in 65536 131072; do
    s=$((seq / 1024))k
    for impl in base fused; do
        tag=q30b_fix_${impl}${s}; extra=""; [ $impl = fused ] && extra="PRIMUS_ULYSSES_FUSED_A2A=1"
        echo "$(date +%T) start $tag" >> $P
        bash $E/run_e2e.sh fix $tag PRIMUS_MODEL=qwen3_30B_A3B PRIMUS_CP=4 PRIMUS_EP=8 PRIMUS_SEQ_LENGTH=$seq PRIMUS_MAX_POSITION_EMBEDDINGS=$seq $extra
        pkill -f "pretrain" 2>/dev/null; sleep 10
        echo "$(date +%T) end   $tag $(tail -1 $E/$tag.log)" >> $P
    done
done
echo "$(date +%T) LONG DONE" >> $P
