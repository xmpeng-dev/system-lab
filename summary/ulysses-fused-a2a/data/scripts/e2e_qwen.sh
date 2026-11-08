#!/bin/bash
# Qwen3-30B-A3B (48 layers) CP4 EP8: R10 vs fix must not regress; waits for e2e_all. Progress: e2e_progress.txt
E=/home/xiaompen/uly_reg/e2e
until grep -q "ALL DONE" $E/e2e_progress.txt 2>/dev/null; do sleep 30; done
for seq in 16384 32768; do
    s=$((seq / 1024))k
    for run in "fix base" "5d3e68e6 fused" "fix fused"; do
        set -- $run; c=$1; impl=$2; tag=q30b_${c}_${impl}${s}
        extra=""; [ $impl = fused ] && extra="PRIMUS_ULYSSES_FUSED_A2A=1"
        echo "$(date +%T) start $tag" >> $E/e2e_progress.txt
        bash $E/run_e2e.sh $c $tag PRIMUS_MODEL=qwen3_30B_A3B PRIMUS_CP=4 PRIMUS_EP=8 PRIMUS_SEQ_LENGTH=$seq PRIMUS_MAX_POSITION_EMBEDDINGS=$seq $extra
        pkill -f "pretrain" 2>/dev/null; sleep 10
        echo "$(date +%T) end   $tag $(tail -1 $E/$tag.log)" >> $E/e2e_progress.txt
    done
done
echo "$(date +%T) QWEN DONE" >> $E/e2e_progress.txt
