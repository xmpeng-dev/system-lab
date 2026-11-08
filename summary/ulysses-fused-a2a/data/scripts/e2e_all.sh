#!/bin/bash
# GLM-4.7 4-layer proxy, CP8 / CP4 x 16K / 32K x base / fused, per Turbo; waits for val2. Progress: e2e_progress.txt
E=/home/xiaompen/uly_reg/e2e
until grep -q "ALL DONE" /home/xiaompen/uly_reg/logs/tests_progress.txt 2>/dev/null; do sleep 30; done
for c in 183450cc 5d3e68e6 fix; do
    for cp in 8 4; do
        for seq in 16384 32768; do
            s=$((seq / 1024))k
            for impl in base fused; do
                tag=glm_cp${cp}_${c}_${impl}${s}
                extra=""; [ $impl = fused ] && extra="PRIMUS_ULYSSES_FUSED_A2A=1"
                echo "$(date +%T) start $tag" >> $E/e2e_progress.txt
                bash $E/run_e2e.sh $c $tag PRIMUS_CP=$cp PRIMUS_EP=8 PRIMUS_SEQ_LENGTH=$seq PRIMUS_MAX_POSITION_EMBEDDINGS=$seq $extra
                pkill -f "pretrain" 2>/dev/null; sleep 10
                echo "$(date +%T) end   $tag $(tail -1 $E/$tag.log)" >> $E/e2e_progress.txt
            done
        done
    done
done
echo "$(date +%T) ALL DONE" >> $E/e2e_progress.txt
