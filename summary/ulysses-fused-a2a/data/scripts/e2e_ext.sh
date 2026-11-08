#!/bin/bash
# final code (turbo_fix = 50d6c80a): Qwen3-235B proxy, MiniMax-M2.5 proxy (CP4 EP8), Llama3.1-8B (CP4 EP1), 16K / 32K, base / fused
E=/home/xiaompen/uly_reg/e2e
run() {  # tag model ep seq extra...
    tag=$1; m=$2; ep=$3; seq=$4; shift 4
    echo "$(date +%T) start $tag" >> $E/ext_progress.txt
    bash $E/run_e2e.sh fix $tag PRIMUS_MODEL=$m PRIMUS_CP=4 PRIMUS_EP=$ep PRIMUS_SEQ_LENGTH=$seq PRIMUS_MAX_POSITION_EMBEDDINGS=$seq "$@"
    pkill -f "pretrain" 2>/dev/null; sleep 10
    echo "$(date +%T) end   $tag $(tail -1 $E/$tag.log)" >> $E/ext_progress.txt
}
for seq in 16384 32768; do
    s=$((seq / 1024))k
    for p in "q235 qwen3_235B_A22B_4layer_proxy 8" "mm25 minimax_m2.5_4layer_proxy 8" "l8b llama3.1_8B 1"; do
        set -- $p
        run ${1}_fix_base$s $2 $3 $seq
        run ${1}_fix_fused$s $2 $3 $seq PRIMUS_ULYSSES_FUSED_A2A=1
    done
    run l8b_fix_fusedaiter$s llama3.1_8B 1 $seq PRIMUS_ULYSSES_FUSED_A2A=1 PRIMUS_TURBO_ATTN_BACKEND=AITER
done
echo "$(date +%T) EXT DONE" >> $E/ext_progress.txt
