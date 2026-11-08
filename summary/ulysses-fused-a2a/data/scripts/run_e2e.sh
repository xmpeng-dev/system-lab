#!/bin/bash
# run_e2e.sh <commit> <tag> [ENV=VAL ...]   (inside the container); log: /home/xiaompen/uly_reg/e2e/<tag>.log
c=$1; tag=$2; shift 2
for kv in "$@"; do export "$kv"; done
E=/home/xiaompen/uly_reg/e2e
export PYTHONPATH=$E/pyboot_$c:/home/xiaompen/pydeps/flydsl0341:/home/xiaompen/uly_reg/turbo_$c${PYTHONPATH:+:$PYTHONPATH}
export FLYDSL_RUNTIME_CACHE_DIR=/root/.flydsl/cache_$c
export PRIMUS_EXP_NAME=$tag PRIMUS_TRAIN_ITERS=12 PRIMUS_GBS=16 PRIMUS_MODEL=${PRIMUS_MODEL:-glm4_7_4layer_proxy}
cd /home/xiaompen/Primus-dev
{
    echo "### commit=$c tag=$tag env: $(env | grep -E '^PRIMUS_' | sort | tr '\n' ' ')"
    python3 -c "import primus_turbo, flydsl; print('### primus_turbo', primus_turbo.__file__, 'flydsl', flydsl.__version__, flydsl.__file__)"
} > $E/$tag.log 2>&1
./runner/primus-cli direct -- train pretrain --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml >> $E/$tag.log 2>&1
echo "exit=$?" >> $E/$tag.log
