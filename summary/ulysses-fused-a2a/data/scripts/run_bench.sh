#!/bin/bash
# run_bench.sh <commit> <tag> <nproc> <layer|overlap> <args...>   (inside the container)
# log: /home/xiaompen/uly_reg/logs/<tag>.log
c=$1; tag=$2; np=$3; shift 3
W=/home/xiaompen/uly_reg/turbo_$c
mkdir -p /home/xiaompen/uly_reg/logs
export TURBO=$W PYTHONPATH=/home/xiaompen/pydeps/flydsl0341:$W FLYDSL_RUNTIME_CACHE_DIR=/root/.flydsl/cache_$c
cd /home/xiaompen/uly_reg
{
    echo "### commit=$c np=$np args=$* env: $(env | grep -E '^ULY_' | tr '\n' ' ')"
    timeout 5400 torchrun --nproc_per_node $np --master_port $((29500 + RANDOM % 1000)) uly_bench.py "$@" 2>&1 \
        | grep -vE "^W[0-9]{4}|warnings.warn|^\s*$|opaque|return func|FutureWarning|UserWarning"
    echo "### exit=${PIPESTATUS[0]}"
} > /home/xiaompen/uly_reg/logs/$tag.log 2>&1
