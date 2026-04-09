#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

NPROC="${1:-2,4,8}"
COLLECTIVE="${2:-allreduce,reduce_scatter,allgather,alltoall}"

export NCCL_NVLS_ENABLE=0
export NCCL_NTHREADS=256
export NCCL_MAX_CTAS=16
export PCCL_NUM_BLOCKS=16


python tests/bench_algo_comparison.py \
    --nproc="$NPROC" \
    --collective="$COLLECTIVE" \
    --executor=tma \
    --min-size=4096 \
    --max-size=536870912 \
    --warmup=2 \
    --iters=3 \
    --dtype=bf16
