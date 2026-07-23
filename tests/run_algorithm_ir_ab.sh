#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PCCL_BUFFER_SIZE="${PCCL_BUFFER_SIZE:-536870912}"
export PCCL_DISABLE_FUSED="${PCCL_DISABLE_FUSED:-1}"

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  tests/bench_algorithm_ir_ab.py "$@"
