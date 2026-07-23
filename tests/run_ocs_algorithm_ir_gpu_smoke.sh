#!/usr/bin/env bash
set -euo pipefail

lowering="${1:-generated}"
iterations="${2:-3}"
elements="${3:-4096}"

if [[ "${lowering}" != "generated" && "${lowering}" != "template" ]]; then
  echo "lowering must be generated or template" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PCCL_BUFFER_SIZE="${PCCL_BUFFER_SIZE:-536870912}"
export PCCL_DISABLE_FUSED="${PCCL_DISABLE_FUSED:-1}"

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  tests/ocs_algorithm_ir_smoke.py \
  --lowering "${lowering}" \
  --iterations "${iterations}" \
  --elements "${elements}"
