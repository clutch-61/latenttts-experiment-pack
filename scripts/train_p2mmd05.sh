#!/usr/bin/env bash
# Warm-start dualBeat (or coconut) + dual_beat SF-C + λ_mmd=0.05.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:-outputs/p123_dualBeat/model}"
python -u scripts/train_p123_system.py \
  --ckpt "$CKPT" --teacher_ckpt checkpoints/coconut \
  --select_mode dual_beat --sf_mode C \
  --lambda_mmd 0.05 --lambda_lat 0 --lambda_rank 0 \
  --mmd_ramp_steps 100 --dropout_p 0.2 \
  --k 64 --q_batch 16 --grad_accum 8 \
  --lr 3e-6 --max_steps 1500 --save_every 1500 \
  --frozen_dtype bf16 --auto_pack \
  --outdir "${OUTDIR:-outputs/p123_p2mmd05}"
