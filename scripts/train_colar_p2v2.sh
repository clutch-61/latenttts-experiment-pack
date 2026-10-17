#!/usr/bin/env bash
# CoLaR-p2v2: strategy-aligned fixed T=16, noise sample, no MCLinear, answer_beat filters.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU="${1:-0}"
OUT="${2:-outputs/colar_p2v2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT" logs
export CUDA_VISIBLE_DEVICES="$GPU"
python scripts/train_p123_system.py \
  --generator_type=colar \
  --ckpt=checkpoints/colar \
  --teacher_ckpt=checkpoints/colar \
  --selector_family=none \
  --select_mode=answer_beat \
  --sf_mode=C \
  --latent_length=16 \
  --max_new_tokens=128 \
  --k=16 \
  --no_auto_pack \
  --grad_accum=4 \
  --q_batch=2 \
  --max_steps=400 \
  --lr=2e-6 \
  --dropout_p=0.03 \
  --lambda_mmd=0 \
  --lambda_lat=0 \
  --lambda_rank=0 \
  --save_every=100 \
  --frozen_dtype=bf16 \
  --outdir="$OUT" \
  2>&1 | tee "logs/$(basename "$OUT").log"
