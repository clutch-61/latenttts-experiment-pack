#!/usr/bin/env bash
# CoLaR eval under p2v2 strategy protocol: fixed_t prompt + latent_length=16.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU="${1:?gpu}"
CKPT="${2:?ckpt}"
TAG="${3:?tag}"
N="${4:-1}"
mkdir -p "results/full/${TAG}" logs
LOG="logs/${TAG}_n${N}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
if [[ "$N" == "1" ]]; then
  python -m src.infer_llama \
    --model_id="$CKPT" --model_type=colar --model_dtype=bf16 \
    --data_path=data/gsm_test.json \
    --prompt_style=fixed_t --latent_length=16 \
    --do_sample=False --n_samples=1 --batch_size=16 --max_new_tokens=128 \
    2>&1 | tee "$LOG"
else
  python -m src.infer_llama \
    --model_id="$CKPT" --model_type=colar --model_dtype=bf16 \
    --data_path=data/gsm_test.json \
    --prompt_style=fixed_t --latent_length=16 \
    --do_sample=True --sampling_by=noise --noise_std=0.003 \
    --n_samples="$N" --batch_size=4 --max_new_tokens=128 \
    2>&1 | tee "$LOG"
fi
rg -n 'Coverage:|Pass@|Voting Accuracy:' "$LOG" | tee "results/full/${TAG}/gsm_n${N}.txt"
