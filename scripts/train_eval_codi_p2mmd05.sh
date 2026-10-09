#!/usr/bin/env bash
# CODI init + dual_beat SF-C + λ_mmd=0.05. Teacher = CODI (not coconut).
# Kill: GSM N16 dropout-off O2 top1 must beat frozen CODI+B0 40.03; collapse → discard.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-outputs/p123_codi_p2mmd05_${STAMP}}"
LOG=logs/p123/train_codi_p2mmd05_${STAMP}.log
mkdir -p logs/p123 "$OUTDIR"

# CUDA_VISIBLE_DEVICES: student+O2 = :0, teacher+B0 = :1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3}"

python -u scripts/train_p123_system.py \
  --generator_type=codi \
  --ckpt checkpoints/codi --teacher_ckpt checkpoints/codi \
  --select_mode dual_beat --sf_mode C \
  --lambda_mmd 0.05 --lambda_lat 0 --lambda_rank 0 \
  --mmd_ramp_steps 100 --dropout_p 0.2 \
  --k 64 --q_batch 16 --grad_accum 8 \
  --lr 3e-6 --max_steps 1500 --save_every 1500 \
  --frozen_dtype bf16 --frozen_device cuda:1 --auto_pack \
  --outdir "$OUTDIR" \
  > "$LOG" 2>&1

echo TRAIN_DONE "$OUTDIR" | tee -a "$LOG"

# Eval: dropout off, O2, GSM N16 first (kill metric)
python -u -m src.infer_gpt2_rm \
  --generator_type=codi --generator_id="$OUTDIR/model" \
  --prm_id=outputs/latentrm_order_pref/best --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/gsm_test.json --num_return_sequences=16 --seed=42 \
  --batch_size=16 --sort_by_len=False \
  --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
  --result_json="results/full/ext/codi_ours/bon_CODIp2_O2_gsm_N16_dnone_s42_${STAMP}.json" \
  >> "$LOG" 2>&1

echo EVAL_GSM16_DONE | tee -a "$LOG"
