#!/usr/bin/env bash
# Isolate P2: dualBeat warm-start + λ_mmd=0.05, no λ_lat/rank.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG=p2mmd05
OUT="outputs/p123_${TAG}_${STAMP}"
L=logs/p123
mkdir -p "$L" "$OUT"

# student+O2 on physical GPU3; teacher+B0 on physical GPU0
export CUDA_VISIBLE_DEVICES=3,0
echo "STAMP=$STAMP OUT=$OUT" | tee "$L/train_${TAG}_${STAMP}.log"

python -u scripts/train_p123_system.py \
  --ckpt outputs/p123_dualBeat_20261001_202106/model \
  --teacher_ckpt checkpoints/coconut \
  --select_mode dual_beat \
  --sf_mode C \
  --lambda_mmd 0.05 \
  --lambda_lat 0 \
  --lambda_rank 0 \
  --mmd_ramp_steps 100 \
  --k 64 --q_batch 16 --grad_accum 8 \
  --lr 3e-6 --max_steps 1500 --save_every 1500 \
  --frozen_dtype bf16 --auto_pack \
  --frozen_device cuda:1 \
  --outdir "$OUT" \
  >> "$L/train_${TAG}_${STAMP}.log" 2>&1
echo TRAIN_EXIT=$? | tee -a "$L/train_${TAG}_${STAMP}.log"

# eval: official dropout=None (pipeline) on 3+0; paper p=0.2 GSM N16 on GPU3 after
bash scripts/pipeline_p123_eval.sh "$TAG" "$STAMP" "$OUT" "$L/train_${TAG}_${STAMP}.log" $$ 3 0 \
  >> "$L/pipe_${TAG}_${STAMP}.log" 2>&1 || true

GEN="$OUT/model"
O2=outputs/latentrm_order_pref/best
R=results/full/p123
CUDA_VISIBLE_DEVICES=3 python -u -m src.infer_gpt2_rm \
  --generator_type=coconut --generator_id="$GEN" \
  --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/gsm_test.json --num_return_sequences=16 --seed=42 \
  --batch_size=16 --sort_by_len=False --max_new_tokens=128 \
  --claim_agg=top1 --dropout_p=0.2 --model_dtype=bf16 \
  --result_json="$R/bon_${TAG}_O2_N16_dp02_gsm_${STAMP}.json" \
  > "$L/bon_gsm_o2_dp02_${TAG}_${STAMP}.log" 2>&1 || true

echo ALL_P2_CHAIN_DONE | tee -a "$L/train_${TAG}_${STAMP}.log"
