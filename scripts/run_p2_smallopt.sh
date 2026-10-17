#!/usr/bin/env bash
# (1) seed 43/44 GSM N16 drop=None p2 vs dualBeat
# (2) short continue: train dropout 0.05 from p2mmd05, then eval GSM N16 none
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
P2=outputs/p123_p2mmd05_20261004_215700/model
DB=outputs/p123_dualBeat_20261001_202106/model
O2=outputs/latentrm_order_pref/best
R=results/full/p123
L=logs/p123
mkdir -p "$R" "$L"

bon_none() {
  local gpu="$1" tag="$2" gen="$3" seed="$4"
  local json="$R/bon_${tag}_O2_N16_dnone_s${seed}_${STAMP}.json"
  echo "======== GPU$gpu $tag seed=$seed $(date -Is) ========" | tee -a "$L/smallopt_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$gen" \
    --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path=data/gsm_test.json --num_return_sequences=16 --seed="$seed" \
    --batch_size=16 --sort_by_len=False --max_new_tokens=128 \
    --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_N16_dnone_s${seed}_${STAMP}.log" 2>&1
  echo "DONE $tag s$seed" | tee -a "$L/smallopt_${STAMP}.log"
}

echo "STAMP=$STAMP" | tee "$L/smallopt_${STAMP}.log"

# avoid GPU6; 3/7/1/2 for seeds (colocate OK)
bon_none 3 p2 "$P2" 43 &
P43=$!
bon_none 7 p2 "$P2" 44 &
P44=$!
bon_none 1 db "$DB" 43 &
D43=$!
bon_none 2 db "$DB" 44 &
D44=$!

# train on GPU0 (empty-ish 89GB)
OUT="outputs/p123_p2dp05_${STAMP}"
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_p123_system.py \
  --ckpt "$P2" --teacher_ckpt checkpoints/coconut \
  --select_mode dual_beat --sf_mode C \
  --lambda_mmd 0.05 --lambda_lat 0 --lambda_rank 0 \
  --mmd_ramp_steps 50 --dropout_p 0.05 \
  --k 64 --q_batch 16 --grad_accum 8 \
  --lr 3e-6 --max_steps 800 --save_every 800 \
  --frozen_dtype bf16 --auto_pack \
  --outdir "$OUT" \
  > "$L/train_p2dp05_${STAMP}.log" 2>&1
echo TRAIN_DP05_EXIT=$? | tee -a "$L/smallopt_${STAMP}.log"

wait $P43 $P44 $D43 $D44 || true

# eval new ckpt GSM N16 none on GPU0 after train
if [[ -d "$OUT/model" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$OUT/model" \
    --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path=data/gsm_test.json --num_return_sequences=16 --seed=42 \
    --batch_size=16 --sort_by_len=False --max_new_tokens=128 \
    --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$R/bon_p2dp05_O2_N16_dnone_s42_${STAMP}.json" \
    > "$L/bon_p2dp05_N16_dnone_s42_${STAMP}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python -u -m src.infer_gpt2 \
    --model_type=coconut --model_id="$OUT/model" --model_dtype=bf16 \
    --data_path=data/gsm_test.json --n_samples=1 --batch_size=16 \
    --latent_length=6 --max_new_tokens=128 --do_sample=False \
    > "$L/det_p2dp05_gsm_N1_${STAMP}.log" 2>&1
fi
echo ALL_SMALLOPT_DONE | tee -a "$L/smallopt_${STAMP}.log"
