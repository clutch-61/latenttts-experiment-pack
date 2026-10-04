#!/usr/bin/env bash
# Wait for an sf_full train to finish, then run generator / +O2 / diag / gsm_valid evals and summarize.
# Usage: pipeline_sf_full_eval.sh TAG STAMP OUTDIR TRAIN_LOG TRAIN_PID
set -uo pipefail
TAG=$1; STAMP=$2; OUT=$3; TRAIN_LOG=$4; TRAIN_PID=$5
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

while ! grep -q '^DONE steps=' "$TRAIN_LOG" 2>/dev/null; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 5
    grep -q '^DONE steps=' "$TRAIN_LOG" && break
    echo TRAIN_DIED; tail -30 "$TRAIN_LOG"; exit 1
  fi
  sleep 20
done
echo TRAIN_DONE
grep '^DONE' "$TRAIN_LOG" | tail -1

CKPT=$OUT/model
R=results/full/sf_full
L=logs/sf_full
CUDA_VISIBLE_DEVICES=6 python scripts/eval_sf_generator.py --model_id "$CKPT" --data_path data/sf_pilot/valid.json \
  --n_samples 4 --seed 42 --batch_size 8 --out_json $R/${TAG}_pilotvalid_${STAMP}.json > $L/eval_${TAG}_pilot_${STAMP}.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -m src.infer_gpt2_rm --generator_type=coconut --generator_id="$CKPT" \
  --prm_id=outputs/latentrm_order_pref/best --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/sf_pilot/valid.json --num_return_sequences=4 --seed=42 --batch_size=32 --sort_by_len=False \
  --result_json=$R/bon_${TAG}_O2_N4_pilot_${STAMP}.json > $L/bon_${TAG}_pilot_${STAMP}.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python scripts/diag_sf_transfer.py --a_id checkpoints/coconut \
  --b_id outputs/sf_full_O2gateT_k4_m0.5_20260930_231108/model --c_id "$CKPT" \
  --data_path data/sf_pilot/valid.json --n_samples 4 --seed 42 --batch_size 8 \
  --outdir $R/diag_${TAG}_${STAMP} > $L/diag_${TAG}_${STAMP}.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python scripts/eval_sf_generator.py --model_id "$CKPT" --data_path data/gsm_valid.json \
  --n_samples 4 --seed 42 --batch_size 8 --out_json $R/${TAG}_gsmvalid_${STAMP}.json > $L/eval_${TAG}_gsm_${STAMP}.log 2>&1 &
wait
echo ALL_EVAL_DONE
python scripts/summarize_sf_full.py --tag "$TAG" --stamp "$STAMP" --outdir "$OUT"
echo PIPELINE_DONE
