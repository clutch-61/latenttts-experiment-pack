#!/usr/bin/env bash
# Wait for P123 train, then: gen / +O2 / +B0 / pilot+O2 / mid diag vs A+B0.
# Usage: pipeline_p123_eval.sh TAG STAMP OUTDIR TRAIN_LOG TRAIN_PID [GPU_A] [GPU_B]
set -uo pipefail
TAG=$1; STAMP=$2; OUT=$3; TRAIN_LOG=$4; TRAIN_PID=$5
GPU_A=${6:-5}
GPU_B=${7:-6}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

while ! grep -q '^DONE steps=' "$TRAIN_LOG" 2>/dev/null; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 5
    grep -q '^DONE steps=' "$TRAIN_LOG" && break
    echo TRAIN_DIED; tail -40 "$TRAIN_LOG"; exit 1
  fi
  sleep 30
done
echo TRAIN_DONE
grep '^DONE' "$TRAIN_LOG" | tail -1

CKPT=$OUT/model
R=results/full/p123
L=logs/p123
mkdir -p "$R" "$L"
O2=outputs/latentrm_order_pref/best
B0=outputs/latentrm_baseline/best
if [[ -d "$OUT/o2" ]]; then
  O2="$OUT/o2"
  echo "using trained O2 $O2"
fi

# Generator-only (pilot + gsm_test subsample via full gsm_valid for speed; gsm_test in second wave)
CUDA_VISIBLE_DEVICES=$GPU_A python scripts/eval_sf_generator.py --model_id "$CKPT" \
  --data_path data/sf_pilot/valid.json --n_samples 4 --seed 42 --batch_size 8 \
  --out_json "$R/${TAG}_pilot_gen_${STAMP}.json" > "$L/eval_pilot_${TAG}_${STAMP}.log" 2>&1 &
PID_GEN_P=$!

CUDA_VISIBLE_DEVICES=$GPU_A python -m src.infer_gpt2_rm --generator_type=coconut --generator_id="$CKPT" \
  --prm_id=$O2 --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/sf_pilot/valid.json --num_return_sequences=4 --seed=42 --batch_size=32 --sort_by_len=False \
  --max_new_tokens=128 --claim_agg=wvote \
  --result_json="$R/bon_${TAG}_O2_N4_pilot_${STAMP}.json" > "$L/bon_pilot_o2_${TAG}_${STAMP}.log" 2>&1 &
PID_O2_P=$!

wait $PID_GEN_P $PID_O2_P

# GSM-Test N=16 system O2 + health B0 (two free cards)
# dropout_p omit → model default (matches locked dualBeat / A+B0 dumps; do not force 0.2)
CUDA_VISIBLE_DEVICES=$GPU_A python -m src.infer_gpt2_rm --generator_type=coconut --generator_id="$CKPT" \
  --prm_id=$O2 --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/gsm_test.json --num_return_sequences=16 --seed=42 --batch_size=16 --sort_by_len=False \
  --max_new_tokens=128 --claim_agg=wvote \
  --result_json="$R/bon_${TAG}_O2_N16_gsm_${STAMP}.json" > "$L/bon_gsm_o2_${TAG}_${STAMP}.log" 2>&1 &
PID_O2=$!
CUDA_VISIBLE_DEVICES=$GPU_B python -m src.infer_gpt2_rm --generator_type=coconut --generator_id="$CKPT" \
  --prm_id=$B0 --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/gsm_test.json --num_return_sequences=16 --seed=42 --batch_size=16 --sort_by_len=False \
  --max_new_tokens=128 --claim_agg=wvote \
  --result_json="$R/bon_${TAG}_B0_N16_gsm_${STAMP}.json" > "$L/bon_gsm_b0_${TAG}_${STAMP}.log" 2>&1 &
PID_B0=$!
wait $PID_O2 $PID_B0

# Prefer recent A+B0 dump if present
BASE_AB0=$(ls -t results/full/sf_full/bon_*_B0_N16_gsm_*.json results/full/bon_*B0*N16*gsm*.json 2>/dev/null | head -1 || true)
# Also try artifacts from last S0
if [[ -z "${BASE_AB0:-}" ]]; then
  BASE_AB0=$(ls -t results/full/sf_full/*A*B0*N16* 2>/dev/null | head -1 || true)
fi

python scripts/diag_p123_mids.py \
  --system_o2 "$R/bon_${TAG}_O2_N16_gsm_${STAMP}.json" \
  --system_b0 "$R/bon_${TAG}_B0_N16_gsm_${STAMP}.json" \
  --baseline_ab0 "${BASE_AB0:-}" \
  --gen_only "$R/${TAG}_pilot_gen_${STAMP}.json" \
  --pilot_o2 "$R/bon_${TAG}_O2_N4_pilot_${STAMP}.json" \
  --mid_jsonl "$OUT/mid_metrics.jsonl" \
  --a_b0_acc 0.3374 \
  --out_json "$R/mids_${TAG}_${STAMP}.json"

echo "=== kill criteria vs dualBeat (wvote ≥+1pp; top ≥+0.5 secondary) ==="
python scripts/summarize_bon.py \
  dualBeat_O2=results/full/p123/bon_dualBeat_O2_N16_gsm_20261001_202106.json \
  "${TAG}_O2=$R/bon_${TAG}_O2_N16_gsm_${STAMP}.json" || true

echo PIPELINE_DONE
cat "$R/mids_${TAG}_${STAMP}.json"
