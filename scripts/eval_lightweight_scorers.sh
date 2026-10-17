#!/usr/bin/env bash
# SWIFT-Latent / HSRM-Latent BoN N=16, 3 seeds, frozen COCONUT.
# Same NEW-protocol rig as B0 (eval_ext_baselines.sh): batch_size=16, sort_by_len=False,
# claim_agg=top1, dnone (dropout_p=null), model_dtype=bf16.
# Usage: SCORER=swift GPU=3 bash scripts/eval_lightweight_scorers.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCORER="${SCORER:?}"   # swift | hsrm
GPU="${GPU:?}"
MODEL="${MODEL:-checkpoints/coconut}"
PRM_ID="outputs/${SCORER}_latent/best.pt"
R="results/full/lightweight"
L="logs/lightweight"
mkdir -p "$R" "$L"
STAMP="$(date +%Y%m%d_%H%M%S)"

for s in 42 43 44; do
  json="$R/bon_${SCORER}_gsm_N16_dnone_s${s}_${STAMP}.json"
  if [[ -f "$json" ]]; then echo "SKIP seed $s"; continue; fi
  echo "======== GPU$GPU ${SCORER} seed=$s $(date -Is) ========" | tee -a "$L/eval_${SCORER}_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$MODEL" \
    --prm_id="$PRM_ID" --prm_model_family="$SCORER" --prm_mode=best_of_n \
    --data_path=data/gsm_test.json --num_return_sequences=16 --seed=$s \
    --batch_size=16 --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    >> "$L/eval_${SCORER}_${STAMP}.log" 2>&1
  echo "DONE GPU$GPU ${SCORER} seed=$s" | tee -a "$L/eval_${SCORER}_${STAMP}.log"
done
echo "ALL DONE ${SCORER}" | tee -a "$L/eval_${SCORER}_${STAMP}.log"
