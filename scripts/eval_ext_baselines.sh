#!/usr/bin/env bash
# Unified NEW-protocol eval for ext RL baselines: B0 scorer, N=16.
# Usage: MODEL= TAG= GPU= [SEEDS="42 43 44 45 46"] bash eval_ext_baselines.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:?}"
TAG="${TAG:?}"
GPU="${GPU:?}"
SEEDS="${SEEDS:-42 43 44 45 46}"
R="results/full/ext_baselines"
L="logs/ext_baselines"
mkdir -p "$R" "$L"
STAMP="$(date +%Y%m%d_%H%M%S)"

for s in $SEEDS; do
  json="$R/bon_${TAG}_B0_gsm_N16_dnone_s${s}.json"
  if ! compgen -G "$R/bon_${TAG}_B0_gsm_N16_dnone_s${s}_*.json" >/dev/null; then
    echo "======== GPU$GPU ${TAG} seed=$s $(date -Is) ========" | tee -a "$L/eval_${TAG}_${STAMP}.log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m src.infer_gpt2_rm \
      --generator_type=coconut --generator_id="$MODEL" \
      --prm_id=outputs/latentrm_baseline/best --prm_model_family=gpt2 --prm_mode=best_of_n \
      --data_path=data/gsm_test.json --num_return_sequences=16 --seed=$s \
      --batch_size=16 --sort_by_len=False \
      --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
      --result_json="${json%.json}_${STAMP}.json" \
      >> "$L/eval_${TAG}_${STAMP}.log" 2>&1
    echo "DONE GPU$GPU ${TAG} seed=$s" | tee -a "$L/eval_${TAG}_${STAMP}.log"
  else
    echo "SKIP ${TAG} seed $s"
  fi
done
echo "ALL DONE ${TAG}" | tee -a "$L/eval_${TAG}_${STAMP}.log"
