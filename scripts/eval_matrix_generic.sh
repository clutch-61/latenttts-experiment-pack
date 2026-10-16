#!/usr/bin/env bash
# Generic same-protocol BoN matrix: one generator + one scorer, dropout off, seed 42.
# Usage: GEN_TYPE= GEN= PRM= TAG= R= L= GPU= STAMP= bash eval_matrix_generic.sh
#   GEN_TYPE: coconut|codi
#   GEN: generator id/path
#   PRM: scorer id/path (gpt2 family)
#   TAG: filename prefix (e.g. p2mmd05_B0)
#   R / L: results / logs dirs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GEN_TYPE="${GEN_TYPE:-coconut}"
GEN="${GEN:-checkpoints/coconut}"
PRM="${PRM:-outputs/latentrm_baseline/best}"
TAG="${TAG:-GEN_PRM}"
R="${R:-results/full/generic}"
L="${L:-logs/generic}"
GPU="${GPU:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$R" "$L"

bon() {
  local split="$1" n="$2" data="$3" bs="$4"
  local t="${TAG}_${split}_N${n}_dnone_s42"
  local json="$R/bon_${t}_${STAMP}.json"
  if [[ -f "$json" ]]; then echo "SKIP $t"; return 0; fi
  echo "======== GPU$GPU $t $(date -Is) ========" | tee -a "$L/master_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -u -m src.infer_gpt2_rm \
    --generator_type="$GEN_TYPE" --generator_id="$GEN" \
    --prm_id="$PRM" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${t}_${STAMP}.log" 2>&1
  echo "DONE GPU$GPU $t" | tee -a "$L/master_${STAMP}.log"
}

echo "STAMP=$STAMP GPU=$GPU GEN_TYPE=$GEN_TYPE GEN=$GEN PRM=$PRM TAG=$TAG" | tee "$L/master_${STAMP}.log"

bon gsm 4 data/gsm_test.json 16
bon gsm 16 data/gsm_test.json 16
bon gsm 64 data/gsm_test.json 64
bon hard 4 data/gsm_hard.json 16
bon hard 16 data/gsm_hard.json 16
bon hard 64 data/gsm_hard.json 64
bon ma 4 data/multiarith.json 16
bon ma 16 data/multiarith.json 16
bon ma 64 data/multiarith.json 64
echo ALL_DONE_$TAG | tee -a "$L/master_${STAMP}.log"
