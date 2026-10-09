#!/usr/bin/env bash
# Generator axis: LatentTTS-CODI GPT-2, T=6. Scorer = B0, dropout off.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

GEN="${GEN:-checkpoints/codi}"
B0=outputs/latentrm_baseline/best
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
R=results/full/ext/codi_b0
L=logs/ext/codi_b0
mkdir -p "$R" "$L"

if [[ ! -f "$GEN/model.safetensors" && ! -f "$GEN/pytorch_model.bin" ]]; then
  echo "MISSING $GEN" | tee "$L/master_${STAMP}.log"
  exit 1
fi

bon() {
  local split="$1" n="$2" data="$3" bs="$4"
  local tag="CODI_B0_${split}_N${n}_dnone_s42"
  local json="$R/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then echo "SKIP $tag"; return 0; fi
  echo "======== $tag $(date -Is) ========"
  python -u -m src.infer_gpt2_rm \
    --generator_type=codi --generator_id="$GEN" \
    --prm_id="$B0" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE $tag" | tee -a "$L/master_${STAMP}.log"
}

echo "STAMP=$STAMP GPU=$CUDA_VISIBLE_DEVICES GEN=$GEN" | tee "$L/master_${STAMP}.log"
bon gsm 4 data/gsm_test.json 16
bon gsm 16 data/gsm_test.json 16
bon hard 16 data/gsm_hard.json 16
bon ma 16 data/multiarith.json 16
bon gsm 64 data/gsm_test.json 64
bon hard 4 data/gsm_hard.json 16
bon ma 4 data/multiarith.json 16
bon hard 64 data/gsm_hard.json 64
bon ma 64 data/multiarith.json 64
echo ALL_CODI_B0_DONE | tee -a "$L/master_${STAMP}.log"
