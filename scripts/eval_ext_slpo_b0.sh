#!/usr/bin/env bash
# Generator axis: SLPO-COCONUT, fixed T=6, no gate. Scorer = LatentTTS B0, dropout off.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GEN="${GEN:-checkpoints/slpo-coconut-gpt2}"
B0=outputs/latentrm_baseline/best
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
R=results/full/ext/slpo_b0
L=logs/ext/slpo_b0
mkdir -p "$R" "$L"

if [[ ! -f "$GEN/model.safetensors" && ! -f "$GEN/pytorch_model.bin" ]]; then
  echo "MISSING $GEN — download first" | tee "$L/master_${STAMP}.log"
  exit 1
fi

bon() {
  local split="$1" n="$2" data="$3" bs="$4"
  local tag="SLPO_B0_${split}_N${n}_dnone_s42"
  local json="$R/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "SKIP $tag"
    return 0
  fi
  echo "======== $tag $(date -Is) ========"
  python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$GEN" \
    --prm_id="$B0" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE $tag" | tee -a "$L/master_${STAMP}.log"
}

echo "STAMP=$STAMP GPU=$CUDA_VISIBLE_DEVICES GEN=$GEN" | tee "$L/master_${STAMP}.log"

python -u -m src.infer_gpt2 \
  --model_type=coconut --model_dtype=bf16 --model_id="$GEN" \
  --data_path=data/gsm_test.json --n_samples=1 --batch_size=16 \
  --latent_length=6 --max_new_tokens=128 --do_sample=False \
  > "$L/det_SLPO_gsm_N1_${STAMP}.log" 2>&1

bon gsm 4 data/gsm_test.json 16
bon gsm 16 data/gsm_test.json 16
bon hard 16 data/gsm_hard.json 16
bon ma 16 data/multiarith.json 16
bon hard 4 data/gsm_hard.json 16
bon ma 4 data/multiarith.json 16
bon gsm 64 data/gsm_test.json 64
bon hard 64 data/gsm_hard.json 64
bon ma 64 data/multiarith.json 64

echo ALL_SLPO_B0_DONE | tee -a "$L/master_${STAMP}.log"
