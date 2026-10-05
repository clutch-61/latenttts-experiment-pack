#!/usr/bin/env bash
# Paper-protocol baseline: frozen COCONUT + B0, MC-dropout p=0.2.
# Readout in JSON: top1 = Best-of-N; also dump wvote / majority.
# Priority: N=1 det; then Hard/MA N=16; then GSM N=4/16/64.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate latenttts 2>/dev/null || true
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

GEN=checkpoints/coconut
B0=outputs/latentrm_baseline/best
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
R=results/full/p123/paper_b0
L=logs/p123/paper_b0
mkdir -p "$R" "$L"

bon() {
  local split="$1" n="$2" data="$3" bs="$4"
  local tag="A_B0_${split}_N${n}_dp02_s42"
  echo "======== $tag $(date -Is) ========"
  python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$GEN" \
    --prm_id="$B0" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --max_new_tokens=128 --claim_agg=top1 --dropout_p=0.2 --model_dtype=bf16 \
    --result_json="$R/bon_${tag}_${STAMP}.json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE $tag" | tee -a "$L/master_${STAMP}.log"
}

echo "STAMP=$STAMP GPU=$CUDA_VISIBLE_DEVICES" | tee "$L/master_${STAMP}.log"

# N=1 deterministic (no dropout), generator-only
echo "======== A_det gsm_test N1 $(date -Is) ========" | tee -a "$L/master_${STAMP}.log"
python -u -m src.infer_gpt2 \
  --model_type=coconut --model_dtype=bf16 \
  --data_path=data/gsm_test.json --n_samples=1 --batch_size=16 \
  --latent_length=6 --max_new_tokens=128 --do_sample=False \
  > "$L/det_A_gsm_test_N1_${STAMP}.log" 2>&1
echo "DONE det gsm_test" | tee -a "$L/master_${STAMP}.log"

echo "======== A_det gsm_hard N1 $(date -Is) ========" | tee -a "$L/master_${STAMP}.log"
python -u -m src.infer_gpt2 \
  --model_type=coconut --model_dtype=bf16 \
  --data_path=data/gsm_hard.json --n_samples=1 --batch_size=16 \
  --latent_length=6 --max_new_tokens=128 --do_sample=False \
  > "$L/det_A_gsm_hard_N1_${STAMP}.log" 2>&1
echo "DONE det gsm_hard" | tee -a "$L/master_${STAMP}.log"

echo "======== A_det multiarith N1 $(date -Is) ========" | tee -a "$L/master_${STAMP}.log"
python -u -m src.infer_gpt2 \
  --model_type=coconut --model_dtype=bf16 \
  --data_path=data/multiarith.json --n_samples=1 --batch_size=16 \
  --latent_length=6 --max_new_tokens=128 --do_sample=False \
  > "$L/det_A_multiarith_N1_${STAMP}.log" 2>&1
echo "DONE det multiarith" | tee -a "$L/master_${STAMP}.log"

# Hard / MA N=16 first (user priority), then GSM N=16/4/64
bon hard 16 data/gsm_hard.json 16
bon ma 16 data/multiarith.json 16
bon gsm 16 data/gsm_test.json 16
bon gsm 4 data/gsm_test.json 16
bon hard 4 data/gsm_hard.json 16
bon ma 4 data/multiarith.json 16
bon gsm 64 data/gsm_test.json 64
bon hard 64 data/gsm_hard.json 64
bon ma 64 data/multiarith.json 64

echo ALL_PAPER_B0_DONE | tee -a "$L/master_${STAMP}.log"
