#!/usr/bin/env bash
# Fill p2mmd05 paper-protocol matrix (N=1 + BoN N=4/16/64 × gsm/hard/ma, p=0.2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GEN=outputs/p123_p2mmd05_20261004_215700/model
O2=outputs/latentrm_order_pref/best
STAMP="${STAMP:-20261005_131900}"
R=results/full/p123/paper_p2mmd05
L=logs/p123/paper_p2mmd05
mkdir -p "$R" "$L"

bon() {
  local gpu="$1" split="$2" n="$3" data="$4" bs="$5"
  local tag="p2mmd05_O2_${split}_N${n}_dp02_s42"
  local json="$R/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "SKIP $tag" | tee -a "$L/parallel_${STAMP}.log"
    return 0
  fi
  echo "======== GPU$gpu $tag $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$GEN" \
    --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --max_new_tokens=128 --claim_agg=top1 --dropout_p=0.2 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE GPU$gpu $tag" | tee -a "$L/parallel_${STAMP}.log"
}

det() {
  local gpu="$1" split="$2" data="$3"
  echo "======== GPU$gpu det $split $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2 \
    --model_type=coconut --model_id="$GEN" --model_dtype=bf16 \
    --data_path="$data" --n_samples=1 --batch_size=16 \
    --latent_length=6 --max_new_tokens=128 --do_sample=False \
    > "$L/det_p2mmd05_${split}_N1_${STAMP}.log" 2>&1
  echo "DONE GPU$gpu det $split" | tee -a "$L/parallel_${STAMP}.log"
}

echo "STAMP=$STAMP GEN=$GEN" | tee "$L/parallel_${STAMP}.log"

# copy already-run GSM N16 p=0.2 into this folder as symlink for the table
if [[ ! -f "$R/bon_p2mmd05_O2_gsm_N16_dp02_s42_${STAMP}.json" ]]; then
  ln -sf "$(pwd)/results/full/p123/bon_p2mmd05_O2_N16_dp02_gsm_20261004_215700.json" \
    "$R/bon_p2mmd05_O2_gsm_N16_dp02_s42_${STAMP}.json" || true
fi

(
  det 1 gsm data/gsm_test.json
  bon 1 gsm 4 data/gsm_test.json 16
  bon 1 gsm 64 data/gsm_test.json 64
) > "$L/chain_gpu1_${STAMP}.log" 2>&1 &
echo CHAIN1=$!

(
  det 2 hard data/gsm_hard.json
  bon 2 hard 16 data/gsm_hard.json 16
  bon 2 hard 4 data/gsm_hard.json 16
  bon 2 hard 64 data/gsm_hard.json 64
) > "$L/chain_gpu2_${STAMP}.log" 2>&1 &
echo CHAIN2=$!

(
  det 4 ma data/multiarith.json
  bon 4 ma 16 data/multiarith.json 16
  bon 4 ma 4 data/multiarith.json 16
  bon 4 ma 64 data/multiarith.json 64
) > "$L/chain_gpu4_${STAMP}.log" 2>&1 &
echo CHAIN4=$!

# Hard N16 first is on 2; extra N64 gsm already on 1 chain.
wait
echo ALL_P2_MATRIX_DONE | tee -a "$L/parallel_${STAMP}.log"
