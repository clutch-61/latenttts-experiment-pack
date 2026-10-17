#!/usr/bin/env bash
# Parallel leftover paper-B0 jobs. Hard N16 already running on GPU7.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GEN=checkpoints/coconut
B0=outputs/latentrm_baseline/best
STAMP="${STAMP:-20261004_203149}"
R=results/full/p123/paper_b0
L=logs/p123/paper_b0
mkdir -p "$R" "$L"

bon() {
  local gpu="$1" split="$2" n="$3" data="$4" bs="$5"
  local tag="A_B0_${split}_N${n}_dp02_s42"
  local json="$R/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "SKIP exists $tag" | tee -a "$L/parallel_${STAMP}.log"
    return 0
  fi
  echo "======== GPU$gpu $tag $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$GEN" \
    --prm_id="$B0" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --max_new_tokens=128 --claim_agg=top1 --dropout_p=0.2 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE GPU$gpu $tag" | tee -a "$L/parallel_${STAMP}.log"
}

# GPU3: Hard/MA first remaining (MA N16), then MA 4/64
(
  bon 3 ma 16 data/multiarith.json 16
  bon 3 ma 4 data/multiarith.json 16
  bon 3 ma 64 data/multiarith.json 64
) > "$L/chain_gpu3_${STAMP}.log" 2>&1 &
echo CHAIN3=$!

# GPU0: GSM N16 then 4 then 64
(
  bon 0 gsm 16 data/gsm_test.json 16
  bon 0 gsm 4 data/gsm_test.json 16
  bon 0 gsm 64 data/gsm_test.json 64
) > "$L/chain_gpu0_${STAMP}.log" 2>&1 &
echo CHAIN0=$!

# GPU4: Hard N4/N64 (N16 is on GPU7)
(
  bon 4 hard 4 data/gsm_hard.json 16
  bon 4 hard 64 data/gsm_hard.json 64
) > "$L/chain_gpu4_${STAMP}.log" 2>&1 &
echo CHAIN4=$!

wait
echo ALL_PARALLEL_DONE | tee -a "$L/parallel_${STAMP}.log"
