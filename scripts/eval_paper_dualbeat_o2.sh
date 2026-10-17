#!/usr/bin/env bash
# Locked system, same protocol as paper A+B0: dualBeat gen + O2, p=0.2, seed 42,
# N=1 det + BoN N=4/16/64, top1/wvote/maj in JSON. Skip GSM N16 (already on disk).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GEN=outputs/p123_dualBeat_20261001_202106/model
O2=outputs/latentrm_order_pref/best
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
R=results/full/p123/paper_dualbeat_o2
L=logs/p123/paper_dualbeat_o2
mkdir -p "$R" "$L"

bon() {
  local gpu="$1" split="$2" n="$3" data="$4" bs="$5"
  local tag="dualBeat_O2_${split}_N${n}_dp02_s42"
  local json="$R/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "SKIP exists $tag" | tee -a "$L/parallel_${STAMP}.log"
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
  local tag="dualBeat_det_${split}_N1"
  echo "======== GPU$gpu $tag $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2 \
    --model_type=coconut --model_id="$GEN" --model_dtype=bf16 \
    --data_path="$data" --n_samples=1 --batch_size=16 \
    --latent_length=6 --max_new_tokens=128 --do_sample=False \
    > "$L/det_${tag}_${STAMP}.log" 2>&1
  echo "DONE GPU$gpu $tag" | tee -a "$L/parallel_${STAMP}.log"
}

echo "STAMP=$STAMP GEN=$GEN O2=$O2" | tee "$L/parallel_${STAMP}.log"

# N=1 on 0/3/7, then BoN chains (skip GSM N16: already have dp02 json)
(
  det 0 gsm data/gsm_test.json
  bon 0 hard 16 data/gsm_hard.json 16
  bon 0 hard 4 data/gsm_hard.json 16
  bon 0 hard 64 data/gsm_hard.json 64
) > "$L/chain_gpu0_${STAMP}.log" 2>&1 &
echo CHAIN0=$!

(
  det 3 ma data/multiarith.json
  bon 3 ma 16 data/multiarith.json 16
  bon 3 ma 4 data/multiarith.json 16
  bon 3 ma 64 data/multiarith.json 64
) > "$L/chain_gpu3_${STAMP}.log" 2>&1 &
echo CHAIN3=$!

(
  det 7 hard data/gsm_hard.json
  bon 7 gsm 4 data/gsm_test.json 16
  bon 7 gsm 64 data/gsm_test.json 64
) > "$L/chain_gpu7_${STAMP}.log" 2>&1 &
echo CHAIN7=$!

wait
echo ALL_PAPER_DUALBEAT_O2_DONE | tee -a "$L/parallel_${STAMP}.log"
