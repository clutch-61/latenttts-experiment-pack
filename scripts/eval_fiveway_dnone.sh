#!/usr/bin/env bash
# dropout=None BoN matrix for dualBeat and p2mmd05 (official-lock protocol).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="${STAMP:-20261005_132900}"
R=results/full/p123/fiveway
L=logs/p123/fiveway
mkdir -p "$R" "$L"
O2=outputs/latentrm_order_pref/best

bon() {
  local gpu="$1" sys="$2" gen="$3" split="$4" n="$5" data="$6" bs="$7"
  local tag="${sys}_O2_${split}_N${n}_dnone_s42"
  local json="$R/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "SKIP $tag" | tee -a "$L/parallel_${STAMP}.log"
    return 0
  fi
  echo "======== GPU$gpu $tag $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$gen" \
    --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE GPU$gpu $tag" | tee -a "$L/parallel_${STAMP}.log"
}

DB=outputs/p123_dualBeat_20261001_202106/model
P2=outputs/p123_p2mmd05_20261004_215700/model

echo "STAMP=$STAMP" | tee "$L/parallel_${STAMP}.log"

# GPU3: dualBeat missing Hard/N4 (GSM N16/N64 already on disk)
(
  bon 3 db "$DB" hard 16 data/gsm_hard.json 16
  bon 3 db "$DB" hard 4 data/gsm_hard.json 16
  bon 3 db "$DB" hard 64 data/gsm_hard.json 64
  bon 3 db "$DB" gsm 4 data/gsm_test.json 16
) > "$L/chain_gpu3_${STAMP}.log" 2>&1 &
echo CHAIN3=$!

# GPU7: dualBeat MA (avoid GPU6)
(
  bon 7 db "$DB" ma 16 data/multiarith.json 16
  bon 7 db "$DB" ma 4 data/multiarith.json 16
  bon 7 db "$DB" ma 64 data/multiarith.json 64
) > "$L/chain_gpu7_${STAMP}.log" 2>&1 &
echo CHAIN7=$!

# GPU0: p2 dropout=None Hard → MA → GSM N4/N64 (N16 exists)
(
  bon 0 p2 "$P2" hard 16 data/gsm_hard.json 16
  bon 0 p2 "$P2" hard 4 data/gsm_hard.json 16
  bon 0 p2 "$P2" hard 64 data/gsm_hard.json 64
  bon 0 p2 "$P2" ma 16 data/multiarith.json 16
  bon 0 p2 "$P2" ma 4 data/multiarith.json 16
  bon 0 p2 "$P2" ma 64 data/multiarith.json 64
  bon 0 p2 "$P2" gsm 4 data/gsm_test.json 16
  bon 0 p2 "$P2" gsm 64 data/gsm_test.json 64
) > "$L/chain_gpu0_${STAMP}.log" 2>&1 &
echo CHAIN0=$!

wait
echo ALL_DNONE_DONE | tee -a "$L/parallel_${STAMP}.log"
