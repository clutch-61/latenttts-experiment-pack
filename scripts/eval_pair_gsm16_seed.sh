#!/usr/bin/env bash
# Same-pool paired eval: given one generator + one seed, run B0 and O2 on the SAME
# candidate pool (identical gen+seed+batch+sort_by_len => identical pool; scorer only
# changes ranking/top1). Used to test "B0 validation failure after p2 drift".
# Usage: TAG= GEN_TYPE= GEN= SEED= GPU= [STAMP=] bash eval_pair_gsm16_seed.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG="${TAG:-gen}"
GEN_TYPE="${GEN_TYPE:-coconut}"
GEN="${GEN:-checkpoints/coconut}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
R="results/full/paired_seeds"
mkdir -p "$R" "logs/paired_seeds"

for entry in "B0:outputs/latentrm_baseline/best" "O2:outputs/latentrm_order_pref/best"; do
  name="${entry%%:*}"; prm="${entry##*:}"
  t="${TAG}_${name}_gsm_N16_s${SEED}"
  json="$R/bon_${t}_${STAMP}.json"
  log="logs/paired_seeds/${t}_${STAMP}.log"
  if [[ -f "$json" ]]; then echo "SKIP $t"; continue; fi
  echo "==== GPU$GPU $t $(date -Is) ===="
  CUDA_VISIBLE_DEVICES="$GPU" python -u -m src.infer_gpt2_rm \
    --generator_type="$GEN_TYPE" --generator_id="$GEN" \
    --prm_id="$prm" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path=data/gsm_test.json --num_return_sequences=16 --seed="$SEED" \
    --batch_size=16 --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" > "$log" 2>&1
  echo "DONE GPU$GPU $t"
done
echo "PAIR_DONE $TAG s$SEED"
