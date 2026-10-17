#!/usr/bin/env bash
# Single-GPU GSM N=16 seed=42 dumps for O2/O1/B1, then paired stats.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export NUM_PROCESSES=1

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/full/bon_paired_gsm16_${STAMP}"
mkdir -p "$OUTDIR" logs/bon_paired
MASTER="logs/bon_paired/run_${STAMP}.log"
PORT="${MASTER_PORT:-29580}"

run_one() {
  local name="$1" prm="$2"
  local json="$OUTDIR/${name}.json"
  local log="logs/bon_paired/${name}_${STAMP}.log"
  echo "======== START $name $(date -Is) ========" | tee -a "$MASTER" "$log"
  set +e
  accelerate launch --main_process_port "$PORT" --num_processes 1 \
    -m src.infer_gpt2_rm \
    --generator_type=coconut \
    --prm_id="$prm" \
    --prm_model_family=gpt2 \
    --prm_mode=best_of_n \
    --data_path=data/gsm_test.json \
    --num_return_sequences=16 \
    --seed=42 \
    --batch_size=1024 \
    --progress_bar=True \
    --result_json="$json" \
    >>"$log" 2>&1
  local ec=$?
  set -e
  echo "======== END $name ec=$ec $(date -Is) ========" | tee -a "$MASTER" "$log"
  PORT=$((PORT + 1))
}

run_one O2 outputs/latentrm_order_pref/best
run_one O1 outputs/latentrm_order_only/best
run_one B1 outputs/latentrm_baseline_bce/best

echo "==== pair stats $(date -Is) ====" | tee -a "$MASTER"
python scripts/pair_bon_stats.py \
  --o2 "$OUTDIR/O2.json" \
  --o1 "$OUTDIR/O1.json" \
  --b1 "$OUTDIR/B1.json" \
  --out "$OUTDIR/pair_summary.json" | tee -a "$MASTER"

echo "DONE outdir=$OUTDIR" | tee -a "$MASTER"
