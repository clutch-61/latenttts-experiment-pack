#!/usr/bin/env bash
# Full valid-64 eval_order for O2 / B0 / B1 (max_batches=0 => entire dataset).
# Single-GPU; queue-friendly.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Prefer a free card if caller set CUDA_VISIBLE_DEVICES; default single GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs/eval_order results/full
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="results/full/eval_order_valid64_full_${STAMP}.tsv"
LOG="logs/eval_order/valid64_full_${STAMP}.log"

echo -e "model\tprm\torder_pair_acc\thierarchy_consistency\tlog" >"$SUMMARY"
echo "summary=$SUMMARY device=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

run_one() {
  local name="$1" prm="$2"
  local run_log="logs/eval_order/${name}_valid64_full_${STAMP}.log"
  echo "======== START $name $(date -Is) ========" | tee -a "$LOG" "$run_log"
  set +e
  python -m src.eval_order_pref \
    --prm_id="$prm" \
    --data_dir=latent-data/coconut/valid-64 \
    --batch_size=16 \
    --max_batches=0 \
    --device=cuda \
    >>"$run_log" 2>&1
  local ec=$?
  set -e
  echo "======== END $name ec=$ec $(date -Is) ========" | tee -a "$LOG" "$run_log"
  local pair hier
  pair=$(grep -oE "'order_pair_acc': [0-9.]+" "$run_log" | tail -1 | awk '{print $2}' || true)
  hier=$(grep -oE "'hierarchy_consistency': [0-9.]+" "$run_log" | tail -1 | awk '{print $2}' || true)
  echo -e "${name}\t${prm}\t${pair:-FAIL}\t${hier:-}\t${run_log}" | tee -a "$SUMMARY"
}

run_one O2 outputs/latentrm_order_pref/best
run_one B0 outputs/latentrm_baseline/best
run_one B1 outputs/latentrm_baseline_bce/best

echo "DONE $SUMMARY" | tee -a "$LOG"
cat "$SUMMARY" | tee -a "$LOG"
