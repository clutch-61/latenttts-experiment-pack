#!/usr/bin/env bash
# Fill GSM8K-Test seed=42 Cov/Voting for N=4/16/64.
# Majority vote does not use PRM ranking; run once with official RM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PORT="${MASTER_PORT:-29525}"
SEED=42
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/bon_mv results/full
SUMMARY="results/full/bon_mv_gsm42_${STAMP}.tsv"
echo -e "split\tn\tseed\tacc\tcov\tvot\tlog" >"$SUMMARY"

for n in 4 16 64; do
  run_log="logs/bon_mv/gsm_test_N${n}_s${SEED}_${STAMP}.log"
  echo "======== START N=${n} $(date -Is) ========" | tee -a "$run_log"
  set +e
  accelerate launch --main_process_port "$PORT" --num_processes "$NUM_PROCESSES" \
    -m src.infer_gpt2_rm \
    --generator_type=coconut \
    --prm_id=checkpoints/latentRM \
    --prm_model_family=gpt2 \
    --prm_mode=best_of_n \
    --data_path=data/gsm_test.json \
    --num_return_sequences="$n" \
    --seed="$SEED" \
    --batch_size=1024 \
    --progress_bar=True \
    >>"$run_log" 2>&1
  ec=$?
  set -e
  echo "======== END N=${n} ec=$ec $(date -Is) ========" | tee -a "$run_log"
  acc=$(grep -E '^Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
  cov=$(grep -E '^Coverage:' "$run_log" | tail -1 | awk '{print $2}' || true)
  vot=$(grep -E '^Voting Accuracy:' "$run_log" | tail -1 | awk '{print $3}' || true)
  echo -e "gsm_test\t${n}\t${SEED}\t${acc:-FAIL}\t${cov:-}\t${vot:-}\t${run_log}" | tee -a "$SUMMARY"
  PORT=$((PORT + 1))
done

echo "DONE $SUMMARY"
cat "$SUMMARY"
