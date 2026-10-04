#!/usr/bin/env bash
# B1 GSM-Test BoN N=16 × inference seeds {43,44} (seed 42 already in bon_b1 TSV).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PORT="${MASTER_PORT:-29560}"
PRM_ID="outputs/latentrm_baseline_bce/best"

mkdir -p logs/bon_b1 results/full
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="results/full/bon_b1_gsm_n16_seeds_${STAMP}.tsv"
MASTER="logs/bon_b1/seeds_${STAMP}.log"
echo -e "split\tprm\tn\tseed\tacc\tcov\tvot\tlog" >"$SUMMARY"
echo "note=inference_seeds_only fixed_checkpoint=$PRM_ID" | tee -a "$MASTER"

for seed in 43 44; do
  tag="gsm_test_B1_N16_s${seed}"
  run_log="logs/bon_b1/${tag}_${STAMP}.log"
  echo "======== START $tag $(date -Is) ========" | tee -a "$MASTER" "$run_log"
  set +e
  accelerate launch --main_process_port "$PORT" --num_processes "$NUM_PROCESSES" \
    -m src.infer_gpt2_rm \
    --generator_type=coconut \
    --prm_id="$PRM_ID" \
    --prm_model_family=gpt2 \
    --prm_mode=best_of_n \
    --data_path=data/gsm_test.json \
    --num_return_sequences=16 \
    --seed="$seed" \
    --batch_size=1024 \
    --progress_bar=True \
    >>"$run_log" 2>&1
  ec=$?
  set -e
  echo "======== END $tag ec=$ec $(date -Is) ========" | tee -a "$MASTER" "$run_log"
  acc=$(grep -E '^Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
  cov=$(grep -E '^Coverage:' "$run_log" | tail -1 | awk '{print $2}' || true)
  vot=$(grep -E '^Voting Accuracy:' "$run_log" | tail -1 | awk '{print $3}' || true)
  echo -e "gsm_test\tB1\t16\t${seed}\t${acc:-FAIL}\t${cov:-}\t${vot:-}\t${run_log}" | tee -a "$SUMMARY"
  PORT=$((PORT + 1))
done

echo "DONE $SUMMARY" | tee -a "$MASTER"
# append prior seed42 row for convenience
echo -e "gsm_test\tB1\t16\t42\t32.1456%\t51.0993%\t34.3442%\tlogs/bon_b1/gsm_test_B1_N16_s42_20260928_152912.log (prior)" | tee -a "$SUMMARY"
cat "$SUMMARY" | tee -a "$MASTER"
