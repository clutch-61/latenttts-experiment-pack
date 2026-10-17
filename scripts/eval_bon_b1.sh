#!/usr/bin/env bash
# B1 BoN matrix: {gsm_test, hard, multiarith} × N∈{4,16,64} × seed=42
# PRM = outputs/latentrm_baseline_bce/best
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PORT="${MASTER_PORT:-29540}"
SEED="${SEED:-42}"
PRM_ID="outputs/latentrm_baseline_bce/best"

mkdir -p logs/bon_b1 results/full
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="results/full/bon_b1_${STAMP}.tsv"
MASTER_LOG="logs/bon_b1/run_${STAMP}.log"

echo -e "split\tprm\tn\tseed\tacc\tcov\tvot\tlog" >"$SUMMARY"
echo "summary=$ROOT/$SUMMARY prm=$PRM_ID" | tee -a "$MASTER_LOG"

declare -A DATA=(
  [gsm_test]="data/gsm_test.json"
  [hard]="data/gsm_hard.json"
  [multiarith]="data/multiarith.json"
)

for split in gsm_test hard multiarith; do
  data_path="${DATA[$split]}"
  for n in 4 16 64; do
    tag="${split}_B1_N${n}_s${SEED}"
    run_log="logs/bon_b1/${tag}_${STAMP}.log"
    echo "======== START $tag $(date -Is) ========" | tee -a "$MASTER_LOG" "$run_log"
    set +e
    accelerate launch --main_process_port "$PORT" --num_processes "$NUM_PROCESSES" \
      -m src.infer_gpt2_rm \
      --generator_type=coconut \
      --prm_id="$PRM_ID" \
      --prm_model_family=gpt2 \
      --prm_mode=best_of_n \
      --data_path="$data_path" \
      --num_return_sequences="$n" \
      --seed="$SEED" \
      --batch_size=1024 \
      --progress_bar=True \
      >>"$run_log" 2>&1
    ec=$?
    set -e
    echo "======== END $tag ec=$ec $(date -Is) ========" | tee -a "$MASTER_LOG" "$run_log"
    acc=$(grep -E '^Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
    cov=$(grep -E '^Coverage:' "$run_log" | tail -1 | awk '{print $2}' || true)
    vot=$(grep -E '^Voting Accuracy:' "$run_log" | tail -1 | awk '{print $3}' || true)
    echo -e "${split}\tB1\t${n}\t${SEED}\t${acc:-FAIL}\t${cov:-}\t${vot:-}\t${run_log}" | tee -a "$SUMMARY"
    PORT=$((PORT + 1))
  done
done

echo "DONE summary=$SUMMARY" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
