#!/usr/bin/env bash
# Focused BoN matrix: GSM8K-Test × {O2, B0, official} × N∈{4,16,64}
# Detach-friendly: nohup setsid bash scripts/eval_bon_matrix.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PORT="${MASTER_PORT:-29503}"
SEED="${SEED:-42}"

mkdir -p logs/bon_matrix results/full
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="results/full/bon_gsm_test_${STAMP}.tsv"
MASTER_LOG="logs/bon_matrix/run_${STAMP}.log"

echo -e "prm\tn\tacc\tcov\tvot\tlog" >"$SUMMARY"
echo "summary=$ROOT/$SUMMARY" | tee -a "$MASTER_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_PROCESSES=$NUM_PROCESSES seed=$SEED" | tee -a "$MASTER_LOG"

declare -A PRMS=(
  [O2]="outputs/latentrm_order_pref/best"
  [B0]="outputs/latentrm_baseline/best"
  [official]="checkpoints/latentRM"
)

# N=1 not supported by best_of_n assert (needs >1)
NS=(4 16 64)

for prm_name in O2 B0 official; do
  prm_id="${PRMS[$prm_name]}"
  if [[ ! -e "$prm_id" ]]; then
    echo "SKIP missing $prm_name -> $prm_id" | tee -a "$MASTER_LOG"
    continue
  fi
  for n in "${NS[@]}"; do
    tag="${prm_name}_N${n}"
    run_log="logs/bon_matrix/${tag}_${STAMP}.log"
    echo "======== START $tag $(date -Is) ========" | tee -a "$MASTER_LOG" "$run_log"
    set +e
    accelerate launch --main_process_port "$PORT" --num_processes "$NUM_PROCESSES" \
      -m src.infer_gpt2_rm \
      --generator_type=coconut \
      --prm_id="$prm_id" \
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
    echo "======== END $tag ec=$ec $(date -Is) ========" | tee -a "$MASTER_LOG" "$run_log"

    # parse last Acc/Coverage/Voting from log
    acc=$(grep -E '^Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
    cov=$(grep -E '^Coverage:' "$run_log" | tail -1 | awk '{print $2}' || true)
    vot=$(grep -E '^Voting Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
    echo -e "${prm_name}\t${n}\t${acc:-FAIL}\t${cov:-}\t${vot:-}\t${run_log}" | tee -a "$SUMMARY"
    # bump port to avoid TIME_WAIT collisions between launches
    PORT=$((PORT + 1))
  done
done

echo "DONE summary=$SUMMARY" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
