#!/usr/bin/env bash
# Stability BoN (post Phase-1 soft signal):
#   A) Hard + MultiArith × {O2,B0,official} × N∈{4,16,64}  seed=42
#   B) GSM8K-Test N=16 × seeds 43,44 × {O2,B0,official}
# Detach: nohup setsid bash scripts/eval_bon_stability.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PORT="${MASTER_PORT:-29520}"

mkdir -p logs/bon_stability results/full
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="results/full/bon_stability_${STAMP}.tsv"
MASTER_LOG="logs/bon_stability/run_${STAMP}.log"

echo -e "split\tprm\tn\tseed\tacc\tcov\tvot\tlog" >"$SUMMARY"
echo "summary=$ROOT/$SUMMARY" | tee -a "$MASTER_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_PROCESSES=$NUM_PROCESSES" | tee -a "$MASTER_LOG"

declare -A PRMS=(
  [O2]="outputs/latentrm_order_pref/best"
  [B0]="outputs/latentrm_baseline/best"
  [official]="checkpoints/latentRM"
)

run_one() {
  local split="$1" data_path="$2" prm_name="$3" n="$4" seed="$5"
  local prm_id="${PRMS[$prm_name]}"
  local tag="${split}_${prm_name}_N${n}_s${seed}"
  local run_log="logs/bon_stability/${tag}_${STAMP}.log"

  if [[ ! -e "$prm_id" ]]; then
    echo "SKIP missing $prm_name -> $prm_id" | tee -a "$MASTER_LOG"
    echo -e "${split}\t${prm_name}\t${n}\t${seed}\tSKIP\t\t\t" | tee -a "$SUMMARY"
    return 0
  fi
  if [[ ! -f "$data_path" ]]; then
    echo "SKIP missing data $data_path" | tee -a "$MASTER_LOG"
    echo -e "${split}\t${prm_name}\t${n}\t${seed}\tNODATA\t\t\t" | tee -a "$SUMMARY"
    return 0
  fi

  echo "======== START $tag $(date -Is) ========" | tee -a "$MASTER_LOG" "$run_log"
  set +e
  accelerate launch --main_process_port "$PORT" --num_processes "$NUM_PROCESSES" \
    -m src.infer_gpt2_rm \
    --generator_type=coconut \
    --prm_id="$prm_id" \
    --prm_model_family=gpt2 \
    --prm_mode=best_of_n \
    --data_path="$data_path" \
    --num_return_sequences="$n" \
    --seed="$seed" \
    --batch_size=1024 \
    --progress_bar=True \
    >>"$run_log" 2>&1
  local ec=$?
  set -e
  echo "======== END $tag ec=$ec $(date -Is) ========" | tee -a "$MASTER_LOG" "$run_log"

  local acc cov vot
  acc=$(grep -E '^Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
  cov=$(grep -E '^Coverage:' "$run_log" | tail -1 | awk '{print $2}' || true)
  vot=$(grep -E '^Voting Accuracy:' "$run_log" | tail -1 | awk '{print $2}' || true)
  echo -e "${split}\t${prm_name}\t${n}\t${seed}\t${acc:-FAIL}\t${cov:-}\t${vot:-}\t${run_log}" | tee -a "$SUMMARY"
  PORT=$((PORT + 1))
}

# ---- A: Hard / MultiArith full N grid, seed=42 ----
for split_data in "hard:data/gsm_hard.json" "multiarith:data/multiarith.json"; do
  split="${split_data%%:*}"
  data_path="${split_data#*:}"
  for prm_name in O2 B0 official; do
    for n in 4 16 64; do
      run_one "$split" "$data_path" "$prm_name" "$n" 42
    done
  done
done

# ---- B: GSM-Test N=16 multi-seed (best-N check) ----
for seed in 43 44; do
  for prm_name in O2 B0 official; do
    run_one "gsm_test" "data/gsm_test.json" "$prm_name" 16 "$seed"
  done
done

echo "DONE summary=$SUMMARY" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
