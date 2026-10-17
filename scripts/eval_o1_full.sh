#!/usr/bin/env bash
# O1: full valid-64 eval_order + BoN matrix + GSM N16 inference seeds 43/44
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRM="outputs/latentrm_order_only/best"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/eval_order logs/bon_o1 results/full

ORDER_TSV="results/full/eval_order_o1_valid64_full_${STAMP}.tsv"
BON_TSV="results/full/bon_o1_${STAMP}.tsv"
MASTER="logs/bon_o1/run_${STAMP}.log"

echo -e "model\tprm\torder_pair_acc\thierarchy_consistency\tlog" >"$ORDER_TSV"
echo -e "split\tprm\tn\tseed\tacc\tcov\tvot\tlog" >"$BON_TSV"

echo "==== O1 eval_order full valid-64 $(date -Is) ====" | tee -a "$MASTER"
export CUDA_VISIBLE_DEVICES="${ORDER_GPU:-0}"
run_log="logs/eval_order/O1_valid64_full_${STAMP}.log"
set +e
python -m src.eval_order_pref \
  --prm_id="$PRM" \
  --data_dir=latent-data/coconut/valid-64 \
  --batch_size=16 \
  --max_batches=0 \
  --device=cuda \
  >>"$run_log" 2>&1
ec=$?
set -e
echo "eval_order ec=$ec" | tee -a "$MASTER"
pair=$(grep -oE "'order_pair_acc': [0-9.]+" "$run_log" | tail -1 | awk '{print $2}' || true)
hier=$(grep -oE "'hierarchy_consistency': [0-9.]+" "$run_log" | tail -1 | awk '{print $2}' || true)
echo -e "O1\t${PRM}\t${pair:-FAIL}\t${hier:-}\t${run_log}" | tee -a "$ORDER_TSV" "$MASTER"

echo "==== O1 BoN matrix $(date -Is) ====" | tee -a "$MASTER"
export CUDA_VISIBLE_DEVICES="${BON_GPUS:-0,2,6}"
export NUM_PROCESSES="${NUM_PROCESSES:-3}"
PORT="${MASTER_PORT:-29570}"

declare -A DATA=(
  [gsm_test]=data/gsm_test.json
  [hard]=data/gsm_hard.json
  [multiarith]=data/multiarith.json
)

run_bon() {
  local split="$1" n="$2" seed="$3"
  local tag="${split}_O1_N${n}_s${seed}"
  local rlog="logs/bon_o1/${tag}_${STAMP}.log"
  echo "======== START $tag $(date -Is) ========" | tee -a "$MASTER" "$rlog"
  set +e
  accelerate launch --main_process_port "$PORT" --num_processes "$NUM_PROCESSES" \
    -m src.infer_gpt2_rm \
    --generator_type=coconut \
    --prm_id="$PRM" \
    --prm_model_family=gpt2 \
    --prm_mode=best_of_n \
    --data_path="${DATA[$split]}" \
    --num_return_sequences="$n" \
    --seed="$seed" \
    --batch_size=1024 \
    --progress_bar=True \
    >>"$rlog" 2>&1
  local ec=$?
  set -e
  echo "======== END $tag ec=$ec $(date -Is) ========" | tee -a "$MASTER" "$rlog"
  local acc cov vot
  acc=$(grep -E '^Accuracy:' "$rlog" | tail -1 | awk '{print $2}' || true)
  cov=$(grep -E '^Coverage:' "$rlog" | tail -1 | awk '{print $2}' || true)
  vot=$(grep -E '^Voting Accuracy:' "$rlog" | tail -1 | awk '{print $3}' || true)
  echo -e "${split}\tO1\t${n}\t${seed}\t${acc:-FAIL}\t${cov:-}\t${vot:-}\t${rlog}" | tee -a "$BON_TSV" "$MASTER"
  PORT=$((PORT + 1))
}

for split in gsm_test hard multiarith; do
  for n in 4 16 64; do
    run_bon "$split" "$n" 42
  done
done

# GSM N=16 inference seeds to match B1/O2 comparison
for seed in 43 44; do
  run_bon gsm_test 16 "$seed"
done

echo "DONE order=$ORDER_TSV bon=$BON_TSV" | tee -a "$MASTER"
cat "$ORDER_TSV" | tee -a "$MASTER"
cat "$BON_TSV" | tee -a "$MASTER"
