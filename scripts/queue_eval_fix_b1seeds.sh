#!/usr/bin/env bash
# Wait until our O1 train exits (or chosen GPUs free), then run:
#   1) full valid-64 eval_order (O2/B0/B1) on one GPU
#   2) B1 GSM N=16 inference seeds 43/44 on 4 GPUs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
CHAIN="logs/queue_eval_fix_b1seeds_$(date +%Y%m%d_%H%M%S).log"

wait_gpus_free() {
  # Wait until GPUs 0,2,6,7 each have < 2GiB used (O1 released).
  local i
  for i in $(seq 1 720); do  # up to ~6h @ 30s
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$1==0 || $1==2 || $1==6 || $1==7 { if ($2+0 > 2048) c++ } END{print c+0}')
    echo "$(date -Is) busy_among_0267=$busy" | tee -a "$CHAIN"
    if [[ "$busy" -eq 0 ]]; then
      return 0
    fi
    # Also proceed if O1 pid gone even if another job took cards — then re-check.
    if ! pgrep -f 'training_args/train_order_only.yaml' >/dev/null 2>&1; then
      sleep 15
      busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$1==0 || $1==2 || $1==6 || $1==7 { if ($2+0 > 2048) c++ } END{print c+0}')
      echo "$(date -Is) O1 gone; recheck busy=$busy" | tee -a "$CHAIN"
      if [[ "$busy" -eq 0 ]]; then
        return 0
      fi
      # If free cards exist among 0,2,6,7 individually, pick them later.
      if [[ "$busy" -lt 4 ]]; then
        return 0
      fi
    fi
    sleep 30
  done
  echo "TIMEOUT waiting for GPUs" | tee -a "$CHAIN"
  return 1
}

pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2+0 < 2048 {print $1; exit}'
}

{
  echo "==== queue start $(date -Is) ===="
  wait_gpus_free
  FREE=$(pick_free_gpu)
  if [[ -z "${FREE}" ]]; then
    echo "No free GPU after wait" >&2
    exit 1
  fi
  echo "==== eval_order full on GPU $FREE $(date -Is) ===="
  CUDA_VISIBLE_DEVICES="$FREE" bash scripts/eval_order_valid64_full.sh
  echo "==== B1 multi-seed BoN $(date -Is) ===="
  # Prefer 0,2,6,7 if free; else whatever is free among them
  export CUDA_VISIBLE_DEVICES=0,2,6,7
  export NUM_PROCESSES=4
  bash scripts/eval_bon_b1_seeds.sh
  echo "==== queue done $(date -Is) ===="
} >>"$CHAIN" 2>&1

echo "chain_log=$ROOT/$CHAIN"
