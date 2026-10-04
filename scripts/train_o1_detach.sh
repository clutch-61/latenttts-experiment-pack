#!/usr/bin/env bash
# Detach O1 (order only, λ_hier=0) from Cursor/IDE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

mkdir -p logs outputs
LOG="logs/train_o1_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
PORT="${MASTER_PORT:-29550}"

nohup setsid bash -c "
  accelerate launch --main_process_port ${PORT} --num_processes ${NUM_PROCESSES} \
    -m src.train_order_pref training_args/train_order_only.yaml
" >>"$LOG" 2>&1 < /dev/null &

echo "started pid=$!"
echo "log=$ROOT/$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_PROCESSES=$NUM_PROCESSES port=$PORT RESUME=${RESUME_FROM_CHECKPOINT:-none}"
sleep 4
pgrep -af "src.train_order_pref|train_order_only" | grep -v grep || echo "WARN: not visible yet"
