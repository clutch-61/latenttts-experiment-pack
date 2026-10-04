#!/usr/bin/env bash
# Detach B1 (BCE LatentRM, no pref) from Cursor/IDE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

mkdir -p logs outputs
LOG="logs/train_b1_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
PORT="${MASTER_PORT:-29530}"

nohup setsid bash -c "
  accelerate launch --main_process_port ${PORT} --num_processes ${NUM_PROCESSES} \
    -m src.train training_args/train_baseline_bce.yaml
" >>"$LOG" 2>&1 < /dev/null &

echo "started pid=$!"
echo "log=$ROOT/$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_PROCESSES=$NUM_PROCESSES port=$PORT RESUME=${RESUME_FROM_CHECKPOINT:-none}"
sleep 3
pgrep -af "src.train|train_baseline_bce" | grep -v grep || echo "WARN: not visible yet"
