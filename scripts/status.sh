#!/usr/bin/env bash
# One-shot progress view: running jobs, latest line of recent logs, GPU usage.
# Usage: bash scripts/status.sh [hours_back=6]
cd "$(dirname "$0")/.." || exit 1
HOURS=${1:-6}

echo "=== running jobs ($(date '+%m-%d %H:%M')) ==="
ps -u "$USER" -o pid,etime,args --no-headers \
  | grep -E 'train_p123|annotate_data|train_order_pref|infer_gpt2_rm|eval_sf_generator|chain_|pipeline_p123' \
  | grep -vE 'grep|accelerate launch' | cut -c1-160 || true

echo
echo "=== recent logs (last ${HOURS}h) ==="
find logs/p123 logs/onpolicy -name '*.log' -mmin -$((HOURS * 60)) -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -12 | cut -d' ' -f2 | while read -r f; do
    last=$(tail -c 3000 "$f" | tr '\r' '\n' | grep -av '^\s*$' | grep -avE '^\*+$' | tail -1 | cut -c1-170)
    printf '%-58s | %s\n' "$(basename "$f")" "$last"
  done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
