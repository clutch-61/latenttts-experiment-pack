#!/usr/bin/env bash
# Wait for in-flight scorer-axis evals, then run gate ablation for seeds 45,46.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[chain] waiting for scorer-axis evals to finish $(date -Is)"
while pgrep -f 'extend_scorer_axis_seed.sh' >/dev/null 2>&1; do sleep 30; done
while pgrep -f 'src.infer_gpt2_rm.*scorer_axis_5seed' >/dev/null 2>&1; do sleep 30; done
echo "[chain] evals done, starting gate ablation $(date -Is)"

for s in 45 46; do
  GPU_O2=0 GPU_SWIFT=2 SEED="$s" STAMP="$(date +%Y%m%d_%H%M%S)" \
    bash scripts/gate_ablation.sh
  echo "[chain] gate ablation seed $s done $(date -Is)"
done
echo "[chain] ALL DONE $(date -Is)"
