#!/usr/bin/env bash
# After scorer-axis evals finish, extend the generator-axis baselines to seeds 45,46.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[chain-b] waiting $(date -Is)"
while pgrep -f 'extend_scorer_axis_seed.sh' >/dev/null 2>&1; do sleep 30; done
while pgrep -f 'src.infer_gpt2_rm.*scorer_axis_5seed' >/dev/null 2>&1; do sleep 30; done
echo "[chain-b] start ext_baselines seeds 45,46 $(date -Is)"

GPU=3 MODEL=outputs/ext_dropoutgrpo_coconut_grpo_b3000 TAG=dropoutgrpo \
  SEEDS="45 46" bash scripts/eval_ext_baselines.sh
GPU=3 MODEL=outputs/ext_slpo_coconut_rloo_b3000 TAG=slpo_rloo \
  SEEDS="45 46" bash scripts/eval_ext_baselines.sh
echo "[chain-b] DONE $(date -Is)"
