#!/usr/bin/env bash
# Order-Decision test over multiple corruption seeds.
# Writes results/full/order_decision/od_{model}_N{n}_s{seed}.json and matching
# per(type) npz: results/full/order_decision/odnpz_{model}_N{n}_s{seed}_{type}.npz
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R="results/full/order_decision"
L="logs/order_decision"
mkdir -p "$R" "$L"
MAX_POS="${MAX_POS:-1000}"
NS="${NS:-4 16}"
SEEDS="${SEEDS:-43 44 45 46}"
GPU_LIGHT="${GPU_LIGHT:-1}"
GPU_GPT2="${GPU_GPT2:-1}"

run_one () {
  local gpu="$1" model="$2" family="$3" prm="$4" scorer="${5:-}" seed="$6"
  for n in $NS; do
    local chunk=512
    if [[ "$family" == "gpt2" ]]; then
      if [[ "$n" -ge 16 ]]; then chunk=32; else chunk=128; fi
    fi
    local tag="od_${model}_N${n}_s${seed}"
    local args=(--family "$family" --prm_id "$prm" --n "$n" --max_pos "$MAX_POS"
                --seed "$seed" --chunk "$chunk" --out_json "$R/${tag}.json")
    [[ -n "$scorer" ]] && args+=(--scorer "$scorer")
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/order_decision_test.py "${args[@]}" \
      >"$L/${tag}.log" 2>&1
    echo "done $model N$n s$seed (gpu$gpu) -> $R/${tag}__*.npz"
  done
}

for seed in $SEEDS; do
  run_one "$GPU_LIGHT" swift light outputs/swift_latent/best.pt swift "$seed" &
  run_one "$GPU_LIGHT" hsrm  light outputs/hsrm_latent/best.pt  hsrm  "$seed" &
  wait
  run_one "$GPU_GPT2" o2 gpt2 outputs/latentrm_order_pref/best "" "$seed" &
  run_one "$GPU_GPT2" b0 gpt2 outputs/latentrm_baseline/best  "" "$seed" &
  wait
  echo "seed $seed done"
done
echo "ORDER_DECISION_SEEDS_DONE"
