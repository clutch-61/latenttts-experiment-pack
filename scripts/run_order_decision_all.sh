#!/usr/bin/env bash
# Order-decision test for all 4 scorers, N in {4,16}. Writes a TSV summary.
# Different GPUs run in parallel to cut wall-clock (row files are per-model, no races).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1

R="results/full/order_decision"
mkdir -p "$R" logs/order_decision
MAX_POS="${MAX_POS:-1000}"
NS="${NS:-4 16}"

run_one () {
  local gpu="$1" model="$2" family="$3" prm="$4" scorer="${5:-}"
  local rows="$R/rows_${model}.tsv"
  : > "$rows"
  for n in $NS; do
    local log="logs/order_decision/${model}_N${n}.log"
    local chunk=512
    if [[ "$family" == "gpt2" ]]; then
      if [[ "$n" -ge 16 ]]; then chunk=32; else chunk=128; fi
    fi
    local args=(--family "$family" --prm_id "$prm" --n "$n" --max_pos "$MAX_POS"
                --chunk "$chunk" --out_json "$R/${model}_N${n}.json")
    [[ -n "$scorer" ]] && args+=(--scorer "$scorer")
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/order_decision_test.py "${args[@]}" >"$log" 2>&1
    python3 - "$R/${model}_N${n}.json" "$model" "$family" >>"$rows" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for c,v in d["results"].items():
    print(f'{sys.argv[2]}\t{sys.argv[3]}\t{d["n_pool"]}\t{c}\t{v["pair_acc"]:.4f}\t{v["strict_win"]:.4f}\t{v["tie_frac"]:.4f}\t{d["n_pos"]}')
PY
    echo "done $model N$n (gpu$gpu)"
  done
}

run_one 3 o2     gpt2  outputs/latentrm_order_pref/best &
run_one 2 b0     gpt2  outputs/latentrm_baseline/best &
run_one 0 swift  light outputs/swift_latent/best.pt swift &
run_one 0 hsrm   light outputs/hsrm_latent/best.pt  hsrm &
wait

TSV="$R/order_decision_summary.tsv"
echo -e "model\tfamily\tn_pool\tcorruption\tpair_acc\tstrict_win\ttie_frac\tn_pos" > "$TSV"
for m in o2 b0 swift hsrm; do cat "$R/rows_${m}.tsv" >> "$TSV"; done
echo "ALL DONE -> $TSV"
column -t -s$'\t' "$TSV"
