#!/usr/bin/env bash
# P3->P1 loop: wait dualBeat on-policy annotate -> build mix -> continue O2 -> BoN eval.
# Usage: nohup bash scripts/chain_onpolicy_o2.sh > logs/onpolicy/chain_STAMP.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GEN=outputs/p123_dualBeat_20261001_202106/model
OUTRM=outputs/latentrm_order_pref_onpolicy
R=results/full/p123
L=logs/onpolicy
STAMP=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] waiting for annotate"
while pgrep -f "annotate_data.*latent-data/dualBeat" > /dev/null; do sleep 60; done
echo "[$(date)] annotate done"
ls latent-data/dualBeat/train_a latent-data/dualBeat/train_b latent-data/dualBeat/valid-64 | head -20

python scripts/build_onpolicy_mix.py || { echo BUILD_FAILED; exit 1; }

# use every one of GPU 0/5 that has >= 60GB free
GPUS=""
for g in 0 5; do
  free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "${free:-0}" -ge 60000 ]; then GPUS="${GPUS:+$GPUS,}$g"; fi
done
[ -z "$GPUS" ] && GPUS=5
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "[$(date)] train O2-onpolicy on GPUS=$GPUS NP=$NP"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --main_process_port 29621 --num_processes $NP \
  -m src.train_order_pref training_args/train_order_pref_onpolicy.yaml > $L/train_o2_onpolicy_${STAMP}.log 2>&1
echo "[$(date)] train exit=$?"
RM=$OUTRM/best
[ -d "$RM" ] || { echo NO_BEST; ls $OUTRM; exit 1; }

bon() {  # gpu gen prm N tag
  CUDA_VISIBLE_DEVICES=$1 python -m src.infer_gpt2_rm --generator_type=coconut --generator_id=$2 \
    --prm_id=$3 --prm_model_family=gpt2 --prm_mode=best_of_n --data_path=data/gsm_test.json \
    --num_return_sequences=$4 --seed=42 --batch_size=16 --sort_by_len=False \
    --result_json=$R/bon_$5_${STAMP}.json > $L/bon_$5_${STAMP}.log 2>&1
}
G1=$(echo "$GPUS" | cut -d, -f1); G2=$(echo "$GPUS" | cut -d, -f2)
bon $G1 $GEN $RM 16 dualBeat_O2op_N16 &
bon $G2 checkpoints/coconut $RM 16 A_O2op_N16 &
wait
bon $G1 $GEN $RM 64 dualBeat_O2op_N64
echo "[$(date)] eval done"

python - <<PY
import json, math
from collections import defaultdict
base = 0.3373768006065201
def row(tag, path):
    d = json.load(open(path)); ex = d["examples"]; n = len(ex)
    top = wv = 0
    for e in ex:
        ans = [str(a) for a in e["answers"]]; s = [float(x) for x in e["scores"]]; c = e["corrects"]
        ok = {a: bool(ci) for a, ci in zip(ans, c)}
        top += bool(c[max(range(len(s)), key=lambda i: s[i])])
        m = max(s); w = defaultdict(float)
        for a, x in zip(ans, s): w[a] += math.exp(x - m)
        wv += ok[max(w, key=w.get)]
    m = d["meta"]
    print(f"{tag:<22} top={100*top/n:.2f} wvote={100*wv/n:.2f} cov={100*m['coverage']:.2f} "
          f"d_top_vs_AB0={100*(top/n-base):+.2f} d_wvote_vs_AB0={100*(wv/n-base):+.2f}")
for tag in ["dualBeat_O2op_N16", "dualBeat_O2op_N64", "A_O2op_N16"]:
    row(tag, f"$R/bon_{tag}_${STAMP}.json")
PY
echo CHAIN_DONE
