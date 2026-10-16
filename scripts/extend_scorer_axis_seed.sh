#!/usr/bin/env bash
# Add seeds 45,46 to the scorer axis, writing into a single uniform 5-seed dir.
#   results/full/scorer_axis_5seed/{pool}_{scorer}_N{n}_s{seed}.json
# Pools: frozen | p2.  Scorers: B0 O2 SWIFT HSRM.  N: 16 (all), 64 (p2 only).
# Usage: SEED=45 bash scripts/extend_scorer_axis_seed.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:?}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-3}"

GEN_FROZEN=checkpoints/coconut
GEN_P2=outputs/p123_p2mmd05_20261004_215700/model
R=results/full/scorer_axis_5seed
L=logs/scorer_axis_5seed
mkdir -p "$R" "$L"

run_one () {
  local gpu="$1" pool="$2" scorer="$3" n="$4"
  local gen prm fam
  case "$pool" in
    frozen) gen="$GEN_FROZEN" ;;
    p2)     gen="$GEN_P2" ;;
  esac
  case "$scorer" in
    B0)    prm=outputs/latentrm_baseline/best; fam=gpt2 ;;
    O2)    prm=outputs/latentrm_order_pref/best; fam=gpt2 ;;
    SWIFT) prm=outputs/swift_latent/best.pt; fam=swift ;;
    HSRM)  prm=outputs/hsrm_latent/best.pt; fam=hsrm ;;
  esac
  local json="$R/${pool}_${scorer}_N${n}_s${SEED}.json"
  [[ -f "$json" ]] && { echo "SKIP $pool $scorer N$n s$SEED"; return; }
  local log="$L/${pool}_${scorer}_N${n}_s${SEED}.log"
  echo "==== GPU$gpu $pool $scorer N$n s$SEED $(date -Is) ===="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$gen" \
    --prm_id="$prm" --prm_model_family="$fam" --prm_mode=best_of_n \
    --data_path=data/gsm_test.json --num_return_sequences="$n" --seed="$SEED" \
    --batch_size=16 --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" > "$log" 2>&1
  echo "  done $pool $scorer N$n s$SEED"
}

# group A: gpt2 scorers on all pools (bigger model, give own GPU)
( run_one "$GPU_A" frozen B0 16; run_one "$GPU_A" frozen O2 16; run_one "$GPU_A" p2 B0 16; run_one "$GPU_A" p2 O2 16 ) &
PA=$!
# group B: swift
( run_one "$GPU_B" frozen SWIFT 16; run_one "$GPU_B" p2 SWIFT 16; run_one "$GPU_B" p2 SWIFT 64 ) &
PB=$!
# group C: hsrm
( run_one "$GPU_C" frozen HSRM 16; run_one "$GPU_C" p2 HSRM 16; run_one "$GPU_C" p2 HSRM 64 ) &
PC=$!
wait $PA $PB $PC
echo "SEED_DONE $SEED"
ls "$R" | grep "s${SEED}" | sort
