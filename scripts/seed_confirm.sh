#!/usr/bin/env bash
# Re-run dualBeat+O2 and the A+B0 baseline on GSM-Test N16 under extra seeds (paired per seed).
# Usage: bash scripts/seed_confirm.sh [GPU_A=0] [GPU_B=5]
cd "$(dirname "$0")/.." || exit 1
GA=${1:-0}; GB=${2:-5}
GEN=outputs/p123_dualBeat_20261001_202106/model
O2=outputs/latentrm_order_pref/best
B0=outputs/latentrm_baseline/best
R=results/full/p123/seeds; L=logs/p123/seeds
mkdir -p $R $L

bon() {  # gpu gen rm seed tag
  CUDA_VISIBLE_DEVICES=$1 python -m src.infer_gpt2_rm --generator_type=coconut --generator_id=$2 \
    --prm_id=$3 --prm_model_family=gpt2 --prm_mode=best_of_n --data_path=data/gsm_test.json \
    --num_return_sequences=16 --seed=$4 --batch_size=16 --sort_by_len=False \
    --result_json=$R/bon_$5_s$4.json > $L/bon_$5_s$4.log 2>&1
}

for s in 43 44; do
  bon $GA $GEN $O2 $s dualBeat_O2 &
  bon $GB checkpoints/coconut $B0 $s A_B0 &
done
wait
echo "[$(date)] evals done"

ARGS=""
for s in 42 43 44; do
  if [ $s = 42 ]; then
    ARGS+=" dualBeat_O2_s42=results/full/p123/bon_dualBeat_O2_N16_gsm_20261001_202106.json"
    ARGS+=" A_B0_s42=results/full/sf_full/bon_A_B0_N16_gsm_20261001_192426.json"
  else
    ARGS+=" dualBeat_O2_s$s=$R/bon_dualBeat_O2_s$s.json A_B0_s$s=$R/bon_A_B0_s$s.json"
  fi
done
python scripts/summarize_bon.py $ARGS
echo SEEDS_DONE
