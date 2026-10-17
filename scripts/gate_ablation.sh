#!/usr/bin/env bash
# Gate ablation: same data / seed / budget, only the gating selector changes.
#   arm O2    : selector_family=gpt2   (outputs/latentrm_order_pref/best)
#   arm SWIFT : selector_family=swift  (outputs/swift_latent/best.pt)
# Both use --score_norm zscore so margins are on the same scale.
# Then eval each generator with frozen O2 and frozen B0 (BoN N=16, seed 42).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-400}"
K="${K:-32}"
SEED="${SEED:-42}"
GPU_O2="${GPU_O2:-3}"
GPU_SWIFT="${GPU_SWIFT:-0}"
L=logs/gate_ablation
R=results/full/gate_ablation
mkdir -p "$L" "$R"

OUT_O2="outputs/gateabl_o2_s${SEED}_${STAMP}"
OUT_SWIFT="outputs/gateabl_swift_s${SEED}_${STAMP}"

common_args=(
  --ckpt checkpoints/coconut --teacher_ckpt checkpoints/coconut
  --select_mode dual_beat --sf_mode C
  --score_norm zscore --lambda_mmd 0 --lambda_lat 0 --lambda_rank 0
  --k "$K" --q_batch 8 --grad_accum 4 --lr 5e-6 --seed "$SEED"
  --max_steps "$STEPS" --save_every "$STEPS" --dropout_p 0.2
  --frozen_dtype bf16 --no_auto_pack
)

echo "STAMP=$STAMP SEED=$SEED STEPS=$STEPS K=$K GPU_O2=$GPU_O2 GPU_SWIFT=$GPU_SWIFT" | tee "$L/run_${STAMP}.log"

CUDA_VISIBLE_DEVICES="$GPU_O2" python -u scripts/train_p123_system.py \
  "${common_args[@]}" --selector_family gpt2 \
  --o2_id outputs/latentrm_order_pref/best \
  --outdir "$OUT_O2" > "$L/train_o2_${STAMP}.log" 2>&1 &
P_O2=$!
echo "O2-gate train pid=$P_O2 gpu=$GPU_O2 -> $OUT_O2"

CUDA_VISIBLE_DEVICES="$GPU_SWIFT" python -u scripts/train_p123_system.py \
  "${common_args[@]}" --selector_family swift \
  --o2_id outputs/swift_latent/best.pt \
  --outdir "$OUT_SWIFT" > "$L/train_swift_${STAMP}.log" 2>&1 &
P_SW=$!
echo "SWIFT-gate train pid=$P_SW gpu=$GPU_SWIFT -> $OUT_SWIFT"

wait $P_O2; echo "O2-gate train exit=$?" | tee -a "$L/run_${STAMP}.log"
wait $P_SW; echo "SWIFT-gate train exit=$?" | tee -a "$L/run_${STAMP}.log"

# --- eval both arms with frozen O2 and frozen B0 (N=16, seed 42) ---
eval_arm() {
  local arm="$1" gen="$2" prm="$3" fam="$4" tag="$5" gpu="$6"
  local json="$R/bon_${arm}_${tag}_N16_s${SEED}.json"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=coconut --generator_id="$gen" \
    --prm_id="$prm" --prm_model_family="$fam" --prm_mode=best_of_n \
    --data_path=data/gsm_test.json --num_return_sequences=16 --seed="$SEED" \
    --batch_size=16 --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" > "$L/eval_${arm}_${tag}_${STAMP}.log" 2>&1
  echo "  eval $arm $tag done"
}

for arm in o2 swift; do
  gen="outputs/gateabl_${arm}_s${SEED}_${STAMP}/model"
  [[ -d "$gen" ]] || { echo "SKIP $arm (no model)"; continue; }
  eval_arm "$arm" "$gen" outputs/latentrm_order_pref/best gpt2 "o2" "$GPU_O2"
  eval_arm "$arm" "$gen" outputs/latentrm_baseline/best gpt2 "b0" "$GPU_O2"
done

echo "GATE_ABLATION_DONE STAMP=$STAMP SEED=$SEED" | tee -a "$L/run_${STAMP}.log"
