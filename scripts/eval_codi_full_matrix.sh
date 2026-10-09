#!/usr/bin/env bash
# Full matrix: frozen CODI+O2 and CODIp2+O2. Dropout off, seed 42, T=6.
# N=1 det (no RM) + BoN N=4/16/64 × GSM/Hard/MA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CODIP2=outputs/p123_codi_p2mmd05_20261006_104417/model
CODI=checkpoints/codi
O2=outputs/latentrm_order_pref/best
R1=results/full/ext/codi_ours
R2=results/full/ext/codi_o2
L=logs/ext/codi_full
mkdir -p "$R1" "$R2" "$L"

echo "STAMP=$STAMP" | tee "$L/parallel_${STAMP}.log"

bon() {
  local gpu="$1" gen="$2" gtag="$3" rdir="$4" split="$5" n="$6" data="$7" bs="$8"
  local tag="${gtag}_${split}_N${n}_dnone_s42"
  local json="$rdir/bon_${tag}_${STAMP}.json"
  if [[ -f "$json" ]]; then
    echo "SKIP $tag" | tee -a "$L/parallel_${STAMP}.log"
    return 0
  fi
  echo "======== GPU$gpu $tag $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2_rm \
    --generator_type=codi --generator_id="$gen" \
    --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
    --data_path="$data" --num_return_sequences="$n" --seed=42 \
    --batch_size="$bs" --sort_by_len=False \
    --latent_length=6 --max_new_tokens=128 --claim_agg=top1 --model_dtype=bf16 \
    --result_json="$json" \
    > "$L/bon_${tag}_${STAMP}.log" 2>&1
  echo "DONE GPU$gpu $tag" | tee -a "$L/parallel_${STAMP}.log"
}

det() {
  local gpu="$1" gen="$2" gtag="$3" split="$4" data="$5"
  local tag="${gtag}_det_${split}_N1"
  local log="$L/det_${tag}_${STAMP}.log"
  if [[ -f "$log" ]] && grep -q "Pass@1:" "$log"; then
    echo "SKIP $tag" | tee -a "$L/parallel_${STAMP}.log"
    return 0
  fi
  echo "======== GPU$gpu $tag $(date -Is) ========" | tee -a "$L/parallel_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.infer_gpt2 \
    --model_type=codi --model_id="$gen" --model_dtype=bf16 \
    --data_path="$data" --n_samples=1 --batch_size=16 \
    --latent_length=6 --max_new_tokens=128 --do_sample=False \
    > "$log" 2>&1
  echo "DONE GPU$gpu $tag" | tee -a "$L/parallel_${STAMP}.log"
}

# GPU0: isolation GSM + remaining GSM for frozen CODI+O2; frozen N1 gsm
(
  det 0 "$CODI" CODI gsm data/gsm_test.json
  bon 0 "$CODI" CODI_O2 "$R2" gsm 16 data/gsm_test.json 16
  bon 0 "$CODI" CODI_O2 "$R2" gsm 4 data/gsm_test.json 16
  bon 0 "$CODI" CODI_O2 "$R2" gsm 64 data/gsm_test.json 64
) > "$L/chain_gpu0_${STAMP}.log" 2>&1 &
echo CHAIN0=$!

# GPU3: frozen CODI+O2 Hard/MA + frozen N1 hard
(
  det 3 "$CODI" CODI hard data/gsm_hard.json
  bon 3 "$CODI" CODI_O2 "$R2" hard 16 data/gsm_hard.json 16
  bon 3 "$CODI" CODI_O2 "$R2" ma 16 data/multiarith.json 16
  bon 3 "$CODI" CODI_O2 "$R2" hard 4 data/gsm_hard.json 16
  bon 3 "$CODI" CODI_O2 "$R2" ma 4 data/multiarith.json 16
  bon 3 "$CODI" CODI_O2 "$R2" hard 64 data/gsm_hard.json 64
  bon 3 "$CODI" CODI_O2 "$R2" ma 64 data/multiarith.json 64
) > "$L/chain_gpu3_${STAMP}.log" 2>&1 &
echo CHAIN3=$!

# GPU6: CODIp2 Hard/MA + CODIp2 N1 hard/ma
(
  det 6 "$CODIP2" CODIp2 hard data/gsm_hard.json
  det 6 "$CODIP2" CODIp2 ma data/multiarith.json
  bon 6 "$CODIP2" CODIp2_O2 "$R1" hard 16 data/gsm_hard.json 16
  bon 6 "$CODIP2" CODIp2_O2 "$R1" ma 16 data/multiarith.json 16
  bon 6 "$CODIP2" CODIp2_O2 "$R1" hard 4 data/gsm_hard.json 16
  bon 6 "$CODIP2" CODIp2_O2 "$R1" ma 4 data/multiarith.json 16
  bon 6 "$CODIP2" CODIp2_O2 "$R1" hard 64 data/gsm_hard.json 64
  bon 6 "$CODIP2" CODIp2_O2 "$R1" ma 64 data/multiarith.json 64
) > "$L/chain_gpu6_${STAMP}.log" 2>&1 &
echo CHAIN6=$!

# GPU7: CODIp2 GSM leftover + CODIp2 N1 gsm + frozen N1 ma
(
  det 7 "$CODIP2" CODIp2 gsm data/gsm_test.json
  det 7 "$CODI" CODI ma data/multiarith.json
  bon 7 "$CODIP2" CODIp2_O2 "$R1" gsm 4 data/gsm_test.json 16
  bon 7 "$CODIP2" CODIp2_O2 "$R1" gsm 64 data/gsm_test.json 64
) > "$L/chain_gpu7_${STAMP}.log" 2>&1 &
echo CHAIN7=$!

wait
echo ALL_CODI_FULL_DONE | tee -a "$L/parallel_${STAMP}.log"
