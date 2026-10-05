#!/usr/bin/env bash
# Best-system eval: p2 generator + frozen O2, dropout=None, GSM-Test N=16.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
GEN="${GEN:-outputs/p123_p2mmd05/model}"
O2="${O2:-outputs/latentrm_order_pref/best}"
python -u -m src.infer_gpt2_rm \
  --generator_type=coconut --generator_id="$GEN" \
  --prm_id="$O2" --prm_model_family=gpt2 --prm_mode=best_of_n \
  --data_path=data/gsm_test.json --num_return_sequences=16 --seed=42 \
  --batch_size=16 --sort_by_len=False --max_new_tokens=128 \
  --claim_agg=top1 --model_dtype=bf16 \
  --result_json="${OUT:-results/bon_p2mmd05_O2_N16_dnone_s42.json}"
