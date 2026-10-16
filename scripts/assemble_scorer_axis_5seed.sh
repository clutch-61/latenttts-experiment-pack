#!/usr/bin/env bash
# Assemble a uniform 5-seed scorer-axis dir from existing (3-seed) + new (45,46) files.
# Result: results/full/scorer_axis_5seed/{pool}_{scorer}_N{n}_s{seed}.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=results/full/scorer_axis_5seed
mkdir -p "$OUT"

link() {  # src dst
  [[ -f "$1" ]] || { echo "MISSING $1"; return; }
  ln -sf "$(realpath --relative-to="$OUT" "$1")" "$OUT/$2"
}

# ---- frozen, N16 ----
for s in 42 43 44; do
  link "results/full/paired_seeds/bon_frozen_B0_gsm_N16_s${s}_20261016_002036.json" "frozen_B0_N16_s${s}.json"
  link "results/full/paired_seeds/bon_frozen_O2_gsm_N16_s${s}_20261016_002036.json" "frozen_O2_N16_s${s}.json"
  link "results/full/lightweight/bon_swift_gsm_N16_dnone_s${s}_test.json" "frozen_SWIFT_N16_s${s}.json"
  link "results/full/lightweight/bon_hsrm_gsm_N16_dnone_s${s}_20261016_174900.json" "frozen_HSRM_N16_s${s}.json"
done

# ---- p2, N16 ----
link "results/full/p123/exp1_ours_b0/bon_p2mmd05_B0_gsm_N16_dnone_s42_20261015_230629.json" "p2_B0_N16_s42.json"
link "results/full/p123/exp1_ours_o2/bon_p2mmd05_O2_gsm_N16_dnone_s42_20261016_001127.json" "p2_O2_N16_s42.json"
for s in 43 44; do
  link "results/full/paired_seeds/bon_p2mmd05_B0_gsm_N16_s${s}_20261016_011103.json" "p2_B0_N16_s${s}.json"
  link "results/full/paired_seeds/bon_p2mmd05_O2_gsm_N16_s${s}_20261016_011103.json" "p2_O2_N16_s${s}.json"
done
for s in 42 43 44; do
  link "results/full/lightweight/bon_p2mmd05_swift_gsm_N16_s${s}.json" "p2_SWIFT_N16_s${s}.json"
  link "results/full/lightweight/bon_p2mmd05_hsrm_gsm_N16_s${s}.json" "p2_HSRM_N16_s${s}.json"
done

# ---- p2, N64 (swift/hsrm) ----
for s in 42 43 44; do
  link "results/full/lightweight/bon_p2mmd05_swift_gsm_N64_s${s}.json" "p2_SWIFT_N64_s${s}.json"
  link "results/full/lightweight/bon_p2mmd05_hsrm_gsm_N64_s${s}.json" "p2_HSRM_N64_s${s}.json"
done

echo "=== assembled ==="
ls "$OUT" | sort
echo "count: $(ls "$OUT" | wc -l)"
