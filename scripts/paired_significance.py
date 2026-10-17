#!/usr/bin/env python3
"""Per-question paired significance for O2 vs B0, on frozen vs p2 pools.

Reads per-example JSON dumps from infer_gpt2_rm (--result_json). For each seed:
  - same-pool O2 vs B0 contingency (O2 wins / B0 wins / both right / both wrong)
  - McNemar test (exact binomial on discordant pairs)
  - paired bootstrap CI for mean(O2-B0)
Then tests whether the O2-over-B0 gap is larger on the p2 pool than the frozen pool
(per-question interaction, bootstrap CI on mean[(O2-B0)_p2 - (O2-B0)_frozen]).
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {e["idx"]: e for e in d["examples"]}


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b (O2 right, B0 wrong)
    and c (B0 right, O2 wrong). Binomial test on b ~ Bin(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    p = 0.5
    # two-sided: sum probabilities <= observed probability
    obs = comb(n, b) * (p ** n)
    total = 0.0
    for k in range(n + 1):
        prob = comb(n, k) * (p ** n)
        if prob <= obs + 1e-15:
            total += prob
    return min(1.0, total)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = d[idx].mean()
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen_b0", required=True)
    ap.add_argument("--frozen_o2", required=True)
    ap.add_argument("--p2_b0", required=True)
    ap.add_argument("--p2_o2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fb0 = load(Path(args.frozen_b0))
    fo2 = load(Path(args.frozen_o2))
    pb0 = load(Path(args.p2_b0))
    po2 = load(Path(args.p2_o2))

    # same-pool identity check
    frozen_common = sorted(set(fb0) & set(fo2))
    p2_common = sorted(set(pb0) & set(po2))
    frozen_hash_ok = all(fb0[i]["answers_sha1"] == fo2[i]["answers_sha1"] for i in frozen_common)
    p2_hash_ok = all(pb0[i]["answers_sha1"] == po2[i]["answers_sha1"] for i in p2_common)

    def pair(b0, o2, common):
        o2c = np.array([int(o2[i]["selected_correct"]) for i in common])
        b0c = np.array([int(b0[i]["selected_correct"]) for i in common])
        b = int(((o2c == 1) & (b0c == 0)).sum())  # O2 right, B0 wrong
        c = int(((o2c == 0) & (b0c == 1)).sum())  # B0 right, O2 wrong
        both_r = int(((o2c == 1) & (b0c == 1)).sum())
        both_w = int(((o2c == 0) & (b0c == 0)).sum())
        mean, lo, hi = bootstrap_mean_diff(o2c, b0c)
        return {
            "n": len(common),
            "o2_acc": float(o2c.mean()),
            "b0_acc": float(b0c.mean()),
            "o2_only": b,
            "b0_only": c,
            "both_right": both_r,
            "both_wrong": both_w,
            "mcnemar_p": exact_mcnemar(b, c),
            "mean_diff_pp": mean * 100,
            "ci95_pp": [lo * 100, hi * 100],
        }

    frozen = pair(fb0, fo2, frozen_common)
    p2 = pair(pb0, po2, p2_common)

    # per-question interaction: (O2-B0)_p2 - (O2-B0)_frozen, on common question ids
    inter_common = sorted(set(frozen_common) & set(p2_common))
    d_frozen = np.array([int(fo2[i]["selected_correct"]) - int(fb0[i]["selected_correct"]) for i in inter_common])
    d_p2 = np.array([int(po2[i]["selected_correct"]) - int(pb0[i]["selected_correct"]) for i in inter_common])
    inter = d_p2 - d_frozen
    imean, ilo, ihi = bootstrap_mean_diff(d_p2, d_frozen)
    # also report how often gap is larger on p2 vs frozen
    inter_pos = int((inter > 0).sum())
    inter_neg = int((inter < 0).sum())
    inter_zero = int((inter == 0).sum())

    out = {
        "frozen_pool": frozen,
        "p2_pool": p2,
        "frozen_candidates_identical": bool(frozen_hash_ok),
        "p2_candidates_identical": bool(p2_hash_ok),
        "interaction": {
            "n_common_questions": len(inter_common),
            "gap_p2_minus_gap_frozen_mean_pp": float(imean * 100),
            "ci95_pp": [ilo * 100, ihi * 100],
            "p2_gap_larger": inter_pos,
            "frozen_gap_larger": inter_neg,
            "tie": inter_zero,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
