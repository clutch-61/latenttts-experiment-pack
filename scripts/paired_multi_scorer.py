#!/usr/bin/env python3
"""Paired significance of O2 vs a challenger scorer, frozen vs p2 pools.

Generalizes ``scripts/paired_significance.py`` (which hard-codes B0) to any
challenger that was evaluated on the SAME candidate pools as O2 (verified via
``answers_sha1``). For each seed it reports the same-pool contingency, exact
McNemar, paired bootstrap CI for mean(O2 - challenger), and the per-question
interaction (gap_p2 - gap_frozen).

Example:
    python scripts/paired_multi_scorer.py \
        --challenger swift \
        --o2_frozen results/full/paired_seeds/bon_frozen_O2_gsm_N16_s42_20261016_002036.json \
        --ch_frozen results/full/lightweight/bon_swift_gsm_N16_dnone_s42_test.json \
        --o2_p2 results/full/paired_seeds/bon_p2mmd05_O2_gsm_N16_s42_...json \
        --ch_p2 results/full/lightweight/bon_p2mmd05_swift_gsm_N16_s42.json \
        --out results/full/lightweight/paired_o2_vs_swift_s42.json
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {e["idx"]: e for e in d["examples"]}


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    obs = comb(n, b) * (0.5 ** n)
    total = 0.0
    for k in range(n + 1):
        prob = comb(n, k) * (0.5 ** n)
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


def pair(o2: dict, ch: dict, common: list):
    o2c = np.array([int(o2[i]["selected_correct"]) for i in common])
    chc = np.array([int(ch[i]["selected_correct"]) for i in common])
    b = int(((o2c == 1) & (chc == 0)).sum())  # O2 right, challenger wrong
    c = int(((o2c == 0) & (chc == 1)).sum())  # challenger right, O2 wrong
    mean, lo, hi = bootstrap_mean_diff(o2c, chc)
    return {
        "n": len(common),
        "o2_acc": float(o2c.mean()),
        "ch_acc": float(chc.mean()),
        "o2_only": b,
        "ch_only": c,
        "both_right": int(((o2c == 1) & (chc == 1)).sum()),
        "both_wrong": int(((o2c == 0) & (chc == 0)).sum()),
        "mcnemar_p": exact_mcnemar(b, c),
        "mean_diff_pp": mean * 100,
        "ci95_pp": [lo * 100, hi * 100],
        "hash_identical": bool(all(o2[i]["answers_sha1"] == ch[i]["answers_sha1"] for i in common)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--o2_frozen", required=True)
    ap.add_argument("--ch_frozen", required=True)
    ap.add_argument("--o2_p2", required=True)
    ap.add_argument("--ch_p2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    o2f, chf = load(args.o2_frozen), load(args.ch_frozen)
    o2p, chp = load(args.o2_p2), load(args.ch_p2)

    fc = sorted(set(o2f) & set(chf))
    pc = sorted(set(o2p) & set(chp))
    frozen = pair(o2f, chf, fc)
    p2 = pair(o2p, chp, pc)

    ic = sorted(set(fc) & set(pc))
    d_frozen = np.array([int(o2f[i]["selected_correct"]) - int(chf[i]["selected_correct"]) for i in ic])
    d_p2 = np.array([int(o2p[i]["selected_correct"]) - int(chp[i]["selected_correct"]) for i in ic])
    inter = d_p2 - d_frozen
    imean, ilo, ihi = bootstrap_mean_diff(d_p2, d_frozen)

    out = {
        "challenger": args.challenger,
        "frozen_pool": frozen,
        "p2_pool": p2,
        "interaction": {
            "n_common_questions": len(ic),
            "gap_p2_minus_gap_frozen_mean_pp": float(imean * 100),
            "ci95_pp": [float(ilo * 100), float(ihi * 100)],
            "p2_gap_larger": int((inter > 0).sum()),
            "frozen_gap_larger": int((inter < 0).sum()),
            "tie": int((inter == 0).sum()),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
