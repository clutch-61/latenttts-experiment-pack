#!/usr/bin/env python3
"""Paired BoN comparison given per-example JSON dumps from infer_gpt2_rm --result_json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """Paired bootstrap CI for mean(a-b)."""
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = d[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o2", required=True)
    ap.add_argument("--o1", required=True)
    ap.add_argument("--b1", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dumps = {k: load(Path(getattr(args, k))) for k in ("o2", "o1", "b1")}
    by_idx = {}
    for name, dump in dumps.items():
        by_idx[name] = {ex["idx"]: ex for ex in dump["examples"]}

    common = sorted(set(by_idx["o2"]) & set(by_idx["o1"]) & set(by_idx["b1"]))
    hash_mismatch = []
    for i in common:
        h = {n: by_idx[n][i]["answers_sha1"] for n in ("o2", "o1", "b1")}
        if len(set(h.values())) != 1:
            hash_mismatch.append({"idx": i, **h})

    rows = []
    for i in common:
        rows.append(
            {
                "idx": i,
                "any_correct": by_idx["o2"][i]["any_correct"],
                "o2": int(by_idx["o2"][i]["selected_correct"]),
                "o1": int(by_idx["o1"][i]["selected_correct"]),
                "b1": int(by_idx["b1"][i]["selected_correct"]),
                "answers_sha1": by_idx["o2"][i]["answers_sha1"],
            }
        )

    o2 = np.array([r["o2"] for r in rows])
    o1 = np.array([r["o1"] for r in rows])
    b1 = np.array([r["b1"] for r in rows])

    def contingency(a, b):
        # a wins over b / b wins over a / both right / both wrong
        return {
            "a_only": int(((a == 1) & (b == 0)).sum()),
            "b_only": int(((a == 0) & (b == 1)).sum()),
            "both_right": int(((a == 1) & (b == 1)).sum()),
            "both_wrong": int(((a == 0) & (b == 0)).sum()),
        }

    summary = {
        "n_common": len(common),
        "n_hash_mismatch": len(hash_mismatch),
        "candidates_identical": len(hash_mismatch) == 0,
        "acc": {
            "o2": float(o2.mean()),
            "o1": float(o1.mean()),
            "b1": float(b1.mean()),
        },
        "o2_vs_o1": contingency(o2, o1),
        "o2_vs_b1": contingency(o2, b1),
        "o1_vs_b1": contingency(o1, b1),
        "paired_diff_pp": {},
        "hash_mismatch_examples": hash_mismatch[:20],
        "meta": {k: dumps[k]["meta"] for k in dumps},
    }
    for label, a, b in (("o2_minus_o1", o2, o1), ("o2_minus_b1", o2, b1), ("o1_minus_b1", o1, b1)):
        mean, lo, hi = bootstrap_mean_diff(a, b)
        summary["paired_diff_pp"][label] = {
            "mean_pp": mean * 100,
            "ci95_pp": [lo * 100, hi * 100],
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
