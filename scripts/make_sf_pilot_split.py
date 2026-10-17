#!/usr/bin/env python3
"""Write fixed Phase-G pilot splits (same questions for A/B/C)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/gsm_train.json")
    ap.add_argument("--valid_json", default="data/gsm_valid.json")
    ap.add_argument("--n_train", type=int, default=2048)
    ap.add_argument("--n_valid", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data/sf_pilot")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train = json.load(open(root / args.train_json))
    valid = json.load(open(root / args.valid_json))
    rng = np.random.default_rng(args.seed)

    tr_idx = rng.choice(len(train), size=min(args.n_train, len(train)), replace=False)
    tr_idx = sorted(int(i) for i in tr_idx)
    va_idx = list(range(min(args.n_valid, len(valid))))  # prefix of official valid for stability

    train_sub = [train[i] for i in tr_idx]
    valid_sub = [valid[i] for i in va_idx]

    meta = {
        "seed": args.seed,
        "n_train": len(train_sub),
        "n_valid": len(valid_sub),
        "train_indices": tr_idx,
        "valid_indices": va_idx,
        "train_source": args.train_json,
        "valid_source": args.valid_json,
        "note": "Fixed Phase-G A/B/C subset; do not reshuffle mid-pilot.",
    }
    json.dump(train_sub, open(outdir / "train.json", "w"), ensure_ascii=False)
    json.dump(valid_sub, open(outdir / "valid.json", "w"), ensure_ascii=False)
    json.dump(meta, open(outdir / "meta.json", "w"), indent=2)
    print(f"wrote {outdir}: train={len(train_sub)} valid={len(valid_sub)}")


if __name__ == "__main__":
    main()
