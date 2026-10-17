#!/usr/bin/env python3
"""Symlink on-policy shards + an equal-size random slice of coconut shards into one train dir.

CachedPickleDatasetV2 lists *.safetensors in a single dir and keys rows by question idx;
on-policy idxs are disjoint from coconut's (see make_onpolicy_subset.py), so merging is safe.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from safetensors import safe_open


def n_questions(path: Path) -> int:
    with safe_open(str(path), framework="pt") as f:
        return sum(1 for k in f.keys() if k.endswith(".estimations"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onpolicy_dirs", nargs="+", default=["latent-data/dualBeat/train_a", "latent-data/dualBeat/train_b", "latent-data/dualBeat/train_c"])
    ap.add_argument("--coconut_dir", default="latent-data/coconut/train")
    ap.add_argument("--coconut_ratio", type=float, default=1.0, help="coconut questions per on-policy question")
    ap.add_argument("--out", default="latent-data/onpolicy_mix/train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.safetensors"):
        old.unlink()

    n_on = 0
    for d in args.onpolicy_dirs:
        tag = Path(d).name
        for f in sorted(Path(d).glob("*.safetensors")):
            n_on += n_questions(f)
            (out / f"op_{tag}_{f.name}").symlink_to(f.resolve())
    print(f"on-policy questions={n_on}")

    target = int(n_on * args.coconut_ratio)
    shards = sorted(Path(args.coconut_dir).glob("*.safetensors"))
    random.seed(args.seed)
    random.shuffle(shards)
    n_coco = 0
    for f in shards:
        if n_coco >= target:
            break
        n_coco += n_questions(f)
        (out / f"coco_{f.name}").symlink_to(f.resolve())
    print(f"coconut questions={n_coco} (target {target}) -> {out}")


if __name__ == "__main__":
    main()
