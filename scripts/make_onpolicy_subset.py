#!/usr/bin/env python3
"""Sample gsm_train idxs never annotated for coconut, split into shards for on-policy annotate.

Disjoint from latent-data/coconut/train_done_idx.txt so on-policy shards can be merged
with coconut shards (CachedPickleDatasetV2 keys rows by question idx).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/gsm_train.json")
    ap.add_argument("--done_idx", default="latent-data/coconut/train_done_idx.txt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n_shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_prefix", default="latent-data/dualBeat/subset_idx")
    args = ap.parse_args()

    n_total = len(json.load(open(args.train_json)))
    done = {int(l) for l in open(args.done_idx) if l.strip()}
    pool = [i for i in range(n_total) if i not in done]
    random.seed(args.seed)
    pick = sorted(random.sample(pool, args.n))
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for s in range(args.n_shards):
        part = pick[s :: args.n_shards]
        path = f"{args.out_prefix}_{s}.txt"
        Path(path).write_text("\n".join(map(str, part)) + "\n")
        print(f"{path}: {len(part)}")
    print(f"pool={len(pool)} done={len(done)} total={n_total}")


if __name__ == "__main__":
    main()
