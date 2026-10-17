#!/usr/bin/env python3
"""Build hard-only on-policy mix: keep mixed questions with correct-rate <= max_correct_frac.

Addresses O2op failure mode (easy mixed + heavy coconut dilution). Writes filtered
safetensors under --hard_dir, then optionally mixes coconut at --coconut_ratio.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def n_questions(path: Path) -> int:
    with safe_open(str(path), framework="pt") as f:
        return sum(1 for k in f.keys() if k.endswith(".estimations"))


def filter_shard(src: Path, dst: Path, max_correct_frac: float) -> tuple[int, int]:
    kept: dict[str, torch.Tensor] = {}
    n_in = n_keep = 0
    with safe_open(str(src), framework="pt") as f:
        idxs = sorted({k.split(".", 1)[0] for k in f.keys() if k.endswith(".estimations")})
        for idx in idxs:
            n_in += 1
            corrects = f.get_tensor(f"{idx}.corrects")
            n = int(corrects.numel())
            nc = int(corrects.sum().item())
            if nc <= 0 or nc >= n:
                continue
            if nc / n > max_correct_frac:
                continue
            n_keep += 1
            for k in f.keys():
                if k.startswith(idx + "."):
                    kept[k] = f.get_tensor(k)
    if kept:
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_file(kept, str(dst))
    return n_in, n_keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--onpolicy_dirs",
        nargs="+",
        default=[
            "latent-data/dualBeat/train_a",
            "latent-data/dualBeat/train_b",
            "latent-data/dualBeat/train_c",
        ],
    )
    ap.add_argument("--hard_dir", default="latent-data/dualBeat/hard_mixed")
    ap.add_argument("--coconut_dir", default="latent-data/coconut/train")
    ap.add_argument("--coconut_ratio", type=float, default=0.25)
    ap.add_argument("--max_correct_frac", type=float, default=0.5)
    ap.add_argument("--out", default="latent-data/onpolicy_hard_mix/train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hard_dir = Path(args.hard_dir)
    if hard_dir.exists():
        for old in hard_dir.glob("*.safetensors"):
            old.unlink()
    hard_dir.mkdir(parents=True, exist_ok=True)

    total_in = total_keep = 0
    for d in args.onpolicy_dirs:
        tag = Path(d).name
        for f in sorted(Path(d).glob("*.safetensors")):
            n_in, n_keep = filter_shard(
                f, hard_dir / f"{tag}_{f.name}", args.max_correct_frac
            )
            total_in += n_in
            total_keep += n_keep
    print(
        f"filtered {total_keep}/{total_in} questions "
        f"(correct_frac<={args.max_correct_frac}) -> {hard_dir}"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.safetensors"):
        old.unlink()

    n_hard = 0
    for f in sorted(hard_dir.glob("*.safetensors")):
        n_hard += n_questions(f)
        (out / f"hard_{f.name}").symlink_to(f.resolve())
    print(f"hard questions={n_hard}")

    target = int(n_hard * args.coconut_ratio)
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
