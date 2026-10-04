#!/usr/bin/env python3
"""O2-only multi-thief leftover hinge. Generator is not loaded.

Pairs come from mine_o2_hard_leftover.py (gsm_train). Latents stay detached.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.system_scoring import load_prm, sum_logit_score  # noqa: E402


class LeftoverPairs(Dataset):
    def __init__(self, path: Path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.pairs = blob["pairs"]
        self.meta = blob.get("meta") or {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def collate(batch):
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="latent-data/o2_hard_leftover/train_k16.pt")
    ap.add_argument("--o2_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--tok_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir or f"outputs/p123_o2hardrank_{stamp}")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ds = LeftoverPairs(ROOT / args.pairs if not Path(args.pairs).is_absolute() else Path(args.pairs))
    print(f"pairs={len(ds)} mine_meta={ds.meta}", flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tok_id))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    o2 = load_prm(str(ROOT / args.o2_id), tok, device, dtype=torch.float32, trainable=True)
    opt = torch.optim.AdamW([p for p in o2.parameters() if p.requires_grad], lr=args.lr)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    hist = []
    step = 0
    for ep in range(args.epochs):
        o2.train()
        losses = []
        n_hit = 0
        n_pair = 0
        pbar = tqdm(loader, desc=f"o2hard ep{ep}")
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            loss_acc = []
            for ex in batch:
                gold = {"input_ids": ex["gold_ids"], "latents": ex["gold_lat"]}
                for tid, tlat in zip(ex["thief_ids"], ex["thief_lat"]):
                    s_pos = sum_logit_score(o2, gold["input_ids"], gold["latents"].detach())
                    s_neg = sum_logit_score(o2, tid, tlat.detach())
                    hinge = torch.relu(args.margin + s_neg - s_pos)
                    loss_acc.append(hinge)
                    n_pair += 1
                    if float((s_pos - s_neg).detach()) > 0:
                        n_hit += 1
            loss = torch.stack(loss_acc).mean()
            loss.backward()
            opt.step()
            step += 1
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}", hit=n_hit / max(n_pair, 1))
        row = {
            "epoch": ep,
            "mean_loss": sum(losses) / max(len(losses), 1),
            "train_gold_beats_thief": n_hit / max(n_pair, 1),
            "steps": step,
        }
        hist.append(row)
        print(f"EPOCH {row}", flush=True)

    o2.eval()
    o2.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    json.dump(
        {"args": vars(args), "hist": hist, "mine": ds.meta, "n_pairs": len(ds)},
        open(outdir / "train_stats.json", "w"),
        indent=2,
    )
    print(f"DONE o2={outdir}", flush=True)


if __name__ == "__main__":
    main()
