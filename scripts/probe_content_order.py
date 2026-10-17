#!/usr/bin/env python3
"""Order probe on content-view embeddings (Phase-G2).

If a linear probe can predict latent position from C(h), the view is NOT
position-free — report as set-level regularization only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.order_pref.content_view import ContentProjector, content_view  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive_latents", default="data/sf_pilot/train_A_positive_latents.pt")
    ap.add_argument("--projector", default=None, help="optional trained content_projector.pt")
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ROOT / args.positive_latents, map_location="cpu", weights_only=False)
    xs, ys = [], []
    for lat in blob["by_idx"].values():
        # lat: (L, D)
        for t in range(lat.shape[0]):
            xs.append(lat[t])
            ys.append(t)
    X = torch.stack(xs, dim=0).to(device)  # (N, D)
    Y = torch.tensor(ys, dtype=torch.long, device=device)
    L = int(Y.max().item()) + 1
    dim = X.shape[-1]

    projector = ContentProjector(dim, args.bottleneck).to(device)
    if args.projector:
        projector.load_state_dict(torch.load(ROOT / args.projector, map_location=device))
        projector.eval()
        for p in projector.parameters():
            p.requires_grad_(False)
    else:
        projector.train()

    # Probe on content embeddings
    with torch.no_grad() if args.projector else torch.enable_grad():
        C = content_view(X.unsqueeze(0), mode="projector", projector=projector).squeeze(0)

    # If projector was random/untrained and we want probe on frozen random view:
    if not args.projector:
        with torch.no_grad():
            C = content_view(X.unsqueeze(0), mode="projector", projector=projector).squeeze(0).detach()

    probe = nn.Linear(dim, L).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    # also baseline: probe on raw latents
    probe_raw = nn.Linear(dim, L).to(device)
    opt_raw = torch.optim.Adam(probe_raw.parameters(), lr=1e-2)

    def run(feat, head, optimizer):
        losses = []
        for _ in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = head(feat)
            loss = nn.functional.cross_entropy(logits, Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            acc = (head(feat).argmax(-1) == Y).float().mean().item()
        return acc, losses[-1]

    # detach features for probe-only training
    C_det = C.detach()
    X_det = X.detach()
    acc_c, _ = run(C_det, probe, opt)
    acc_raw, _ = run(X_det, probe_raw, opt_raw)
    chance = 1.0 / L
    summary = {
        "n": int(X.shape[0]),
        "L": L,
        "chance": chance,
        "probe_acc_content_view": acc_c,
        "probe_acc_raw_latent": acc_raw,
        "projector_path": args.projector,
        "claim": (
            "NOT position-free"
            if acc_c > chance * 2
            else "weak order signal in content view"
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(outp, "w"), indent=2)


if __name__ == "__main__":
    main()
