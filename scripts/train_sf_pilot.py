#!/usr/bin/env python3
"""Minimal Phase-G B/C generator pilot trainer (answer-supervised).

B: latents from *original* COCONUT, cached once, detached every step.
C: live differentiable self-rollout with current weights.

Example:
  CUDA_VISIBLE_DEVICES=5 python scripts/train_sf_pilot.py --mode C --max_steps 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.sf_rollout import RolloutBatch, sf_train_step, rollout_latent_thoughts  # noqa: E402
from src.order_pref.content_view import ContentProjector, content_view  # noqa: E402
from src.order_pref.set_align import mmd_rbf_multiscale  # noqa: E402


class SFPilotDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tok,
        max_target_len: int = 16,
        target_mode: str = "short",
        pseudo_path: Path | None = None,
        only_correct_pseudo: bool = True,
    ):
        self.tok = tok
        self.max_target_len = max_target_len
        self.target_mode = target_mode
        if target_mode == "pseudo":
            blob = json.load(open(pseudo_path))
            exs = blob["examples"]
            if only_correct_pseudo:
                exs = [e for e in exs if e.get("correct") and e.get("target_text")]
            self.data = exs
        else:
            self.data = json.load(open(path))

    def __len__(self):
        return len(self.data)

    def _target_text(self, ex) -> str:
        if self.target_mode == "pseudo":
            return ex["target_text"]
        ans = str(ex["answer"]).replace(",", "")
        if self.target_mode == "short":
            return f"### {ans}"
        if self.target_mode == "process":
            steps = ex.get("steps") or []
            body = "\n".join(steps)
            if body:
                return f"{body}\n### {ans}"
            return f"### {ans}"
        raise ValueError(f"unknown target_mode={self.target_mode}")

    def __getitem__(self, i):
        ex = self.data[i]
        q = ex["question"] + "\n<|start-latent|>"
        ans = self._target_text(ex)
        qenc = self.tok(q, add_special_tokens=True)
        aenc = self.tok(ans, add_special_tokens=False)
        aid = list(aenc["input_ids"][: self.max_target_len])
        if self.tok.eos_token_id is not None and (
            len(aid) == 0 or aid[-1] != self.tok.eos_token_id
        ):
            room = max(self.max_target_len - 1, 0)
            aid = aid[:room] + [self.tok.eos_token_id]
        return {
            "input_ids": qenc["input_ids"],
            "attention_mask": qenc["attention_mask"],
            "answer_ids": aid,
            "idx": int(ex["idx"]) if "idx" in ex else i,
        }


def collate(pad_id: int):
    def _fn(batch):
        def pad_right(seqs, pad_val):
            m = max(len(s) for s in seqs)
            out = torch.full((len(seqs), m), pad_val, dtype=torch.long)
            mask = torch.zeros((len(seqs), m), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, : len(s)] = 1
            return out, mask

        ids, qmask = pad_right([b["input_ids"] for b in batch], pad_id)
        # attention_mask from tokenizer is all-ones for tokens; rebuild from lengths
        ans, ans_mask = pad_right([b["answer_ids"] for b in batch], pad_id)
        # zero pad positions in answer mask
        for i, b in enumerate(batch):
            ans_mask[i, len(b["answer_ids"]) :] = 0
        idxs = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
        return {
            "batch": RolloutBatch(
                input_ids=ids,
                attention_mask=qmask,
                answer_ids=ans,
                answer_mask=ans_mask,
            ),
            "idx": idxs,
        }

    return _fn


def load_model(device: str, ckpt: str):
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")

    class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentCOCONUT.from_pretrained(
        ckpt,
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tok.pad_token_id,
        torch_dtype=torch.float32,
    )
    model.to(device)
    return model, tok


@torch.no_grad()
def build_latent_cache(model, ds, tok, device, latent_length: int, path: Path, batch_size: int):
    """Cache latents from *current* (should be original) weights, keyed by dataset idx."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate(tok.pad_token_id),
        drop_last=False,
    )
    by_idx = {}
    for pack in tqdm(loader, desc="cache B latents @ frozen coconut"):
        batch = pack["batch"]
        idxs = pack["idx"].tolist()
        batch.input_ids = batch.input_ids.to(device)
        batch.attention_mask = batch.attention_mask.to(device)
        lat, _, _ = rollout_latent_thoughts(
            model,
            batch.input_ids,
            batch.attention_mask,
            latent_length=latent_length,
            detach=True,
        )
        for j, idx in enumerate(idxs):
            # store only real (unpadded) question — latents are per-row full; OK
            by_idx[int(idx)] = lat[j].cpu()
    torch.save({"by_idx": by_idx, "latent_length": latent_length, "n": len(by_idx)}, path)
    print(f"saved {path} n={len(by_idx)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["B", "C"], required=True)
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Use 1 unless collate left-pads; right-pad breaks latent append.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_path", default="data/sf_pilot/train_latents_B.pt")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument(
        "--target_mode",
        choices=["short", "process", "pseudo"],
        default="short",
        help="short=### ans; process=dataset steps; pseudo=A correct trajectories",
    )
    ap.add_argument("--pseudo_path", default="data/sf_pilot/train_A_pseudo.json")
    ap.add_argument("--max_target_len", type=int, default=None,
                    help="Default 16 for short, 192 for process/pseudo")
    ap.add_argument("--lambda_mmd", type=float, default=0.0,
                    help="Weight for set-level MMD on content view (0=off)")
    ap.add_argument("--positive_latents", default="data/sf_pilot/train_A_positive_latents.pt")
    ap.add_argument("--mmd_bottleneck", type=int, default=64)
    ap.add_argument("--mmd_ramp_steps", type=int, default=50,
                    help="Linear ramp λ_mmd from 0→lambda_mmd over this many steps")
    args = ap.parse_args()
    if args.max_target_len is None:
        args.max_target_len = 16 if args.target_mode == "short" else 192

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{args.mode}_{args.target_mode}"
    if args.lambda_mmd > 0:
        tag += f"_mmd{args.lambda_mmd:g}"
    outdir = Path(args.outdir or f"outputs/sf_pilot_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(device, str(ROOT / args.ckpt))
    ds = SFPilotDataset(
        ROOT / args.train_json,
        tok,
        max_target_len=args.max_target_len,
        target_mode=args.target_mode,
        pseudo_path=ROOT / args.pseudo_path,
    )
    print(f"dataset size={len(ds)} target_mode={args.target_mode} lambda_mmd={args.lambda_mmd}")

    cache_by_idx = None
    cache_path = ROOT / args.cache_path
    if args.mode == "B":
        if args.rebuild_cache or not cache_path.exists():
            build_latent_cache(
                model, ds, tok, device, args.latent_length, cache_path, args.batch_size
            )
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_by_idx = blob["by_idx"]
        print(f"loaded B cache {cache_path}")

    positive_by_idx = None
    projector = None
    if args.lambda_mmd > 0:
        pos = torch.load(ROOT / args.positive_latents, map_location="cpu", weights_only=False)
        positive_by_idx = pos["by_idx"]
        dim = pos["dim"]
        projector = ContentProjector(dim, args.mmd_bottleneck).to(device)
        print(f"MMD on: positive n={pos['n_ok']} bottleneck={args.mmd_bottleneck}")

    model.train()
    if projector is not None:
        projector.train()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate(tok.pad_token_id),
        drop_last=True,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if projector is not None:
        params = params + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    json.dump({**vars(args), "outdir": str(outdir)}, open(outdir / "train_args.json", "w"), indent=2)

    step = 0
    losses = []
    pbar = tqdm(total=args.max_steps, desc=f"SF-{tag}")
    while step < args.max_steps:
        for pack in loader:
            batch = pack["batch"]
            idxs = pack["idx"].tolist()
            batch.input_ids = batch.input_ids.to(device)
            batch.attention_mask = batch.attention_mask.to(device)
            batch.answer_ids = batch.answer_ids.to(device)
            batch.answer_mask = batch.answer_mask.to(device)

            cached = None
            if args.mode == "B":
                cached = torch.stack([cache_by_idx[int(i)] for i in idxs], dim=0).to(device)

            opt.zero_grad(set_to_none=True)
            out = sf_train_step(
                model,
                batch,
                mode=args.mode,
                latent_length=args.latent_length,
                cached_latents=cached,
            )
            loss = out["loss"]
            loss_ans = loss.detach().item()
            loss_mmd_v = 0.0
            lam = 0.0
            if projector is not None and positive_by_idx is not None:
                # ramp
                if args.mmd_ramp_steps > 0:
                    lam = args.lambda_mmd * min(1.0, (step + 1) / args.mmd_ramp_steps)
                else:
                    lam = args.lambda_mmd
                # only rows with positive ref
                sf_lat = out["latents"]
                pos_list = []
                sf_list = []
                for j, qi in enumerate(idxs):
                    if int(qi) in positive_by_idx:
                        pos_list.append(positive_by_idx[int(qi)].to(device))
                        sf_list.append(sf_lat[j])
                if pos_list:
                    H_pos = torch.stack(pos_list, dim=0)  # (B', L, D)
                    H_sf = torch.stack(sf_list, dim=0)
                    C_sf = content_view(H_sf, mode="projector", projector=projector)
                    C_pos = content_view(H_pos, mode="projector", projector=projector)
                    # detach positive path through projector? keep projector training on both
                    loss_mmd = mmd_rbf_multiscale(C_sf, C_pos.detach())
                    loss = loss + lam * loss_mmd
                    loss_mmd_v = loss_mmd.detach().item()

            loss.backward()
            opt.step()
            losses.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "loss_ans": loss_ans,
                    "loss_mmd": loss_mmd_v,
                    "lam_mmd": lam,
                    "g_lat": out["grad_to_latent_norm"],
                }
            )
            step += 1
            pbar.set_postfix(
                ans=f"{loss_ans:.3f}",
                mmd=f"{loss_mmd_v:.3f}",
                lam=f"{lam:.3g}",
                g_lat=out["grad_to_latent_norm"],
            )
            pbar.update(1)
            if step >= args.max_steps:
                break
    pbar.close()

    model.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    if projector is not None:
        torch.save(projector.state_dict(), outdir / "content_projector.pt")
    json.dump(losses, open(outdir / "loss.json", "w"))
    print(f"DONE mode={args.mode} steps={step} out={outdir}")


if __name__ == "__main__":
    main()
