#!/usr/bin/env python3
"""Cache frozen-A latent trajectories for questions where final answer is correct.

Used as H^positive for set-level MMD in Phase-G2 D_mmd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.sf_rollout import rollout_latent_thoughts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--out", default="data/sf_pilot/train_A_positive_latents.pt")
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.ckpt))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")

    class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentCOCONUT.from_pretrained(
        str(ROOT / args.ckpt),
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tok.pad_token_id,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    gen = LatentGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        latent_length=args.latent_length,
        latent_do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    extractor = MODELS["coconut"]["answer_extractor"]
    train = json.load(open(ROOT / args.train_json))

    by_idx = {}
    n_ok = 0
    with torch.no_grad():
        for i, ex in enumerate(tqdm(train, desc="A+ latents")):
            gold = float(str(ex["answer"]).replace(",", ""))
            q = ex["question"] + "\n<|start-latent|>"
            enc = {k: v.to(device) for k, v in tok(q, return_tensors="pt").items()}
            # Use same differentiable rollout path (detached) so H^positive matches SF embeds
            lat, _, _ = rollout_latent_thoughts(
                model,
                enc["input_ids"],
                enc["attention_mask"],
                latent_length=args.latent_length,
                detach=True,
            )
            # Check correctness via full generate (text)
            out = model.generate(**enc, generation_config=gen, use_cache=True)
            seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
            pred = extractor(tok.decode(seq, skip_special_tokens=True))
            ok = pred == gold
            if ok:
                by_idx[i] = lat[0].cpu()  # (L, D)
                n_ok += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "by_idx": by_idx,
            "n": len(train),
            "n_ok": n_ok,
            "latent_length": args.latent_length,
            "dim": next(iter(by_idx.values())).shape[-1] if by_idx else None,
            "note": "Frozen A rollout latents only when final answer correct (greedy text check).",
        },
        out_path,
    )
    print(json.dumps({"out": str(out_path), "n_ok": n_ok, "n": len(train), "frac": n_ok / len(train)}, indent=2))


if __name__ == "__main__":
    main()
