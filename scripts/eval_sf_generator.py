#!/usr/bin/env python3
"""Eval generator Pass@1 / Coverage@N on a JSON split (no LatentRM / O2)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import datasets

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.utils import InferenceCollator, pass_at_k_mean  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="checkpoints/coconut")
    ap.add_argument("--data_path", default="data/sf_pilot/valid.json")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = str(ROOT / args.model_id) if not args.model_id.startswith("/") else args.model_id

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")

    class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentCOCONUT.from_pretrained(
        model_id,
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tok.pad_token_id,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    gen_cfg = LatentGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        latent_length=args.latent_length,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        dropout_p=0.2,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )

    raw = json.load(open(ROOT / args.data_path))
    ds = datasets.Dataset.from_list(raw)
    ds = ds.map(
        lambda x, idx: {
            "idx": idx,
            "question": x["question"] + "\n<|start-latent|>",
            "answer": float(str(x["answer"]).replace(",", "")),
        },
        with_indices=True,
    )
    ds = ds.map(lambda x: tok(x["question"]), batched=True)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=InferenceCollator(tok))

    extractor = MODELS["coconut"]["answer_extractor"]
    corrects = {}
    uniq_answers = {}

    for batch in tqdm(loader, desc="eval"):
        model_inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        out = model.generate(
            **model_inputs,
            generation_config=gen_cfg,
            num_return_sequences=args.n_samples,
            use_cache=True,
        )
        text = tok.batch_decode(out, skip_special_tokens=True)
        pred = [extractor(t) for t in text]
        for i, idx in enumerate(batch["idx"].tolist()):
            chunk = pred[i * args.n_samples : (i + 1) * args.n_samples]
            gold = batch["answer"][i].item() if torch.is_tensor(batch["answer"][i]) else batch["answer"][i]
            corrects[idx] = [c == gold for c in chunk]
            uniq_answers[idx] = len(set(chunk))

    arr = np.array([corrects[i] for i in sorted(corrects)])
    pass1 = float(arr[:, 0].mean()) if arr.shape[1] >= 1 else float("nan")
    # Coverage@N = fraction of questions with ≥1 correct among N
    cov = float(arr.any(axis=1).mean())
    # also pass@k via utility if available
    pass_at = {}
    for k in [1, 2, 4]:
        if k <= args.n_samples:
            pass_at[f"pass@{k}"] = float(pass_at_k_mean(arr, k))

    mean_uniq = float(np.mean([uniq_answers[i] for i in uniq_answers]))
    summary = {
        "model_id": model_id,
        "data_path": args.data_path,
        "n": len(corrects),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "pass@1_first_sample": pass1,
        "coverage@N": cov,
        **pass_at,
        "mean_unique_answers": mean_uniq,
    }
    print(json.dumps(summary, indent=2))
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(outp, "w"), indent=2)


if __name__ == "__main__":
    main()
