#!/usr/bin/env python3
"""Label gsm_train questions as empty / leftover / ok at k=16 (never gsm_test).

Writes a train JSON of empty(+optional leftover) originals for coverage upsample.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.paths import MODELS  # noqa: E402
from src.system_scoring import aggregate_latent_scores, load_prm  # noqa: E402
from train_p123_system import TrainJSON, generate_batch, load_generator  # noqa: E402


def score_pool(prm, seqs, lats, device, chunk=256):
    pad = prm.config.pad_token_id if prm.config.pad_token_id is not None else 0
    out = []
    dtype = next(prm.parameters()).dtype
    for s in range(0, seqs.size(0), chunk):
        ids = seqs[s : s + chunk].to(device)
        lat = lats[s : s + chunk].to(device=device, dtype=dtype)
        logits = prm(
            input_ids=ids,
            attention_mask=(ids != pad).long(),
            latent_embeds=lat,
            return_dict=True,
        ).logits.squeeze(-1)
        mask = ids == prm.config.latent_id
        out.append(aggregate_latent_scores(logits, mask, reduce="sum_logit").detach().float().cpu())
    return torch.cat(out, dim=0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/gsm_train.json")
    ap.add_argument("--ckpt", default="outputs/p123_dualBeat_20261001_202106/model")
    ap.add_argument("--o2_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--out_json", default="data/gsm_train_k16empty.json")
    ap.add_argument("--out_meta", default="latent-data/o2_hard_leftover/k16_buckets.json")
    ap.add_argument("--n_q", type=int, default=12000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--q_batch", type=int, default=32)
    ap.add_argument("--empty_repeat", type=int, default=2)
    ap.add_argument("--include_leftover", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--latent_length", type=int, default=6)
    args = ap.parse_args()
    assert "test" not in Path(args.train_json).name.lower()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    raw_path = ROOT / args.train_json
    raw = json.load(open(raw_path))
    ds = TrainJSON(raw_path)
    rng = random.Random(args.seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    order = order[: args.n_q]

    student, tok = load_generator(device, str(ROOT / args.ckpt), trainable=False, dtype=torch.bfloat16)
    o2 = load_prm(str(ROOT / args.o2_id), tok, device, dtype=torch.bfloat16, trainable=False)
    gen_tok = AutoTokenizer.from_pretrained(str(ROOT / args.ckpt), padding_side="left")
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    extractor = MODELS["coconut"]["answer_extractor"]

    empty_idx, leftover_idx, ok_idx = [], [], []
    pbar = tqdm(range(0, len(order), args.q_batch), desc="mine_k16_buckets")
    for bi in pbar:
        idxs = order[bi : bi + args.q_batch]
        exs = [ds[i] for i in idxs]
        qs = [ex["question"] for ex in exs]
        seqs, lats = generate_batch(
            student,
            gen_tok,
            qs,
            k=args.k,
            sample=True,
            latent_length=args.latent_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dropout_p=args.dropout_p,
        )
        texts = gen_tok.batch_decode(seqs, skip_special_tokens=True)
        scores = score_pool(o2, seqs, lats, device)
        for qi, ex in enumerate(exs):
            gold = ex["answer"]
            rows = []
            for j in range(args.k):
                i = qi * args.k + j
                pred = extractor(texts[i])
                rows.append({"correct": pred == gold, "score": scores[i]})
            qid = int(ex["idx"])
            correct = [r for r in rows if r["correct"]]
            if not correct:
                empty_idx.append(qid)
                continue
            top = max(rows, key=lambda r: r["score"])
            if top["correct"]:
                ok_idx.append(qid)
            else:
                leftover_idx.append(qid)
        pbar.set_postfix(empty=len(empty_idx), left=len(leftover_idx), ok=len(ok_idx))

    keep = []
    for qid in empty_idx:
        keep.extend([raw[qid]] * args.empty_repeat)
    if args.include_leftover:
        keep.extend(raw[qid] for qid in leftover_idx)
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(keep, open(out_json, "w"))
    meta = {
        **{k: v for k, v in vars(args).items() if k != "include_leftover"},
        "include_leftover": args.include_leftover,
        "n_q": len(order),
        "n_empty": len(empty_idx),
        "n_leftover": len(leftover_idx),
        "n_ok": len(ok_idx),
        "n_keep": len(keep),
        "empty_idx": empty_idx,
        "leftover_idx": leftover_idx,
    }
    meta_path = ROOT / args.out_meta
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(meta_path, "w"))
    print(
        f"WROTE {out_json} keep={len(keep)} empty={len(empty_idx)} leftover={len(leftover_idx)} ok={len(ok_idx)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
