#!/usr/bin/env python3
"""Mine leftover gold-vs-thief pairs from gsm_train (never gsm_test).

Frozen dualBeat generator + frozen O2, k matches test BoN so the ranking
problem is the same difficulty as leftover on GSM-Test. Latents detached.
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
    ap.add_argument("--out", default="latent-data/o2_hard_leftover/train_k16.pt")
    ap.add_argument("--n_q", type=int, default=4000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--q_batch", type=int, default=16)
    ap.add_argument("--n_thieves", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--latent_length", type=int, default=6)
    args = ap.parse_args()
    assert "test" not in Path(args.train_json).name.lower(), "do not mine GSM-Test"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = TrainJSON(ROOT / args.train_json)
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

    pairs = []
    n_pool = n_left = n_empty = 0
    pbar = tqdm(range(0, len(order), args.q_batch), desc="mine_leftover")
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
        lats_cpu = lats.detach().float().cpu()
        seqs_cpu = seqs.detach().cpu()
        for qi, ex in enumerate(exs):
            gold = ex["answer"]
            rows = []
            for j in range(args.k):
                i = qi * args.k + j
                pred = extractor(texts[i])
                rows.append(
                    {
                        "correct": pred == gold,
                        "score": scores[i],
                        "ids": seqs_cpu[i],
                        "lat": lats_cpu[i],
                    }
                )
            correct = [r for r in rows if r["correct"]]
            wrong = [r for r in rows if not r["correct"]]
            if not correct:
                n_empty += 1
                continue
            n_pool += 1
            gold_r = max(correct, key=lambda r: r["score"])
            top = max(rows, key=lambda r: r["score"])
            if top["correct"] or not wrong:
                continue
            n_left += 1
            thieves = sorted(wrong, key=lambda r: r["score"], reverse=True)[: args.n_thieves]
            pairs.append(
                {
                    "qid": ex["idx"],
                    "gold_ids": gold_r["ids"],
                    "gold_lat": gold_r["lat"],
                    "gold_score": gold_r["score"],
                    "thief_ids": [t["ids"] for t in thieves],
                    "thief_lat": [t["lat"] for t in thieves],
                    "thief_score": [t["score"] for t in thieves],
                    "gap": thieves[0]["score"] - gold_r["score"],
                }
            )
        pbar.set_postfix(pool=n_pool, left=n_left, pairs=len(pairs))

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        **vars(args),
        "n_q": len(order),
        "n_pool": n_pool,
        "n_empty": n_empty,
        "n_leftover": n_left,
        "n_pairs": len(pairs),
        "note": "gsm_train leftover only; generator frozen; k matches test BoN",
    }
    torch.save({"pairs": pairs, "meta": meta}, out)
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"WROTE {out} leftover={n_left} pool={n_pool} empty={n_empty}", flush=True)


if __name__ == "__main__":
    main()
