#!/usr/bin/env python3
"""SF-style generator + frozen O2 rejection-sampling (RFT) pilot.

Per step:
  1) no_grad: sample K candidates from current generator
  2) freeze O2 scores them (sum of latent-token logits, same as BoN)
  3) among *correct* samples, pick O2-best; if none correct → skip
  4) SF live rollout + CE on gold ``### {ans}`` (grad through rollout)

O2 is never backpropped. MMD off by default (optional later on O2-high set).

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sf_o2_rft.py --max_steps 200 --k 4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig  # noqa: E402
from src.models.gpt2 import COCONUTGPT2ForTokenClassification  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.sf_rollout import RolloutBatch, sf_train_step  # noqa: E402


class TrainJSON(Dataset):
    def __init__(self, path: Path):
        self.data = json.load(open(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        return {
            "idx": i,
            "question": ex["question"],
            "answer": float(str(ex["answer"]).replace(",", "")),
            "answer_str": str(ex["answer"]).replace(",", ""),
        }


def load_generator(device, ckpt: str):
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    target_id = tok.convert_tokens_to_ids(">>")

    class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentCOCONUT.from_pretrained(
        ckpt,
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        target_id=target_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tok.pad_token_id,
        torch_dtype=torch.float32,
    ).to(device)
    return model, tok


def load_o2(device, prm_id: str, tok):
    prm = COCONUTGPT2ForTokenClassification.from_pretrained(
        prm_id,
        latent_id=tok.convert_tokens_to_ids("<|latent|>"),
        latent_start_id=tok.convert_tokens_to_ids("<|start-latent|>"),
        latent_end_id=tok.convert_tokens_to_ids("<|end-latent|>"),
        target_id=tok.convert_tokens_to_ids(">>"),
        pad_token_id=tok.pad_token_id,
        torch_dtype=torch.float32,
    ).to(device)
    prm.eval()
    for p in prm.parameters():
        p.requires_grad_(False)
    return prm


@torch.no_grad()
def sample_and_score(
    model,
    prm,
    tok,
    question: str,
    gold: float,
    *,
    k: int,
    latent_length: int,
    max_new_tokens: int,
    device: str,
):
    """Return list of dicts: text, after, pred, correct, score."""
    extractor = MODELS["coconut"]["answer_extractor"]
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    q = question + "\n<|start-latent|>"
    enc = tok(q, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_cfg = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        dropout_p=0.2,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_cfg,
        num_return_sequences=k,
        return_dict_in_generate=True,
        use_cache=True,
    )
    texts = tok.batch_decode(out.sequences, skip_special_tokens=True)
    preds = [extractor(t) for t in texts]
    corrects = [p == gold for p in preds]

    afters = []
    for i in range(k):
        seq = out.sequences[i].tolist()
        after = ""
        if end_id in seq:
            epos = seq.index(end_id)
            after = tok.decode(seq[epos + 1 :], skip_special_tokens=False)
            # strip trailing pad/eos clutter for cleaner CE
            after = after.replace(tok.pad_token or "", "").strip()
            if tok.eos_token:
                after = after.replace(tok.eos_token, "").strip()
        afters.append(after)

    inputs = tok(texts, return_tensors="pt", padding=True)
    logits = prm(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        latent_embeds=out.latent_thoughts.to(device),
        return_dict=True,
    ).logits.squeeze(-1)
    mask = (inputs["input_ids"] == prm.config.latent_id).to(device)
    scores = torch.where(mask, logits, 0).sum(dim=-1)

    rows = []
    for i in range(k):
        rows.append(
            {
                "text": texts[i],
                "after": afters[i],
                "pred": preds[i],
                "correct": bool(corrects[i]),
                "score": float(scores[i].item()),
            }
        )
    return rows


def pick_target(rows, *, require_process: bool = True):
    """O2-best among correct trajs. If require_process, must contain << (skip short ###)."""
    ok = [r for r in rows if r["correct"] and r.get("after")]
    if require_process:
        ok = [r for r in ok if "<<" in r["after"]]
    if not ok:
        return None
    return max(ok, key=lambda r: r["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--prm_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_target_len", type=int, default=192,
                    help="Use long targets (picked trajectory), not short ### only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_mode", choices=["traj", "short"], default="traj",
                    help="traj=O2-picked after-end-latent text; short=### gold (collapses)")
    ap.add_argument("--require_process", action="store_true", default=True,
                    help="Only update on correct trajs containing << (avoid ### collapse)")
    ap.add_argument("--allow_short_correct", action="store_true",
                    help="Allow ###-only correct targets (sets require_process=False)")
    args = ap.parse_args()
    if args.allow_short_correct:
        args.require_process = False

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir or f"outputs/sf_pilot_O2rft_{args.target_mode}_k{args.k}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id

    model, tok = load_generator(device, ckpt)
    prm = load_o2(device, prm_id, tok)
    ds = TrainJSON(ROOT / args.train_json)
    print(f"dataset={len(ds)} k={args.k} target_mode={args.target_mode} prm={prm_id}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    json.dump({**vars(args), "outdir": str(outdir)}, open(outdir / "train_args.json", "w"), indent=2)

    step = 0
    skips = 0
    losses = []
    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)
    pbar = tqdm(total=args.max_steps, desc=f"SF-O2rft-{args.target_mode}-k{args.k}")

    while step < args.max_steps:
        if not order:
            order = list(range(len(ds)))
            random.shuffle(order)
        qi = order.pop()
        ex = ds[qi]

        model.eval()
        rows = sample_and_score(
            model,
            prm,
            tok,
            ex["question"],
            ex["answer"],
            k=args.k,
            latent_length=args.latent_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        picked = pick_target(rows, require_process=args.require_process)
        if picked is None:
            skips += 1
            pbar.set_postfix(skip=skips, ans="-", n_ok=0)
            continue

        if args.target_mode == "short":
            ans_text = f"### {ex['answer_str']}"
        else:
            ans_text = picked["after"]
        qenc = tok(ex["question"] + "\n<|start-latent|>", add_special_tokens=True)
        aenc = tok(ans_text, add_special_tokens=False)
        aid = list(aenc["input_ids"][: args.max_target_len])
        if tok.eos_token_id is not None and (not aid or aid[-1] != tok.eos_token_id):
            aid = aid + [tok.eos_token_id]
        batch = RolloutBatch(
            input_ids=torch.tensor([qenc["input_ids"]], dtype=torch.long, device=device),
            attention_mask=torch.tensor([qenc["attention_mask"]], dtype=torch.long, device=device),
            answer_ids=torch.tensor([aid], dtype=torch.long, device=device),
            answer_mask=torch.ones(1, len(aid), dtype=torch.long, device=device),
        )

        model.train()
        opt.zero_grad(set_to_none=True)
        out = sf_train_step(model, batch, mode="C", latent_length=args.latent_length)
        loss = out["loss"]
        loss.backward()
        opt.step()

        n_ok = sum(1 for r in rows if r["correct"])
        losses.append(
            {
                "step": step,
                "idx": qi,
                "loss": float(loss.item()),
                "n_ok": n_ok,
                "pick_score": picked["score"],
                "target_len": len(aid),
                "g_lat": float(out.get("grad_to_latent_norm") or 0.0),
            }
        )
        step += 1
        pbar.update(1)
        pbar.set_postfix(
            skip=skips,
            ans=f"{loss.item():.3f}",
            n_ok=n_ok,
            tlen=len(aid),
            sc=f"{picked['score']:.2f}",
        )

    pbar.close()
    model.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    json.dump(losses, open(outdir / "loss.json", "w"))
    json.dump({"skips": skips, "updates": step, "target_mode": args.target_mode}, open(outdir / "stats.json", "w"))
    print(f"DONE steps={step} skips={skips} out={outdir}")


if __name__ == "__main__":
    main()
