#!/usr/bin/env python3
"""Fundamental SF+O2 pilot: **no cloning A-pseudo**.

Only learn from the model's own samples:
  correct AND contains << AND O2-best among K (with margin vs wrong)
  → SF live CE on that self trajectory.
Otherwise skip. Never use A-pseudo / dataset steps / short ### gold.

Optional: prefer higher O2 among process-correct; light format shaping via
selection only (not a separate clone loss).

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sf_o2_noclone.py --max_steps 200 --k 16
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, deque
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
        }


def load_generator(device, ckpt: str):
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentCOCONUT.from_pretrained(
        ckpt,
        latent_id=tok.convert_tokens_to_ids("<|latent|>"),
        latent_start_id=tok.convert_tokens_to_ids("<|start-latent|>"),
        latent_end_id=tok.convert_tokens_to_ids("<|end-latent|>"),
        target_id=tok.convert_tokens_to_ids(">>"),
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
def score_ids_latents(prm, input_ids: torch.Tensor, latents: torch.Tensor) -> float:
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    logits = prm(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        latent_embeds=latents.to(input_ids.device),
        return_dict=True,
    ).logits.squeeze(-1)
    mask = input_ids == prm.config.latent_id
    return float(torch.where(mask, logits, 0).sum().item())


@torch.no_grad()
def sample_and_score(model, prm, tok, question, gold, *, k, latent_length, max_new_tokens, device, dropout_p: float):
    extractor = MODELS["coconut"]["answer_extractor"]
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    enc = tok(question + "\n<|start-latent|>", return_tensors="pt")
    gen_cfg = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        dropout_p=dropout_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    out = model.generate(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        generation_config=gen_cfg,
        num_return_sequences=k,
        return_dict_in_generate=True,
        use_cache=True,
    )
    texts = tok.batch_decode(out.sequences, skip_special_tokens=True)
    preds = [extractor(t) for t in texts]
    rows = []
    for i in range(k):
        seq = out.sequences[i]
        if tok.pad_token_id is not None:
            nz = (seq != tok.pad_token_id).nonzero(as_tuple=False)
            if len(nz):
                seq = seq[: int(nz[-1]) + 1]
        after = ""
        seq_list = seq.tolist()
        if end_id in seq_list:
            epos = seq_list.index(end_id)
            after = tok.decode(seq_list[epos + 1 :], skip_special_tokens=False)
            after = after.replace(tok.pad_token or "", "")
            if tok.eos_token:
                after = after.replace(tok.eos_token, "")
            after = after.strip()
        score = score_ids_latents(prm, seq.unsqueeze(0).to(device), out.latent_thoughts[i])
        rows.append(
            {
                "after": after,
                "pred": preds[i],
                "correct": preds[i] == gold,
                "score": score,
                "has_process": "<<" in after,
                "starts_hash3": after.lstrip().startswith("###"),
                "after_len": len(after),
            }
        )
    return rows


def pick_winner(rows, *, margin_min: float):
    """Only process+correct winners. No A-pseudo."""
    proc_ok = [r for r in rows if r["correct"] and r["has_process"] and r["after"]]
    bad = [r for r in rows if not r["correct"]]
    info = {
        "n_ok": sum(1 for r in rows if r["correct"]),
        "n_proc_ok": len(proc_ok),
        "n_proc": sum(1 for r in rows if r["has_process"]),
        "n_hash3": sum(1 for r in rows if r["starts_hash3"]),
        "mean_after_len": sum(r["after_len"] for r in rows) / max(1, len(rows)),
    }
    if not proc_ok:
        return None, "no_process_correct", info
    # Prefer more process markers, then O2 score — still self-only, no A-pseudo.
    def rank(r):
        return (r["after"].count("<<"), r["score"], r["after_len"])

    winner = max(proc_ok, key=rank)
    info["win_score"] = winner["score"]
    info["win_alen"] = winner["after_len"]
    info["win_n_lt"] = winner["after"].count("<<")
    if bad:
        best_bad = max(bad, key=lambda r: r["score"])
        margin = winner["score"] - best_bad["score"]
        info["margin"] = margin
        if margin < margin_min:
            return None, "low_margin", info
    else:
        info["margin"] = None
    return winner, "update", info


def make_batch(tok, question, target_text, max_target_len, device):
    qenc = tok(question + "\n<|start-latent|>", add_special_tokens=True)
    aenc = tok(target_text, add_special_tokens=False)
    aid = list(aenc["input_ids"][:max_target_len])
    if tok.eos_token_id is not None and (not aid or aid[-1] != tok.eos_token_id):
        aid = aid + [tok.eos_token_id]
    return RolloutBatch(
        input_ids=torch.tensor([qenc["input_ids"]], dtype=torch.long, device=device),
        attention_mask=torch.tensor([qenc["attention_mask"]], dtype=torch.long, device=device),
        answer_ids=torch.tensor([aid], dtype=torch.long, device=device),
        answer_mask=torch.ones(1, len(aid), dtype=torch.long, device=device),
    ), len(aid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--prm_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin_min", type=float, default=0.0)
    ap.add_argument("--dropout_p", type=float, default=0.3, help="higher → more diverse / more << chance")
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--health_window", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"O2noclone_k{args.k}_p{args.dropout_p:g}"
    outdir = Path(args.outdir or f"outputs/sf_pilot_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id
    model, tok = load_generator(device, ckpt)
    prm = load_o2(device, prm_id, tok)
    ds = TrainJSON(ROOT / args.train_json)
    print(f"NO-CLONE n={len(ds)} k={args.k} dropout_p={args.dropout_p} margin_min={args.margin_min}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    json.dump({**vars(args), "outdir": str(outdir), "clone": False}, open(outdir / "train_args.json", "w"), indent=2)

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    skips: Counter = Counter()
    losses = []
    roll_proc_ok = deque(maxlen=args.health_window)
    roll_hash3 = deque(maxlen=args.health_window)
    attempts = 0
    max_attempts = args.max_steps * 80  # expect high skip

    pbar = tqdm(total=args.max_steps, desc=tag)
    dbg = open(outdir / "debug.jsonl", "w")

    while step < args.max_steps and attempts < max_attempts:
        attempts += 1
        if not order:
            order = list(range(len(ds)))
            random.shuffle(order)
        ex = ds[order.pop()]

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
            dropout_p=args.dropout_p,
        )
        winner, reason, info = pick_winner(rows, margin_min=args.margin_min)

        roll_proc_ok.append(info["n_proc_ok"] / args.k)
        roll_hash3.append(info["n_hash3"] / args.k)
        rec = {
            "attempt": attempts,
            "step": step,
            "idx": ex["idx"],
            "gate": reason,
            **info,
            "roll_proc_ok": sum(roll_proc_ok) / len(roll_proc_ok),
            "roll_hash3": sum(roll_hash3) / len(roll_hash3),
        }

        if winner is None:
            skips[reason] += 1
            rec["updated"] = False
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            pbar.set_postfix(
                skip=sum(skips.values()),
                why=reason[:16],
                pok=f"{rec['roll_proc_ok']:.2f}",
                rh=f"{rec['roll_hash3']:.2f}",
            )
            continue

        batch, tlen = make_batch(tok, ex["question"], winner["after"], args.max_target_len, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        out = sf_train_step(model, batch, mode="C", latent_length=args.latent_length)
        loss = out["loss"]
        loss.backward()
        opt.step()

        losses.append(
            {
                "step": step,
                "idx": ex["idx"],
                "loss": float(loss.item()),
                "tlen": tlen,
                "win_score": winner["score"],
                "margin": info.get("margin"),
                "n_proc_ok": info["n_proc_ok"],
            }
        )
        rec.update({"updated": True, "loss": float(loss.item()), "tlen": tlen})
        dbg.write(json.dumps(rec) + "\n")
        dbg.flush()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            tlen=tlen,
            pok=f"{rec['roll_proc_ok']:.2f}",
            skip=sum(skips.values()),
            m=f"{(info.get('margin') if info.get('margin') is not None else 9):.1f}",
        )

    pbar.close()
    dbg.close()
    model.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    json.dump(losses, open(outdir / "loss.json", "w"))
    stats = {
        "updates": step,
        "attempts": attempts,
        "skips": dict(skips),
        "update_rate": step / max(1, attempts),
        "final_roll_proc_ok": float(sum(roll_proc_ok) / len(roll_proc_ok)) if roll_proc_ok else None,
        "clone": False,
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"DONE steps={step} attempts={attempts} rate={stats['update_rate']:.3f} skips={dict(skips)} out={outdir}")


if __name__ == "__main__":
    main()
