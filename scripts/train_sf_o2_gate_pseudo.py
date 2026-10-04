#!/usr/bin/env python3
"""SF + frozen O2 gate + A-pseudo CE target (Phase-G2).

Idea: O2 decides *whether/which question* to update; CE target is A-pseudo
trajectory text (known non-collapsing), not short ### and not noisy self-after.

Per step:
  1) no_grad sample K from current generator
  2) freeze O2 scores (BoN aggregation)
  3) gate (configurable): need ≥1 correct; optional O2 margin vs best wrong
  4) SF live rollout + CE on A-pseudo target_text for that idx
  5) log rich debug signals every step; print rolling health

O2 never gets gradients.

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sf_o2_gate_pseudo.py --max_steps 200 --k 4
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


class PseudoTrain(Dataset):
    """Only questions that have A-correct pseudo targets."""

    def __init__(self, train_json: Path, pseudo_json: Path):
        train = json.load(open(train_json))
        blob = json.load(open(pseudo_json))
        by_idx = {
            int(e["idx"]): e
            for e in blob["examples"]
            if e.get("correct") and e.get("target_text")
        }
        self.rows = []
        for i, ex in enumerate(train):
            if i not in by_idx:
                continue
            pe = by_idx[i]
            self.rows.append(
                {
                    "idx": i,
                    "question": ex["question"],
                    "answer": float(str(ex["answer"]).replace(",", "")),
                    "answer_str": str(ex["answer"]).replace(",", ""),
                    "target_text": pe["target_text"],
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


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
def sample_and_score(model, prm, tok, question, gold, *, k, latent_length, max_new_tokens, device):
    extractor = MODELS["coconut"]["answer_extractor"]
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    q = question + "\n<|start-latent|>"
    enc = tok(q, return_tensors="pt")
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
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
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
            after = after.replace(tok.pad_token or "", "")
            if tok.eos_token:
                after = after.replace(tok.eos_token, "")
            after = after.strip()
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
                "after": afters[i],
                "pred": preds[i],
                "correct": bool(corrects[i]),
                "score": float(scores[i].item()),
                "has_process": "<<" in afters[i],
                "starts_hash3": afters[i].lstrip().startswith("###"),
                "after_len": len(afters[i]),
            }
        )
    return rows


def gate_decision(rows, *, margin_min: float, require_best_process: bool):
    """Return (ok, reason, info)."""
    n_ok = sum(1 for r in rows if r["correct"])
    n_proc = sum(1 for r in rows if r["has_process"])
    n_hash = sum(1 for r in rows if r["starts_hash3"])
    info = {
        "n_ok": n_ok,
        "n_proc": n_proc,
        "n_hash3": n_hash,
        "mean_after_len": sum(r["after_len"] for r in rows) / max(1, len(rows)),
        "scores": [r["score"] for r in rows],
        "corrects": [r["correct"] for r in rows],
    }
    if n_ok == 0:
        return False, "no_correct", info

    best_ok = max((r for r in rows if r["correct"]), key=lambda r: r["score"])
    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        best_bad = max(wrong, key=lambda r: r["score"])
        margin = best_ok["score"] - best_bad["score"]
    else:
        margin = None  # all correct — treat as pass
    info["best_ok_score"] = best_ok["score"]
    info["margin"] = margin
    info["best_ok_process"] = best_ok["has_process"]
    info["best_ok_after_len"] = best_ok["after_len"]

    if require_best_process and not best_ok["has_process"]:
        return False, "best_not_process", info
    if margin_min > 0 and margin is not None and margin < margin_min:
        return False, "low_margin", info
    return True, "update", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--pseudo_path", default="data/sf_pilot/train_A_pseudo.json")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--prm_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin_min", type=float, default=0.5,
                    help="Require O2(best_correct)-O2(best_wrong) ≥ this; 0=off")
    ap.add_argument("--require_best_process", action="store_true", default=False,
                    help="Require O2-best correct sample to contain <<")
    ap.add_argument("--health_window", type=int, default=30)
    ap.add_argument("--abort_neg_margin_rate", type=float, default=0.5,
                    help="Abort if rolling fraction of updates with margin<0 exceeds this")
    ap.add_argument("--abort_hash3_rate", type=float, default=0.98,
                    help="Abort if rolling sampled starts_### rate exceeds this (near-total)")
    ap.add_argument("--abort_mean_after_len", type=float, default=3.0,
                    help="Abort only if after_len nearly empty (short <<### is normal ~8-20)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"O2gate_pseudo_k{args.k}"
    if args.margin_min > 0:
        tag += f"_m{args.margin_min:g}"
    outdir = Path(args.outdir or f"outputs/sf_pilot_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    debug_path = outdir / "debug.jsonl"

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id

    model, tok = load_generator(device, ckpt)
    prm = load_o2(device, prm_id, tok)
    ds = PseudoTrain(ROOT / args.train_json, ROOT / args.pseudo_path)
    print(f"pseudo_usable={len(ds)} k={args.k} margin_min={args.margin_min}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    json.dump({**vars(args), "outdir": str(outdir)}, open(outdir / "train_args.json", "w"), indent=2)

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    skip_counts: Counter = Counter()
    losses = []
    roll_hash3 = deque(maxlen=args.health_window)
    roll_alen = deque(maxlen=args.health_window)
    roll_nok = deque(maxlen=args.health_window)
    roll_neg_m = deque(maxlen=args.health_window)
    aborted = None

    pbar = tqdm(total=args.max_steps, desc=tag)
    dbg = open(debug_path, "w")

    attempts = 0
    max_attempts = args.max_steps * 30

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
        )
        ok, reason, info = gate_decision(
            rows, margin_min=args.margin_min, require_best_process=args.require_best_process
        )

        hash3_rate = info["n_hash3"] / args.k
        roll_hash3.append(hash3_rate)
        roll_alen.append(info["mean_after_len"])
        roll_nok.append(info["n_ok"] / args.k)

        rec = {
            "attempt": attempts,
            "step": step,
            "idx": ex["idx"],
            "gate": reason,
            **{
                k: info[k]
                for k in (
                    "n_ok",
                    "n_proc",
                    "n_hash3",
                    "mean_after_len",
                    "margin",
                    "best_ok_score",
                    "best_ok_process",
                    "best_ok_after_len",
                )
                if k in info
            },
            "roll_hash3": sum(roll_hash3) / len(roll_hash3),
            "roll_alen": sum(roll_alen) / len(roll_alen),
            "roll_nok": sum(roll_nok) / len(roll_nok),
        }

        if not ok:
            skip_counts[reason] += 1
            rec["updated"] = False
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            pbar.set_postfix(
                skip=sum(skip_counts.values()),
                why=reason[:12],
                rh=f"{rec['roll_hash3']:.2f}",
                ra=f"{rec['roll_alen']:.0f}",
                rok=f"{rec['roll_nok']:.2f}",
            )
            continue

        ans_text = ex["target_text"]
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

        m = info.get("margin")
        roll_neg_m.append(1.0 if (m is not None and m < 0) else 0.0)

        losses.append(
            {
                "step": step,
                "idx": ex["idx"],
                "loss": float(loss.item()),
                "target_len": len(aid),
                "n_ok": info["n_ok"],
                "margin": m,
                "g_lat": float(out.get("grad_to_latent_norm") or 0.0),
            }
        )
        rec.update(
            {
                "updated": True,
                "loss": float(loss.item()),
                "target_len": len(aid),
                "g_lat": float(out.get("grad_to_latent_norm") or 0.0),
                "roll_neg_m": sum(roll_neg_m) / len(roll_neg_m) if roll_neg_m else 0.0,
            }
        )
        dbg.write(json.dumps(rec) + "\n")
        dbg.flush()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            skip=sum(skip_counts.values()),
            loss=f"{loss.item():.3f}",
            tlen=len(aid),
            rh=f"{rec['roll_hash3']:.2f}",
            ra=f"{rec['roll_alen']:.0f}",
            rok=f"{rec['roll_nok']:.2f}",
            m=f"{(m if m is not None else 9):.1f}",
            nm=f"{rec['roll_neg_m']:.2f}",
        )

        if len(roll_hash3) >= args.health_window:
            if rec["roll_hash3"] >= args.abort_hash3_rate and rec["roll_alen"] < args.abort_mean_after_len:
                aborted = f"format_collapse hash3={rec['roll_hash3']:.2f} alen={rec['roll_alen']:.1f}"
                break
            if roll_neg_m and (sum(roll_neg_m) / len(roll_neg_m)) > args.abort_neg_margin_rate:
                aborted = f"neg_margin_rate={sum(roll_neg_m)/len(roll_neg_m):.2f}"
                break

    pbar.close()
    dbg.close()

    model.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    json.dump(losses, open(outdir / "loss.json", "w"))
    stats = {
        "updates": step,
        "attempts": attempts,
        "skips": dict(skip_counts),
        "aborted": aborted,
        "final_roll_hash3": float(sum(roll_hash3) / len(roll_hash3)) if roll_hash3 else None,
        "final_roll_alen": float(sum(roll_alen) / len(roll_alen)) if roll_alen else None,
        "final_roll_nok": float(sum(roll_nok) / len(roll_nok)) if roll_nok else None,
        "final_roll_neg_m": float(sum(roll_neg_m) / len(roll_neg_m)) if roll_neg_m else None,
        "margin_min": args.margin_min,
        "require_best_process": args.require_best_process,
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"DONE steps={step} skips={dict(skip_counts)} aborted={aborted} out={outdir}")


if __name__ == "__main__":
    main()
