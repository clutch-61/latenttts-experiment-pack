#!/usr/bin/env python3
"""SF + frozen O2 with *beat-A* target switch (Phase-G2).

Defect addressed: O2gate still CE-clones A-pseudo, so cannot beat A.
Fix: score A-pseudo (cached latents) with O2; self-sample K; if best
correct self scores higher than A by margin_beat → CE on *self* traj;
else fall back to A-pseudo CE (same as O2gate). O2 never backpropped.

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sf_o2_beat_a.py --max_steps 200 --k 8
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


class BeatADataset(Dataset):
    def __init__(self, train_json: Path, pseudo_json: Path, pos_latents: Path):
        train = json.load(open(train_json))
        blob = json.load(open(pseudo_json))
        pos = torch.load(pos_latents, map_location="cpu", weights_only=False)
        by_pseudo = {
            int(e["idx"]): e
            for e in blob["examples"]
            if e.get("correct") and e.get("target_text")
        }
        self.rows = []
        for i, ex in enumerate(train):
            if i not in by_pseudo or i not in pos["by_idx"]:
                continue
            pe = by_pseudo[i]
            self.rows.append(
                {
                    "idx": i,
                    "question": ex["question"],
                    "answer": float(str(ex["answer"]).replace(",", "")),
                    "answer_str": str(ex["answer"]).replace(",", ""),
                    "target_text": pe["target_text"],
                    "a_latents": pos["by_idx"][i],  # (L, D)
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
def score_ids_latents(prm, input_ids: torch.Tensor, latents: torch.Tensor) -> float:
    """input_ids (1, T); latents (1, L, D) or (L, D). BoN-style sum on latent positions."""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    attn = torch.ones_like(input_ids)
    logits = prm(
        input_ids=input_ids,
        attention_mask=attn,
        latent_embeds=latents.to(input_ids.device),
        return_dict=True,
    ).logits.squeeze(-1)
    mask = input_ids == prm.config.latent_id
    return float(torch.where(mask, logits, 0).sum().item())


@torch.no_grad()
def score_a_pseudo(prm, tok, question: str, target_text: str, a_latents: torch.Tensor, device: str) -> float:
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    L = a_latents.shape[0]
    q_ids = tok(question + "\n<|start-latent|>", add_special_tokens=True)["input_ids"]
    a_ids = tok(target_text, add_special_tokens=False)["input_ids"]
    ids = q_ids + [latent_id] * L + [end_id] + a_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return score_ids_latents(prm, input_ids, a_latents.to(device))


@torch.no_grad()
def sample_and_score(model, prm, tok, question, gold, *, k, latent_length, max_new_tokens, device):
    extractor = MODELS["coconut"]["answer_extractor"]
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    enc = tok(question + "\n<|start-latent|>", return_tensors="pt")
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

    # Score each sample the same way as A-pseudo (explicit ids + latents), not tok(text).
    scores = []
    for i in range(k):
        # trim pad
        seq = out.sequences[i]
        if tok.pad_token_id is not None:
            nz = (seq != tok.pad_token_id).nonzero(as_tuple=False)
            if len(nz):
                seq = seq[: int(nz[-1]) + 1]
        sc = score_ids_latents(prm, seq.unsqueeze(0).to(device), out.latent_thoughts[i].to(device))
        scores.append(sc)

    rows = []
    for i in range(k):
        rows.append(
            {
                "after": afters[i],
                "pred": preds[i],
                "correct": preds[i] == gold,
                "score": float(scores[i]),
                "has_process": "<<" in afters[i],
                "starts_hash3": afters[i].lstrip().startswith("###"),
                "after_len": len(afters[i]),
            }
        )
    return rows


def choose_target(
    rows,
    a_score: float,
    *,
    margin_beat: float,
    margin_gate: float,
    match_slack: float,
):
    """Return (target_text, reason, info)."""
    ok = [r for r in rows if r["correct"] and r["after"]]
    bad = [r for r in rows if not r["correct"]]
    proc_ok = [r for r in ok if r["has_process"]]
    info = {
        "n_ok": len(ok),
        "n_bad": len(bad),
        "n_proc": sum(1 for r in rows if r["has_process"]),
        "n_proc_ok": len(proc_ok),
        "n_hash3": sum(1 for r in rows if r["starts_hash3"]),
        "mean_after_len": sum(r["after_len"] for r in rows) / max(1, len(rows)),
        "a_score": a_score,
    }
    if not ok:
        return None, "no_correct", info

    if bad:
        best_any = max(ok, key=lambda r: r["score"])
        best_bad = max(bad, key=lambda r: r["score"])
        info["margin_vs_wrong"] = best_any["score"] - best_bad["score"]
    else:
        info["margin_vs_wrong"] = None

    if proc_ok:
        best_p = max(proc_ok, key=lambda r: r["score"])
        gap = best_p["score"] - a_score
        info.update(
            {
                "best_score": best_p["score"],
                "best_proc": True,
                "best_alen": best_p["after_len"],
                "gap_vs_a": gap,
            }
        )
        if gap >= margin_beat:
            return best_p["after"], "beat_a_self", info
        if gap >= -match_slack:
            return best_p["after"], "match_a_self", info
    else:
        best = max(ok, key=lambda r: r["score"])
        info.update(
            {
                "best_score": best["score"],
                "best_proc": False,
                "best_alen": best["after_len"],
                "gap_vs_a": best["score"] - a_score,
            }
        )

    if info["margin_vs_wrong"] is not None and info["margin_vs_wrong"] < margin_gate:
        return None, "low_margin", info
    return None, "fallback_pseudo", info


def make_batch(tok, question, target_text, max_target_len, device):
    qenc = tok(question + "\n<|start-latent|>", add_special_tokens=True)
    aenc = tok(target_text, add_special_tokens=False)
    aid = list(aenc["input_ids"][:max_target_len])
    if tok.eos_token_id is not None and (not aid or aid[-1] != tok.eos_token_id):
        aid = aid + [tok.eos_token_id]
    batch = RolloutBatch(
        input_ids=torch.tensor([qenc["input_ids"]], dtype=torch.long, device=device),
        attention_mask=torch.tensor([qenc["attention_mask"]], dtype=torch.long, device=device),
        answer_ids=torch.tensor([aid], dtype=torch.long, device=device),
        answer_mask=torch.ones(1, len(aid), dtype=torch.long, device=device),
    )
    return batch, len(aid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--pseudo_path", default="data/sf_pilot/train_A_pseudo.json")
    ap.add_argument("--positive_latents", default="data/sf_pilot/train_A_positive_latents.pt")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--prm_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin_beat", type=float, default=0.0, help="self process must be >= A+this")
    ap.add_argument("--match_slack", type=float, default=1.0,
                    help="if process score >= A-match_slack, still use self (near-A)")
    ap.add_argument("--margin_gate", type=float, default=0.5, help="for A-pseudo fallback gate vs wrong")
    ap.add_argument("--fallback_pseudo", action="store_true", default=True)
    ap.add_argument("--no_fallback_pseudo", action="store_true")
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--health_window", type=int, default=40)
    args = ap.parse_args()
    if args.no_fallback_pseudo:
        args.fallback_pseudo = False

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"O2beatA_k{args.k}_b{args.margin_beat:g}_s{args.match_slack:g}"
    outdir = Path(args.outdir or f"outputs/sf_pilot_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id
    model, tok = load_generator(device, ckpt)
    prm = load_o2(device, prm_id, tok)
    ds = BeatADataset(ROOT / args.train_json, ROOT / args.pseudo_path, ROOT / args.positive_latents)
    print(
        f"n={len(ds)} k={args.k} margin_beat={args.margin_beat} "
        f"match_slack={args.match_slack} fallback={args.fallback_pseudo}"
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    json.dump({**vars(args), "outdir": str(outdir)}, open(outdir / "train_args.json", "w"), indent=2)

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    skips: Counter = Counter()
    src_counts: Counter = Counter()
    losses = []
    roll_beat = deque(maxlen=args.health_window)
    roll_gap = deque(maxlen=args.health_window)
    attempts = 0
    max_attempts = args.max_steps * 30
    aborted = None

    pbar = tqdm(total=args.max_steps, desc=tag)
    dbg = open(outdir / "debug.jsonl", "w")

    while step < args.max_steps and attempts < max_attempts:
        attempts += 1
        if not order:
            order = list(range(len(ds)))
            random.shuffle(order)
        ex = ds[order.pop()]

        model.eval()
        a_score = score_a_pseudo(
            prm, tok, ex["question"], ex["target_text"], ex["a_latents"], device
        )
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
        target, reason, info = choose_target(
            rows,
            a_score,
            margin_beat=args.margin_beat,
            margin_gate=args.margin_gate,
            match_slack=args.match_slack,
        )

        # resolve fallback
        if reason == "fallback_pseudo":
            if args.fallback_pseudo:
                target = ex["target_text"]
                reason = "a_pseudo"
            else:
                reason = "no_beat_skip"
                target = None

        rec = {
            "attempt": attempts,
            "step": step,
            "idx": ex["idx"],
            "gate": reason,
            **{k: info.get(k) for k in info},
        }

        if target is None:
            skips[reason] += 1
            rec["updated"] = False
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            pbar.set_postfix(skip=sum(skips.values()), why=reason[:14], beat=f"{sum(roll_beat)/max(1,len(roll_beat)):.2f}")
            continue

        batch, tlen = make_batch(tok, ex["question"], target, args.max_target_len, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        out = sf_train_step(model, batch, mode="C", latent_length=args.latent_length)
        loss = out["loss"]
        loss.backward()
        opt.step()

        is_self = reason.startswith("beat_a") or reason.startswith("match_a")
        roll_beat.append(1.0 if is_self else 0.0)
        if info.get("gap_vs_a") is not None:
            roll_gap.append(info["gap_vs_a"])
        src_counts[reason] += 1

        losses.append(
            {
                "step": step,
                "idx": ex["idx"],
                "loss": float(loss.item()),
                "tlen": tlen,
                "src": reason,
                "gap_vs_a": info.get("gap_vs_a"),
                "a_score": a_score,
                "best_score": info.get("best_score"),
            }
        )
        rec.update(
            {
                "updated": True,
                "loss": float(loss.item()),
                "tlen": tlen,
                "roll_beat": sum(roll_beat) / len(roll_beat),
                "roll_gap": sum(roll_gap) / len(roll_gap) if roll_gap else None,
            }
        )
        dbg.write(json.dumps(rec) + "\n")
        dbg.flush()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            src=reason[:10],
            tlen=tlen,
            beat=f"{rec['roll_beat']:.2f}",
            gap=f"{(info.get('gap_vs_a') or 0):.1f}",
            skip=sum(skips.values()),
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
        "sources": dict(src_counts),
        "aborted": aborted,
        "beat_rate": float(sum(roll_beat) / len(roll_beat)) if roll_beat else None,
        "mean_gap": float(sum(roll_gap) / len(roll_gap)) if roll_gap else None,
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"DONE steps={step} sources={dict(src_counts)} skips={dict(skips)} out={outdir}")


if __name__ == "__main__":
    main()
