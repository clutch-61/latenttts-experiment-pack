#!/usr/bin/env python3
"""Full-data O2gate with **online frozen-A teacher** (no offline A-pseudo cache).

Bottleneck addressed: selfcot@scale CE'd short self answers → flat/↓ vs A.
Best pilot signal was O2gate+A-pseudo; this scales that recipe to gsm_train
without pre-caching 386k trajectories.

Per step:
  1) student samples K; frozen O2 scores / margin gate
  2) frozen A greedy-decodes once; if A correct → CE target = A.after
     else skip (no bad teacher)
  3) SF live CE on student; O2/A never backpropped

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sf_o2_gate_teacher.py \\
    --train_json data/gsm_train.json --max_steps 3000 --k 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
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


def parse_gold(ans) -> float:
    s = str(ans).replace(",", "").replace("$", "").replace("~", "").strip()
    try:
        return float(s)
    except ValueError:
        if "/" in s:
            a, b = s.split("/", 1)
            try:
                return float(a.strip()) / float(b.strip())
            except ValueError:
                pass
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            return float(m.group(0))
        raise ValueError(f"unparseable answer: {ans!r}")


class TrainJSON(Dataset):
    def __init__(self, path: Path):
        raw = json.load(open(path))
        self.data = []
        skipped = 0
        for i, ex in enumerate(raw):
            try:
                gold = parse_gold(ex["answer"])
            except (ValueError, ZeroDivisionError):
                skipped += 1
                continue
            self.data.append({"idx": i, "question": ex["question"], "answer": gold})
        if skipped:
            print(f"TrainJSON skipped {skipped}; usable={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def load_generator(device, ckpt: str, *, trainable: bool):
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
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
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


def _after_from_seq(tok, seq, end_id):
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
    return after, seq


@torch.no_grad()
def sample_student(model, prm, tok, question, gold, *, k, latent_length, max_new_tokens, device, dropout_p):
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
        after, seq = _after_from_seq(tok, out.sequences[i], end_id)
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


@torch.no_grad()
def teacher_target(teacher, tok, question, gold, *, latent_length, max_new_tokens, device):
    extractor = MODELS["coconut"]["answer_extractor"]
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    enc = tok(question + "\n<|start-latent|>", return_tensors="pt")
    gen_cfg = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    out = teacher.generate(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        generation_config=gen_cfg,
        num_return_sequences=1,
        return_dict_in_generate=True,
        use_cache=True,
    )
    after, _ = _after_from_seq(tok, out.sequences[0], end_id)
    pred = extractor(tok.decode(out.sequences[0], skip_special_tokens=True))
    return after, pred == gold, pred


def gate(rows, *, margin_min: float):
    ok = [r for r in rows if r["correct"]]
    bad = [r for r in rows if not r["correct"]]
    info = {
        "n_ok": len(ok),
        "n_hash3": sum(1 for r in rows if r["starts_hash3"]),
        "mean_after_len": sum(r["after_len"] for r in rows) / max(1, len(rows)),
    }
    if not ok:
        return False, "no_correct", info
    best_ok = max(ok, key=lambda r: r["score"])
    info["best_ok_score"] = best_ok["score"]
    if bad:
        margin = best_ok["score"] - max(bad, key=lambda r: r["score"])["score"]
        info["margin"] = margin
        if margin < margin_min:
            return False, "low_margin", info
    else:
        info["margin"] = None
    return True, "pass", info


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
    ap.add_argument("--train_json", default="data/gsm_train.json")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--teacher_ckpt", default="checkpoints/coconut")
    ap.add_argument("--prm_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin_min", type=float, default=0.5)
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--health_window", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"O2gateT_k{args.k}_m{args.margin_min:g}"
    outdir = Path(args.outdir or f"outputs/sf_full_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    tckpt = str(ROOT / args.teacher_ckpt) if not args.teacher_ckpt.startswith("/") else args.teacher_ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id

    student, tok = load_generator(device, ckpt, trainable=True)
    teacher, _ = load_generator(device, tckpt, trainable=False)
    prm = load_o2(device, prm_id, tok)
    ds = TrainJSON(ROOT / args.train_json)
    print(f"GATE-TEACHER n={len(ds)} k={args.k} margin_min={args.margin_min} steps={args.max_steps}")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    json.dump(
        {**vars(args), "outdir": str(outdir), "recipe": "o2gate_online_teacher"},
        open(outdir / "train_args.json", "w"),
        indent=2,
    )

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    skips: Counter = Counter()
    losses = []
    roll_hash3 = deque(maxlen=args.health_window)
    roll_alen = deque(maxlen=args.health_window)
    attempts = 0
    max_attempts = args.max_steps * 40

    pbar = tqdm(total=args.max_steps, desc=tag)
    dbg = open(outdir / "debug.jsonl", "w")

    while step < args.max_steps and attempts < max_attempts:
        attempts += 1
        if not order:
            order = list(range(len(ds)))
            random.shuffle(order)
        ex = ds[order.pop()]

        student.eval()
        rows = sample_student(
            student,
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
        ok, reason, info = gate(rows, margin_min=args.margin_min)
        roll_hash3.append(info["n_hash3"] / args.k)
        roll_alen.append(info["mean_after_len"])

        rec = {
            "attempt": attempts,
            "step": step,
            "idx": ex["idx"],
            "gate": reason,
            **info,
            "roll_hash3": sum(roll_hash3) / len(roll_hash3),
            "roll_alen": sum(roll_alen) / len(roll_alen),
        }

        if not ok:
            skips[reason] += 1
            rec["updated"] = False
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            pbar.set_postfix(skip=sum(skips.values()), why=reason[:12], h3=f"{rec['roll_hash3']:.2f}")
            continue

        target, a_ok, a_pred = teacher_target(
            teacher,
            tok,
            ex["question"],
            ex["answer"],
            latent_length=args.latent_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        rec["teacher_ok"] = a_ok
        rec["teacher_alen"] = len(target) if target else 0
        if not a_ok or not target:
            skips["teacher_wrong"] += 1
            rec["updated"] = False
            rec["gate"] = "teacher_wrong"
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            pbar.set_postfix(skip=sum(skips.values()), why="teacher_wrong")
            continue

        batch, tlen = make_batch(tok, ex["question"], target, args.max_target_len, device)
        student.train()
        opt.zero_grad(set_to_none=True)
        out = sf_train_step(student, batch, mode="C", latent_length=args.latent_length)
        loss = out["loss"]
        loss.backward()
        opt.step()

        losses.append({"step": step, "idx": ex["idx"], "loss": float(loss.item()), "tlen": tlen, "margin": info.get("margin")})
        rec.update({"updated": True, "loss": float(loss.item()), "tlen": tlen})
        dbg.write(json.dumps(rec) + "\n")
        dbg.flush()

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.3f}", tlen=tlen, h3=f"{rec['roll_hash3']:.2f}", skip=sum(skips.values()))

        if args.save_every > 0 and step % args.save_every == 0:
            ck = outdir / f"checkpoint-{step}"
            ck.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(ck)
            tok.save_pretrained(ck)
            json.dump(losses, open(outdir / "loss.json", "w"))
            json.dump(
                {
                    "updates": step,
                    "attempts": attempts,
                    "skips": dict(skips),
                    "update_rate": step / max(1, attempts),
                },
                open(outdir / "stats.json", "w"),
                indent=2,
            )

    pbar.close()
    dbg.close()
    student.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    json.dump(losses, open(outdir / "loss.json", "w"))
    stats = {
        "updates": step,
        "attempts": attempts,
        "skips": dict(skips),
        "update_rate": step / max(1, attempts),
        "recipe": "o2gate_online_teacher",
        "clone_teacher": True,
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"DONE steps={step} attempts={attempts} rate={stats['update_rate']:.3f} skips={dict(skips)} out={outdir}")


if __name__ == "__main__":
    main()
