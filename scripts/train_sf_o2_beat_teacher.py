#!/usr/bin/env python3
"""Full-data O2: prefer self when it **beats frozen teacher** on O2; else CE teacher.

Addresses gateT bottleneck: cloning A raises Pass@1 but drops +O2 BoN.
Here, among student K samples, if best correct O2 score > teacher_O2 + beat_margin
→ CE on that self after; else if O2 gate passes and teacher correct → CE teacher; else skip.

Typical warm-start from a gateT checkpoint:
  --ckpt outputs/sf_full_O2gateT_k4_m0.5_*/model
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
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


def load_generator(device, ckpt: str, *, trainable: bool, dtype=torch.float32):
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
        torch_dtype=dtype,
    ).to(device)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tok


def load_o2(device, prm_id: str, tok, dtype=torch.float32):
    prm = COCONUTGPT2ForTokenClassification.from_pretrained(
        prm_id,
        latent_id=tok.convert_tokens_to_ids("<|latent|>"),
        latent_start_id=tok.convert_tokens_to_ids("<|start-latent|>"),
        latent_end_id=tok.convert_tokens_to_ids("<|end-latent|>"),
        target_id=tok.convert_tokens_to_ids(">>"),
        pad_token_id=tok.pad_token_id,
        torch_dtype=dtype,
    ).to(device)
    prm.eval()
    for p in prm.parameters():
        p.requires_grad_(False)
    return prm


def _vram_gb(device):
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(device)
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
        "alloc_gb": alloc / 1e9,
        "reserved_gb": reserved / 1e9,
        "used_frac": 1.0 - free / max(total, 1),
    }


@torch.no_grad()
def score_ids_latents(prm, input_ids: torch.Tensor, latents: torch.Tensor) -> float:
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    logits = prm(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        latent_embeds=latents.to(device=input_ids.device, dtype=next(prm.parameters()).dtype),
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
                "after_len": len(after),
                "starts_hash3": after.lstrip().startswith("###"),
                "latents": out.latent_thoughts[i].detach().float().cpu(),
            }
        )
    return rows


@torch.no_grad()
def teacher_decode_score(teacher, prm, tok, question, gold, *, latent_length, max_new_tokens, device):
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
    after, seq = _after_from_seq(tok, out.sequences[0], end_id)
    pred = extractor(tok.decode(out.sequences[0], skip_special_tokens=True))
    score = score_ids_latents(prm, seq.unsqueeze(0).to(device), out.latent_thoughts[0])
    return after, pred == gold, score


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
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--margin_min", type=float, default=0.5, help="student vs wrong for teacher-fallback gate")
    ap.add_argument("--beat_margin", type=float, default=0.0, help="self must beat teacher O2 by this")
    ap.add_argument(
        "--min_alen",
        type=int,
        default=0,
        help="absolute min after_len (chars) for beat_self; 0=off",
    )
    ap.add_argument(
        "--beat_len_eps",
        type=int,
        default=0,
        help="require after_len >= teacher_alen + beat_len_eps for beat_self (relative length)",
    )
    ap.add_argument(
        "--lambda_lat",
        type=float,
        default=0.0,
        help="MSE match live C-rollout latents to winning beat_self latents",
    )
    ap.add_argument("--grad_accum", type=int, default=1, help="optimizer step every N successful updates")
    ap.add_argument(
        "--frozen_dtype",
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="dtype for frozen teacher + O2 (pack VRAM)",
    )
    ap.add_argument(
        "--auto_pack",
        action="store_true",
        default=True,
        help="if VRAM still empty after load, raise k (default on)",
    )
    ap.add_argument("--no_auto_pack", action="store_false", dest="auto_pack")
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"O2beatT_k{args.k}_b{args.beat_margin:g}"
    if args.min_alen > 0:
        tag += f"_a{args.min_alen}"
    if args.beat_len_eps > 0:
        tag += f"_le{args.beat_len_eps}"
    if args.lambda_lat > 0:
        tag += f"_ll{args.lambda_lat:g}"
    if args.grad_accum > 1:
        tag += f"_ga{args.grad_accum}"
    outdir = Path(args.outdir or f"outputs/sf_full_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    tckpt = str(ROOT / args.teacher_ckpt) if not args.teacher_ckpt.startswith("/") else args.teacher_ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id

    frozen_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.frozen_dtype]
    student, tok = load_generator(device, ckpt, trainable=True, dtype=torch.float32)
    teacher, _ = load_generator(device, tckpt, trainable=False, dtype=frozen_dtype)
    prm = load_o2(device, prm_id, tok, dtype=frozen_dtype)

    mem = _vram_gb(device)
    k_before = args.k
    if args.auto_pack and mem is not None:
        # Empty card after 3 models → raise K so generate fills activation memory.
        if mem["used_frac"] < 0.25 and mem["free_gb"] > 60:
            args.k = max(args.k, 32)
        elif mem["used_frac"] < 0.35 and mem["free_gb"] > 40:
            args.k = max(args.k, 16)
        if args.grad_accum < 4 and mem["free_gb"] > 40:
            args.grad_accum = max(args.grad_accum, 4)
    if args.k != k_before or args.grad_accum > 1:
        # refresh tag/outdir note only in logs; keep user outdir
        pass
    mem2 = _vram_gb(device)
    print(
        f"PACK k={args.k} (req {k_before}) grad_accum={args.grad_accum} "
        f"frozen_dtype={args.frozen_dtype} vram={mem2}"
    )

    ds = TrainJSON(ROOT / args.train_json)
    print(
        f"BEAT-TEACHER n={len(ds)} k={args.k} beat_margin={args.beat_margin} "
        f"min_alen={args.min_alen} beat_len_eps={args.beat_len_eps} lambda_lat={args.lambda_lat} "
        f"grad_accum={args.grad_accum} margin_min={args.margin_min} lr={args.lr} ckpt={ckpt}"
    )

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    json.dump(
        {**vars(args), "outdir": str(outdir), "recipe": "o2_beat_teacher", "vram": _vram_gb(device)},
        open(outdir / "train_args.json", "w"),
        indent=2,
    )

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    accum_i = 0
    skips: Counter = Counter()
    sources: Counter = Counter()
    losses = []
    attempts = 0
    max_attempts = args.max_steps * 40
    pbar = tqdm(total=args.max_steps, desc=tag)
    dbg = open(outdir / "debug.jsonl", "w")
    opt.zero_grad(set_to_none=True)

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
        ok_rows = [r for r in rows if r["correct"] and r["after"]]
        bad_rows = [r for r in rows if not r["correct"]]
        t_after, t_ok, t_score = teacher_decode_score(
            teacher,
            prm,
            tok,
            ex["question"],
            ex["answer"],
            latent_length=args.latent_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        rec = {
            "attempt": attempts,
            "step": step,
            "idx": ex["idx"],
            "n_ok": len(ok_rows),
            "teacher_ok": t_ok,
            "teacher_score": t_score,
            "teacher_alen": len(t_after) if t_after else 0,
        }

        target = None
        src = None
        win_latents = None
        t_alen = len(t_after) if t_after else 0
        if ok_rows:
            best_any = max(ok_rows, key=lambda r: (r["score"], r["after_len"]))
            rec["best_self_score"] = best_any["score"]
            rec["gap_vs_teacher"] = best_any["score"] - t_score
            rec["best_self_alen"] = best_any["after_len"]
            if t_ok:
                cand = ok_rows
                if args.min_alen > 0:
                    cand = [r for r in cand if r["after_len"] >= args.min_alen]
                if args.beat_len_eps > 0:
                    need = t_alen + args.beat_len_eps
                    cand = [r for r in cand if r["after_len"] >= need]
                if cand:
                    best = max(cand, key=lambda r: (r["score"], r["after_len"]))
                    if best["score"] - t_score > args.beat_margin:
                        target, src = best["after"], "beat_self"
                        win_latents = best.get("latents")
                else:
                    rec["short_self"] = True
            elif best_any["after"]:
                # teacher wrong: accept any correct self (length filter would strand the step)
                target, src = best_any["after"], "self_teacher_wrong"
                win_latents = best_any.get("latents")

        if target is None:
            # teacher fallback if student gate vs wrong ok
            if not ok_rows:
                skips["no_correct"] += 1
                rec.update({"updated": False, "gate": "no_correct"})
                dbg.write(json.dumps(rec) + "\n")
                dbg.flush()
                pbar.set_postfix(skip=sum(skips.values()), why="no_correct")
                continue
            best = max(ok_rows, key=lambda r: r["score"])
            if bad_rows:
                margin = best["score"] - max(bad_rows, key=lambda r: r["score"])["score"]
                rec["margin"] = margin
                if margin < args.margin_min:
                    skips["low_margin"] += 1
                    rec.update({"updated": False, "gate": "low_margin"})
                    dbg.write(json.dumps(rec) + "\n")
                    dbg.flush()
                    pbar.set_postfix(skip=sum(skips.values()), why="low_margin")
                    continue
            if not t_ok or not t_after:
                skips["teacher_wrong"] += 1
                rec.update({"updated": False, "gate": "teacher_wrong"})
                dbg.write(json.dumps(rec) + "\n")
                dbg.flush()
                pbar.set_postfix(skip=sum(skips.values()), why="teacher_wrong")
                continue
            target, src = t_after, "teacher"

        batch, tlen = make_batch(tok, ex["question"], target, args.max_target_len, device)
        student.train()
        out = sf_train_step(student, batch, mode="C", latent_length=args.latent_length)
        loss = out["loss"]
        loss_lat = None
        if (
            args.lambda_lat > 0
            and win_latents is not None
            and src in ("beat_self", "self_teacher_wrong")
        ):
            live = out["latents"]  # (1, L, D)
            tgt_lat = win_latents.to(device=live.device, dtype=live.dtype)
            if tgt_lat.dim() == 2:
                tgt_lat = tgt_lat.unsqueeze(0)
            loss_lat = torch.nn.functional.mse_loss(live, tgt_lat)
            loss = loss + args.lambda_lat * loss_lat
        (loss / args.grad_accum).backward()
        accum_i += 1
        if accum_i >= args.grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum_i = 0

        sources[src] += 1
        losses.append(
            {
                "step": step,
                "idx": ex["idx"],
                "loss": float(loss.item()),
                "loss_lat": float(loss_lat.item()) if loss_lat is not None else None,
                "tlen": tlen,
                "src": src,
            }
        )
        rec.update(
            {
                "updated": True,
                "src": src,
                "loss": float(loss.item()),
                "loss_lat": float(loss_lat.item()) if loss_lat is not None else None,
                "tlen": tlen,
                "gate": "update",
            }
        )
        dbg.write(json.dumps(rec) + "\n")
        dbg.flush()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            tlen=tlen,
            src=src[:8],
            beat=sources.get("beat_self", 0),
            tea=sources.get("teacher", 0),
            ll=f"{loss_lat.item():.3f}" if loss_lat is not None else "-",
            k=args.k,
        )

        if args.save_every > 0 and step % args.save_every == 0:
            ck = outdir / f"checkpoint-{step}"
            ck.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(ck)
            tok.save_pretrained(ck)
            json.dump(losses, open(outdir / "loss.json", "w"))
            json.dump(
                {"updates": step, "attempts": attempts, "skips": dict(skips), "sources": dict(sources),
                 "update_rate": step / max(1, attempts)},
                open(outdir / "stats.json", "w"),
                indent=2,
            )

    pbar.close()
    if accum_i > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    dbg.close()
    student.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    json.dump(losses, open(outdir / "loss.json", "w"))
    stats = {
        "updates": step,
        "attempts": attempts,
        "skips": dict(skips),
        "sources": dict(sources),
        "update_rate": step / max(1, attempts),
        "self_rate": (sources.get("beat_self", 0) + sources.get("self_teacher_wrong", 0)) / max(1, step),
        "recipe": "o2_beat_teacher",
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"DONE steps={step} attempts={attempts} sources={dict(sources)} skips={dict(skips)} out={outdir}")


if __name__ == "__main__":
    main()
