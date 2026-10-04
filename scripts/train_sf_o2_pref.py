#!/usr/bin/env python3
"""SF + frozen O2 best-vs-worst preference pilot.

Per step:
  1) no_grad: sample K candidates, O2 scores them
  2) winner = O2-best among *correct* (prefer << process)
  3) loser  = O2-best among *wrong*
  4) require margin = score(win)-score(lose) ≥ margin_min
  5) L = CE_SF(winner_after) + λ_ul * Unlikelihood_SF(loser_after)
  6) O2 never backpropped; debug.jsonl every attempt

Example:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sf_o2_pref.py --max_steps 200 --k 4 --lambda_ul 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig  # noqa: E402
from src.models.gpt2 import COCONUTGPT2ForTokenClassification  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.sf_rollout import (  # noqa: E402
    RolloutBatch,
    answer_ce_loss,
    append_end_and_answer_embeds,
    build_prefix_embeds_with_latents,
    rollout_latent_thoughts,
    sf_train_step,
)


class TrainJSON(Dataset):
    def __init__(self, path: Path, pseudo_path: Path | None = None):
        self.data = json.load(open(path))
        self.pseudo_by_idx = {}
        if pseudo_path and pseudo_path.exists():
            blob = json.load(open(pseudo_path))
            for e in blob["examples"]:
                if e.get("correct") and e.get("target_text"):
                    self.pseudo_by_idx[int(e["idx"])] = e["target_text"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        return {
            "idx": i,
            "question": ex["question"],
            "answer": float(str(ex["answer"]).replace(",", "")),
            "answer_str": str(ex["answer"]).replace(",", ""),
            "pseudo": self.pseudo_by_idx.get(i),
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


def make_batch(tok, question: str, target_text: str, max_target_len: int, device: str) -> RolloutBatch:
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


def unlikelihood_loss(model, batch: RolloutBatch, *, latent_length: int) -> torch.Tensor:
    """Live SF rollout then -log(1-p) on loser target tokens."""
    latents, _, _ = rollout_latent_thoughts(
        model, batch.input_ids, batch.attention_mask, latent_length=latent_length, detach=False
    )
    prefix_embeds, prefix_mask = build_prefix_embeds_with_latents(
        model, batch.input_ids, batch.attention_mask, latents
    )
    embeds, attn, labels, _ = append_end_and_answer_embeds(
        model, prefix_embeds, prefix_mask, batch.answer_ids, batch.answer_mask
    )
    out = model(inputs_embeds=embeds, attention_mask=attn, use_cache=False, return_dict=True)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    valid = shift_labels != -100
    if valid.sum() == 0:
        return shift_logits.sum() * 0.0
    # gather log p of target ids
    tgt = shift_labels.clamp(min=0)
    token_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    p = token_lp.exp().clamp(max=1.0 - 1e-4)
    ul = -torch.log((1.0 - p).clamp(min=1e-4))
    return ul[valid].mean()


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
                "correct": preds[i] == gold,
                "score": float(scores[i].item()),
                "has_process": "<<" in afters[i],
                "starts_hash3": afters[i].lstrip().startswith("###"),
                "after_len": len(afters[i]),
            }
        )
    return rows


def pick_pair(rows, *, margin_min: float, prefer_process: bool, mode: str = "correct_vs_wrong"):
    """Pick winner/loser.

    mode:
      correct_vs_wrong: best correct vs best wrong (legacy)
      correct_vs_correct: best vs worst among *correct* (O2 ranking among valid)
    """
    ok = [r for r in rows if r["correct"] and r["after"]]
    bad = [r for r in rows if (not r["correct"]) and r["after"]]
    info = {
        "n_ok": sum(1 for r in rows if r["correct"]),
        "n_bad": len(bad),
        "n_proc": sum(1 for r in rows if r["has_process"]),
        "n_hash3": sum(1 for r in rows if r["starts_hash3"]),
        "mean_after_len": sum(r["after_len"] for r in rows) / max(1, len(rows)),
        "pair_mode": mode,
    }
    if prefer_process:
        proc = [r for r in ok if r["has_process"]]
        win_pool = proc if proc else ok
    else:
        win_pool = ok

    if mode == "correct_vs_correct":
        if len(ok) < 2:
            return None, None, "need_2correct", info
        # rank all correct by O2; winner=best, loser=worst (must differ)
        ranked = sorted(ok, key=lambda r: r["score"], reverse=True)
        # prefer process winner if available at top ranks
        if prefer_process:
            proc_sorted = [r for r in ranked if r["has_process"]]
            winner = proc_sorted[0] if proc_sorted else ranked[0]
        else:
            winner = ranked[0]
        loser = ranked[-1]
        if loser is winner or abs(winner["score"] - loser["score"]) < 1e-6:
            return None, None, "no_score_gap", info
    else:
        if not win_pool:
            return None, None, "no_winner", info
        if not bad:
            return None, None, "no_loser", info
        winner = max(win_pool, key=lambda r: r["score"])
        loser = max(bad, key=lambda r: r["score"])

    margin = winner["score"] - loser["score"]
    info.update(
        {
            "margin": margin,
            "win_score": winner["score"],
            "lose_score": loser["score"],
            "win_alen": winner["after_len"],
            "lose_alen": loser["after_len"],
            "win_proc": winner["has_process"],
            "lose_correct": loser["correct"],
        }
    )
    if margin < margin_min:
        return None, None, "low_margin", info
    return winner, loser, "update", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/sf_pilot/train.json")
    ap.add_argument("--pseudo_path", default="data/sf_pilot/train_A_pseudo.json",
                    help="Fallback CE target when winner lacks <<")
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--prm_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pair_mode", choices=["correct_vs_wrong", "correct_vs_correct"], default="correct_vs_correct",
                    help="correct_vs_correct: O2 ranks among valid (safer); vs_wrong uses UL on errors")
    ap.add_argument("--lambda_ul", type=float, default=0.05)
    ap.add_argument("--margin_min", type=float, default=0.5)
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefer_process", action="store_true", default=True)
    ap.add_argument("--no_prefer_process", action="store_true")
    ap.add_argument("--health_window", type=int, default=30)
    ap.add_argument("--abort_hash3_rate", type=float, default=0.98)
    ap.add_argument("--abort_mean_after_len", type=float, default=3.0)
    args = ap.parse_args()
    if args.no_prefer_process:
        args.prefer_process = False

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"O2pref_{args.pair_mode}_k{args.k}_ul{args.lambda_ul:g}_m{args.margin_min:g}"
    outdir = Path(args.outdir or f"outputs/sf_pilot_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    prm_id = str(ROOT / args.prm_id) if not args.prm_id.startswith("/") else args.prm_id
    model, tok = load_generator(device, ckpt)
    prm = load_o2(device, prm_id, tok)
    ds = TrainJSON(ROOT / args.train_json, ROOT / args.pseudo_path)
    print(
        f"n={len(ds)} pseudo={len(ds.pseudo_by_idx)} k={args.k} mode={args.pair_mode} "
        f"λ_ul={args.lambda_ul} margin_min={args.margin_min} prefer_process={args.prefer_process}"
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    json.dump({**vars(args), "outdir": str(outdir)}, open(outdir / "train_args.json", "w"), indent=2)

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    skips: Counter = Counter()
    losses = []
    roll_hash3 = deque(maxlen=args.health_window)
    roll_alen = deque(maxlen=args.health_window)
    roll_nok = deque(maxlen=args.health_window)
    aborted = None
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
        winner, loser, reason, info = pick_pair(
            rows,
            margin_min=args.margin_min,
            prefer_process=args.prefer_process,
            mode=args.pair_mode,
        )

        roll_hash3.append(info["n_hash3"] / args.k)
        roll_alen.append(info["mean_after_len"])
        roll_nok.append(info["n_ok"] / args.k)
        rec = {
            "attempt": attempts,
            "step": step,
            "idx": ex["idx"],
            "gate": reason,
            **info,
            "roll_hash3": sum(roll_hash3) / len(roll_hash3),
            "roll_alen": sum(roll_alen) / len(roll_alen),
            "roll_nok": sum(roll_nok) / len(roll_nok),
        }

        if winner is None:
            skips[reason] += 1
            rec["updated"] = False
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            pbar.set_postfix(
                skip=sum(skips.values()),
                why=reason[:11],
                rh=f"{rec['roll_hash3']:.2f}",
                ra=f"{rec['roll_alen']:.0f}",
                rok=f"{rec['roll_nok']:.2f}",
            )
            continue

        # CE target: process winner after, else A-pseudo fallback
        if winner["has_process"]:
            win_text = winner["after"]
            win_src = "self_process"
        elif ex.get("pseudo"):
            win_text = ex["pseudo"]
            win_src = "a_pseudo"
        else:
            skips["no_process_target"] += 1
            rec["updated"] = False
            rec["gate"] = "no_process_target"
            dbg.write(json.dumps(rec) + "\n")
            dbg.flush()
            continue

        # For correct_vs_correct, UL target should be loser's after (also correct path)
        lose_text = loser["after"]
        if not lose_text:
            skips["empty_loser"] += 1
            continue

        win_batch, win_tlen = make_batch(tok, ex["question"], win_text, args.max_target_len, device)
        lose_batch, lose_tlen = make_batch(tok, ex["question"], lose_text, args.max_target_len, device)

        model.train()
        opt.zero_grad(set_to_none=True)
        # winner CE (SF-C)
        out_w = sf_train_step(model, win_batch, mode="C", latent_length=args.latent_length)
        loss_ce = out_w["loss"]
        # loser unlikelihood (separate live rollout)
        loss_ul = unlikelihood_loss(model, lose_batch, latent_length=args.latent_length)
        loss = loss_ce + args.lambda_ul * loss_ul
        loss.backward()
        opt.step()

        losses.append(
            {
                "step": step,
                "idx": ex["idx"],
                "loss": float(loss.item()),
                "loss_ce": float(loss_ce.item()),
                "loss_ul": float(loss_ul.item()),
                "margin": info.get("margin"),
                "win_tlen": win_tlen,
                "lose_tlen": lose_tlen,
                "g_lat": float(out_w.get("grad_to_latent_norm") or 0.0),
            }
        )
        rec.update(
            {
                "updated": True,
                "loss": float(loss.item()),
                "loss_ce": float(loss_ce.item()),
                "loss_ul": float(loss_ul.item()),
                "win_tlen": win_tlen,
                "lose_tlen": lose_tlen,
                "win_src": win_src,
            }
        )
        dbg.write(json.dumps(rec) + "\n")
        dbg.flush()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            skip=sum(skips.values()),
            ce=f"{loss_ce.item():.3f}",
            ul=f"{loss_ul.item():.3f}",
            m=f"{info['margin']:.1f}",
            wt=win_tlen,
            rh=f"{rec['roll_hash3']:.2f}",
            ra=f"{rec['roll_alen']:.0f}",
            rok=f"{rec['roll_nok']:.2f}",
        )

        if len(roll_hash3) >= args.health_window:
            if rec["roll_hash3"] >= args.abort_hash3_rate and rec["roll_alen"] < args.abort_mean_after_len:
                aborted = f"format_collapse h3={rec['roll_hash3']:.2f} alen={rec['roll_alen']:.1f}"
                break

    pbar.close()
    dbg.close()
    model.save_pretrained(outdir / "model")
    tok.save_pretrained(outdir / "model")
    json.dump(losses, open(outdir / "loss.json", "w"))
    stats = {
        "updates": step,
        "attempts": attempts,
        "skips": dict(skips),
        "aborted": aborted,
        "final_roll_hash3": float(sum(roll_hash3) / len(roll_hash3)) if roll_hash3 else None,
        "final_roll_alen": float(sum(roll_alen) / len(roll_alen)) if roll_alen else None,
        "final_roll_nok": float(sum(roll_nok) / len(roll_nok)) if roll_nok else None,
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"DONE steps={step} skips={dict(skips)} aborted={aborted} out={outdir}")


if __name__ == "__main__":
    main()
