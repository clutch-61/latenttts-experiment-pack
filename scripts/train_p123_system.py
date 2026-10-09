#!/usr/bin/env python3
"""Phase-1/2/3 integrated system trainer with mid-quantity diagnostics.

P1: frozen O2 (selection) + frozen B0 (health / dual report)
P2: optional content-view MMD (default off; no blind λ↑)
P3: differentiable SF-C CE on selected target (self-beat or teacher)

Selection (`dual_beat`):
  prefer correct student that beats teacher on O2 AND B0 ≥ teacher_B0 − b0_slack;
  else teacher if O2 gate passes; else skip.

Main claim later: (gen+O2) − (A+B0). Mid logs expose leftover / O2–B0 agree / format.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.sf_rollout import RolloutBatch, sf_train_step  # noqa: E402
from src.system_scoring import load_prm, sum_logit_score, wvote_select  # noqa: E402
from src.order_pref.content_view import ContentProjector, content_view  # noqa: E402
from src.order_pref.set_align import mmd_rbf_multiscale  # noqa: E402


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
            steps = ex.get("steps") or []
            self.data.append(
                {"idx": i, "question": ex["question"], "answer": gold, "steps": steps}
            )
        if skipped:
            print(f"TrainJSON skipped {skipped}; usable={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def latent_prefix(question: str, generator_type: str) -> str:
    if generator_type == "coconut":
        return question + "\n<|start-latent|>"
    return question + "<|start-latent|>"


def load_generator(
    device, ckpt: str, *, trainable: bool, dtype=torch.float32, generator_type: str = "coconut"
):
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cls = MODELS[generator_type]["class"]

    class LatentGen(cls, LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    load_kw = dict(
        latent_id=tok.convert_tokens_to_ids("<|latent|>"),
        latent_start_id=tok.convert_tokens_to_ids("<|start-latent|>"),
        latent_end_id=tok.convert_tokens_to_ids("<|end-latent|>"),
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tok.pad_token_id,
        torch_dtype=dtype,
    )
    if generator_type == "coconut":
        load_kw["target_id"] = tok.convert_tokens_to_ids(">>")
    model = LatentGen.from_pretrained(ckpt, **load_kw).to(device)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tok


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


def _set_cuda(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)


@torch.no_grad()
def prm_scores_from_texts(prm, prm_tok, texts, latents, device, chunk):
    """Batched BoN-convention score (same path as infer_gpt2_rm best_of_n)."""
    _set_cuda(device)
    scores = []
    dtype = next(prm.parameters()).dtype
    for s in range(0, len(texts), chunk):
        enc = prm_tok(texts[s : s + chunk], return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        logits = prm(
            input_ids=ids,
            attention_mask=enc["attention_mask"].to(device),
            latent_embeds=latents[s : s + chunk].to(device=device, dtype=dtype),
            return_dict=True,
        ).logits.squeeze(-1)
        scores.append(torch.where(ids == prm.config.latent_id, logits, 0).float().sum(-1).cpu())
    return torch.cat(scores).tolist()


@torch.no_grad()
def generate_batch(
    model,
    gen_tok,
    questions,
    *,
    k,
    sample,
    latent_length,
    max_new_tokens,
    device,
    dropout_p,
    generator_type: str = "coconut",
):
    _set_cuda(device)
    prompts = [latent_prefix(q, generator_type) for q in questions]
    enc = gen_tok(prompts, return_tensors="pt", padding=True)
    kw = dict(latent_do_sample=True, latent_do_sample_by="dropout", dropout_p=dropout_p) if sample else dict(
        latent_do_sample=False
    )
    gen_cfg = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        pad_token_id=gen_tok.pad_token_id,
        eos_token_id=gen_tok.eos_token_id,
        bos_token_id=gen_tok.bos_token_id,
        **kw,
    )
    out = model.generate(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        generation_config=gen_cfg,
        num_return_sequences=k,
        return_dict_in_generate=True,
        use_cache=True,
    )
    return out.sequences, out.latent_thoughts


def _sync_replica(src, dst):
    dst.load_state_dict(src.state_dict())


def _merge_k_halves(seq_a, lat_a, seq_b, lat_b, nq, k2, pad_id):
    """Interleave two (nq*k2, L) generates into (nq*2k2, L). Left-pad to common L."""

    def _pad(s, L):
        if s.size(1) >= L:
            return s
        return torch.nn.functional.pad(s, (L - s.size(1), 0), value=pad_id)

    L = max(seq_a.size(1), seq_b.size(1))
    sa = _pad(seq_a, L).view(nq, k2, L)
    sb = _pad(seq_b, L).view(nq, k2, L)
    seq = torch.cat([sa, sb], dim=1).reshape(nq * (2 * k2), L)
    la = lat_a.view(nq, k2, *lat_a.shape[1:])
    lb = lat_b.view(nq, k2, *lat_b.shape[1:])
    lat = torch.cat([la, lb], dim=1).reshape(nq * (2 * k2), *lat_a.shape[1:])
    return seq, lat


def sample_and_score_safe(*args, exs, **kw):
    """Long prompts can blow up generate's first-step full-vocab logits; halve the batch on OOM."""
    try:
        return sample_and_score(*args, exs=exs, **kw)
    except torch.OutOfMemoryError:
        if len(exs) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(exs) // 2
        print(f"OOM at q_batch={len(exs)}; splitting", flush=True)
        return sample_and_score_safe(*args, exs=exs[:mid], **kw) + sample_and_score_safe(
            *args, exs=exs[mid:], **kw
        )


@torch.no_grad()
def sample_and_score(
    student,
    teacher,
    o2,
    b0,
    gen_tok,
    prm_tok,
    exs,
    *,
    k,
    latent_length,
    max_new_tokens,
    device,
    dropout_p,
    chunk,
    frozen_device=None,
    replica=None,
    replica_device=None,
    timing_out=None,
    generator_type: str = "coconut",
):
    """Returns per-question (rows, teacher_tuple) for a batch of questions.

    If frozen_device is set, teacher/B0 generate+score there; student stays on device.
    If replica is set on a *third* card, student k is split (k/2 ‖ k/2); that is the
    only split that can beat single-card student generate (teacher k=1 is already cheap).
    """
    fdev = frozen_device or device
    extractor = MODELS[generator_type]["answer_extractor"]
    end_id = gen_tok.convert_tokens_to_ids("<|end-latent|>")
    qs = [ex["question"] for ex in exs]
    nq = len(qs)
    t_gen0 = time.perf_counter()
    k2 = (
        k // 2
        if (
            replica is not None
            and replica_device
            and replica_device != device
            and k >= 2
            and k % 2 == 0
        )
        else 0
    )
    t_kw = dict(
        sample=False,
        latent_length=latent_length,
        max_new_tokens=max_new_tokens,
        dropout_p=0.0,
        generator_type=generator_type,
    )
    s_kw = dict(
        sample=True,
        latent_length=latent_length,
        max_new_tokens=max_new_tokens,
        dropout_p=dropout_p,
        generator_type=generator_type,
    )
    if k2:
        _sync_replica(student, replica)
        workers = 3 if fdev not in (device, replica_device) else 2
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_s = pool.submit(generate_batch, student, gen_tok, qs, k=k2, device=device, **s_kw)
            fut_r = pool.submit(generate_batch, replica, gen_tok, qs, k=k2, device=replica_device, **s_kw)
            fut_t = pool.submit(generate_batch, teacher, gen_tok, qs, k=1, device=fdev, **t_kw)
            a_seq, a_lat = fut_s.result()
            b_seq, b_lat = fut_r.result()
            t_seq, t_lat = fut_t.result()
        pad_id = gen_tok.pad_token_id if gen_tok.pad_token_id is not None else 0
        s_seq, s_lat = _merge_k_halves(a_seq, a_lat, b_seq, b_lat, nq, k2, pad_id)
    elif fdev != device:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_s = pool.submit(generate_batch, student, gen_tok, qs, k=k, device=device, **s_kw)
            fut_t = pool.submit(generate_batch, teacher, gen_tok, qs, k=1, device=fdev, **t_kw)
            s_seq, s_lat = fut_s.result()
            t_seq, t_lat = fut_t.result()
    else:
        s_seq, s_lat = generate_batch(student, gen_tok, qs, k=k, device=device, **s_kw)
        t_seq, t_lat = generate_batch(teacher, gen_tok, qs, k=1, device=fdev, **t_kw)
    t_gen = time.perf_counter() - t_gen0
    t_sc0 = time.perf_counter()
    s_text = gen_tok.batch_decode(s_seq, skip_special_tokens=True)
    t_text = gen_tok.batch_decode(t_seq, skip_special_tokens=True)
    texts = s_text + t_text
    o2_dev = str(next(o2.parameters()).device)
    b0_dev = str(next(b0.parameters()).device) if b0 is not None else o2_dev
    lats_o2 = torch.cat(
        [s_lat.to(o2_dev, non_blocking=True), t_lat.to(device=o2_dev, dtype=s_lat.dtype)],
        dim=0,
    )
    if b0 is not None and b0_dev != o2_dev:
        lats_b0 = torch.cat(
            [s_lat.to(b0_dev, non_blocking=True), t_lat.to(device=b0_dev, dtype=s_lat.dtype)],
            dim=0,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_o2 = pool.submit(prm_scores_from_texts, o2, prm_tok, texts, lats_o2, o2_dev, chunk)
            fut_b0 = pool.submit(prm_scores_from_texts, b0, prm_tok, texts, lats_b0, b0_dev, chunk)
            sc_o2 = fut_o2.result()
            sc_b0 = fut_b0.result()
    else:
        sc_o2 = prm_scores_from_texts(o2, prm_tok, texts, lats_o2, o2_dev, chunk)
        sc_b0 = (
            prm_scores_from_texts(b0, prm_tok, texts, lats_o2, o2_dev, chunk)
            if b0 is not None
            else [None] * len(texts)
        )
    n_s = len(s_text)
    s_lat_cpu = s_lat.detach().float().cpu()
    t_lat_cpu = t_lat.detach().float().cpu()

    results = []
    for qi, ex in enumerate(exs):
        gold = ex["answer"]
        rows = []
        for j in range(k):
            i = qi * k + j
            after, _ = _after_from_seq(gen_tok, s_seq[i], end_id)
            pred = extractor(s_text[i])
            rows.append(
                {
                    "after": after,
                    "pred": pred,
                    "correct": pred == gold,
                    "score_o2": sc_o2[i],
                    "score_b0": sc_b0[i],
                    "after_len": len(after),
                    "starts_hash3": after.lstrip().startswith("###"),
                    "has_cot": "<<" in after,
                    "latents": s_lat_cpu[i],
                    "input_ids": s_seq[i].detach().cpu(),
                }
            )
        t_after, _ = _after_from_seq(gen_tok, t_seq[qi], end_id)
        t_ok = extractor(t_text[qi]) == gold
        # teacher latents always returned (P2 online H^pos when t_ok)
        results.append(
            (rows, (t_after, t_ok, sc_o2[n_s + qi], sc_b0[n_s + qi], t_lat_cpu[qi]))
        )
    if timing_out is not None:
        timing_out["t_gen"] = t_gen
        timing_out["t_score"] = time.perf_counter() - t_sc0
        timing_out["split_k"] = bool(k2)
        timing_out["nq"] = nq
        timing_out["k"] = k
    return results


def make_batch(tok, question, target_text, max_target_len, device, generator_type: str = "coconut"):
    qenc = tok(latent_prefix(question, generator_type), add_special_tokens=True)
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


def o2_wvote_answer(rows):
    """Softmax-weighted vote over answers using O2 scores (T=1). Returns (winner_pred, mass_dict)."""
    answers = [str(r["pred"]) for r in rows]
    scores = [float(r["score_o2"]) for r in rows]
    wi = wvote_select(answers, scores)
    winner = answers[wi]
    # mass only for mid diagnostics
    m = max(scores)
    mass = defaultdict(float)
    for a, s in zip(answers, scores):
        mass[a] += math.exp(s - m)
    return winner, mass


def mid_from_rows(rows, t_o2, t_b0, t_ok):
    """Per-question intermediate quantities for root-cause analysis."""
    n = len(rows)
    preds = [r["pred"] for r in rows]
    correct_idxs = [i for i, r in enumerate(rows) if r["correct"]]
    top_o2 = max(range(n), key=lambda i: rows[i]["score_o2"])
    top_b0 = (
        max(range(n), key=lambda i: rows[i]["score_b0"] if rows[i]["score_b0"] is not None else -1e9)
        if rows[0]["score_b0"] is not None
        else None
    )
    pool_ok = len(correct_idxs) > 0
    o2_pick_ok = rows[top_o2]["correct"]
    leftover = pool_ok and (not o2_pick_ok)
    wv_ans, _ = o2_wvote_answer(rows)
    # wvote correct iff any row with that pred is correct (preds share label)
    wv_ok = any(r["correct"] for r in rows if str(r["pred"]) == wv_ans)
    top_pred = str(rows[top_o2]["pred"])
    return {
        "k": n,
        "n_correct": len(correct_idxs),
        "pool_has_correct": pool_ok,
        "o2_top_correct": o2_pick_ok,
        "leftover": leftover,
        "wvote_correct": wv_ok,
        "top_wvote_disagree": top_pred != wv_ans,
        "o2_b0_top_agree": (top_b0 is not None and top_o2 == top_b0),
        "unique_preds": len({str(p) for p in preds}),
        "hash3_rate": sum(1 for r in rows if r["starts_hash3"]) / max(n, 1),
        "cot_rate": sum(1 for r in rows if r["has_cot"]) / max(n, 1),
        "mean_after_len": sum(r["after_len"] for r in rows) / max(n, 1),
        "teacher_ok": t_ok,
        "best_o2": max(r["score_o2"] for r in rows),
        "teacher_o2": t_o2,
        "best_b0": max((r["score_b0"] for r in rows if r["score_b0"] is not None), default=None),
        "teacher_b0": t_b0,
    }


def select_target(
    rows,
    t_after,
    t_ok,
    t_o2,
    t_b0,
    t_lat=None,
    *,
    margin_min,
    beat_margin,
    b0_slack,
    prefer_cot,
    select_mode="dual_beat",
):
    """Return (source, target_text, win_latents_or_None, reason).

    select_mode:
      dual_beat — original: beat_self if O2+B0 gate, else teacher if O2 gate, else skip.
      coverage — if student pool has no correct and teacher is correct, always take teacher
        (no O2 gate). Otherwise fall back to dual_beat.
      wvote — if O2 softmax-weighted-vote winner answer is correct, CE on the highest-O2
        sample of that answer (align generator with the aggregation that works). Else dual_beat.

    Teacher rows return teacher latents (for P2 MMD / λ_lat) when available.
    """
    wrong = [r for r in rows if not r["correct"]]
    max_wrong_o2 = max((r["score_o2"] for r in wrong), default=-1e9)
    correct = [r for r in rows if r["correct"]]
    if select_mode == "coverage" and not correct and t_ok and t_after:
        return "teacher", t_after, t_lat, "cover_teacher"
    if select_mode == "wvote":
        wv_ans, _ = o2_wvote_answer(rows)
        cand = [r for r in rows if str(r["pred"]) == wv_ans]
        if cand and any(r["correct"] for r in cand):
            # only correct wvote winners; pick best O2 among that answer
            best = max((r for r in cand if r["correct"]), key=lambda r: r["score_o2"])
            # optional B0 health vs teacher (same slack as dual_beat)
            if t_b0 is not None and best["score_b0"] is not None and best["score_b0"] < t_b0 - b0_slack:
                pass  # fall through to dual_beat rather than reinforce unhealthy trajectory
            else:
                return "wvote_self", best["after"], best["latents"], "wvote_correct"
    if correct:
        # rank by O2; require B0 health vs teacher
        ranked = sorted(correct, key=lambda r: r["score_o2"], reverse=True)
        for r in ranked:
            if r["score_o2"] <= t_o2 + beat_margin:
                continue
            if t_b0 is not None and r["score_b0"] is not None:
                if r["score_b0"] < t_b0 - b0_slack:
                    continue
            if prefer_cot and not r["has_cot"] and any(x["has_cot"] for x in ranked):
                # skip pure ### if a CoT correct exists that also beats
                cot_ok = [
                    x
                    for x in ranked
                    if x["has_cot"]
                    and x["score_o2"] > t_o2 + beat_margin
                    and (t_b0 is None or x["score_b0"] is None or x["score_b0"] >= t_b0 - b0_slack)
                ]
                if cot_ok:
                    continue
            return "beat_self", r["after"], r["latents"], "o2_beat+b0_ok"
    # teacher fallback
    if t_ok and t_o2 >= max_wrong_o2 + margin_min:
        return "teacher", t_after, t_lat, "o2_gate_teacher"
    return "skip", None, None, "no_signal"


def mixed_rank_pair(rows):
    """Best-O2 gold vs best-O2 thief — the leftover pattern on the live k-pool."""
    correct = [r for r in rows if r["correct"] and r.get("input_ids") is not None]
    wrong = [r for r in rows if (not r["correct"]) and r.get("input_ids") is not None]
    if not correct or not wrong:
        return None
    gold = max(correct, key=lambda r: r["score_o2"])
    thief = max(wrong, key=lambda r: r["score_o2"])
    return gold, thief


def _safe_ckpt(student, tok, o2, projector, losses, outdir, *, fatal: bool = False):
    """Write ckpt; disk-full must not kill a mid-run (final save still raises)."""
    try:
        student.save_pretrained(outdir / "model")
        tok.save_pretrained(outdir / "model")
        if o2 is not None:
            o2.save_pretrained(outdir / "o2")
            tok.save_pretrained(outdir / "o2")
        if projector is not None:
            torch.save(projector.state_dict(), outdir / "projector.pt")
        json.dump(losses, open(outdir / "losses.json", "w"))
    except OSError as e:
        print(f"WARN ckpt save failed ({e}); continue", flush=True)
        if fatal:
            raise
    except Exception as e:
        # safetensors raises SafetensorError, not always OSError
        msg = str(e)
        print(f"WARN ckpt save failed ({type(e).__name__}: {msg}); continue", flush=True)
        if fatal:
            raise


def o2_rank_hinge(o2, gold_row, thief_row, *, margin: float):
    """BoN-aligned hinge: sum_logit(gold) >= sum_logit(thief) + margin. Latents detached."""
    s_pos = sum_logit_score(o2, gold_row["input_ids"], gold_row["latents"].detach())
    s_neg = sum_logit_score(o2, thief_row["input_ids"], thief_row["latents"].detach())
    loss = torch.relu(margin + s_neg - s_pos)
    gap = float((s_neg - s_pos).detach().item())
    return loss, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/gsm_train.json")
    ap.add_argument(
        "--generator_type",
        choices=["coconut", "codi"],
        default="coconut",
        help="student/teacher family; must match --ckpt / --teacher_ckpt",
    )
    ap.add_argument("--ckpt", default="checkpoints/coconut")
    ap.add_argument("--teacher_ckpt", default="checkpoints/coconut")
    ap.add_argument("--o2_id", default="outputs/latentrm_order_pref/best")
    ap.add_argument("--b0_id", default="outputs/latentrm_baseline/best")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--margin_min", type=float, default=0.5)
    ap.add_argument("--beat_margin", type=float, default=0.0)
    ap.add_argument(
        "--b0_slack",
        type=float,
        default=0.5,
        help="allow beat_self if student B0 >= teacher_B0 - slack",
    )
    ap.add_argument("--prefer_cot", action="store_true", default=True)
    ap.add_argument("--no_prefer_cot", action="store_false", dest="prefer_cot")
    ap.add_argument("--lambda_lat", type=float, default=0.0)
    ap.add_argument("--lambda_mmd", type=float, default=0.0, help="P2; 0=off")
    ap.add_argument(
        "--lambda_rank",
        type=float,
        default=0.0,
        help="P1 online leftover hinge on mixed k-pool (BoN sum_logit). 0=O2 frozen.",
    )
    ap.add_argument("--rank_margin", type=float, default=1.0)
    ap.add_argument("--o2_lr", type=float, default=1e-5)
    ap.add_argument("--positive_latents", default="data/sf_pilot/train_A_positive_latents.pt")
    ap.add_argument("--mmd_bottleneck", type=int, default=64)
    ap.add_argument("--mmd_ramp_steps", type=int, default=100)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--frozen_dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument(
        "--frozen_device",
        default=None,
        help="cuda device for teacher/B0 (e.g. cuda:1). Default: same as student. "
        "Use with CUDA_VISIBLE_DEVICES=A,B for 2-card packing.",
    )
    ap.add_argument(
        "--gen_replica_device",
        default=None,
        help="third card for student generate replica (k/2‖k/2). Needs CUDA_VISIBLE_DEVICES=A,B,C "
        "with A=student+O2, B=replica, C=teacher+B0. Do not colocate replica with teacher.",
    )
    ap.add_argument("--auto_pack", action="store_true", default=True)
    ap.add_argument("--no_auto_pack", action="store_false", dest="auto_pack")
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument(
        "--target_mode",
        choices=["after", "process"],
        default="after",
        help="after=CE on selected decode text; process=CE on dataset steps+### (anti short-###)",
    )
    ap.add_argument("--latent_length", type=int, default=6)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="must match infer_gpt2_rm default (was 96; caused train/eval protocol drift)",
    )
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=400)
    ap.add_argument("--q_batch", type=int, default=8, help="questions per batched generate")
    ap.add_argument("--score_chunk", type=int, default=512, help="PRM forward chunk size")
    ap.add_argument("--diag_every", type=int, default=50, help="flush mid JSONL every N updates")
    ap.add_argument(
        "--select_mode",
        choices=["dual_beat", "coverage", "wvote"],
        default="wvote",
        help="default wvote: align train target with claim aggregation (softmax T=1)",
    )
    ap.add_argument(
        "--easy_keep_prob",
        type=float,
        default=1.0,
        help="in coverage mode, keep beat_self updates with this prob (rest dropped as easy_drop)",
    )
    ap.add_argument(
        "--sf_mode",
        choices=["auto", "B", "C"],
        default="C",
        help="C (default): live latent rollout with grad through latents. "
        "B: CE on cached selected latents (detached; severs latent-gen grads — not recommended as default). "
        "auto: currently aliases to C after fixseam root-cause (B hurt latent learning).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    frozen_device = args.frozen_device or device
    if frozen_device.startswith("cuda") and not torch.cuda.is_available():
        frozen_device = device
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"P123_k{args.k}_b0s{args.b0_slack:g}"
    if args.generator_type != "coconut":
        tag += f"_{args.generator_type}"
    if args.select_mode != "wvote":
        tag += f"_{args.select_mode}"
    if args.sf_mode != "auto":
        tag += f"_sf{args.sf_mode}"
    if args.easy_keep_prob < 1.0:
        tag += f"_ek{args.easy_keep_prob:g}"
    if args.dropout_p != 0.2:
        tag += f"_dp{args.dropout_p:g}"
    if args.target_mode != "after":
        tag += f"_{args.target_mode}"
    if args.lambda_lat > 0:
        tag += f"_ll{args.lambda_lat:g}"
    if args.lambda_mmd > 0:
        tag += f"_mmd{args.lambda_mmd:g}"
    if args.lambda_rank > 0:
        tag += f"_rk{args.lambda_rank:g}"
    outdir = Path(args.outdir or f"outputs/p123_{tag}_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    mid_path = outdir / "mid_metrics.jsonl"

    ckpt = str(ROOT / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt
    tckpt = str(ROOT / args.teacher_ckpt) if not args.teacher_ckpt.startswith("/") else args.teacher_ckpt
    o2_id = str(ROOT / args.o2_id) if not args.o2_id.startswith("/") else args.o2_id
    b0_id = str(ROOT / args.b0_id) if not args.b0_id.startswith("/") else args.b0_id

    frozen_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.frozen_dtype
    ]
    # 2-card packing: student+O2 on device0; teacher+B0 on frozen_device → parallel gen + parallel score
    student, tok = load_generator(
        device, ckpt, trainable=True, dtype=torch.float32, generator_type=args.generator_type
    )
    teacher, _ = load_generator(
        frozen_device, tckpt, trainable=False, dtype=frozen_dtype, generator_type=args.generator_type
    )
    o2 = load_prm(
        o2_id,
        tok,
        device,
        dtype=torch.float32 if args.lambda_rank > 0 else frozen_dtype,
        trainable=args.lambda_rank > 0,
    )
    if args.lambda_rank > 0:
        o2.eval()  # dropout off for k-pool scoring; train() only on rank forward
    b0 = load_prm(b0_id, tok, frozen_device, dtype=frozen_dtype, trainable=False)
    replica = None
    replica_device = args.gen_replica_device
    if replica_device:
        if replica_device == device:
            raise SystemExit("--gen_replica_device must differ from student device")
        if replica_device == frozen_device:
            print("WARN replica==frozen_device: both generate on same card, expect little speedup", flush=True)
        replica, _ = load_generator(
            replica_device, ckpt, trainable=False, dtype=frozen_dtype, generator_type=args.generator_type
        )
        print(f"GEN replica on {replica_device} (k split {args.k}//2)", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(ckpt, padding_side="left")
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    prm_tok = tok
    print(
        f"DEVICES student+O2={device} teacher+B0={frozen_device} "
        f"(parallel gen + parallel O2‖B0 score)",
        flush=True,
    )

    mem = _vram_gb(device)
    k_before = args.k
    if args.auto_pack and mem is not None:
        # GPT2-scale student leaves most of an H20 idle even at k=32; pack harder.
        if mem["used_frac"] < 0.25 and mem["free_gb"] > 60:
            args.k = max(args.k, 64)
            args.grad_accum = max(args.grad_accum, 8)
        elif mem["used_frac"] < 0.35 and mem["free_gb"] > 40:
            args.k = max(args.k, 32)
            args.grad_accum = max(args.grad_accum, 4)
        elif mem["used_frac"] < 0.45 and mem["free_gb"] > 25:
            args.k = max(args.k, 16)
    print(
        f"PACK k={args.k} (req {k_before}) grad_accum={args.grad_accum} "
        f"frozen_dtype={args.frozen_dtype} vram={_vram_gb(device)}"
    )

    projector = None
    positive_by_idx = {}
    if args.lambda_mmd > 0:
        # Prefer online teacher/correct latents (full gsm_train). Optional cache is bonus.
        dim = None
        pos_path = ROOT / args.positive_latents
        if pos_path.is_file():
            pos = torch.load(pos_path, map_location="cpu", weights_only=False)
            positive_by_idx = pos.get("by_idx") or {}
            dim = pos.get("dim")
            print(
                f"P2 cache: n_ok={pos.get('n_ok', len(positive_by_idx))} "
                f"(pilot-scale; online H^pos used when missing)"
            )
        if dim is None:
            dim = int(getattr(student.config, "n_embd", None) or student.config.hidden_size)
        projector = ContentProjector(dim, args.mmd_bottleneck).to(device)
        projector.train()
        print(f"P2 MMD on: λ={args.lambda_mmd} bn={args.mmd_bottleneck} dim={dim} online_Hpos=1")

    ds = TrainJSON(ROOT / args.train_json)
    print(
        f"P123 n={len(ds)} k={args.k} b0_slack={args.b0_slack} prefer_cot={args.prefer_cot} "
        f"select={args.select_mode} sf_mode={args.sf_mode} easy_keep={args.easy_keep_prob} "
        f"max_new_tokens={args.max_new_tokens} dropout_p={args.dropout_p} "
        f"λ_lat={args.lambda_lat} λ_mmd={args.lambda_mmd} λ_rank={args.lambda_rank} lr={args.lr}"
    )

    params = [p for p in student.parameters() if p.requires_grad]
    if projector is not None:
        params = params + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    opt_o2 = None
    if args.lambda_rank > 0:
        opt_o2 = torch.optim.AdamW([p for p in o2.parameters() if p.requires_grad], lr=args.o2_lr)
        print(f"P1 O2 rank on: λ={args.lambda_rank} margin={args.rank_margin} o2_lr={args.o2_lr}", flush=True)

    meta = {
        **vars(args),
        "outdir": str(outdir),
        "recipe": f"p123_{args.select_mode}_sf{args.sf_mode}",
        "vram": _vram_gb(device),
        "phases": {
            "P1": (
                f"O2 leftover-rank λ={args.lambda_rank} + B0 health"
                if args.lambda_rank > 0
                else "frozen O2 select + B0 health"
            ),
            "P2": f"MMD λ={args.lambda_mmd}",
            "P3": f"SF CE on {args.select_mode} target; sf_mode={args.sf_mode}",
        },
        "claim_metric": "(gen+O2 wvote) - (A+B0 top1)",
        "protocol": {
            "max_new_tokens": args.max_new_tokens,
            "dropout_p": args.dropout_p,
            "select_mode": args.select_mode,
            "sf_mode": args.sf_mode,
            "claim_agg": "wvote",
        },
        "kill_criteria": {
            "vs": "dualBeat N16 O2 wvote",
            "need_wvote_pp": 1.0,
            "need_top_pp": 0.5,
            "note": "wvote primary; top closing toward wvote is secondary alignment signal",
        },
    }
    json.dump(meta, open(outdir / "train_args.json", "w"), indent=2)

    order = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(order)

    step = 0
    accum_i = 0
    skips: Counter = Counter()
    sources: Counter = Counter()
    mid_buf = []
    losses = []
    attempts = 0
    max_attempts = args.max_steps * 50
    pbar = tqdm(total=args.max_steps, desc=tag)
    oi = 0
    opt.zero_grad(set_to_none=True)

    pending = []
    while step < args.max_steps and attempts < max_attempts:
        if not pending:
            exs = [ds[order[(oi + j) % len(order)]] for j in range(args.q_batch)]
            oi += args.q_batch
            student.eval()
            timing_out = {}
            pending = list(
                zip(
                    exs,
                    sample_and_score_safe(
                        student, teacher, o2, b0, gen_tok, prm_tok, exs=exs,
                        k=args.k, latent_length=args.latent_length,
                        max_new_tokens=args.max_new_tokens, device=device,
                        dropout_p=args.dropout_p, chunk=args.score_chunk,
                        frozen_device=frozen_device,
                        replica=replica, replica_device=replica_device,
                        timing_out=timing_out,
                        generator_type=args.generator_type,
                    ),
                )
            )
            if timing_out:
                print(
                    f"TIMING t_gen={timing_out.get('t_gen', 0):.2f}s "
                    f"t_score={timing_out.get('t_score', 0):.2f}s "
                    f"split_k={timing_out.get('split_k')} "
                    f"q={timing_out.get('nq')} k={timing_out.get('k')}",
                    flush=True,
                )
            if attempts == 0 and torch.cuda.is_available():
                print(
                    f"BATCH q={args.q_batch} k={args.k} seqs={args.q_batch * args.k} "
                    f"peak_alloc_gb={torch.cuda.max_memory_allocated(device) / 1e9:.1f} vram={_vram_gb(device)}",
                    flush=True,
                )
        ex, (rows, (t_after, t_ok, t_o2, t_b0, t_lat)) = pending.pop(0)
        attempts += 1
        q, gold, qidx = ex["question"], ex["answer"], ex["idx"]
        mid = mid_from_rows(rows, t_o2, t_b0, t_ok)
        src, target, win_lat, reason = select_target(
            rows,
            t_after,
            t_ok,
            t_o2,
            t_b0,
            t_lat,
            margin_min=args.margin_min,
            beat_margin=args.beat_margin,
            b0_slack=args.b0_slack,
            prefer_cot=args.prefer_cot,
            select_mode=args.select_mode,
        )
        if (
            args.select_mode == "coverage"
            and src == "beat_self"
            and args.easy_keep_prob < 1.0
            and random.random() > args.easy_keep_prob
        ):
            src, target, win_lat, reason = "skip", None, None, "easy_drop"
        mid["source"] = src
        mid["reason"] = reason
        mid["qid"] = qidx
        mid["step"] = step

        rank_pair = mixed_rank_pair(rows) if args.lambda_rank > 0 else None
        do_ce = src != "skip" and bool(target)
        if not do_ce and rank_pair is None:
            skips[reason] += 1
            mid_buf.append(mid)
            continue
        mid_buf.append(mid)

        if args.target_mode == "process" and do_ce and ex.get("steps"):
            body = "\n".join(ex["steps"])
            if body:
                ans_str = (
                    str(int(gold)) if abs(gold - round(gold)) < 1e-9 else f"{gold:g}"
                )
                target = f"{body}\n### {ans_str}"

        loss = None
        out = {"grad_to_latent_norm": 0.0, "mode": None}
        loss_ans = 0.0
        loss_lat_v = 0.0
        loss_mmd_v = 0.0
        loss_rank_v = 0.0
        rank_gap = None
        lam_mmd = 0.0
        mmd_src = None

        if do_ce:
            batch, _ = make_batch(
                tok, q, target, args.max_target_len, device, generator_type=args.generator_type
            )
            student.train()
            # Close sample↔backprop gap with C + λ_lat (MSE to selected latents).
            # Do NOT default to SF-B: cached latents are detached and kill g_lat (~99.5% steps in fixseam).
            use_cached = win_lat is not None and args.sf_mode == "B"
            if args.sf_mode in ("auto", "C"):
                use_cached = False
            if use_cached:
                cached = win_lat.to(device=device, dtype=torch.float32)
                if cached.dim() == 2:
                    cached = cached.unsqueeze(0)
                out = sf_train_step(
                    student, batch, mode="B", latent_length=args.latent_length, cached_latents=cached
                )
            else:
                out = sf_train_step(student, batch, mode="C", latent_length=args.latent_length)
            loss = out["loss"]
            loss_ans = float(loss.detach().item())

            if args.lambda_lat > 0 and win_lat is not None and out["mode"] == "C":
                live = out["latents"]  # (1, L, D)
                ref = win_lat.to(device=device, dtype=live.dtype).unsqueeze(0)
                loss_lat = torch.nn.functional.mse_loss(live, ref)
                loss = loss + args.lambda_lat * loss_lat
                loss_lat_v = float(loss_lat.detach().item())

            # P2: H^pos = teacher latents if correct, else cache, else selected correct latents
            H_pos = None
            if projector is not None:
                if t_ok and t_lat is not None:
                    H_pos, mmd_src = t_lat, "teacher"
                elif int(qidx) in positive_by_idx:
                    H_pos, mmd_src = positive_by_idx[int(qidx)], "cache"
                elif win_lat is not None and src in ("beat_self", "wvote_self"):
                    H_pos, mmd_src = win_lat, "win_self"
            if projector is not None and H_pos is not None:
                if args.mmd_ramp_steps > 0:
                    lam_mmd = args.lambda_mmd * min(1.0, (step + 1) / args.mmd_ramp_steps)
                else:
                    lam_mmd = args.lambda_mmd
                H_sf = out["latents"]
                href = H_pos.to(device=device, dtype=H_sf.dtype)
                if href.dim() == 2:
                    href = href.unsqueeze(0)
                C_sf = content_view(H_sf, mode="projector", projector=projector)
                C_pos = content_view(href, mode="projector", projector=projector)
                loss_mmd = mmd_rbf_multiscale(C_sf, C_pos.detach())
                loss = loss + lam_mmd * loss_mmd
                loss_mmd_v = float(loss_mmd.detach().item())
            mid["mmd_src"] = mmd_src

        if rank_pair is not None:
            o2.train()
            loss_rk, rank_gap = o2_rank_hinge(
                o2, rank_pair[0], rank_pair[1], margin=args.rank_margin
            )
            o2.eval()
            loss_rank_v = float(loss_rk.detach().item())
            mid["rank_gap"] = rank_gap
            mid["rank_active"] = True
            loss = loss_rk * args.lambda_rank if loss is None else loss + args.lambda_rank * loss_rk
        else:
            mid["rank_active"] = False

        (loss / args.grad_accum).backward()
        accum_i += 1
        sources[src if do_ce else "o2_rank"] += 1
        losses.append(
            {
                "step": step,
                "loss": float(loss.detach().item()),
                "loss_ans": loss_ans,
                "loss_lat": loss_lat_v,
                "loss_mmd": loss_mmd_v,
                "loss_rank": loss_rank_v,
                "rank_gap": rank_gap,
                "lam_mmd": lam_mmd,
                "mmd_src": mmd_src,
                "g_lat": out["grad_to_latent_norm"],
                "source": src if do_ce else "o2_rank",
                "sf_mode": out["mode"],
                "reason": reason,
            }
        )

        if accum_i >= args.grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
            if opt_o2 is not None:
                opt_o2.step()
                opt_o2.zero_grad(set_to_none=True)
            accum_i = 0
            step += 1
            pbar.update(1)
            pbar.set_postfix(
                src=src,
                ans=f"{loss_ans:.3f}",
                left=f"{sum(1 for m in mid_buf[-args.diag_every:] if m.get('leftover'))}",
                agree=f"{sum(1 for m in mid_buf[-args.diag_every:] if m.get('o2_b0_top_agree'))}",
            )

            if step % args.diag_every == 0:
                window = mid_buf[-args.diag_every :]
                n = max(len(window), 1)
                summary = {
                    "step": step,
                    "n": len(window),
                    "pool_ok_rate": sum(m["pool_has_correct"] for m in window) / n,
                    "leftover_rate": sum(m["leftover"] for m in window) / n,
                    "o2_top_ok_rate": sum(m["o2_top_correct"] for m in window) / n,
                    "wvote_ok_rate": sum(m.get("wvote_correct", False) for m in window) / n,
                    "top_wv_disagree": sum(m.get("top_wvote_disagree", False) for m in window) / n,
                    "o2_b0_agree": sum(m["o2_b0_top_agree"] for m in window) / n,
                    "mmd_hit_rate": sum(1 for m in window if m.get("mmd_src")) / n,
                    "rank_hit_rate": sum(1 for m in window if m.get("rank_active")) / n,
                    "mean_rank_gap": (
                        sum(m["rank_gap"] for m in window if m.get("rank_gap") is not None)
                        / max(sum(1 for m in window if m.get("rank_gap") is not None), 1)
                    ),
                    "hash3_rate": sum(m["hash3_rate"] for m in window) / n,
                    "cot_rate": sum(m["cot_rate"] for m in window) / n,
                    "mean_unique": sum(m["unique_preds"] for m in window) / n,
                    "mean_alen": sum(m["mean_after_len"] for m in window) / n,
                    "sources": dict(sources),
                    "skips": dict(skips),
                }
                with open(mid_path, "a") as f:
                    f.write(json.dumps(summary) + "\n")
                print(f"\nMID step={step} {summary}", flush=True)

            if step % args.save_every == 0:
                _safe_ckpt(student, tok, o2 if args.lambda_rank > 0 else None, projector, losses, outdir)

    pbar.close()
    _safe_ckpt(student, tok, o2 if args.lambda_rank > 0 else None, projector, losses, outdir, fatal=True)
    json.dump(
        {"sources": dict(sources), "skips": dict(skips), "attempts": attempts, "steps": step},
        open(outdir / "train_stats.json", "w"),
        indent=2,
    )
    # final mid dump
    if mid_buf:
        with open(mid_path, "a") as f:
            f.write(json.dumps({"final_tail": mid_buf[-min(200, len(mid_buf)) :]}) + "\n")
    print(f"DONE steps={step} sources={dict(sources)} skips={dict(skips)} out={outdir}")


if __name__ == "__main__":
    main()
