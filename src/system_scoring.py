"""Shared LatentRM scoring for the P1–P3 system (O2 / B0).

BoN-aligned score: sum of PRM logits on <|latent|> positions (matches infer_gpt2_rm).
Claim aggregation: O2 softmax-weighted vote over answers (T=1).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.models.gpt2 import COCONUTGPT2ForTokenClassification


def wvote_select(
    answers: Sequence[str],
    scores: Sequence[float],
    *,
    temperature: float = 1.0,
) -> int:
    """Index of a sample whose answer wins O2 softmax-weighted vote (T=1).

    Among samples sharing the winning answer, pick the highest raw score.
    """
    if not answers:
        raise ValueError("empty answers")
    if len(answers) != len(scores):
        raise ValueError("answers/scores length mismatch")
    t = max(float(temperature), 1e-8)
    m = max(float(s) for s in scores)
    mass: dict[str, float] = defaultdict(float)
    for a, s in zip(answers, scores):
        mass[str(a)] += math.exp((float(s) - m) / t)
    winner = max(mass, key=mass.get)
    best_i, best_s = -1, -1e30
    for i, (a, s) in enumerate(zip(answers, scores)):
        if str(a) != winner:
            continue
        fs = float(s)
        if fs > best_s:
            best_i, best_s = i, fs
    return best_i


def load_prm(prm_id: str, tok, device, *, dtype=torch.float32, trainable: bool = False):
    prm = COCONUTGPT2ForTokenClassification.from_pretrained(
        prm_id,
        latent_id=tok.convert_tokens_to_ids("<|latent|>"),
        latent_start_id=tok.convert_tokens_to_ids("<|start-latent|>"),
        latent_end_id=tok.convert_tokens_to_ids("<|end-latent|>"),
        target_id=tok.convert_tokens_to_ids(">>"),
        pad_token_id=tok.pad_token_id,
        torch_dtype=dtype,
    ).to(device)
    if trainable:
        prm.train()
        for p in prm.parameters():
            p.requires_grad_(True)
    else:
        prm.eval()
        for p in prm.parameters():
            p.requires_grad_(False)
    return prm


def sum_logit_score(prm, input_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Differentiable BoN sum-logit on latent positions. Latents should be detached for RM-only grads."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    device = next(prm.parameters()).device
    dtype = next(prm.parameters()).dtype
    input_ids = input_ids.to(device)
    logits = prm(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        latent_embeds=latents.to(device=device, dtype=dtype),
        return_dict=True,
    ).logits.squeeze(-1)
    mask = input_ids == prm.config.latent_id
    return torch.where(mask, logits, torch.zeros_like(logits)).sum()


def aggregate_latent_scores(
    logits: torch.Tensor,
    mask: torch.Tensor,
    reduce: str = "sum_logit",
) -> torch.Tensor:
    """Aggregate per-position PRM logits on latent mask → one score per row.

    reduce:
      - sum_logit / mean_logit: raw logits (legacy BoN / infer default)
      - sum_log_prob / mean_log_prob: logsigmoid (matches O2 pref train)
    """
    if logits.dim() == 3:
        logits = logits.squeeze(-1)
    if reduce in ("sum_log_prob", "mean_log_prob"):
        vals = F.logsigmoid(logits)
    elif reduce in ("sum_logit", "mean_logit", "sum", "mean"):
        vals = logits
    else:
        raise ValueError(f"unknown score reduce={reduce!r}")
    masked = torch.where(mask, vals, torch.zeros_like(vals))
    scored = masked.sum(dim=-1)
    if reduce in ("mean_log_prob", "mean_logit", "mean"):
        denom = mask.float().sum(dim=-1).clamp(min=1.0)
        scored = scored / denom
    return scored


@torch.no_grad()
def score_ids_latents(
    prm,
    input_ids: torch.Tensor,
    latents: torch.Tensor,
    *,
    reduce: str = "sum_logit",
) -> float:
    """Score on latent positions; default sum_logit matches historical BoN dumps."""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    logits = prm(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        latent_embeds=latents.to(device=input_ids.device, dtype=next(prm.parameters()).dtype),
        return_dict=True,
    ).logits.squeeze(-1)
    mask = input_ids == prm.config.latent_id
    return float(aggregate_latent_scores(logits, mask, reduce=reduce)[0].item())


@torch.no_grad()
def score_both(
    o2,
    b0: Optional[object],
    input_ids: torch.Tensor,
    latents: torch.Tensor,
    *,
    reduce: str = "sum_logit",
) -> tuple[float, Optional[float]]:
    s_o2 = score_ids_latents(o2, input_ids, latents, reduce=reduce)
    s_b0 = score_ids_latents(b0, input_ids, latents, reduce=reduce) if b0 is not None else None
    return s_o2, s_b0


def ensure_tokenizer(ckpt: str):
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
