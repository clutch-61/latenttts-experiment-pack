"""Differentiable COCONUT latent rollout for Phase-G SF-style pilot.

B: cached / detached latent prefix + answer CE (grad does NOT go through rollout)
C: live self-rollout + same answer CE (grad DOES go through latent steps)

Inference-aligned: each latent embed is the last-layer hidden state of the previous
step (see LatentGenerationMixin._sample), not the <|latent|> token embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


@dataclass
class RolloutBatch:
    """Question prefix already includes <|start-latent|> as last token."""

    input_ids: torch.LongTensor  # (B, Lq)
    attention_mask: torch.LongTensor  # (B, Lq)
    answer_ids: torch.LongTensor  # (B, La)  e.g. tokenized "#{answer}"
    answer_mask: torch.LongTensor  # (B, La)  1 = supervised


def _token_embeds(model, token_ids: torch.LongTensor) -> torch.Tensor:
    # Match generate(): map latent_id -> 0 before embedding lookup; latent positions
    # are overwritten with hidden states separately.
    latent_id = model.config.latent_id
    safe = torch.where(token_ids == latent_id, torch.zeros_like(token_ids), token_ids)
    return model.get_input_embeddings()(safe)


def rollout_latent_thoughts(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    latent_length: int = 6,
    detach: bool = False,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Autoregressively emit `latent_length` latent embeds (inference-style).

    Returns:
        latent_embeds: (B, latent_length, D)
        ids_after: input_ids with <|latent|>*L appended (for bookkeeping)
        mask_after: attention_mask extended
    """
    embeds = _token_embeds(model, input_ids)
    ids = input_ids
    mask = attention_mask
    latents = []
    B = input_ids.size(0)
    device = input_ids.device
    latent_id = model.config.latent_id

    for _ in range(latent_length):
        out = model(
            inputs_embeds=embeds,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # Last *valid* token (handles right-padding in a batch). After the first
        # latent append, mask grows on the right so -1 is always valid; still use
        # mask lengths for safety.
        lengths = mask.long().sum(dim=1)  # (B,)
        last_idx = (lengths - 1).clamp(min=0)
        hs = out.hidden_states[-1]  # (B, T, D)
        h = hs[torch.arange(B, device=device), last_idx]  # (B, D)
        if detach:
            h = h.detach()
        latents.append(h)
        embeds = torch.cat([embeds, h.unsqueeze(1)], dim=1)
        ids = torch.cat(
            [ids, torch.full((B, 1), latent_id, dtype=ids.dtype, device=device)],
            dim=1,
        )
        mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=device)], dim=1)

    latent_embeds = torch.stack(latents, dim=1)  # (B, L, D)
    return latent_embeds, ids, mask


def append_end_and_answer_embeds(
    model,
    prefix_embeds: torch.Tensor,
    prefix_mask: torch.LongTensor,
    answer_ids: torch.LongTensor,
    answer_mask: torch.LongTensor,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, int]:
    """prefix_embeds already include question + start + latents.

    Appends <|end-latent|> then answer token embeds.
    Returns embeds, attn_mask, labels (-100 ignore), n_prefix (length before answer).
    """
    B = prefix_embeds.size(0)
    device = prefix_embeds.device
    end_id = model.config.latent_end_id
    end_ids = torch.full((B, 1), end_id, dtype=torch.long, device=device)
    end_emb = _token_embeds(model, end_ids)

    ans_emb = _token_embeds(model, answer_ids)
    embeds = torch.cat([prefix_embeds, end_emb, ans_emb], dim=1)

    end_mask = torch.ones((B, 1), dtype=prefix_mask.dtype, device=device)
    attn = torch.cat([prefix_mask, end_mask, answer_mask], dim=1)

    n_prefix = prefix_embeds.size(1) + 1  # + end-latent
    ignore = torch.full((B, n_prefix), -100, dtype=torch.long, device=device)
    # Only supervise positions where answer_mask==1
    supervised = torch.where(answer_mask.bool(), answer_ids, torch.full_like(answer_ids, -100))
    labels = torch.cat([ignore, supervised], dim=1)
    return embeds, attn, labels, n_prefix


def answer_ce_loss(
    model,
    embeds: torch.Tensor,
    attention_mask: torch.LongTensor,
    labels: torch.LongTensor,
) -> torch.Tensor:
    out = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits  # (B, T, V)
    # Shift for causal LM: predict token t from positions < t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def build_prefix_embeds_with_latents(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    latent_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """question(+start) token embeds + provided latent embeds."""
    q_emb = _token_embeds(model, input_ids)
    embeds = torch.cat([q_emb, latent_embeds], dim=1)
    B, Llat, _ = latent_embeds.shape
    lat_mask = torch.ones((B, Llat), dtype=attention_mask.dtype, device=attention_mask.device)
    mask = torch.cat([attention_mask, lat_mask], dim=1)
    return embeds, mask


def sf_train_step(
    model,
    batch: RolloutBatch,
    *,
    mode: Literal["B", "C"],
    latent_length: int = 6,
    cached_latents: Optional[torch.Tensor] = None,
) -> dict:
    """One forward+loss for B or C.

    B: use cached_latents (or no_grad rollout) detached.
    C: live rollout with gradients through latent steps.
    """
    if mode == "B":
        if cached_latents is None:
            with torch.no_grad():
                cached_latents, _, _ = rollout_latent_thoughts(
                    model,
                    batch.input_ids,
                    batch.attention_mask,
                    latent_length=latent_length,
                    detach=True,
                )
        latents = cached_latents.detach()
        # Ensure no grad edge into latents even if caller forgot detach
        assert not latents.requires_grad
    elif mode == "C":
        latents, _, _ = rollout_latent_thoughts(
            model,
            batch.input_ids,
            batch.attention_mask,
            latent_length=latent_length,
            detach=False,
        )
    else:
        raise ValueError(f"mode must be B or C, got {mode}")

    prefix_embeds, prefix_mask = build_prefix_embeds_with_latents(
        model, batch.input_ids, batch.attention_mask, latents
    )
    embeds, attn, labels, _ = append_end_and_answer_embeds(
        model, prefix_embeds, prefix_mask, batch.answer_ids, batch.answer_mask
    )
    loss = answer_ce_loss(model, embeds, attn, labels)

    # Diagnostics: gradient into first latent vector (C should be nonzero)
    grad_to_latent_norm = None
    if mode == "C" and latents.requires_grad:
        # retain for autograd.grad without freeing graph needed for .backward()
        g = torch.autograd.grad(loss, latents, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            grad_to_latent_norm = g.detach().float().norm().item()
        else:
            grad_to_latent_norm = 0.0
    elif mode == "B":
        grad_to_latent_norm = 0.0  # by construction (detached)

    return {
        "loss": loss,
        "latents": latents,
        "grad_to_latent_norm": grad_to_latent_norm,
        "mode": mode,
    }
