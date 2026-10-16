"""Lightweight latent scorers adapted from recent open-source works.

Adapted (X-Latent) versions — they consume the *latent hidden-state sequence*
of a COCONUT/CODI trajectory (shape (B, T, D)) and return a per-trajectory
scalar score, used for best-of-N selection. The two methods:

- ``SwiftLatentRM``: SWIFT / ELHSR (arXiv:2505.12225, KDD 2026). A linear
  token-level reward model with an optional sigmoid gate, aggregated as
  ``sum(gate * reward) / sum(gate)`` over latent steps.
- ``HSRMLatentRM``: HSRM (arXiv:2608.30841, EMNLP 2026). A 2-layer Transformer
  encoder over latent-step hidden states, mean-pooled and scored (~2M params).

Both use per-trajectory binary cross-entropy on outcome labels in this repo
(unified loss for a clean structural comparison against B0/B1/O2); the upstream
papers use DPO / ranking objectives on their own datasets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SwiftLatentRM(nn.Module):
    """SWIFT-style linear latent reward model.

    ``Linear(D, 2)`` predicts a per-step gate and reward; scores are
    ``sum(gate * reward) / sum(gate)`` over the latent steps.
    """

    def __init__(self, hidden_size: int = 768, disable_gate: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.disable_gate = disable_gate
        if not disable_gate:
            self.fused_layer = nn.Linear(hidden_size, 2)
        else:
            self.reward_layer = nn.Linear(hidden_size, 1)

    def forward(self, latent_embeds: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """latent_embeds: (B, T, D) -> scores (B,)."""
        if not self.disable_gate:
            fused = self.fused_layer(latent_embeds)  # (B, T, 2)
            gates = torch.sigmoid(fused[..., 0])
            rewards = fused[..., 1]
        else:
            gates = torch.ones_like(latent_embeds[..., 0])
            rewards = self.reward_layer(latent_embeds).squeeze(-1)
        if mask is not None:
            gates = gates * mask
            rewards = rewards * mask
        score = (gates * rewards).sum(dim=1) / gates.sum(dim=1).clamp(min=1e-8)
        return score  # (B,)


class HSRMLatentRM(nn.Module):
    """HSRM-style 2-layer Transformer ranker over latent-step hidden states."""

    def __init__(
        self,
        hidden_size: int = 768,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        max_steps: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.input_proj = nn.Linear(hidden_size, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_steps, model_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_mult,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(model_dim, 1)

    def forward(self, latent_embeds: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """latent_embeds: (B, T, D) -> scores (B,)."""
        x = self.input_proj(latent_embeds)  # (B, T, M)
        x = x + self.pos_embed[:, : x.size(1)]
        if mask is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = (mask == 0)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, M)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = x.sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)
        score = self.score_head(pooled).squeeze(-1)  # (B,)
        return score

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_lightweight_scorer(name: str, hidden_size: int = 768, **kwargs) -> nn.Module:
    if name == "swift":
        return SwiftLatentRM(hidden_size=hidden_size, disable_gate=kwargs.pop("disable_gate", False))
    if name == "hsrm":
        return HSRMLatentRM(hidden_size=hidden_size, **kwargs)
    raise ValueError(f"Unknown lightweight scorer: {name} (expected 'swift' or 'hsrm')")
