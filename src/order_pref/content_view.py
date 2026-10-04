"""Position-stripped content views for Phase-2 set alignment.

Guard: reject transforms that keep full ordered information invertibly.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


FAKE_VIEW_NAMES = {
    "identity",
    "clone",
    "detach",
    "uniform_scale",
    "mean_center",
    "diagonal_affine",
}


def reject_fake_content_view(name: str) -> None:
    if name in FAKE_VIEW_NAMES:
        raise ValueError(
            f"content view {name!r} is forbidden: it does not strip order information "
            f"and would leak ordered tokens into a permutation-invariant set loss."
        )


class ContentProjector(nn.Module):
    """Independent bottleneck projector; positions must be removed before call."""

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., d)
        return self.net(h)


def content_view(
    h: torch.Tensor,
    mode: Literal["projector", "pool_shuffle_invariant"] = "projector",
    projector: ContentProjector | None = None,
) -> torch.Tensor:
    """Build a content-only set representation.

    For continuous latents, true position stripping is imperfect (order is the content).
    Phase-2 should only turn this on after decision-AI confirms the claim still holds
    for T~6 continuous thoughts. Default projector path requires an injected module.
    """
    if mode in FAKE_VIEW_NAMES:
        reject_fake_content_view(mode)
    if mode == "projector":
        if projector is None:
            raise ValueError("projector mode requires ContentProjector")
        return projector(h)
    if mode == "pool_shuffle_invariant":
        # deliberately weak: mean over time after random feature dropout — ablation only
        return h.mean(dim=-2, keepdim=True).expand_as(h)
    raise ValueError(f"unknown content view mode {mode!r}")
