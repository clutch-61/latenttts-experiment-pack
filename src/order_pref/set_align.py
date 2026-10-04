"""Phase-2 set alignment losses (Chamfer / multi-scale MMD). Do not mix into ordered stream."""

from __future__ import annotations

from typing import Sequence

import torch


def bidirectional_chamfer(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x,y: (T, d) or (B, T, d). Returns scalar (mean over batch if present)."""
    if x.dim() == 2:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    # (B, T, T)
    dist = torch.cdist(x, y, p=2).pow(2)
    return dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean()


def mmd_rbf_multiscale(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
) -> torch.Tensor:
    """Unbiased-looking batch MMD^2 with multi-scale RBF. x,y: (T,d) or (B,T,d)."""
    if x.dim() == 2:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    b, t, _ = x.shape

    def _kernel(a, b, sigma: float):
        # a,b: (B, T, d)
        dist = torch.cdist(a, b, p=2).pow(2)
        return torch.exp(-dist / (2 * sigma * sigma))

    loss = x.new_zeros(())
    for sigma in sigmas:
        kxx = _kernel(x, x, sigma)
        kyy = _kernel(y, y, sigma)
        kxy = _kernel(x, y, sigma)
        # exclude diagonal for xx/yy
        eye = torch.eye(t, device=x.device, dtype=torch.bool).unsqueeze(0)
        kxx = kxx.masked_fill(eye, 0.0)
        kyy = kyy.masked_fill(eye, 0.0)
        mmd = kxx.sum(dim=(-1, -2)) / (t * (t - 1)) + kyy.sum(dim=(-1, -2)) / (t * (t - 1))
        mmd = mmd - 2.0 * kxy.mean(dim=(-1, -2))
        loss = loss + mmd.mean()
    return loss / len(sigmas)
