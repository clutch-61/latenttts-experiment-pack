"""Runnable self-check for Phase-1 order preference primitives (no GPU / weights)."""

from __future__ import annotations

import torch

from src.order_pref.corruption import CORRUPTION_SPECS, apply_corruption, build_negative_bundle
from src.order_pref.preference_loss import hierarchical_preference_loss, sequence_score_from_logits
from src.order_pref.metrics import hierarchy_consistency, order_discrimination_accuracy
from src.order_pref.content_view import reject_fake_content_view
from src.order_pref.set_align import bidirectional_chamfer, mmd_rbf_multiscale


def main() -> None:
    torch.manual_seed(0)
    t, d = 6, 8
    h = torch.randn(t, d)

    # corruptions change order, preserve set of vectors for reverse/adjacent cases roughly
    h_rev = apply_corruption(h, "full_reverse")
    assert h_rev.shape == h.shape
    assert not torch.allclose(h_rev, h)

    negs, ranks, names = build_negative_bundle(h)
    assert len(negs) == 4  # Phase-1 default excludes composite
    assert len(set(ranks)) > 1
    print("corruptions:", list(zip(names, ranks)))

    # preference loss: perfect ranking -> near zero
    score_pos = torch.tensor([2.0, 2.0])
    # ranks ascending severity: mild should score higher (4 corruptions)
    ordered_scores = torch.tensor([[1.5, 1.0, 0.5, 0.0], [1.5, 1.0, 0.5, 0.0]])
    loss_ok = hierarchical_preference_loss(score_pos, ordered_scores, ranks, order_margin=0.1, hierarchy_margin=0.05)
    assert float(loss_ok) < 1e-5, loss_ok

    # inverted hierarchy should hurt
    loss_bad = hierarchical_preference_loss(
        score_pos, ordered_scores.flip(dims=[1]), ranks, order_margin=0.1, hierarchy_margin=0.05
    )
    assert float(loss_bad) > float(loss_ok)

    try:
        hierarchical_preference_loss(score_pos, ordered_scores, None)
        raise AssertionError("expected ValueError for missing ranks")
    except ValueError:
        pass

    logits = torch.randn(2, 10, 1)
    labels = torch.full((2, 10), -100)
    labels[:, 3:9] = 1
    s = sequence_score_from_logits(logits, labels, reduce="mean_log_prob")
    assert s.shape == (2,)
    s2 = sequence_score_from_logits(logits, labels, reduce="sum_log_prob")
    # fixed-length latent mask (6) => mean * 6 == sum
    assert torch.allclose(s * 6, s2, atol=1e-5)

    disc = order_discrimination_accuracy(score_pos, ordered_scores)
    hier = hierarchy_consistency(ordered_scores, ranks)
    assert disc["order_pair_acc"] == 1.0
    assert hier["hierarchy_consistency"] == 1.0

    try:
        reject_fake_content_view("identity")
        raise AssertionError("expected reject")
    except ValueError:
        pass

    x, y = torch.randn(6, 4), torch.randn(6, 4)
    _ = bidirectional_chamfer(x, y)
    _ = mmd_rbf_multiscale(x, y)

    print("order_pref self-check OK")


if __name__ == "__main__":
    main()
