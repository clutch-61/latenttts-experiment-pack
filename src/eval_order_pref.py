"""Evaluate order discrimination / hierarchy consistency of a LatentRM checkpoint."""

from __future__ import annotations

from typing import List

import torch
from fire import Fire
from transformers import AutoTokenizer
from tqdm import tqdm

from src.dataset import CachedPickleDatasetV2, DataCollatorForLatentRM
from src.models.gpt2 import COCONUTGPT2ForTokenClassification
from src.order_pref.corruption import DEFAULT_PHASE1_CORRUPTIONS, CORRUPTION_SPECS, build_negative_bundle
from src.order_pref.metrics import hierarchy_consistency, order_discrimination_accuracy
from src.order_pref.preference_loss import sequence_score_from_logits


@torch.no_grad()
def main(
    prm_id: str = "checkpoints/latentRM",
    data_dir: str = "latent-data/coconut/valid-4",
    batch_size: int = 8,
    max_batches: int = 0,
    device: str = "cuda",
):
    """Evaluate order metrics.

    Args:
        max_batches: If <=0, scan the full dataset. If >0, cap at
            ``max_batches * batch_size`` flat trajectory rows
            (``get_single_sample=True`` indexing).
    """
    tok = AutoTokenizer.from_pretrained(prm_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")

    model = COCONUTGPT2ForTokenClassification.from_pretrained(
        prm_id, latent_id=latent_id, latent_start_id=start_id, latent_end_id=end_id
    ).to(device)
    model.eval()

    ds = CachedPickleDatasetV2(data_dir=data_dir, include_gt=True, get_single_sample=True)
    collator = DataCollatorForLatentRM(tok, latent_token_id=latent_id, remove_pad_token=True)

    names = list(DEFAULT_PHASE1_CORRUPTIONS)
    ranks = [CORRUPTION_SPECS[n].rank for n in names]

    pair_accs: List[float] = []
    hier_accs: List[float] = []
    per_rank = {n: [] for n in names}
    n_pos_scored = 0
    n_traj_seen = 0

    if max_batches is None or int(max_batches) <= 0:
        n = len(ds)
        cap_desc = "full"
    else:
        n = min(len(ds), int(max_batches) * int(batch_size))
        cap_desc = f"capped max_batches={max_batches}"
    # With get_single_sample, consecutive blocks of n_samples belong to one question idx.
    n_questions_touched = (n + ds.n_samples - 1) // ds.n_samples if ds.n_samples else n
    print(
        {
            "data_dir": data_dir,
            "prm_id": prm_id,
            "dataset_len_traj": len(ds),
            "n_samples_per_q": ds.n_samples,
            "num_questions_in_ds": ds.num_test_samples,
            "scan_traj": n,
            "scan_questions_approx": n_questions_touched,
            "batch_size": batch_size,
            "cap": cap_desc,
        }
    )

    for start in tqdm(range(0, n, batch_size), desc="order-eval"):
        feats = [ds[i] for i in range(start, min(start + batch_size, n))]
        n_traj_seen += len(feats)
        batch = collator(feats)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        latent_list = [h.to(device) for h in batch["latent_embeds"]]

        # Prefer corrects=1 rows when available (decision-locked analysis set).
        keep = list(range(len(latent_list)))
        if "corrects" in feats[0]:
            keep = [i for i, f in enumerate(feats) if float(f["corrects"].reshape(-1)[0]) > 0.5]
            if not keep:
                continue
            input_ids = input_ids[keep]
            labels = labels[keep]
            if attn is not None:
                attn = attn[keep]
            latent_list = [latent_list[i] for i in keep]

        n_pos_scored += len(latent_list)
        out_pos = model(input_ids=input_ids, attention_mask=attn, latent_embeds=latent_list)
        s_pos = sequence_score_from_logits(out_pos.logits, labels, reduce="mean_log_prob")

        all_negs = []
        rows = []
        for i, h in enumerate(latent_list):
            negs, _, _ = build_negative_bundle(h, names=names)
            all_negs.extend(negs)
            rows.extend([i] * len(negs))
        out_neg = model(
            input_ids=input_ids[rows],
            attention_mask=attn[rows] if attn is not None else None,
            latent_embeds=all_negs,
        )
        s_neg = sequence_score_from_logits(out_neg.logits, labels[rows], reduce="mean_log_prob")
        s_neg = s_neg.view(len(latent_list), len(names))

        disc = order_discrimination_accuracy(s_pos, s_neg)
        hier = hierarchy_consistency(s_neg, ranks)
        pair_accs.append(disc["order_pair_acc"])
        hier_accs.append(hier["hierarchy_consistency"])
        for j, name in enumerate(names):
            per_rank[name].append(float((s_pos > s_neg[:, j]).float().mean().item()))

    def avg(xs):
        return sum(xs) / max(1, len(xs))

    print(
        {
            "n_traj_seen": n_traj_seen,
            "n_pos_correct_scored": n_pos_scored,
            "n_metric_batches": len(pair_accs),
        }
    )
    print({"order_pair_acc": avg(pair_accs), "hierarchy_consistency": avg(hier_accs)})
    print({k: avg(v) for k, v in per_rank.items()})


if __name__ == "__main__":
    Fire(main)
