"""Train a lightweight latent scorer (SWIFT-Latent / HSRM-Latent) on the
frozen-COCONUT candidate pool (``latent-data/coconut``).

Per-trajectory BCE on outcome labels (``corrects``). Model is a tiny scorer
over latent hidden-state sequences (see ``src/lightweight_scorers.py``).

Example:
    python scripts/train_lightweight_scorer.py \
        --scorer swift --data_dir latent-data/coconut \
        --output_dir outputs/swift_latent --epochs 5 --lr 1e-3 --batch_size 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CachedPickleDatasetV2  # noqa: E402
from src.lightweight_scorers import build_lightweight_scorer  # noqa: E402


def collate_latent(batch):
    """Collate list of {'latent_embeds': (T, D), 'corrects': bool} -> tensors."""
    embeds = torch.stack([x["latent_embeds"].float() for x in batch])  # (B, T, D)
    labels = torch.stack([x["corrects"].float() for x in batch])  # (B,)
    return embeds, labels


@torch.no_grad()
def evaluate(scorer, dataset, batch_size, device):
    """Return BoN top-1 accuracy (select best-scored trajectory per prompt)."""
    scorer.eval()
    # Group trajectories per prompt (get_single_sample=True flattens them).
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_latent,
        num_workers=0,
    )
    n_samples = dataset.n_samples
    correct = 0
    total = 0
    for embeds, labels in loader:
        embeds = embeds.to(device)
        labels = labels.to(device)
        scores = scorer(embeds)  # (B,)
        # reshape (num_prompts, n_samples)
        scores = scores.reshape(-1, n_samples)
        labels = labels.reshape(-1, n_samples)
        best = scores.argmax(dim=1)
        correct += labels.gather(1, best.unsqueeze(1)).sum().item()
        total += labels.shape[0]
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", type=str, required=True, choices=["swift", "hsrm"])
    ap.add_argument("--data_dir", type=str, default="latent-data/coconut")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--model_dim", type=int, default=256, help="HSRM inner dim")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--preload", action="store_true", help="preload train data into RAM (~20GB)")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = full epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = CachedPickleDatasetV2(
        f"{args.data_dir}/train", verbose=True, get_single_sample=True, include_gt=True
    )
    valid64 = CachedPickleDatasetV2(
        f"{args.data_dir}/valid-64", verbose=False, get_single_sample=True, include_gt=True
    )
    print(f"[data] train prompts={train_ds.num_test_samples} n_samples={train_ds.n_samples} "
          f"valid64 prompts={valid64.num_test_samples}")

    scorer = build_lightweight_scorer(args.scorer, hidden_size=args.hidden_size, model_dim=args.model_dim)
    scorer.to(device)
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"[model] {args.scorer} params={n_params:,}")

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    # Positive up-weighting: correct trajectories are the minority for hard prompts.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.preload:
        print("[preload] loading train latent_embeds into RAM ...")
        import glob
        from safetensors.torch import load_file
        from torch.utils.data import TensorDataset

        embeds_list = []
        labels_list = []
        files = sorted(glob.glob(f"{args.data_dir}/train/*.safetensors"))
        for f in tqdm(files, desc="preload"):
            data = load_file(f)
            for key in data:
                if key.endswith(".latent_embeds"):
                    qid = key.split(".")[0]
                    embeds_list.append(data[f"{qid}.latent_embeds"].float())
                    labels_list.append(data[f"{qid}.corrects"].float())
        all_embeds = torch.cat(embeds_list, dim=0)  # (N, 6, 768)
        all_labels = torch.cat(labels_list, dim=0)  # (N,)
        del embeds_list, labels_list
        print(f"[preload] done: {all_embeds.shape} ({all_embeds.numel() * 4 / 1e9:.1f} GB)")
        train_loader = DataLoader(
            TensorDataset(all_embeds, all_labels),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_latent,
            num_workers=args.num_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_acc = 0.0
    for epoch in range(args.epochs):
        scorer.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        running = 0.0
        for i, (embeds, labels) in enumerate(pbar):
            embeds = embeds.to(device)
            labels = labels.to(device)
            logits = scorer(embeds)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running = 0.9 * running + 0.1 * loss.item() if running else loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps and global_step >= args.max_steps:
                break
        scheduler.step()

        acc64 = evaluate(scorer, valid64, args.batch_size, device)
        print(f"[eval] epoch {epoch + 1}: valid64 top1={acc64 * 100:.2f}%")
        if acc64 >= best_acc:
            best_acc = acc64
            ckpt = {
                "scorer": args.scorer,
                "hidden_size": args.hidden_size,
                "model_dim": args.model_dim,
                "state_dict": scorer.state_dict(),
                "epoch": epoch + 1,
                "valid64_top1": acc64,
            }
            torch.save(ckpt, out / "best.pt")
            print(f"[save] best.pt (valid64 top1={acc64 * 100:.2f}%)")
        if args.max_steps and global_step >= args.max_steps:
            break

    # final save + config
    torch.save(
        {
            "scorer": args.scorer,
            "hidden_size": args.hidden_size,
            "model_dim": args.model_dim,
            "state_dict": scorer.state_dict(),
            "epoch": args.epochs,
            "valid64_top1": best_acc,
        },
        out / "last.pt",
    )
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"[done] best valid64 top1={best_acc * 100:.2f}% -> {out}")


if __name__ == "__main__":
    main()
