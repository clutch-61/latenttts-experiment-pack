#!/usr/bin/env python3
"""Per-question A/B/C dump + transfer / format diagnostics (Phase-G).

Saves per-qid answers for frozen A and finetuned B/C under identical decode settings,
then summarizes transfer counts and output-structure rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import datasets
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.utils import InferenceCollator  # noqa: E402


def load_model(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")

    class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentCOCONUT.from_pretrained(
        model_id,
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tok.pad_token_id,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return model, tok, end_id, tok.eos_token_id


@torch.no_grad()
def dump_model(name, model_id, data_path, n_samples, seed, batch_size, device, max_new_tokens=64):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, tok, end_id, eos_id = load_model(model_id, device)
    gen_cfg = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=6,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        dropout_p=0.2,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    extractor = MODELS["coconut"]["answer_extractor"]
    raw = json.load(open(data_path))
    ds = datasets.Dataset.from_list(raw)
    ds = ds.map(
        lambda x, idx: {
            "idx": idx,
            "question": x["question"] + "\n<|start-latent|>",
            "answer": float(str(x["answer"]).replace(",", "")),
        },
        with_indices=True,
    )
    ds = ds.map(lambda x: tok(x["question"]), batched=True)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=InferenceCollator(tok))

    by_idx = {}
    for batch in tqdm(loader, desc=f"dump-{name}"):
        model_inputs = {
            k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")
        }
        out = model.generate(
            **model_inputs,
            generation_config=gen_cfg,
            num_return_sequences=n_samples,
            use_cache=True,
        )
        # LatentGenerateDecoderOnlyOutput or tensor
        seqs = out.sequences if hasattr(out, "sequences") else out
        for i, qid in enumerate(batch["idx"].tolist()):
            gold = float(batch["answer"][i])
            rows = []
            for s in range(n_samples):
                seq = seqs[i * n_samples + s]
                ids = seq.tolist()
                text_skip = tok.decode(seq, skip_special_tokens=True)
                text_raw = tok.decode(seq, skip_special_tokens=False)
                has_end_latent = end_id in ids
                after = None
                if has_end_latent:
                    epos = ids.index(end_id)
                    after = tok.decode(seq[epos + 1 :], skip_special_tokens=False)
                pred = extractor(text_skip)
                extract_fail = pred == float("inf")
                starts_hash3 = bool(after is not None and after.lstrip().startswith("###"))
                starts_hash1 = bool(after is not None and after.lstrip().startswith("#") and not starts_hash3)
                has_eos = eos_id in ids
                # early stop: ended right after ### with almost nothing? track after length
                after_len = len(after) if after is not None else 0
                rows.append(
                    {
                        "sample": s,
                        "pred": None if extract_fail else pred,
                        "correct": (not extract_fail) and pred == gold,
                        "extract_fail": extract_fail,
                        "has_end_latent": has_end_latent,
                        "has_eos": has_eos,
                        "starts_###": starts_hash3,
                        "starts_single_#": starts_hash1,
                        "after_end_latent": after[:200] if after else None,
                        "after_len_chars": after_len,
                        "text_skip_tail": text_skip[-120:],
                    }
                )
            by_idx[int(qid)] = {
                "gold": gold,
                "any_correct": any(r["correct"] for r in rows),
                "first_correct": rows[0]["correct"],
                "n_unique_pred": len({r["pred"] for r in rows if r["pred"] is not None}),
                "samples": rows,
            }
    del model
    torch.cuda.empty_cache()
    return by_idx


def transfer(a_ok, b_ok):
    both = a_only = b_only = neither = 0
    for ao, bo in zip(a_ok, b_ok):
        if ao and bo:
            both += 1
        elif ao and not bo:
            a_only += 1
        elif (not ao) and bo:
            b_only += 1
        else:
            neither += 1
    return {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither, "n": len(a_ok)}


def structure_rates(by_idx, n_samples):
    keys = [
        "extract_fail",
        "has_end_latent",
        "has_eos",
        "starts_###",
        "starts_single_#",
    ]
    totals = {k: 0 for k in keys}
    n = 0
    after_lens = []
    for v in by_idx.values():
        for r in v["samples"]:
            n += 1
            for k in keys:
                totals[k] += int(bool(r[k]))
            after_lens.append(r["after_len_chars"])
    rates = {k: totals[k] / n for k in keys}
    rates["mean_after_len"] = float(np.mean(after_lens)) if after_lens else 0.0
    rates["mean_unique_pred"] = float(np.mean([v["n_unique_pred"] for v in by_idx.values()]))
    rates["pass1"] = float(np.mean([v["first_correct"] for v in by_idx.values()]))
    rates["coverage"] = float(np.mean([v["any_correct"] for v in by_idx.values()]))
    rates["n_questions"] = len(by_idx)
    rates["n_samples"] = n_samples
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_id", default="checkpoints/coconut")
    ap.add_argument("--b_id", required=True)
    ap.add_argument("--c_id", required=True)
    ap.add_argument("--data_path", default="data/sf_pilot/valid.json")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--outdir", default="results/full/sf_pilot/diag_transfer")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = str(ROOT / args.data_path)

    dumps = {}
    for name, mid in [("A", args.a_id), ("B", args.b_id), ("C", args.c_id)]:
        mid_path = str(ROOT / mid) if not mid.startswith("/") else mid
        dumps[name] = dump_model(
            name, mid_path, data_path, args.n_samples, args.seed, args.batch_size, device
        )
        json.dump(dumps[name], open(outdir / f"{name}_per_qid.json", "w"), ensure_ascii=False)

    qids = sorted(dumps["A"].keys())
    # primary: coverage-style any_correct; also first_sample
    summary = {"seed": args.seed, "n_samples": args.n_samples, "n": len(qids), "structure": {}, "transfer_any": {}, "transfer_first": {}}
    for name in ("A", "B", "C"):
        summary["structure"][name] = structure_rates(dumps[name], args.n_samples)

    for label, key in [("any", "any_correct"), ("first", "first_correct")]:
        a = [dumps["A"][i][key] for i in qids]
        b = [dumps["B"][i][key] for i in qids]
        c = [dumps["C"][i][key] for i in qids]
        bucket = summary[f"transfer_{label}"]
        bucket["A_vs_B"] = transfer(a, b)
        bucket["A_vs_C"] = transfer(a, c)
        bucket["B_vs_C"] = transfer(b, c)
        # paired diff pp for coverage/first
        def diff_pp(x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            d = (x - y).mean() * 100
            # bootstrap CI
            rng = np.random.default_rng(args.seed)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(x), len(x))
                boots.append((x[idx] - y[idx]).mean() * 100)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return {"mean_pp": float(d), "ci95_pp": [float(lo), float(hi)]}

        bucket["paired_diff_pp"] = {
            "B_minus_A": diff_pp(b, a),
            "C_minus_A": diff_pp(c, a),
            "C_minus_B": diff_pp(c, b),
        }

    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
