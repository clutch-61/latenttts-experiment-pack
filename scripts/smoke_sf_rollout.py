#!/usr/bin/env python3
"""Smoke test for Phase-G B vs C latent-rollout distinction.

Proves:
1. C: answer loss has nonzero grad w.r.t. latent thoughts
2. B: latents detached → grad_to_latent == 0; params still update via answer CE
3. Params actually change after one optimizer step (C)
4. Rollout uses hidden-state feedback (same idea as inference)

Usage (from LatentTTS-main):
  CUDA_VISIBLE_DEVICES=5 python -m scripts.smoke_sf_rollout
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation_mixin import LatentGenerationMixin  # noqa: E402
from src.paths import MODELS  # noqa: E402
from src.sf_rollout import RolloutBatch, sf_train_step  # noqa: E402


def load_model(device: str):
    model_id = MODELS["coconut"]["id"]
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
    )
    model.to(device)
    model.train()  # need grad; dropout off via attn/embd pdrop=0; resid may remain
    return model, tok


def make_batch(tok, question: str, answer: str, device: str) -> RolloutBatch:
    q = question + "\n<|start-latent|>"
    enc = tok(q, return_tensors="pt")
    # Match coconut_extract_answer_number: text after '#'
    ans_text = f"#{answer}"
    aenc = tok(ans_text, return_tensors="pt", add_special_tokens=False)
    return RolloutBatch(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        answer_ids=aenc["input_ids"].to(device),
        answer_mask=torch.ones_like(aenc["input_ids"]).to(device),
    )


def param_snapshot(model, n: int = 3):
    snaps = []
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            snaps.append(p.detach().float().reshape(-1)[:64].clone())
        if len(snaps) >= n:
            break
    return snaps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model, tok = load_model(device)

    data = json.load(open(ROOT / "data/gsm_train.json"))
    ex = data[0]
    batch = make_batch(tok, ex["question"], str(ex["answer"]).replace(",", ""), device)
    print(f"q_len={batch.input_ids.shape[1]} ans_len={batch.answer_ids.shape[1]} ans_ids={batch.answer_ids.tolist()}")

    # --- C: grad through latents ---
    out_c = sf_train_step(model, batch, mode="C", latent_length=6)
    loss_c = out_c["loss"]
    g_lat = out_c["grad_to_latent_norm"]
    print(f"[C] loss={loss_c.item():.4f} grad_to_latent_norm={g_lat}")

    ok_grad = g_lat is not None and g_lat > 1e-8
    print(f"[C] PASS nonzero latent grad: {ok_grad}")

    before = param_snapshot(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    opt.zero_grad(set_to_none=True)
    loss_c.backward()
    # also check param grads nonzero
    grad_param = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_param += p.grad.detach().float().norm().item()
    print(f"[C] sum_param_grad_norm={grad_param:.4f}")
    opt.step()
    after = param_snapshot(model)
    delta = sum((a - b).abs().sum().item() for a, b in zip(after, before))
    ok_update = delta > 0
    print(f"[C] param_delta_L1={delta:.6g} PASS update: {ok_update}")

    # --- B: detached latents, grad_to_latent must be 0 ---
    # reload fresh model so B is clean (optional); reuse current weights is fine for grad check
    model.zero_grad(set_to_none=True)
    out_b = sf_train_step(model, batch, mode="B", latent_length=6)
    loss_b = out_b["loss"]
    g_b = out_b["grad_to_latent_norm"]
    print(f"[B] loss={loss_b.item():.4f} grad_to_latent_norm={g_b}")
    ok_b_detach = g_b == 0.0
    print(f"[B] PASS latent grad==0: {ok_b_detach}")

    before_b = param_snapshot(model)
    opt.zero_grad(set_to_none=True)
    loss_b.backward()
    grad_param_b = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_param_b += p.grad.detach().float().norm().item()
    opt.step()
    after_b = param_snapshot(model)
    delta_b = sum((a - b).abs().sum().item() for a, b in zip(after_b, before_b))
    ok_b_update = delta_b > 0 and grad_param_b > 0
    print(f"[B] sum_param_grad_norm={grad_param_b:.4f} param_delta_L1={delta_b:.6g} PASS params still update: {ok_b_update}")

    # --- structure check: latent embeds != token embed of <|latent|> ---
    with torch.no_grad():
        from src.sf_rollout import rollout_latent_thoughts, _token_embeds

        lat, _, _ = rollout_latent_thoughts(
            model, batch.input_ids, batch.attention_mask, latent_length=6, detach=True
        )
        fake = _token_embeds(
            model,
            torch.full(
                (1, 6),
                model.config.latent_id,
                dtype=torch.long,
                device=device,
            ),
        )
        # After mapping latent_id->0, this is actually emb(0); still should differ from hiddens
        dist = (lat - fake).float().norm().item()
        print(f"[struct] ||latent_h - emb(pad-for-latent)||={dist:.4f} (should be >> 0)")
        ok_struct = dist > 1.0

    all_ok = ok_grad and ok_update and ok_b_detach and ok_b_update and ok_struct
    print("=" * 60)
    print("SMOKE_RESULT:", "PASS" if all_ok else "FAIL")
    print(
        json.dumps(
            {
                "C_grad_to_latent": g_lat,
                "C_param_update": ok_update,
                "B_latent_grad_zero": ok_b_detach,
                "B_params_still_update": ok_b_update,
                "struct_hidden_not_token_emb": ok_struct,
            },
            indent=2,
        )
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
