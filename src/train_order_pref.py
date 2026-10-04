"""Train LatentRM with hierarchical order-preference (Phase-1, decision-locked).

L = L_task + λ_pref * L_pref
Order negatives = permutation-corrupted latent trajectories from corrects=1 only.
Score = mean log-sigmoid over latent steps (aligned with BCE eval family).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser

from src.dataset import CachedPickleDatasetV2, DataCollatorForLatentRM
from src.order_pref.corruption import CORRUPTION_SPECS, build_negative_bundle
from src.order_pref.preference_loss import hierarchical_preference_loss, sequence_score_from_logits
from src.trainer import LatentRMConfig, LatentRMTrainer


@dataclass
class OrderPrefConfig(LatentRMConfig):
    pref_lambda: float = field(default=0.5, metadata={"help": "Weight of L_pref. 0 disables preference."})
    order_margin: float = field(
        default=0.05,
        metadata={"help": "Start small; recalibrate via estimate_order_gap_scale on real gaps (α≈0.1–0.25)."},
    )
    hierarchy_margin: float = field(default=0.0125, metadata={"help": "≈ β * order_margin with β=0.25"})
    lambda_hier: float = field(default=1.0)
    pref_score_reduce: str = field(
        default="mean_log_prob",
        metadata={"help": "Must match eval_order_pref. Prefer mean_log_prob."},
    )
    corruption_names: str = field(
        default="adjacent_swap,segment_swap,multi_segment,full_reverse",
        metadata={"help": "No composite by default (decision-locked)."},
    )
    pref_on_correct_only: bool = field(
        default=True,
        metadata={"help": "Only build permutation negatives from corrects=1 trajectories."},
    )


class DataCollatorForOrderPref(DataCollatorForLatentRM):
    """Keeps per-row `corrects` so preference can filter successful trajectories."""

    def torch_call(self, features):
        batch = super().torch_call(features)
        corrects = []
        for f in features:
            if "corrects" not in f:
                corrects = None
                break
            c = f["corrects"]
            corrects.append(c.reshape(-1)[0].float() if torch.is_tensor(c) else float(c))
        if corrects is not None:
            batch["corrects"] = torch.stack(
                [c if torch.is_tensor(c) else torch.tensor(c, dtype=torch.float32) for c in corrects]
            )
        return batch


class OrderPrefTrainer(LatentRMTrainer):
    def __init__(self, args: OrderPrefConfig, **kwargs):
        super().__init__(args=args, **kwargs)
        self.corruption_names = [n.strip() for n in args.corruption_names.split(",") if n.strip()]
        for n in self.corruption_names:
            if n not in CORRUPTION_SPECS:
                raise ValueError(f"unknown corruption {n}")

        # Preference needs GT correctness; rebuild train set / collator if enabled.
        if float(args.pref_lambda) > 0:
            if args.loss_type != "bce":
                raise ValueError(
                    "Order preference path requires loss_type=bce (one trajectory per row). "
                    "Keep CE as a separate baseline run; do not silently switch task loss."
                )
            if args.pref_on_correct_only and args.train_dataset:
                self.train_dataset = CachedPickleDatasetV2(
                    data_dir=args.train_dataset,
                    verbose=False,
                    include_gt=True,
                    get_single_sample=True,
                )
            # replace collators to surface corrects
            self.data_collator = DataCollatorForOrderPref(
                self.processing_class,
                latent_token_id=self.processing_class.convert_tokens_to_ids("<|latent|>"),
                latent_end_id=self.processing_class.convert_tokens_to_ids("<|end-latent|>"),
                remove_pad_token=True,
                generator_tokenizer=self.data_collator.generator_tokenizer,
            )
            self.eval_data_collator = DataCollatorForOrderPref(
                self.processing_class,
                latent_token_id=self.processing_class.convert_tokens_to_ids("<|latent|>"),
                remove_pad_token=True,
                generator_tokenizer=self.eval_data_collator.generator_tokenizer,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            latent_embeds=inputs["latent_embeds"],
        )
        labels = inputs["labels"]
        task_loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)

        pref_lambda = float(self.args.pref_lambda)
        if pref_lambda <= 0.0:
            return (task_loss, outputs) if return_outputs else task_loss

        pref_loss = self._preference_loss(model, inputs, outputs)
        loss = task_loss + pref_lambda * pref_loss
        return (loss, outputs) if return_outputs else loss

    def _preference_loss(self, model, inputs, outputs) -> torch.Tensor:
        labels = inputs["labels"]
        latent_list = inputs["latent_embeds"]
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        reduce = self.args.pref_score_reduce
        corrects = inputs.get("corrects")

        eligible = []
        for i, h in enumerate(latent_list):
            if h.dim() != 2:
                continue
            if self.args.pref_on_correct_only:
                if corrects is None:
                    raise ValueError(
                        "pref_on_correct_only=True but batch has no `corrects`. "
                        "Annotated data must keep GT; train set was rebuilt with include_gt=True."
                    )
                if float(corrects[i]) <= 0.5:
                    continue
            eligible.append(i)

        if not eligible:
            return outputs.logits.new_zeros(())

        score_pos_all = sequence_score_from_logits(outputs.logits, labels, reduce=reduce)

        all_neg_embeds = []
        row_indices = []
        ranks_ref = None
        for i in eligible:
            negs, ranks, _ = build_negative_bundle(latent_list[i], names=self.corruption_names)
            if ranks_ref is None:
                ranks_ref = ranks
            all_neg_embeds.extend(negs)
            row_indices.extend([i] * len(negs))

        k = len(self.corruption_names)
        b_eff = len(eligible)
        assert len(row_indices) == b_eff * k and ranks_ref is not None

        neg_out = model(
            input_ids=input_ids[row_indices],
            attention_mask=attn[row_indices] if attn is not None else None,
            latent_embeds=all_neg_embeds,
        )
        score_negs = sequence_score_from_logits(neg_out.logits, labels[row_indices], reduce=reduce)
        score_negs = score_negs.view(b_eff, k)

        return hierarchical_preference_loss(
            score_pos_all[eligible],
            score_negs,
            ranks_ref,
            order_margin=float(self.args.order_margin),
            hierarchy_margin=float(self.args.hierarchy_margin),
            lambda_hier=float(self.args.lambda_hier),
        )


def main(*args, **kwargs):
    parser = HfArgumentParser(OrderPrefConfig)
    if len(args) == 1 and "yaml" in args[0]:
        trainer_args = parser.parse_yaml_file(args[0])[0]
    elif len(args) == 1 and "json" in args[0]:
        trainer_args = parser.parse_json_file(args[0])[0]
    elif len(args) == 0:
        trainer_args = parser.parse_dict(kwargs)[0]
    else:
        raise ValueError(f"Invalid arguments: {args}")

    trainer = OrderPrefTrainer(args=trainer_args)
    trainer.train()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
