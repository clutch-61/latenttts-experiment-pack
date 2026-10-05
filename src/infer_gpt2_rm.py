from collections import Counter
from typing import Literal
import hashlib
import json
import os
import torch
import numpy as np
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from fire import Fire

from .generation_mixin import LatentGenerationMixin, LatentGenerationConfig
from .paths import MODELS
from .models.gpt2 import COCONUTGPT2ForTokenClassification
from .models.llama import COCONUTLlamaForTokenClassification
from .system_scoring import aggregate_latent_scores, wvote_select
from .utils import set_seed, InferenceCollator


def _answers_hash(answers: list) -> str:
    payload = json.dumps([str(a) for a in answers], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@torch.no_grad()
def main(
    generator_type: Literal["coconut", "codi"] = "coconut",
    generator_id: str | None = None,
    model_dtype: Literal["bf16", "fp16", "fp32"] = "fp32",
    prm_id: str = "../../latentRM-coconut",
    prm_model_family: Literal["llama", "gpt2"] = "gpt2",
    prm_mode: Literal["best_of_n", "beam_search"] = "beam_search",
    data_path: str = "data/gsm_test.json",
    num_return_sequences: int = 1,
    num_beams: int = 4,
    num_beam_candidates: int = 4,
    batch_size: int = 1024,
    latent_length: int = 6,
    max_new_tokens: int | None = 128,
    dropout_p: float | None = None,
    claim_agg: Literal["wvote", "top1"] = "wvote",
    score_reduce: Literal[
        "sum_logit", "mean_logit", "sum_log_prob", "mean_log_prob"
    ] = "sum_logit",
    seed: int = 200,
    sort_by_len: bool = True,
    progress_bar: bool = True,
    result_json: str | None = None,
):
    """BoN eval. Claim Acc defaults to softmax-weighted vote (T=1); top-1 kept as accuracy_top1."""
    new_line_after_input = generator_type == "coconut"
    if prm_mode == "beam_search":
        assert num_return_sequences == 1
        batch_size = max(1, batch_size // num_beam_candidates // num_beams)
    else:
        assert num_return_sequences > 1
        num_beam_candidates = None
        num_beams = 1
        batch_size = max(1, batch_size // num_return_sequences)

    model_id = generator_id or MODELS[generator_type]["id"]
    set_seed(seed)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Loading {data_path}")
        print(
            f"generator_id={model_id} claim_agg={claim_agg} score_reduce={score_reduce} "
            f"dropout_p={dropout_p} max_new_tokens={max_new_tokens}"
        )

    processing_class = AutoTokenizer.from_pretrained(model_id)
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
    start_id = processing_class.convert_tokens_to_ids("<|start-latent|>")
    end_id = processing_class.convert_tokens_to_ids("<|end-latent|>")
    target_id = processing_class.convert_tokens_to_ids(">>")
    assert (
        latent_id != start_id and latent_id != end_id and start_id != end_id
    ), f"latent_id, start_id, end_id must be different, but got {latent_id}, {start_id}, {end_id}"

    generation_config = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        dropout_p=dropout_p,
        generation_mode="beam_search" if prm_mode == "beam_search" else None,
        num_beams=num_beams,
        num_beam_candidates=num_beam_candidates,
        pad_token_id=processing_class.pad_token_id,
        eos_token_id=processing_class.eos_token_id,
        bos_token_id=processing_class.bos_token_id,
    )

    class LatentGPT2LMHeadModel(MODELS[generator_type]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentGPT2LMHeadModel.from_pretrained(
        model_id,
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        target_id=target_id,
        pad_token_id=processing_class.pad_token_id,
        device_map={"": accelerator.process_index},
        torch_dtype=(
            torch.bfloat16 if model_dtype == "bf16" else (torch.float16 if model_dtype == "fp16" else None)
        ),
    )
    if prm_model_family == "llama":

        prm_processing_class = AutoTokenizer.from_pretrained(prm_id)
        prm = COCONUTLlamaForTokenClassification.from_pretrained(
            prm_id,
            latent_id=prm_processing_class.convert_tokens_to_ids("<|latent|>"),
            latent_start_id=prm_processing_class.convert_tokens_to_ids("<|start-latent|>"),
            latent_end_id=prm_processing_class.convert_tokens_to_ids("<|end-latent|>"),
            target_id=prm_processing_class.convert_tokens_to_ids(">>"),
            pad_token_id=prm_processing_class.pad_token_id,
            latent_hidden_size=model.config.hidden_size,
            device_map={"": accelerator.process_index},
            torch_dtype=(
                torch.bfloat16
                if model_dtype == "bf16"
                else (torch.float16 if model_dtype == "fp16" else None)
            ),
        )
    else:
        prm_processing_class = processing_class
        prm = COCONUTGPT2ForTokenClassification.from_pretrained(
            prm_id,
            latent_id=latent_id,
            latent_start_id=start_id,
            latent_end_id=end_id,
            target_id=target_id,
            pad_token_id=processing_class.pad_token_id,
            device_map={"": accelerator.process_index},
            torch_dtype=(
                torch.bfloat16
                if model_dtype == "bf16"
                else (torch.float16 if model_dtype == "fp16" else None)
            ),
        )
    prm.eval()
    model.eval()
    dataset = datasets.Dataset.from_json(data_path)
    postfix = "\n<|start-latent|>" if new_line_after_input else "<|start-latent|>"
    dataset = dataset.map(
        lambda x, idx: {
            "idx": idx,
            "question": x["question"] + postfix,
            "answer": float(str(x["answer"]).replace(",", "")),
        },
        with_indices=True,
    )
    dataset = dataset.map(lambda x: processing_class(x["question"]), batched=True)
    if sort_by_len:
        dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
        dataset = dataset.sort("length")
        dataset = dataset.remove_columns("length")

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=InferenceCollator(processing_class))

    if prm_mode == "best_of_n":
        all_corrects = {}
        voting_accuracies = {}
        top1_accuracies = {}
        per_example = {}
    accuracies = {}

    if accelerator.is_main_process and progress_bar:
        pbar = tqdm(dataloader, colour="green", desc="Generating")
    else:
        pbar = dataloader

    for batch in pbar:
        model_inputs = {
            k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
        }

        output = model.generate(
            **model_inputs,
            process_reward_model=prm if prm_mode == "beam_search" else None,
            generation_config=generation_config,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            use_cache=True,
        )
        text_output = processing_class.batch_decode(output.sequences, skip_special_tokens=True)

        prm_input_texts = []
        if prm_model_family == "llama":
            for text in text_output:
                splited = text.split("\n")
                question = splited[0]
                answer = "\n".join(splited[1:])
                input_text = prm_processing_class.apply_chat_template(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    tokenize=False,
                )
                prm_input_texts.append(input_text)
            inputs = prm_processing_class(prm_input_texts, return_tensors="pt", padding=True)
        else:
            inputs = prm_processing_class(text_output, return_tensors="pt", padding=True)
        if prm_mode == "best_of_n":
            prm_scores = prm(
                input_ids=inputs["input_ids"].to(prm.device),
                attention_mask=inputs["attention_mask"].to(prm.device),
                latent_embeds=output.latent_thoughts.to(prm.device),
                return_dict=True,
            ).logits.squeeze(
                -1
            )  # (B * num_samples, seq)

            latent_mask = (inputs["input_ids"] == prm.config.latent_id).to(prm.device)
            prm_scores = aggregate_latent_scores(prm_scores, latent_mask, reduce=score_reduce)
            prm_scores = prm_scores.reshape(-1, num_return_sequences)
            top_indices = torch.topk(prm_scores, k=1, dim=1)[1]
            prm_scores_cpu = prm_scores.detach().float().cpu()
            del prm_scores
        answer_output = [
            MODELS[generator_type]["answer_extractor"](text_output[i]) for i in range(len(text_output))
        ]
        if prm_mode == "best_of_n":
            voting_result = [
                Counter(answer_output[i : i + num_return_sequences]).most_common(1)[0][0]
                for i in range(0, len(answer_output), num_return_sequences)
            ]
            voting_result = [voting_result[i] == batch["answer"][i] for i in range(len(voting_result))]

        correct = [
            answer_output[i] == batch["answer"][i // num_return_sequences] for i in range(len(answer_output))
        ]
        correct = torch.tensor(correct).view(-1, num_return_sequences)
        del output, inputs
        torch.cuda.empty_cache()
        for i, _idx in enumerate(batch["idx"]):
            qid = int(_idx.item())
            if prm_mode == "best_of_n":
                answers_i = [str(a) for a in answer_output[i * num_return_sequences : (i + 1) * num_return_sequences]]
                corrects_i = correct[i].tolist()
                scores_i = [float(x) for x in prm_scores_cpu[i].tolist()]
                sel_top = int(top_indices[i].item())
                sel_wv = int(wvote_select(answers_i, scores_i))
                sel_claim = sel_wv if claim_agg == "wvote" else sel_top
                all_corrects[qid] = corrects_i
                voting_accuracies[qid] = voting_result[i]
                top1_accuracies[qid] = bool(corrects_i[sel_top])
                accuracies[qid] = bool(corrects_i[sel_claim])
                if result_json is not None:
                    per_example[qid] = {
                        "idx": qid,
                        "gold": float(batch["answer"][i]),
                        "answers": answers_i,
                        "corrects": [bool(c) for c in corrects_i],
                        "scores": scores_i,
                        "selected": sel_top,
                        "selected_correct": bool(corrects_i[sel_top]),
                        "selected_wvote": sel_wv,
                        "selected_wvote_correct": bool(corrects_i[sel_wv]),
                        "claim_selected": sel_claim,
                        "claim_correct": bool(corrects_i[sel_claim]),
                        "claim_agg": claim_agg,
                        "voting_correct": bool(voting_result[i]),
                        "answers_sha1": _answers_hash(answers_i),
                        "any_correct": bool(any(corrects_i)),
                    }
            else:
                accuracies[qid] = correct.item()  # only one item since num_return_sequences == 1

        if accelerator.is_main_process and progress_bar:
            _corrects = sum(1 for v in accuracies.values() if bool(v))
            cur_num_samples = len(accuracies)
            if prm_mode == "best_of_n":
                _cov = sum(
                    1
                    for v in all_corrects.values()
                    if (bool(v.any()) if torch.is_tensor(v) else any(v))
                )
                _voting = sum(1 for v in voting_accuracies.values() if bool(v))
                _top = sum(1 for v in top1_accuracies.values() if bool(v))
                _dict = dict(
                    Cov=f"{_cov/cur_num_samples*100:.4f}% ({_cov}/{cur_num_samples})",
                    Acc=f"{_corrects/cur_num_samples*100:.4f}% ({_corrects}/{cur_num_samples})",
                    Top=f"{_top/cur_num_samples*100:.4f}% ({_top}/{cur_num_samples})",
                    Vot=f"{_voting/cur_num_samples*100:.4f}% ({_voting}/{cur_num_samples})",
                )
            else:
                _dict = dict(
                    Acc=f"{_corrects/cur_num_samples*100:.4f}% ({_corrects}/{cur_num_samples})",
                )
            pbar.set_postfix(_dict)

    accelerator.wait_for_everyone()
    if accelerator.num_processes > 1:
        accuracies = accelerator.gather_for_metrics([accuracies], use_gather_object=True)
        accuracies = {idx: value for d in accuracies for idx, value in d.items()}
        if prm_mode == "best_of_n":
            all_corrects = accelerator.gather_for_metrics([all_corrects], use_gather_object=True)
            all_corrects = {idx: value for d in all_corrects for idx, value in d.items()}
            voting_accuracies = accelerator.gather_for_metrics([voting_accuracies], use_gather_object=True)
            voting_accuracies = {idx: value for d in voting_accuracies for idx, value in d.items()}
            top1_accuracies = accelerator.gather_for_metrics([top1_accuracies], use_gather_object=True)
            top1_accuracies = {idx: value for d in top1_accuracies for idx, value in d.items()}
            if result_json is not None:
                per_example = accelerator.gather_for_metrics([per_example], use_gather_object=True)
                per_example = {idx: value for d in per_example for idx, value in d.items()}

    if accelerator.is_main_process:
        print(f"SEED={seed}")

        corrects = np.array([bool(v) for v in accuracies.values()], dtype=bool)
        print(f"Accuracy ({claim_agg}): {corrects.mean()*100:.4f}%")
        if prm_mode == "best_of_n":
            all_corrects_arr = np.asarray(list(all_corrects.values()), dtype=bool)
            coverages = all_corrects_arr.any(axis=-1).mean()
            voting_accuracies_arr = np.asarray(list(voting_accuracies.values()), dtype=bool)
            top1_arr = np.asarray(list(top1_accuracies.values()), dtype=bool)
            print(f"Coverage: {coverages*100:.4f}%")
            print(f"Top-1 Accuracy: {top1_arr.mean()*100:.4f}%")
            print(f"Majority Voting Accuracy: {voting_accuracies_arr.mean()*100:.4f}%")
            # wvote from per-example if present; else claim==wvote
            if result_json is not None and per_example:
                wvote_acc = float(np.mean([bool(e["selected_wvote_correct"]) for e in per_example.values()]))
            else:
                wvote_acc = float(corrects.mean()) if claim_agg == "wvote" else float(top1_arr.mean())
            print(f"Weighted-vote Accuracy: {wvote_acc*100:.4f}%")
            if result_json is not None:
                os.makedirs(os.path.dirname(result_json) or ".", exist_ok=True)
                payload = {
                    "meta": {
                        "prm_id": prm_id,
                        "generator_type": generator_type,
                        "generator_id": model_id,
                        "data_path": data_path,
                        "num_return_sequences": num_return_sequences,
                        "seed": seed,
                        "sort_by_len": sort_by_len,
                        "prm_mode": prm_mode,
                        "claim_agg": claim_agg,
                        "score_reduce": score_reduce,
                        "dropout_p": dropout_p,
                        "max_new_tokens": max_new_tokens,
                        "n_examples": len(per_example),
                        "accuracy": float(corrects.mean()),
                        "accuracy_top1": float(top1_arr.mean()),
                        "accuracy_wvote": wvote_acc,
                        "coverage": float(coverages),
                        "voting_accuracy": float(voting_accuracies_arr.mean()),
                    },
                    "examples": [per_example[k] for k in sorted(per_example.keys())],
                }
                with open(result_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                print(f"Wrote per-example dump: {result_json}")


if __name__ == "__main__":
    Fire(main)
