#!/usr/bin/env python
"""Sweep target-free decoding controls for one saved fMRI-to-T5 checkpoint.

The model weights remain fixed.  Validation targets rank deterministic decoding
configurations, then fixed brain derangements audit only the leading settings.
No test rows are loaded and no row-wise candidate reranking is performed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration

from probe_fmri_supervised_content_head import ContentHead
from train_fmri_minilm_t5_prefix import (
    MiniLMT5Prefix,
    OrderedDelayT5Prefix,
    DelayWeightedCalibrator,
    ResidualBrainCalibrator,
    compact,
    derangement,
    generate,
    inject_t5_cross_attention_lora,
    strings,
)


def parse_numbers(value: str, cast) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--secondary-content-head-checkpoint",
        default="",
        help=(
            "Optional second train-only lexical head. Its logits and train priors "
            "are mixed globally with the lexical head bundled in --checkpoint; "
            "only their shared vocabulary is used."
        ),
    )
    parser.add_argument(
        "--secondary-content-mix",
        type=float,
        default=0.0,
        help="Global secondary-head weight in [0,1]; zero keeps the bundled head unchanged.",
    )
    parser.add_argument("--oof-semantic-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--topks", default="3,5,8")
    parser.add_argument("--bias-strengths", default="1,2,3,4")
    parser.add_argument("--bias-once-values", default="0,1")
    parser.add_argument("--bias-weightings", default="ranked")
    parser.add_argument("--bias-temperatures", default="1")
    parser.add_argument("--prior-subtractions", default="0.25,0.5,0.75")
    parser.add_argument("--max-prior-probabilities", default="1")
    parser.add_argument("--keyword-contexts", default="0,1")
    parser.add_argument("--keyword-strengths", default="1")
    parser.add_argument("--keyword-templates", default="labeled,raw,summarize,sentence")
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--repetition-penalties", default="1")
    parser.add_argument("--no-repeat-ngram-sizes", default="0")
    parser.add_argument("--max-generation-tokens", type=int, default=16)
    parser.add_argument("--min-generation-tokens", default="0,8,10")
    parser.add_argument("--max-output-words", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--derangements", type=int, default=5)
    parser.add_argument("--top-derangement-configs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def infer_content_hidden(state: dict[str, torch.Tensor], vocabulary_size: int) -> int:
    first = state.get("net.1.weight")
    if first is None:
        raise ValueError("Unrecognized content-head state dict.")
    return 0 if first.shape[0] == vocabulary_size else int(first.shape[0])


class MixedContentHead(torch.nn.Module):
    """Fixed convex mixture of two lexical heads on their shared vocabulary."""

    def __init__(
        self,
        primary: torch.nn.Module,
        secondary: torch.nn.Module,
        primary_indices: list[int],
        secondary_indices: list[int],
        secondary_mix: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= secondary_mix <= 1.0:
            raise ValueError("secondary content mix must be in [0,1]")
        if not primary_indices:
            raise ValueError("lexical heads have no shared vocabulary")
        self.primary = primary
        self.secondary = secondary
        self.register_buffer(
            "primary_indices", torch.as_tensor(primary_indices, dtype=torch.long)
        )
        self.register_buffer(
            "secondary_indices", torch.as_tensor(secondary_indices, dtype=torch.long)
        )
        self.secondary_mix = float(secondary_mix)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        primary = self.primary(inputs).index_select(1, self.primary_indices)
        secondary = self.secondary(inputs).index_select(1, self.secondary_indices)
        return primary.lerp(secondary, self.secondary_mix)


def mix_secondary_content_head(
    *,
    primary_head: torch.nn.Module,
    primary_prior_logit: torch.Tensor,
    primary_vocabulary: list[str],
    secondary_checkpoint: str,
    secondary_mix: float,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, list[str]]:
    """Load and globally mix a second train-only head without validation labels."""
    if not secondary_checkpoint or secondary_mix == 0.0:
        return primary_head, primary_prior_logit, primary_vocabulary
    if not 0.0 < secondary_mix <= 1.0:
        raise ValueError("secondary content mix must be in (0,1] when a head is supplied")
    payload = torch.load(secondary_checkpoint, map_location="cpu", weights_only=False)
    return mix_secondary_content_payload(
        primary_head=primary_head,
        primary_prior_logit=primary_prior_logit,
        primary_vocabulary=primary_vocabulary,
        secondary_payload=payload,
        secondary_mix=secondary_mix,
        device=device,
    )


def mix_secondary_content_payload(
    *,
    primary_head: torch.nn.Module,
    primary_prior_logit: torch.Tensor,
    primary_vocabulary: list[str],
    secondary_payload: dict,
    secondary_mix: float,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, list[str]]:
    """Mix a bundled lexical-head payload with the primary head."""
    payload = secondary_payload
    secondary_vocabulary = [str(word) for word in payload["vocabulary"]]
    secondary_state = payload["head_state_dict"]
    if int(payload.get("input_dim", 384)) != 384:
        raise ValueError("Secondary lexical head must consume 384-D semantic features.")
    secondary = ContentHead(
        384,
        infer_content_hidden(secondary_state, len(secondary_vocabulary)),
        len(secondary_vocabulary),
    ).to(device)
    secondary.load_state_dict(secondary_state, strict=True)
    secondary.eval()
    for parameter in secondary.parameters():
        parameter.requires_grad_(False)

    secondary_lookup = {word: index for index, word in enumerate(secondary_vocabulary)}
    primary_indices: list[int] = []
    secondary_indices: list[int] = []
    shared_vocabulary: list[str] = []
    for primary_index, word in enumerate(primary_vocabulary):
        secondary_index = secondary_lookup.get(word)
        if secondary_index is not None:
            primary_indices.append(primary_index)
            secondary_indices.append(secondary_index)
            shared_vocabulary.append(word)
    mixed_head = MixedContentHead(
        primary_head, secondary, primary_indices, secondary_indices, secondary_mix
    ).to(device).eval()
    secondary_prior_probability = torch.as_tensor(
        payload["logit_prior"], dtype=torch.float32, device=device
    ).clamp(1e-5, 1.0 - 1e-5)
    secondary_prior_logit = torch.logit(secondary_prior_probability)
    mixed_prior = primary_prior_logit[primary_indices].lerp(
        secondary_prior_logit[secondary_indices], secondary_mix
    )
    return mixed_head, mixed_prior, shared_vocabulary


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = payload["args"]
    model_name = saved.get("model_name", "t5-small")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    t5 = T5ForConditionalGeneration.from_pretrained(
        model_name, local_files_only=True
    ).to(device).eval()
    for parameter in t5.parameters():
        parameter.requires_grad_(False)
    inject_t5_cross_attention_lora(
        t5,
        rank=int(saved.get("t5_lora_rank", 0)),
        alpha=float(saved.get("t5_lora_alpha", 8.0)),
        last_n_blocks=int(saved.get("t5_lora_last_n_blocks", 2)),
    )
    lora_state = payload.get("t5_lora_state_dict") or {}
    if lora_state:
        missing, unexpected = t5.load_state_dict(lora_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected LoRA keys: {unexpected}")
        if not any("lora_" in key for key in missing):
            pass

    projector_kind = saved.get("projector_kind", "mlp")
    if projector_kind == "delay_transformer":
        projector = OrderedDelayT5Prefix(
            prefix_length=int(saved["prefix_length"]),
            model_dim=t5.config.d_model,
            dropout=float(saved.get("dropout", 0.1)),
        )
    else:
        projector = MiniLMT5Prefix(
            input_dim=384,
            hidden_dim=int(saved["projector_hidden_dim"]),
            prefix_length=int(saved["prefix_length"]),
            model_dim=t5.config.d_model,
            dropout=float(saved.get("dropout", 0.1)),
        )
    projector.load_state_dict(payload["projector_state_dict"], strict=True)
    projector = projector.to(device).eval()

    calibrator = None
    calibrator_state = payload.get("calibrator_state_dict")
    if calibrator_state is not None:
        calibrator_kind = saved.get("brain_calibrator_kind", "residual_mlp")
        if calibrator_kind == "delay_weighted":
            calibrator = DelayWeightedCalibrator().to(device)
        else:
            input_dim = int(calibrator_state["net.0.weight"].numel())
            hidden_dim = int(calibrator_state["net.1.weight"].shape[0])
            calibrator = ResidualBrainCalibrator(
                input_dim, hidden_dim, float(saved.get("dropout", 0.1))
            ).to(device)
        calibrator.load_state_dict(calibrator_state, strict=True)
        calibrator.eval()

    vocabulary = [str(word) for word in payload.get("content_vocabulary") or []]
    content_state = payload.get("content_head_state_dict")
    prior_logit = payload.get("content_prior_logit")
    if not vocabulary or content_state is None or prior_logit is None:
        raise ValueError("Checkpoint does not bundle a lexical content head.")
    hidden_dim = infer_content_hidden(content_state, len(vocabulary))
    content_head = ContentHead(384, hidden_dim, len(vocabulary)).to(device)
    content_head.load_state_dict(content_state, strict=True)
    content_head.eval()
    bundled_secondary = payload.get("secondary_content_head_bundle")
    if bundled_secondary is not None:
        packaged = payload.get("packaged_system") or {}
        decoding = packaged.get("decoding_config") or {}
        secondary_mix = float(decoding.get("secondary_content_mix", 0.0))
        content_head, prior_logit, vocabulary = mix_secondary_content_payload(
            primary_head=content_head,
            primary_prior_logit=torch.as_tensor(
                prior_logit, dtype=torch.float32, device=device
            ),
            primary_vocabulary=vocabulary,
            secondary_payload=bundled_secondary,
            secondary_mix=secondary_mix,
            device=device,
        )
    token_ids = [
        [
            token for token in tokenizer.encode(word, add_special_tokens=False)
            if token not in set(tokenizer.all_special_ids)
        ]
        for word in vocabulary
    ]
    return (
        payload, saved, tokenizer, t5, projector, calibrator,
        content_head, torch.as_tensor(prior_logit, dtype=torch.float32, device=device),
        vocabulary, token_ids,
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    (
        payload, saved, tokenizer, t5, projector, calibrator,
        content_head, prior_logit, vocabulary, token_ids,
    ) = load_model(args.checkpoint, device)
    content_head, prior_logit, vocabulary = mix_secondary_content_head(
        primary_head=content_head,
        primary_prior_logit=prior_logit,
        primary_vocabulary=vocabulary,
        secondary_checkpoint=args.secondary_content_head_checkpoint,
        secondary_mix=args.secondary_content_mix,
        device=device,
    )
    token_ids = [
        [
            token for token in tokenizer.encode(word, add_special_tokens=False)
            if token not in set(tokenizer.all_special_ids)
        ]
        for word in vocabulary
    ]
    if args.secondary_content_head_checkpoint and args.secondary_content_mix > 0.0:
        print(json.dumps({
            "secondary_content_head_checkpoint": args.secondary_content_head_checkpoint,
            "secondary_content_mix": args.secondary_content_mix,
            "shared_vocabulary_size": len(vocabulary),
        }), flush=True)

    with np.load(args.oof_semantic_npz, allow_pickle=True, mmap_mode="r") as archive:
        condition = np.asarray(strings(archive["condition_source"]))
        rows = np.flatnonzero(condition == "validation_prediction_ensemble")
        delayed = np.asarray(archive["input_embeddings"][rows], dtype=np.float32)
    if len(rows) != 266 or delayed.shape[1] != 1536:
        raise ValueError(f"Expected 266 ordered OOF validation rows, got {delayed.shape}.")
    if saved.get("oof_delay_mode", "mean") == "mean":
        delayed = delayed.reshape(len(delayed), 4, 384).mean(axis=1)
    semantic = F.normalize(torch.as_tensor(delayed), p=2, dim=-1)

    with np.load(args.text_npz, allow_pickle=True) as text:
        targets = strings(text["sentence"])[-266:]

    topks = parse_numbers(args.topks, int)
    bias_strengths = parse_numbers(args.bias_strengths, float)
    bias_once_values = [bool(value) for value in parse_numbers(args.bias_once_values, int)]
    bias_weightings = [
        item.strip() for item in args.bias_weightings.split(",") if item.strip()
    ]
    bias_temperatures = parse_numbers(args.bias_temperatures, float)
    prior_subtractions = parse_numbers(args.prior_subtractions, float)
    max_prior_probabilities = parse_numbers(args.max_prior_probabilities, float)
    keyword_contexts = [bool(value) for value in parse_numbers(args.keyword_contexts, int)]
    keyword_strengths = parse_numbers(args.keyword_strengths, float)
    keyword_templates = [
        item.strip() for item in args.keyword_templates.split(",") if item.strip()
    ]
    minimum_tokens = parse_numbers(args.min_generation_tokens, int)
    repetition_penalties = parse_numbers(args.repetition_penalties, float)
    no_repeat_ngram_sizes = parse_numbers(args.no_repeat_ngram_sizes, int)
    configurations = list(itertools.product(
        topks, bias_strengths, bias_once_values, prior_subtractions,
        max_prior_probabilities, keyword_contexts,
        keyword_strengths, keyword_templates, minimum_tokens,
        repetition_penalties, no_repeat_ngram_sizes,
        bias_weightings, bias_temperatures,
    ))
    # Template choice has no effect when keyword context is disabled; retain a
    # single canonical row instead of evaluating duplicates.
    configurations = [
        row for row in configurations
        if (row[5] or row[7] == keyword_templates[0])
        and (row[11] == "softmax" or row[12] == bias_temperatures[0])
    ]
    records: list[dict] = []
    for (
        topk, bias, bias_once, prior, max_prior_probability,
        keywords, keyword_strength,
        keyword_template, min_tokens, repetition_penalty, no_repeat_ngram_size,
        bias_weighting, bias_temperature,
    ) in configurations:
        generated = generate(
            projector, t5, tokenizer, semantic,
            batch_size=args.eval_batch_size,
            max_target_tokens=args.max_generation_tokens,
            min_target_tokens=min_tokens,
            num_beams=args.num_beams,
            device=device,
            calibrator=calibrator,
            content_head=content_head,
            content_prior_logit=prior_logit,
            content_vocabulary_token_ids=token_ids,
            content_bias_topk=topk,
            content_bias_strength=bias,
            content_bias_once=bias_once,
            content_prior_subtraction=prior,
            content_max_prior_probability=max_prior_probability,
            content_bias_weighting=bias_weighting,
            content_bias_temperature=bias_temperature,
            content_vocabulary=vocabulary,
            content_keyword_context=keywords,
            content_keyword_strength=keyword_strength,
            content_keyword_template=keyword_template,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_output_words=args.max_output_words,
        )
        matched = compact(generated, targets)
        records.append({
            "config": {
                "topk": topk, "bias_strength": bias,
                "bias_once": bias_once,
                "prior_subtraction": prior, "keyword_context": keywords,
                "max_prior_probability": max_prior_probability,
                "keyword_strength": keyword_strength,
                "keyword_template": keyword_template,
                "min_generation_tokens": min_tokens,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "bias_weighting": bias_weighting,
                "bias_temperature": bias_temperature,
                "num_beams": args.num_beams,
                "max_output_words": args.max_output_words,
                "secondary_content_head_checkpoint": (
                    args.secondary_content_head_checkpoint or None
                ),
                "secondary_content_mix": args.secondary_content_mix,
            },
            "matched": matched,
            "generated": generated,
        })
        print(json.dumps({"config": records[-1]["config"], "matched": matched}), flush=True)

    records.sort(key=lambda row: (
        -row["matched"]["content_words_overlap"],
        row["matched"]["word_error_rate"],
        -row["matched"]["words_overlap"],
    ))
    rng = np.random.default_rng(args.seed + 4000)
    permutations = [derangement(len(semantic), rng) for _ in range(args.derangements)]
    for record in records[: args.top_derangement_configs]:
        config = record["config"]
        deranged_metrics = []
        for permutation in permutations:
            generated = generate(
                projector, t5, tokenizer, semantic[torch.as_tensor(permutation)],
                batch_size=args.eval_batch_size,
                max_target_tokens=args.max_generation_tokens,
                min_target_tokens=config["min_generation_tokens"],
                num_beams=args.num_beams,
                device=device,
                calibrator=calibrator,
                content_head=content_head,
                content_prior_logit=prior_logit,
                content_vocabulary_token_ids=token_ids,
                content_bias_topk=config["topk"],
                content_bias_strength=config["bias_strength"],
                content_bias_once=config["bias_once"],
                content_prior_subtraction=config["prior_subtraction"],
                content_max_prior_probability=config.get(
                    "max_prior_probability", 1.0
                ),
                content_bias_weighting=config.get("bias_weighting", "ranked"),
                content_bias_temperature=config.get("bias_temperature", 1.0),
                content_vocabulary=vocabulary,
                content_keyword_context=config["keyword_context"],
                content_keyword_strength=config["keyword_strength"],
                content_keyword_template=config["keyword_template"],
                repetition_penalty=config["repetition_penalty"],
                no_repeat_ngram_size=config["no_repeat_ngram_size"],
                max_output_words=config["max_output_words"],
            )
            deranged_metrics.append(compact(generated, targets))
        values = [row["content_words_overlap"] for row in deranged_metrics]
        record["deranged_content_f1_mean"] = float(np.mean(values))
        record["deranged_content_f1_max"] = float(np.max(values))
        record["conditional_content_margin"] = (
            record["matched"]["content_words_overlap"] - float(np.mean(values))
        )
        record["beats_all_derangements"] = (
            record["matched"]["content_words_overlap"] > float(np.max(values))
        )

    output = {
        "scientific_scope": "single-checkpoint val266 decoding sweep; test107 not loaded",
        "checkpoint": args.checkpoint,
        "checkpoint_best": payload.get("best"),
        "sweep_args": vars(args),
        "targets": targets,
        "records": records,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path), "best": records[0]}), flush=True)


if __name__ == "__main__":
    main()
