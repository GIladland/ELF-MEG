#!/usr/bin/env python
"""Evaluate a packaged deterministic fMRI-to-T5 checkpoint on one sealed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sweep_fmri_t5_checkpoint_decoding import load_model
from train_fmri_minilm_t5_prefix import compact, generate, strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--semantic-npz", required=True)
    parser.add_argument("--semantic-key", default="pred")
    parser.add_argument(
        "--semantic-condition-key",
        default="",
        help="Optional metadata key used to select one semantic condition.",
    )
    parser.add_argument(
        "--semantic-condition-value",
        default="",
        help="Required value when --semantic-condition-key is set.",
    )
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument(
        "--text-tail-count",
        type=int,
        default=0,
        help="Select the last N text rows; useful for the fixed val266 tail.",
    )
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    (
        payload, saved, tokenizer, t5, projector, calibrator,
        content_head, prior_logit, vocabulary, token_ids,
    ) = load_model(args.checkpoint, device)
    packaged = payload.get("packaged_system")
    if not isinstance(packaged, dict):
        raise ValueError("checkpoint lacks packaged_system metadata")
    config = packaged.get("decoding_config")
    if not isinstance(config, dict):
        raise ValueError("packaged checkpoint lacks decoding_config")

    with np.load(args.semantic_npz, allow_pickle=True, mmap_mode="r") as semantic_archive:
        semantic_rows = None
        if args.semantic_condition_key:
            if not args.semantic_condition_value:
                raise ValueError(
                    "--semantic-condition-value is required with --semantic-condition-key"
                )
            if args.semantic_condition_key not in semantic_archive.files:
                raise ValueError(
                    f"semantic archive lacks condition key {args.semantic_condition_key!r}"
                )
            conditions = np.asarray(
                strings(semantic_archive[args.semantic_condition_key])
            )
            semantic_rows = np.flatnonzero(
                conditions == args.semantic_condition_value
            )
        raw_values = semantic_archive[args.semantic_key]
        values = np.asarray(
            raw_values if semantic_rows is None else raw_values[semantic_rows],
            dtype=np.float32,
        )
        semantic_metadata = {
            key: np.asarray(
                semantic_archive[key]
                if semantic_rows is None
                else semantic_archive[key][semantic_rows]
            )
            for key in ("story", "start_tr", "stop_tr")
            if key in semantic_archive.files
        }
    with np.load(args.text_npz, allow_pickle=True) as text_archive:
        targets = strings(text_archive[args.sentence_key])
        text_rows = None
        if args.text_tail_count:
            if args.text_tail_count < 0:
                raise ValueError("--text-tail-count must be non-negative")
            text_rows = np.arange(
                max(0, len(targets) - args.text_tail_count), len(targets)
            )
            targets = [targets[index] for index in text_rows]
        text_metadata = {
            key: np.asarray(
                text_archive[key]
                if text_rows is None
                else text_archive[key][text_rows]
            )
            for key in ("story", "start_tr", "stop_tr")
            if key in text_archive.files
        }
    if len(values) != args.expected_count or len(targets) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} rows, got semantic={len(values)} text={len(targets)}"
        )
    if values.shape[1] != 1536:
        raise ValueError(f"expected delayed 1536-D predictions, got {values.shape}")
    for key in sorted(set(semantic_metadata) & set(text_metadata)):
        if not np.array_equal(semantic_metadata[key].astype(str), text_metadata[key].astype(str)):
            raise ValueError(f"semantic/text rows are misaligned on {key}")
    if saved.get("oof_delay_mode", "mean") == "mean":
        values = values.reshape(len(values), 4, 384).mean(axis=1)
    semantic = F.normalize(torch.as_tensor(values), p=2, dim=-1)
    generated = generate(
        projector, t5, tokenizer, semantic,
        batch_size=args.eval_batch_size,
        max_target_tokens=int(saved.get("max_generation_tokens", 16)),
        min_target_tokens=int(config.get("min_generation_tokens", 0)),
        num_beams=int(config["num_beams"]),
        device=device,
        calibrator=calibrator,
        content_head=content_head,
        content_prior_logit=prior_logit,
        content_vocabulary_token_ids=token_ids,
        content_bias_topk=int(config["topk"]),
        content_bias_strength=float(config["bias_strength"]),
        content_bias_once=bool(config["bias_once"]),
        content_prior_subtraction=float(config["prior_subtraction"]),
        content_max_prior_probability=float(config.get("max_prior_probability", 1.0)),
        content_bias_weighting=str(config.get("bias_weighting", "ranked")),
        content_bias_temperature=float(config.get("bias_temperature", 1.0)),
        content_vocabulary=vocabulary,
        content_keyword_context=bool(config["keyword_context"]),
        content_keyword_strength=float(config.get("keyword_strength", 1.0)),
        content_keyword_template=str(config.get("keyword_template", "labeled")),
        repetition_penalty=float(config.get("repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(config.get("no_repeat_ngram_size", 0)),
        max_output_words=int(config["max_output_words"]),
    )
    metrics = compact(generated, targets)
    output = {
        "scientific_scope": f"frozen packaged checkpoint evaluation on {args.split_name}",
        "checkpoint": args.checkpoint,
        "packaged_system": packaged,
        "split": args.split_name,
        "num_examples": len(targets),
        **metrics,
        "targets": targets,
        "generated": generated,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(destination),
        "split": args.split_name,
        **metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
