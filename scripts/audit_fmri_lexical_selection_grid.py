#!/usr/bin/env python
"""Rank leakage-safe lexical selection rules before expensive ELF decoding.

The lexical head and MRI2SEM model are both frozen and were fit without the
validation rows.  This audit scores the candidate word set on val266 and on
fixed derangements of the same brain rows.  It never loads the sealed test107.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from meg_context_overfit import word_overlap_metrics
from probe_fmri_supervised_content_head import ContentHead, encode_features, target_matrix
from modules.fmri2sem_bridge import load_mri2sem_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--mri-checkpoint", required=True)
    parser.add_argument("--lexical-checkpoint", required=True)
    parser.add_argument("--secondary-lexical-checkpoint", default="")
    parser.add_argument("--secondary-text-npz", default="")
    parser.add_argument("--secondary-mix", default="0,0.1,0.25,0.5")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", default="1,3,5,8,10,12,16,20")
    parser.add_argument("--prior-subtraction", default="0,0.1,0.25,0.5")
    parser.add_argument("--max-prior", default="1,0.2,0.1,0.05")
    parser.add_argument("--derangement-rolls", default="1,17,83")
    return parser.parse_args()


def csv_values(value: str, cast):
    return [
        cast(item.strip())
        for item in value.replace(":", ",").split(",")
        if item.strip()
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lexical_payload = torch.load(
        args.lexical_checkpoint, map_location="cpu", weights_only=False
    )
    vocabulary = [str(word) for word in lexical_payload["vocabulary"]]
    with np.load(args.text_npz, allow_pickle=True) as text:
        sentences = [str(value) for value in text["sentence"].tolist()]
    if len(sentences) != args.train_count + args.val_count:
        raise ValueError("Text archive does not match locked train/validation split")
    targets = sentences[args.train_count :]
    prior = lexical_payload.get("logit_prior")
    if prior is None:
        prior = target_matrix(sentences[: args.train_count], vocabulary).float().mean(dim=0)
    prior = torch.as_tensor(prior).float().clamp(1e-5, 1.0 - 1e-5)
    head = ContentHead(
        input_dim=int(lexical_payload["input_dim"]),
        hidden_dim=int(lexical_payload["hidden_dim"]),
        output_dim=len(vocabulary),
    ).to(device)
    head.load_state_dict(lexical_payload["head_state_dict"])
    head.eval()

    with np.load(args.brain_npz, allow_pickle=True, mmap_mode="r") as brain:
        val_x = brain["val_x"]
        if len(val_x) != args.val_count:
            raise ValueError(f"Expected val_count={args.val_count}, got {len(val_x)}")
        model = load_mri2sem_model(
            args.mri_checkpoint,
            input_dim=int(val_x.shape[1]),
            output_dim=int(lexical_payload["input_dim"]),
            hidden_dim=2048,
            res_blocks=4,
            dropout=0.1,
        ).to(device)
        features = encode_features(
            model,
            val_x,
            feature=str(lexical_payload["feature"]),
            batch_size=args.batch_size,
            device=device,
        )
    del model

    logits = []
    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            logits.append(head(features[start : start + args.batch_size].to(device)).cpu())
    logits = torch.cat(logits)

    lexical_configurations = [(0.0, vocabulary, logits, prior)]
    if args.secondary_lexical_checkpoint:
        secondary_payload = torch.load(
            args.secondary_lexical_checkpoint, map_location="cpu", weights_only=False
        )
        secondary_vocabulary = [str(word) for word in secondary_payload["vocabulary"]]
        secondary_input_dim = int(secondary_payload["input_dim"])
        if secondary_input_dim != 384 or features.shape[1] % secondary_input_dim:
            raise ValueError(
                "Secondary head requires a 384-D mean of ordered MRI2SEM delay blocks"
            )
        secondary_head = ContentHead(
            input_dim=secondary_input_dim,
            hidden_dim=int(secondary_payload["hidden_dim"]),
            output_dim=len(secondary_vocabulary),
        ).to(device)
        secondary_head.load_state_dict(secondary_payload["head_state_dict"])
        secondary_head.eval()
        secondary_features = F.normalize(
            features.reshape(len(features), -1, secondary_input_dim).mean(dim=1),
            p=2,
            dim=-1,
        )
        secondary_logits = []
        with torch.no_grad():
            for start in range(0, len(secondary_features), args.batch_size):
                secondary_logits.append(
                    secondary_head(
                        secondary_features[start : start + args.batch_size].to(device)
                    ).cpu()
                )
        secondary_logits = torch.cat(secondary_logits)
        secondary_prior = secondary_payload.get("logit_prior")
        if secondary_prior is None:
            if not args.secondary_text_npz:
                raise ValueError(
                    "Secondary lexical checkpoint lacks a prior; provide --secondary-text-npz"
                )
            with np.load(args.secondary_text_npz, allow_pickle=True) as secondary_text:
                secondary_sentences = [
                    str(value) for value in secondary_text["sentence"].tolist()
                ]
            if len(secondary_sentences) != args.train_count + args.val_count:
                raise ValueError("Secondary text archive does not match the locked split")
            secondary_prior = target_matrix(
                secondary_sentences[: args.train_count], secondary_vocabulary
            ).float().mean(dim=0)
        secondary_prior = torch.as_tensor(secondary_prior).float().clamp(1e-5, 1.0 - 1e-5)
        secondary_lookup = {
            word: index for index, word in enumerate(secondary_vocabulary)
        }
        primary_indices = []
        secondary_indices = []
        shared_vocabulary = []
        for primary_index, word in enumerate(vocabulary):
            secondary_index = secondary_lookup.get(word)
            if secondary_index is not None:
                primary_indices.append(primary_index)
                secondary_indices.append(secondary_index)
                shared_vocabulary.append(word)
        if not shared_vocabulary:
            raise ValueError("Primary and secondary lexical heads have no shared words")
        primary_indices = torch.as_tensor(primary_indices, dtype=torch.long)
        secondary_indices = torch.as_tensor(secondary_indices, dtype=torch.long)
        for mix in csv_values(args.secondary_mix, float):
            if not 0.0 <= mix <= 1.0:
                raise ValueError("secondary_mix values must be in [0, 1]")
            mixed_logits = logits[:, primary_indices].lerp(
                secondary_logits[:, secondary_indices], mix
            )
            mixed_prior_logits = torch.logit(prior[primary_indices]).lerp(
                torch.logit(secondary_prior[secondary_indices]), mix
            )
            lexical_configurations.append(
                (mix, shared_vocabulary, mixed_logits, torch.sigmoid(mixed_prior_logits))
            )

    topks = csv_values(args.topk, int)
    prior_subtractions = csv_values(args.prior_subtraction, float)
    max_priors = csv_values(args.max_prior, float)
    rolls = csv_values(args.derangement_rolls, int)
    rows = []
    seen_configurations = set()
    for secondary_mix, selected_vocabulary, selected_logits, selected_prior in lexical_configurations:
        # The explicit primary-only configuration and a requested secondary mix
        # of zero are identical; evaluate it once.
        signature = (secondary_mix, len(selected_vocabulary))
        if signature in seen_configurations:
            continue
        seen_configurations.add(signature)
        prior_logits = torch.logit(selected_prior)
        for max_prior in max_priors:
            eligible = selected_prior <= max_prior
            for subtraction in prior_subtractions:
                adjusted = selected_logits - subtraction * prior_logits[None, :]
                adjusted = adjusted.masked_fill(~eligible[None, :], -torch.inf)
                for topk in topks:
                    if int(eligible.sum()) < topk:
                        continue
                    indices = adjusted.topk(k=topk, dim=1).indices
                    predictions = [
                        " ".join(selected_vocabulary[index] for index in row.tolist())
                        for row in indices
                    ]
                    matched = word_overlap_metrics(predictions, targets)["summary"]
                    deranged = []
                    for roll in rolls:
                        rolled = torch.roll(indices, shifts=roll, dims=0)
                        rolled_predictions = [
                            " ".join(selected_vocabulary[index] for index in row.tolist())
                            for row in rolled
                        ]
                        deranged.append(
                            word_overlap_metrics(rolled_predictions, targets)["summary"]
                        )
                    deranged_content = float(
                        np.mean([row["content_words_overlap"] for row in deranged])
                    )
                    rows.append(
                        {
                            "secondary_mix": secondary_mix,
                            "topk": topk,
                            "prior_subtraction": subtraction,
                            "max_prior": max_prior,
                            "eligible_words": int(eligible.sum()),
                            "matched": matched,
                            "deranged": deranged,
                            "conditional_content_margin": matched["content_words_overlap"]
                            - deranged_content,
                        }
                    )

    rows.sort(
        key=lambda row: (
            row["matched"]["content_words_overlap"],
            row["conditional_content_margin"],
        ),
        reverse=True,
    )
    output = {
        "scientific_scope": "frozen train11725 lexical head; val266 only; test107 absent",
        "args": vars(args),
        "rows": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(rows[:20], indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
