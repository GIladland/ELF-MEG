#!/usr/bin/env python
"""Evaluate a trained brain lexical head as a direct held-out text decoder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from meg_context_overfit import word_overlap_metrics
from modules.fmri2sem_bridge import load_mri2sem_model
from probe_fmri_supervised_content_head import ContentHead, encode_features, target_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--mri-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--prior-subtraction",
        type=float,
        default=0.0,
        help=(
            "Subtract this multiple of the train-set content-word logit prior before top-k. "
            "This isolates brain-conditional residual evidence from a frequency-only decoder."
        ),
    )
    parser.add_argument(
        "--prior-text-npz",
        default="",
        help="Optional archive whose train sentences define the lexical prior (defaults to --text-npz).",
    )
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument("--trainbank-retrieval", action="store_true")
    parser.add_argument("--retrieval-query-topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--mri-output-dim", type=int, default=384)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mri-dropout", type=float, default=0.1)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def strings(values: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in values.tolist()]


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(size)
    for _ in range(10_000):
        result = rng.permutation(size)
        if np.all(result != base):
            return result
    return np.roll(base, 1)


def compact_metrics(generated: list[str], targets: list[str]) -> dict[str, float | int]:
    summary = word_overlap_metrics(generated, targets)["summary"]
    return {
        key: summary[key]
        for key in (
            "content_words_overlap",
            "content_words_overlap_precision",
            "content_words_overlap_recall",
            "words_overlap",
            "words_overlap_precision",
            "words_overlap_recall",
            "word_error_rate",
            "word_error_reference_words",
            "word_error_insertions",
            "word_error_deletions",
            "word_error_substitutions",
        )
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    brain = np.load(args.brain_npz, allow_pickle=True, mmap_mode="r")
    val_x = brain["val_x"]
    if len(val_x) != args.val_count:
        raise ValueError(f"Expected {args.val_count} val brain rows, got {len(val_x)}")
    text = np.load(args.text_npz, allow_pickle=True)
    sentences = strings(text["sentence"])
    if len(sentences) != args.train_count + args.val_count:
        raise ValueError("Text archive must contain train11725 followed by val266.")
    targets = sentences[args.train_count :]

    head_payload = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    vocabulary = list(head_payload["vocabulary"])
    feature = str(head_payload["feature"])
    model = load_mri2sem_model(
        args.mri_checkpoint,
        input_dim=int(val_x.shape[1]),
        output_dim=args.mri_output_dim,
        hidden_dim=args.mri_hidden_dim,
        res_blocks=args.mri_res_blocks,
        dropout=args.mri_dropout,
    ).to(device)
    features = encode_features(
        model,
        val_x,
        feature=feature,
        batch_size=args.feature_batch_size,
        device=device,
    )
    del model
    head = ContentHead(
        input_dim=int(head_payload["input_dim"]),
        hidden_dim=int(head_payload["hidden_dim"]),
        output_dim=len(vocabulary),
    ).to(device)
    head.load_state_dict(head_payload["head_state_dict"], strict=True)
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device)).float().cpu()
    prior_text = text
    if args.prior_text_npz:
        prior_text = np.load(args.prior_text_npz, allow_pickle=True)
    prior_sentences = strings(prior_text["sentence"])[: args.train_count]
    train_masks = target_matrix(prior_sentences, vocabulary).float()
    document_frequency = train_masks.sum(dim=0)
    prior = (document_frequency / float(args.train_count)).clamp(1e-5, 1.0 - 1e-5)
    if args.prior_subtraction:
        logits = logits - float(args.prior_subtraction) * torch.logit(prior)[None, :]
    top = logits.topk(k=min(args.topk, len(vocabulary)), dim=1).indices.tolist()
    generated = [" ".join(vocabulary[index] for index in row) for row in top]

    rng = np.random.default_rng(args.seed + 1000)
    shuffled = []
    for repeat in range(args.shuffle_repeats):
        permutation = derangement(len(generated), rng)
        shuffled.append(
            {
                "repeat": repeat,
                "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
                "metrics": compact_metrics([generated[index] for index in permutation], targets),
            }
        )

    output = {
        "status": "complete",
        "scientific_scope": "train11725 lexical head to val266; test107 never loaded",
        "args": vars(args),
        "head_best": head_payload.get("best"),
        "matched": compact_metrics(generated, targets),
        "deranged": shuffled,
        "deranged_content_f1_mean": float(
            np.mean([row["metrics"]["content_words_overlap"] for row in shuffled])
        ) if shuffled else None,
        "deranged_content_f1_max": float(
            np.max([row["metrics"]["content_words_overlap"] for row in shuffled])
        ) if shuffled else None,
        "targets": targets,
        "generated": generated,
    }
    if args.trainbank_retrieval:
        train_sentences = sentences[: args.train_count]
        train_masks = target_matrix(train_sentences, vocabulary).float()
        document_frequency = train_masks.sum(dim=0)
        inverse_document_frequency = torch.log(
            (float(args.train_count) + 1.0) / (document_frequency + 1.0)
        )
        retrieval_prior = (document_frequency / float(args.train_count)).clamp(1e-5, 1.0 - 1e-5)
        residual_scores = logits - torch.logit(retrieval_prior)[None, :]
        use_k = min(args.retrieval_query_topk, residual_scores.shape[1])
        values, indices = residual_scores.topk(k=use_k, dim=1)
        sparse_query = torch.zeros_like(residual_scores)
        sparse_query.scatter_(
            1,
            indices,
            F.relu(values) * inverse_document_frequency[indices],
        )
        candidate_norm = train_masks.sum(dim=1).clamp_min(1.0).sqrt()
        retrieval_scores = sparse_query @ train_masks.T
        retrieval_scores = retrieval_scores / candidate_norm[None, :]
        selected = retrieval_scores.argmax(dim=1).tolist()
        retrieved = [train_sentences[index] for index in selected]
        retrieved_shuffled = []
        for repeat in range(args.shuffle_repeats):
            permutation = derangement(len(retrieved), rng)
            retrieved_shuffled.append(
                compact_metrics([retrieved[index] for index in permutation], targets)
            )
        output["trainbank_residual"] = {
            "query_topk": use_k,
            "metrics": compact_metrics(retrieved, targets),
            "deranged_content_f1_mean": float(
                np.mean([row["content_words_overlap"] for row in retrieved_shuffled])
            ),
            "deranged_content_f1_max": float(
                np.max([row["content_words_overlap"] for row in retrieved_shuffled])
            ),
            "selected_train_indices": selected,
            "generated": retrieved,
        }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "generated_content_words"])
        writer.writerows(zip(targets, generated))
    printed = {
        key: output[key]
        for key in ("matched", "deranged_content_f1_mean", "deranged_content_f1_max")
    }
    if "trainbank_residual" in output:
        printed["trainbank_residual"] = {
            key: output["trainbank_residual"][key]
            for key in ("metrics", "deranged_content_f1_mean", "deranged_content_f1_max")
        }
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
