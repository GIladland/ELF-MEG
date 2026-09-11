#!/usr/bin/env python3
"""Evaluate nearest-sentence retrieval from a strictly training-only semantic bank."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CONTENT_STOPWORDS = set(
    """
    a about after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into
    is it its itself just me more most my no nor not of off on once only or other our ours
    ourselves out over own quite really same she should so some such sure than that the
    their theirs them themselves then there these they this those through to too under
    until up us very was we well were what when where which while who whom whose why will
    with would yes you your yours yourself yourselves
    """.split()
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def overlap_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def text_metrics(generated: Sequence[str], targets: Sequence[str]) -> dict[str, float | int]:
    errors = 0
    reference_words = 0
    word_scores = []
    content_scores = []
    for generated_text, target_text in zip(generated, targets):
        generated_words = words(generated_text)
        target_words = words(target_text)
        errors += edit_distance(target_words, generated_words)
        reference_words += len(target_words)
        word_scores.append(overlap_f1(generated_words, target_words))
        content_scores.append(
            overlap_f1(
                [token for token in generated_words if token not in CONTENT_STOPWORDS],
                [token for token in target_words if token not in CONTENT_STOPWORDS],
            )
        )
    return {
        "word_error_rate": errors / max(1, reference_words),
        "word_error_reference_words": reference_words,
        "words_overlap": float(np.mean(word_scores)),
        "content_words_overlap": float(np.mean(content_scores)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-npz", type=Path, required=True)
    parser.add_argument("--bank-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--bank-split", default="train")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def main() -> None:
    args = parse_args()
    with np.load(args.query_npz, allow_pickle=True) as query_data:
        query = normalize_rows(query_data[args.embedding_key])
        targets = strings(query_data["sentence"])
    with np.load(args.bank_npz, allow_pickle=True) as bank_data:
        split = np.asarray(strings(bank_data[args.split_key]), dtype=object)
        keep = np.flatnonzero(split == args.bank_split)
        if len(keep) == 0:
            raise ValueError(f"No bank rows matched {args.split_key}={args.bank_split!r}")
        bank = normalize_rows(bank_data[args.embedding_key][keep])
        bank_sentences = np.asarray(strings(bank_data["sentence"]), dtype=object)[keep]

    similarity = query @ bank.T
    top_k = min(max(1, args.top_k), bank.shape[0])
    partition = np.argpartition(-similarity, kth=top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(similarity, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1, kind="stable")
    top_indices_local = np.take_along_axis(partition, order, axis=1)
    top_scores = np.take_along_axis(similarity, top_indices_local, axis=1)
    top_indices = keep[top_indices_local]
    top_sentences = bank_sentences[top_indices_local]
    generated = [str(value) for value in top_sentences[:, 0].tolist()]
    metrics = text_metrics(generated, targets)

    output = {
        "step": 0,
        "epoch": 0.0,
        "split": "validation",
        "eval_num_examples": len(targets),
        "num_eval_examples": len(targets),
        "generated": generated,
        "targets": targets,
        "generation_quality": metrics,
        "word_overlap": {"summary": metrics},
        "trainbank_retrieval": {
            "query_npz": str(args.query_npz),
            "bank_npz": str(args.bank_npz),
            "split_filter": {args.split_key: args.bank_split},
            "candidate_count": int(len(keep)),
            "reference_text_used_for_selection": False,
            "validation_rows_in_candidate_bank": 0,
            "mean_top1_cosine": float(top_scores[:, 0].mean()),
            "top_indices": top_indices.tolist(),
            "top_cosines": top_scores.tolist(),
            "top_sentences": top_sentences.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": int(len(keep)),
                "generation_quality": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
