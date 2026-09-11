#!/usr/bin/env python3
"""Evaluate semantic predictions as closed-set text retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--prediction-key", default="input_embeddings")
    parser.add_argument("--target-key", default="source_vectors")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--story-key", default="story")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    return parser.parse_args()


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, reference_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def overlap_f1(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def main() -> None:
    args = parse_args()
    with np.load(args.npz, allow_pickle=True) as data:
        prediction = normalize_rows(data[args.prediction_key])
        target = normalize_rows(data[args.target_key])
        sentences = [str(value) for value in data[args.sentence_key]]
        stories = (
            [str(value) for value in data[args.story_key]]
            if args.story_key in data.files
            else None
        )

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    if prediction.shape[0] != len(sentences):
        raise ValueError(
            f"Embedding/sentence row mismatch: {prediction.shape[0]} != {len(sentences)}"
        )

    similarity = prediction @ target.T
    paired_similarity = np.diag(similarity)
    ranks = 1 + np.sum(similarity > paired_similarity[:, None], axis=1)
    retrieved_indices = np.argmax(similarity, axis=1)
    n = len(sentences)

    tokenized = [tokenize(sentence) for sentence in sentences]
    pairwise_wer = np.empty((n, n), dtype=np.float32)
    for reference_index, reference in enumerate(tokenized):
        reference_words = max(1, len(reference))
        for candidate_index, candidate in enumerate(tokenized):
            pairwise_wer[reference_index, candidate_index] = (
                word_edit_distance(reference, candidate) / reference_words
            )

    row_indices = np.arange(n)
    retrieved_wer = pairwise_wer[row_indices, retrieved_indices]
    reference_word_counts = np.asarray([len(words) for words in tokenized], dtype=np.int64)
    edit_errors = np.asarray(
        [
            word_edit_distance(tokenized[index], tokenized[retrieved_index])
            for index, retrieved_index in enumerate(retrieved_indices)
        ],
        dtype=np.int64,
    )
    retrieved_word_f1 = np.asarray(
        [
            overlap_f1(tokenized[retrieved_index], tokenized[index])
            for index, retrieved_index in enumerate(retrieved_indices)
        ],
        dtype=np.float32,
    )
    content_tokenized = [
        [token for token in tokens if token not in CONTENT_STOPWORDS]
        for tokens in tokenized
    ]
    retrieved_content_f1 = np.asarray(
        [
            overlap_f1(content_tokenized[retrieved_index], content_tokenized[index])
            for index, retrieved_index in enumerate(retrieved_indices)
        ],
        dtype=np.float32,
    )
    mismatch_mask = ~np.eye(n, dtype=bool)
    mismatch_cosine = float(similarity[mismatch_mask].mean()) if n > 1 else None
    chance_wrong_wer = float(pairwise_wer[mismatch_mask].mean()) if n > 1 else None
    per_story = {}
    if stories is not None:
        story_array = np.asarray(stories)
        for story in sorted(set(stories)):
            indices = np.flatnonzero(story_array == story)
            restricted_similarity = similarity[np.ix_(indices, indices)]
            restricted_predictions = indices[np.argmax(restricted_similarity, axis=1)]
            per_story[story] = {
                "num_examples": int(len(indices)),
                "global_candidate_top1": float(
                    np.mean(retrieved_indices[indices] == indices)
                ),
                "global_candidate_top1_hits": int(
                    np.sum(retrieved_indices[indices] == indices)
                ),
                "within_story_top1": float(np.mean(restricted_predictions == indices)),
                "within_story_top1_hits": int(np.sum(restricted_predictions == indices)),
                "within_story_chance_top1": 1.0 / len(indices),
            }

    metrics = {
        "contract": "semantic prediction -> nearest paired target embedding -> target text",
        "source_npz": args.npz,
        "prediction_key": args.prediction_key,
        "target_key": args.target_key,
        "sentence_key": args.sentence_key,
        "num_examples": n,
        "embedding_dim": int(prediction.shape[1]),
        "unique_sentences": len(set(sentences)),
        "retrieval": {
            "top1": float(np.mean(ranks <= 1)),
            "top5": float(np.mean(ranks <= min(5, n))),
            "top10": float(np.mean(ranks <= min(10, n))),
            "top1_hits": int(np.sum(ranks <= 1)),
            "top5_hits": int(np.sum(ranks <= min(5, n))),
            "top10_hits": int(np.sum(ranks <= min(10, n))),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "mean_percentile": (
                float(np.mean(1.0 - (ranks - 1) / (n - 1))) if n > 1 else 1.0
            ),
            "matched_cosine": float(np.mean(paired_similarity)),
            "mismatch_cosine": mismatch_cosine,
        },
        "retrieved_text": {
            "corpus_wer": float(edit_errors.sum() / max(1, reference_word_counts.sum())),
            "mean_example_wer": float(np.mean(retrieved_wer)),
            "word_f1": float(np.mean(retrieved_word_f1)),
            "content_word_f1": float(np.mean(retrieved_content_f1)),
            "edit_errors": int(edit_errors.sum()),
            "reference_words": int(reference_word_counts.sum()),
            "exact_text_fraction": float(np.mean(retrieved_indices == row_indices)),
        },
        "chance": {
            "top1": 1.0 / n,
            "top5": min(5, n) / n,
            "top10": min(10, n) / n,
            "random_candidate_mean_wer": float(pairwise_wer.mean()),
            "random_wrong_candidate_mean_wer": chance_wrong_wer,
        },
        "per_story": per_story,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "target_rank",
                    "retrieved_index",
                    "top1_correct",
                    "paired_cosine",
                    "retrieved_cosine",
                    "word_edit_distance",
                    "reference_words",
                    "wer",
                    "target",
                    "retrieved",
                ],
            )
            writer.writeheader()
            for index, retrieved_index in enumerate(retrieved_indices):
                writer.writerow(
                    {
                        "index": index,
                        "target_rank": int(ranks[index]),
                        "retrieved_index": int(retrieved_index),
                        "top1_correct": bool(retrieved_index == index),
                        "paired_cosine": float(paired_similarity[index]),
                        "retrieved_cosine": float(similarity[index, retrieved_index]),
                        "word_edit_distance": int(edit_errors[index]),
                        "reference_words": int(reference_word_counts[index]),
                        "wer": float(retrieved_wer[index]),
                        "target": sentences[index],
                        "retrieved": sentences[retrieved_index],
                    }
                )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
