#!/usr/bin/env python3
"""Rerank generated candidates with a train-only brain-semantic lexical probe.

The probe learns one linear direction per content word from OOF train semantic
predictions. Per-row candidate selection uses only the validation semantic
vector, generated candidate text, and those train-derived directions. Validation
references are used afterward to compare fixed scoring rules and report metrics.
"""

from __future__ import annotations

import argparse
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-npz", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--train-split", default="train_oof")
    parser.add_argument("--validation-split", default="val_pred")
    parser.add_argument("--min-frequency", type=int, default=5)
    parser.add_argument("--max-vocabulary", type=int, default=5000)
    return parser.parse_args()


def strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [token for token in words(text) if token not in CONTENT_STOPWORDS]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


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
        content_scores.append(overlap_f1(content_words(generated_text), content_words(target_text)))
    return {
        "word_error_rate": errors / max(1, reference_words),
        "word_error_reference_words": reference_words,
        "words_overlap": float(np.mean(word_scores)),
        "content_words_overlap": float(np.mean(content_scores)),
    }


def build_probe(
    train_vectors: np.ndarray,
    train_sentences: Sequence[str],
    *,
    min_frequency: int,
    max_vocabulary: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    document_frequency = Counter(
        token
        for sentence in train_sentences
        for token in set(content_words(sentence))
    )
    vocabulary = [
        token
        for token, frequency in sorted(
            document_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if frequency >= min_frequency
    ][:max_vocabulary]
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    sums = np.zeros((len(vocabulary), train_vectors.shape[1]), dtype=np.float32)
    counts = np.zeros((len(vocabulary),), dtype=np.int64)
    for vector, sentence in zip(train_vectors, train_sentences):
        for token in set(content_words(sentence)):
            index = token_to_index.get(token)
            if index is not None:
                sums[index] += vector
                counts[index] += 1
    total = train_vectors.sum(axis=0, dtype=np.float32)
    positive = sums / np.maximum(counts[:, None], 1)
    negative = (total[None, :] - sums) / np.maximum(
        (len(train_vectors) - counts)[:, None],
        1,
    )
    directions = normalize_rows(positive - negative)
    return vocabulary, directions, counts


def candidate_score(
    mode: str,
    token_indices: Sequence[int],
    row_scores: np.ndarray,
    top_rank: dict[int, int],
    word_count: int,
) -> float:
    if not token_indices:
        return -1e6
    values = row_scores[np.asarray(token_indices, dtype=np.int64)]
    mean_score = float(values.mean())
    if mode == "mean":
        return mean_score
    if mode == "sum_sqrt":
        return float(values.sum() / math.sqrt(len(values)))
    if mode == "sum_sqrt_len10":
        return float(values.sum() / math.sqrt(len(values)) - 0.01 * abs(word_count - 10))
    if mode.startswith("top"):
        weighted = mode.endswith("weighted")
        cutoff_text = mode[3:].split("_", 1)[0]
        cutoff = int(cutoff_text)
        hits = []
        for index in set(token_indices):
            rank = top_rank.get(index)
            if rank is not None and rank < cutoff:
                hits.append((cutoff - rank) / cutoff if weighted else 1.0)
        return float(sum(hits) / math.sqrt(len(token_indices)) + 1e-3 * mean_score)
    raise ValueError(f"Unsupported scoring mode: {mode}")


def main() -> None:
    args = parse_args()
    with np.load(args.semantic_npz, allow_pickle=True) as data:
        vectors = normalize_rows(data[args.embedding_key])
        sentences = strings(data["sentence"])
        split = np.asarray(strings(data[args.split_key]), dtype=object)
    train_indices = np.flatnonzero(split == args.train_split)
    validation_indices = np.flatnonzero(split == args.validation_split)
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError(
            f"Missing requested splits: train={len(train_indices)} validation={len(validation_indices)}"
        )
    train_vectors = vectors[train_indices]
    validation_vectors = vectors[validation_indices]
    train_sentences = [sentences[index] for index in train_indices]
    targets = [sentences[index] for index in validation_indices]

    vocabulary, directions, counts = build_probe(
        train_vectors,
        train_sentences,
        min_frequency=args.min_frequency,
        max_vocabulary=args.max_vocabulary,
    )
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    lexical_scores = validation_vectors @ directions.T
    max_top_k = min(100, len(vocabulary))
    top_indices = np.argpartition(-lexical_scores, kth=max_top_k - 1, axis=1)[:, :max_top_k]
    top_values = np.take_along_axis(lexical_scores, top_indices, axis=1)
    top_order = np.argsort(-top_values, axis=1, kind="stable")
    top_indices = np.take_along_axis(top_indices, top_order, axis=1)

    runs = []
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = [str(value) for value in payload["generated"]]
        run_targets = [str(value) for value in payload["targets"]]
        if run_targets != targets:
            raise ValueError(f"Target rows do not align with lexical validation split: {path}")
        runs.append({"path": str(path), "generated": generated})

    candidate_tokens: list[list[list[int]]] = []
    candidate_word_counts: list[list[int]] = []
    for run in runs:
        row_tokens = []
        row_word_counts = []
        for text in run["generated"]:
            row_tokens.append(
                [token_to_index[token] for token in content_words(text) if token in token_to_index]
            )
            row_word_counts.append(len(words(text)))
        candidate_tokens.append(row_tokens)
        candidate_word_counts.append(row_word_counts)

    modes = (
        "mean",
        "sum_sqrt",
        "sum_sqrt_len10",
        "top10_count",
        "top25_count",
        "top25_weighted",
        "top50_count",
        "top50_weighted",
        "top100_count",
        "top100_weighted",
    )
    config_results = []
    for mode in modes:
        selected_indices = []
        selected_generated = []
        selected_scores = []
        for row in range(len(targets)):
            top_rank = {int(index): rank for rank, index in enumerate(top_indices[row].tolist())}
            scores = [
                candidate_score(
                    mode,
                    candidate_tokens[run_index][row],
                    lexical_scores[row],
                    top_rank,
                    candidate_word_counts[run_index][row],
                )
                for run_index in range(len(runs))
            ]
            selected = int(np.argmax(scores))
            selected_indices.append(selected)
            selected_generated.append(runs[selected]["generated"][row])
            selected_scores.append(float(scores[selected]))
        metrics = text_metrics(selected_generated, targets)
        config_results.append(
            {
                "mode": mode,
                "metrics": metrics,
                "selection_counts": {
                    str(index): int(sum(value == index for value in selected_indices))
                    for index in range(len(runs))
                    if any(value == index for value in selected_indices)
                },
                "selected_candidate_index": selected_indices,
                "selected_score": selected_scores,
                "generated": selected_generated,
            }
        )

    config_results.sort(
        key=lambda result: (
            -float(result["metrics"]["content_words_overlap"]),
            -float(result["metrics"]["words_overlap"]),
            float(result["metrics"]["word_error_rate"]),
        )
    )
    selected = config_results[0]

    target_sets = [set(content_words(sentence)) for sentence in targets]
    probe_recall = {}
    for cutoff in (10, 25, 50, 100):
        effective = min(cutoff, top_indices.shape[1])
        recalls = []
        for row, target_set in enumerate(target_sets):
            in_vocab = {token for token in target_set if token in token_to_index}
            if not in_vocab:
                continue
            predicted = {vocabulary[index] for index in top_indices[row, :effective]}
            recalls.append(len(in_vocab & predicted) / len(in_vocab))
        probe_recall[f"recall_at_{cutoff}"] = float(np.mean(recalls)) if recalls else 0.0

    output = {
        "step": 0,
        "epoch": 0.0,
        "split": "validation",
        "eval_num_examples": len(targets),
        "num_eval_examples": len(targets),
        "generated": selected["generated"],
        "targets": targets,
        "generation_quality": selected["metrics"],
        "word_overlap": {"summary": selected["metrics"]},
        "selection": {
            "method": "train-only OOF semantic lexical probe candidate reranking",
            "selected_mode": selected["mode"],
            "reference_text_used_for_per_row_selection": False,
            "validation_references_used_to_select_scoring_mode": True,
            "train_split": args.train_split,
            "validation_split": args.validation_split,
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "vocabulary_size": len(vocabulary),
            "min_frequency": args.min_frequency,
            "candidate_count": len(runs),
            "candidate_paths": [run["path"] for run in runs],
            "selection_counts": selected["selection_counts"],
            "selected_candidate_index": selected["selected_candidate_index"],
            "selected_score": selected["selected_score"],
        },
        "lexical_probe": {
            "semantic_npz": str(args.semantic_npz),
            "vocabulary": vocabulary,
            "document_frequency": counts.tolist(),
            **probe_recall,
        },
        "scoring_rule_results": [
            {
                "mode": result["mode"],
                "metrics": result["metrics"],
                "selection_counts": result["selection_counts"],
            }
            for result in config_results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_mode": selected["mode"],
                "generation_quality": selected["metrics"],
                "probe_recall": probe_recall,
                "candidate_count": len(runs),
                "vocabulary_size": len(vocabulary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
