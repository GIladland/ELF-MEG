#!/usr/bin/env python
"""Audit leakage-safe nearest-neighbour text signal in OOF MiniLM predictions.

The validation query vectors come only from the held-out validation prediction
ensemble.  The searchable memory contains train11725 rows only.  This is an
analysis/architecture probe: it never searches validation targets and never
loads test107.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meg_context_overfit import _CONTENT_WORD_STOPWORDS, _WORD_RE, word_overlap_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=axis, keepdims=True), 1e-8)


def strings(values: np.ndarray) -> list[str]:
    return [str(value.decode() if isinstance(value, bytes) else value) for value in values.tolist()]


def content_words(text: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in _WORD_RE.finditer(text)
        if match.group(0).lower() not in _CONTENT_WORD_STOPWORDS
        and any(character.isalpha() for character in match.group(0))
    ]


def metrics(generated: list[str], targets: list[str]) -> dict:
    return word_overlap_metrics(generated, targets)["summary"]


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(size)
    for _ in range(10_000):
        result = rng.permutation(size)
        if np.all(result != base):
            return result
    return np.roll(base, 1)


def retrieve(query: np.ndarray, database: np.ndarray, sentences: list[str]) -> tuple[list[str], np.ndarray]:
    similarity = normalize(query) @ normalize(database).T
    indices = similarity.argmax(axis=1)
    return [sentences[index] for index in indices.tolist()], indices


def lexical_consensus(
    query: np.ndarray,
    database: np.ndarray,
    sentences: list[str],
    *,
    neighbours: int,
    output_words: int,
) -> list[str]:
    similarity = normalize(query) @ normalize(database).T
    top = np.argpartition(similarity, -neighbours, axis=1)[:, -neighbours:]
    outputs: list[str] = []
    for row, indices in enumerate(top):
        scores: dict[str, float] = {}
        order: dict[str, int] = {}
        for index in indices.tolist():
            weight = float(max(0.0, similarity[row, index]))
            for position, word in enumerate(content_words(sentences[index])):
                scores[word] = scores.get(word, 0.0) + weight
                order[word] = min(order.get(word, position), position)
        ranked = sorted(scores, key=lambda word: (-scores[word], order[word], word))[:output_words]
        outputs.append(" ".join(ranked))
    return outputs


def main() -> None:
    args = parse_args()
    archive = np.load(args.archive, allow_pickle=True)
    condition = np.asarray(strings(archive["condition_source"]))
    oracle_rows = np.flatnonzero(condition == "oracle")
    oof_rows = np.flatnonzero(condition == "oof_prediction")
    val_rows = np.flatnonzero(condition == "validation_prediction_ensemble")
    if not (len(oracle_rows) == len(oof_rows) == 11725 and len(val_rows) == 266):
        raise ValueError("Unexpected leakage-safe archive split sizes.")
    all_vectors = np.asarray(archive["input_embeddings"], dtype=np.float32).reshape(-1, 4, 384)
    sentences = strings(archive["sentence"])
    train_sentences = [sentences[index] for index in oracle_rows.tolist()]
    targets = [sentences[index] for index in val_rows.tolist()]
    if train_sentences != [sentences[index] for index in oof_rows.tolist()]:
        raise ValueError("Oracle and OOF train text rows are not aligned.")

    representations = {
        "mean_query_to_oracle_train": (
            all_vectors[val_rows].mean(axis=1), all_vectors[oracle_rows].mean(axis=1)
        ),
        "mean_query_to_oof_train": (
            all_vectors[val_rows].mean(axis=1), all_vectors[oof_rows].mean(axis=1)
        ),
        "ordered_query_to_oracle_train": (
            normalize(all_vectors[val_rows], axis=2).reshape(len(val_rows), -1),
            normalize(all_vectors[oracle_rows], axis=2).reshape(len(oracle_rows), -1),
        ),
        "ordered_query_to_oof_train": (
            normalize(all_vectors[val_rows], axis=2).reshape(len(val_rows), -1),
            normalize(all_vectors[oof_rows], axis=2).reshape(len(oof_rows), -1),
        ),
    }
    rng = np.random.default_rng(args.seed)
    permutations = [derangement(len(targets), rng) for _ in range(5)]
    results: dict[str, dict] = {}
    for name, (query, database) in representations.items():
        generated, indices = retrieve(query, database, train_sentences)
        matched = metrics(generated, targets)
        deranged = []
        for permutation in permutations:
            shuffled, _ = retrieve(query[permutation], database, train_sentences)
            deranged.append(metrics(shuffled, targets)["content_words_overlap"])
        consensus = {}
        for neighbours in (3, 5, 10, 20):
            generated_words = lexical_consensus(
                query, database, train_sentences,
                neighbours=neighbours, output_words=8,
            )
            consensus[str(neighbours)] = metrics(generated_words, targets)
        results[name] = {
            "nearest_text": matched,
            "nearest_text_deranged_content_f1": deranged,
            "nearest_text_conditional_margin": (
                matched["content_words_overlap"] - float(np.mean(deranged))
            ),
            "lexical_consensus_top8": consensus,
            "retrieved_train_indices": indices.tolist(),
            "generated": generated,
        }
    payload = {
        "scientific_scope": "train11725 memory to val266; test107 absent",
        "archive": args.archive,
        "targets": targets,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        name: {
            "content_f1": row["nearest_text"]["content_words_overlap"],
            "word_f1": row["nearest_text"]["words_overlap"],
            "wer": row["nearest_text"]["word_error_rate"],
            "margin": row["nearest_text_conditional_margin"],
            "consensus_content_f1": {
                k: v["content_words_overlap"]
                for k, v in row["lexical_consensus_top8"].items()
            },
        }
        for name, row in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
