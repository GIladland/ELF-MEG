#!/usr/bin/env python3
"""Snap semantic predictions to their nearest vector in a closed text bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-npz", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-key", default="input_embeddings")
    parser.add_argument("--candidate-key", default="input_embeddings")
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()],
        dtype=object,
    )


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def main() -> None:
    args = parse_args()
    query = load(args.query_npz)
    candidates = load(args.candidate_npz)
    query_vectors = normalize(query[args.query_key])
    candidate_vectors = normalize(candidates[args.candidate_key])
    if query_vectors.shape[1] != candidate_vectors.shape[1]:
        raise ValueError(f"Embedding dimensions differ: {query_vectors.shape} vs {candidate_vectors.shape}")
    if "sentence" not in query or "sentence" not in candidates:
        raise KeyError("Both NPZs must contain sentence arrays")

    similarity = query_vectors @ candidate_vectors.T
    retrieved_index = np.argmax(similarity, axis=1)
    retrieved_cosine = similarity[np.arange(len(query_vectors)), retrieved_index]
    retrieved_sentence = strings(candidates["sentence"])[retrieved_index]
    target_sentence = strings(query["sentence"])

    aligned_candidate_bank = len(query_vectors) == len(candidate_vectors)
    metrics: dict[str, float | int | bool] = {
        "n_queries": int(len(query_vectors)),
        "n_candidates": int(len(candidate_vectors)),
        "aligned_candidate_bank": aligned_candidate_bank,
        "mean_retrieval_cosine": float(np.mean(retrieved_cosine)),
    }
    if aligned_candidate_bank:
        target_similarity = similarity[np.arange(len(query_vectors)), np.arange(len(query_vectors))]
        ranks = 1 + np.sum(similarity > target_similarity[:, None], axis=1)
        metrics.update(
            {
                "top1": float(np.mean(ranks <= 1)),
                "top5": float(np.mean(ranks <= 5)),
                "top10": float(np.mean(ranks <= 10)),
                "mean_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
            }
        )

    output = {
        key: value
        for key, value in query.items()
        if key not in {"input_embeddings", "source_vectors", "schema_json"}
    }
    output.update(
        {
            "input_embeddings": candidate_vectors[retrieved_index].astype(np.float32),
            "source_vectors": query_vectors.astype(np.float32),
            "retrieved_index": retrieved_index.astype(np.int64),
            "retrieved_cosine": retrieved_cosine.astype(np.float32),
            "retrieved_sentence": retrieved_sentence,
            "retrieval_correct": (retrieved_sentence == target_sentence),
            "schema_json": np.asarray(
                json.dumps(
                    {
                        "role": "closed_set_nearest_candidate_semantic_input",
                        "query_npz": str(args.query_npz),
                        "candidate_npz": str(args.candidate_npz),
                        "selection": "argmax cosine(query, candidate_embedding)",
                        "reference_labels_used_for_selection": False,
                        "candidate_text_bank_available_at_inference": True,
                        "metrics": metrics,
                    },
                    indent=2,
                ),
                dtype=object,
            ),
        }
    )
    if aligned_candidate_bank:
        output["target_vectors"] = candidate_vectors.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
