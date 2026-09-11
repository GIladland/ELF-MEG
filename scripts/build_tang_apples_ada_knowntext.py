#!/usr/bin/env python3
"""Build a semantic-only, test-first Tang/Apples ADA known-text corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_split(path: Path, split: str) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=True)
    embeddings = np.asarray(source["input_embeddings"], dtype=np.float32)
    sentences = np.asarray(source["sentence"], dtype=object)
    if embeddings.ndim != 2 or embeddings.shape[1] != 1536:
        raise ValueError(f"Unexpected ADA shape in {path}: {embeddings.shape}")
    if sentences.shape != (embeddings.shape[0],):
        raise ValueError(f"Sentence alignment mismatch in {path}")
    result = {
        "input_embeddings": embeddings,
        "sentence": sentences,
        "source_split": np.full(embeddings.shape[0], split, dtype=object),
    }
    for key in ("subject", "session", "task", "run", "start_samples"):
        if key in source.files:
            value = np.asarray(source[key])
            if value.ndim == 1 and value.shape[0] == embeddings.shape[0]:
                result[key] = value
    return result


def main() -> None:
    args = parse_args()
    # Test-first ordering makes the first 26 rows a deterministic exact-ADA
    # generation audit while all rows remain available for intentional
    # known-text overfitting.
    parts = [
        load_split(args.test, "test"),
        load_split(args.val, "val"),
        load_split(args.train, "train"),
    ]
    keys = set.intersection(*(set(part) for part in parts))
    payload = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    payload["metadata"] = np.asarray(
        json.dumps(
            {
                "schema": "tang_apples_ada_knowntext_testfirst_v1",
                "embedding_model": "text-embedding-ada-002",
                "embedding_dim": 1536,
                "intent": "known-text ELF overfit; MEG predictions excluded",
                "rows": {"test": 26, "val": 110, "train": 2652},
            },
            sort_keys=True,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    embeddings = payload["input_embeddings"]
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"saved={args.output} rows={embeddings.shape[0]} dim={embeddings.shape[1]} "
        f"norm_mean={norms.mean():.8f} norm_min={norms.min():.8f} "
        f"first_split={payload['source_split'][0]}"
    )


if __name__ == "__main__":
    main()
