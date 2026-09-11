#!/usr/bin/env python3
"""Require exact ordered fMRI-oracle/MEG-pack target identity before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-test-npz", type=Path, required=True)
    parser.add_argument("--meg-test-npz", type=Path, required=True)
    parser.add_argument("--story", default="birthofanation")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in values]


def main() -> None:
    args = parse_args()
    with np.load(args.oracle_test_npz, allow_pickle=True) as oracle:
        mask = np.asarray(oracle["story"], dtype=str) == args.story
        indices = np.flatnonzero(mask)
        oracle_sentence = strings(oracle["sentence"][indices])
        oracle_start = np.asarray(oracle["start_tr"][indices], dtype=np.int64)
        oracle_stop = np.asarray(oracle["stop_tr"][indices], dtype=np.int64)
        oracle_embedding = np.asarray(oracle["input_embeddings"][indices], dtype=np.float32)

    with np.load(args.meg_test_npz, allow_pickle=True) as packed:
        packed_sentence = strings(packed["sentence"])
        packed_embedding = np.asarray(packed["input_embeddings"], dtype=np.float32)
        sessions = strings(packed["session"])
        metadata = json.loads(str(packed["metadata"].item()))

    n = len(oracle_sentence)
    if n != 26:
        raise ValueError(f"Expected 26 {args.story} oracle rows, got {n}")
    if packed_sentence != oracle_sentence:
        mismatch = next(
            index
            for index, pair in enumerate(zip(packed_sentence, oracle_sentence))
            if pair[0] != pair[1]
        )
        raise ValueError(
            f"Ordered target mismatch at row {mismatch}: "
            f"MEG={packed_sentence[mismatch]!r} oracle={oracle_sentence[mismatch]!r}"
        )
    if packed_embedding.shape != oracle_embedding.shape or not np.allclose(
        packed_embedding, oracle_embedding, rtol=0.0, atol=2e-4
    ):
        delta = float(np.max(np.abs(packed_embedding - oracle_embedding)))
        raise ValueError(
            f"Semantic target mismatch: packed={packed_embedding.shape} "
            f"oracle={oracle_embedding.shape} max_abs_delta={delta}"
        )
    if set(sessions) != {"19"}:
        raise ValueError(f"Expected only MEG session 19, got {sorted(set(sessions))}")
    if len(set(packed_sentence)) != n:
        raise ValueError("Locked target bank contains duplicate sentences")

    digest = hashlib.sha256("\n".join(packed_sentence).encode("utf-8")).hexdigest()
    result = {
        "status": "PASS",
        "story": args.story,
        "rows": n,
        "ordered_sentences_equal": True,
        "ordered_embeddings_equal_atol": 2e-4,
        "unique_sentences": n,
        "session": "19",
        "start_tr": oracle_start.tolist(),
        "stop_tr": oracle_stop.tolist(),
        "target_bank_sha256": digest,
        "oracle_test_npz": str(args.oracle_test_npz.resolve()),
        "meg_test_npz": str(args.meg_test_npz.resolve()),
        "packed_metadata": metadata,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
