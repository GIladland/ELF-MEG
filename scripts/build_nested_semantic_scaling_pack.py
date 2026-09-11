#!/usr/bin/env python3
"""Build an auditable, anchor-replayed corpus for ELF scaling experiments.

The output order is intentional: exact audit rows come first so the existing
ELF trainer can evaluate them while still training on the entire known-text
corpus.  The audit rows contain text/semantic embeddings only; brain vectors
must never be passed to this builder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-npz", required=True, type=Path)
    parser.add_argument("--augmentation-npz", required=True, type=Path)
    parser.add_argument("--audit-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--augmentation-limit", required=True, type=int)
    parser.add_argument("--anchor-repeats", required=True, type=int)
    parser.add_argument("--audit-tail", type=int, default=0)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--label", default="")
    return parser.parse_args()


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            for value in np.asarray(values).tolist()
        ],
        dtype=object,
    )


def canonical(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", value.casefold()))


def load(path: Path, input_key: str, sentence_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        vectors = np.asarray(source[input_key], dtype=np.float32)
        sentences = strings(source[sentence_key])
    if vectors.ndim != 2 or len(vectors) != len(sentences):
        raise ValueError(f"Invalid semantic NPZ {path}: {vectors.shape=} {len(sentences)=}")
    return vectors, sentences


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def main() -> None:
    args = parse_args()
    if args.anchor_repeats < 1:
        raise ValueError("--anchor-repeats must be positive")
    if args.augmentation_limit < 1:
        raise ValueError("--augmentation-limit must be positive")

    exact_vectors, exact_sentences = load(args.exact_npz, args.input_key, args.sentence_key)
    aug_vectors, aug_sentences = load(args.augmentation_npz, args.input_key, args.sentence_key)
    audit_vectors, audit_sentences = load(args.audit_npz, args.input_key, args.sentence_key)
    if args.audit_tail:
        audit_vectors = audit_vectors[-args.audit_tail :]
        audit_sentences = audit_sentences[-args.audit_tail :]
    if not (exact_vectors.shape[1] == aug_vectors.shape[1] == audit_vectors.shape[1]):
        raise ValueError(
            "Embedding dimensions differ: "
            f"exact={exact_vectors.shape[1]} augmentation={aug_vectors.shape[1]} "
            f"audit={audit_vectors.shape[1]}"
        )

    limit = min(args.augmentation_limit, len(aug_sentences))
    aug_vectors = aug_vectors[:limit]
    aug_sentences = aug_sentences[:limit]
    exact_keys = {canonical(value) for value in exact_sentences.tolist()}
    overlap = sum(canonical(value) in exact_keys for value in aug_sentences.tolist())
    if overlap:
        raise ValueError(f"Augmentation contains {overlap} exact-sentence duplicates")

    vector_parts = [audit_vectors]
    sentence_parts = [audit_sentences]
    source_parts = [np.full(len(audit_sentences), "exact_audit_replay", dtype=object)]
    for replay in range(args.anchor_repeats):
        vector_parts.append(exact_vectors)
        sentence_parts.append(exact_sentences)
        source_parts.append(np.full(len(exact_sentences), f"exact_anchor_replay_{replay}", dtype=object))
    vector_parts.append(aug_vectors)
    sentence_parts.append(aug_sentences)
    source_parts.append(np.full(len(aug_sentences), "synthetic_same_corpus", dtype=object))

    vectors = normalize(np.concatenate(vector_parts, axis=0))
    sentences = np.concatenate(sentence_parts, axis=0)
    condition_source = np.concatenate(source_parts, axis=0)
    rows = np.asarray(
        [
            {"condition_source": str(source), "row_index": int(index)}
            for index, source in enumerate(condition_source.tolist())
        ],
        dtype=object,
    )
    schema = {
        "schema": "nested_semantic_scaling_pack_v1",
        "label": args.label,
        "exact_npz": str(args.exact_npz),
        "augmentation_npz": str(args.augmentation_npz),
        "audit_npz": str(args.audit_npz),
        "audit_rows_first": int(len(audit_sentences)),
        "unique_exact_rows": int(len(exact_sentences)),
        "anchor_repeats": int(args.anchor_repeats),
        "unique_augmentation_rows": int(limit),
        "unique_semantic_text_rows": int(len(exact_sentences) + limit),
        "physical_training_rows": int(len(sentences)),
        "embedding_dim": int(vectors.shape[1]),
        "brain_vectors_included": False,
        "known_text_audit_is_trainable": True,
        "augmentation_exact_overlap": int(overlap),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{args.output.stem}.", suffix=".npz", dir=args.output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(
            temporary,
            input_embeddings=vectors,
            sentence=sentences,
            rows=rows,
            condition_source=condition_source,
            split=np.full(len(sentences), "train", dtype=object),
            schema_json=np.asarray(json.dumps(schema, sort_keys=True)),
        )
        with np.load(temporary, allow_pickle=True) as audit:
            if audit["input_embeddings"].shape != vectors.shape:
                raise RuntimeError("Atomic output audit failed")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(schema, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
