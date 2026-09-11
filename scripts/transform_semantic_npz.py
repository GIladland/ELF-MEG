#!/usr/bin/env python
"""Apply auditable row-preserving transformations to semantic NPZ vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--operation", choices=["mean-blocks"], required=True)
    parser.add_argument("--block-count", type=int, default=4)
    return parser.parse_args()


def parse_schema(value: np.ndarray | None) -> dict:
    if value is None:
        return {}
    raw = str(value.tolist())
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError:
        return {"source_schema_raw": raw}
    return schema if isinstance(schema, dict) else {"source_schema": schema}


def main() -> None:
    args = parse_args()
    source_path = Path(args.input_npz)
    with np.load(source_path, allow_pickle=True) as source:
        if args.input_key not in source.files:
            raise KeyError(f"{source_path} has no {args.input_key!r}")
        arrays = {key: np.asarray(source[key]) for key in source.files if key != "schema_json"}
        schema = parse_schema(source["schema_json"] if "schema_json" in source.files else None)

    vectors = np.asarray(arrays[args.input_key], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2-D semantic matrix, got {vectors.shape}")
    if args.block_count <= 0 or vectors.shape[1] % args.block_count:
        raise ValueError(
            f"Input dimension {vectors.shape[1]} is not divisible by block_count={args.block_count}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Input semantic vectors contain non-finite values")

    block_dim = vectors.shape[1] // args.block_count
    if args.operation == "mean-blocks":
        transformed = vectors.reshape(len(vectors), args.block_count, block_dim).mean(axis=1, dtype=np.float32)
    else:  # pragma: no cover - argparse enforces the choices.
        raise ValueError(args.operation)

    arrays[args.input_key] = np.asarray(transformed, dtype=np.float32)
    transform = {
        "operation": args.operation,
        "block_count": args.block_count,
        "source_dim": int(vectors.shape[1]),
        "output_dim": int(transformed.shape[1]),
        "row_count": int(len(transformed)),
        "row_order_preserved": True,
        "non_semantic_arrays_preserved": True,
        "source_npz": str(source_path),
    }
    schema["semantic_transform"] = transform
    schema["shape"] = list(transformed.shape)
    arrays["schema_json"] = np.asarray(json.dumps(schema, indent=2), dtype=object)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)

    with np.load(output_path, allow_pickle=True) as saved:
        saved_vectors = np.asarray(saved[args.input_key], dtype=np.float32)
        if not np.array_equal(saved_vectors, transformed):
            raise AssertionError("Saved semantic vectors differ from the requested transformation")
        for key, value in arrays.items():
            if key in {args.input_key, "schema_json"}:
                continue
            if not np.array_equal(saved[key], value):
                raise AssertionError(f"Saved row-aligned field changed: {key}")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(transform, indent=2), encoding="utf-8")
    print(json.dumps(transform, indent=2))


if __name__ == "__main__":
    main()
