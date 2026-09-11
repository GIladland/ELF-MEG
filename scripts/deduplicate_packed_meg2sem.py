#!/usr/bin/env python
"""Remove redundant packed MEG rows with the same recording and start sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--embedding-key", default="embeddings_minilm")
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    with np.load(args.input, allow_pickle=True) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}

    required = {"meg", "session", "start_samples", "sentence", args.embedding_key}
    missing = sorted(required - arrays.keys())
    if missing:
        raise KeyError(f"Missing required arrays: {missing}")
    n = len(arrays["meg"])
    for key in required - {"meg"}:
        if len(arrays[key]) != n:
            raise ValueError(f"Row-count mismatch for {key}: {len(arrays[key])} != {n}")

    sessions = np.asarray([str(value) for value in arrays["session"]], dtype=object)
    starts = np.asarray(arrays["start_samples"], dtype=np.int64)
    sentences = np.asarray([str(value) for value in arrays["sentence"]], dtype=object)
    embeddings = np.asarray(arrays[args.embedding_key], dtype=np.float32)

    first_by_key: dict[tuple[str, int], int] = {}
    keep: list[int] = []
    duplicate_pairs: list[tuple[int, int]] = []
    max_embedding_delta = 0.0
    for idx, (session, start) in enumerate(zip(sessions, starts)):
        row_key = (session, int(start))
        first = first_by_key.get(row_key)
        if first is None:
            first_by_key[row_key] = idx
            keep.append(idx)
            continue
        if sentences[first] != sentences[idx]:
            raise ValueError(
                f"Conflicting text for duplicate brain window {row_key}: "
                f"row {first}={sentences[first]!r}, row {idx}={sentences[idx]!r}"
            )
        delta = float(np.max(np.abs(embeddings[first] - embeddings[idx])))
        max_embedding_delta = max(max_embedding_delta, delta)
        if delta > args.atol:
            raise ValueError(
                f"Conflicting semantic labels for duplicate brain window {row_key}: "
                f"rows {first}/{idx}, max_abs_delta={delta:.6g} > {args.atol:.6g}"
            )
        duplicate_pairs.append((first, idx))

    keep_idx = np.asarray(keep, dtype=np.int64)
    output_arrays: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if value.ndim >= 1 and value.shape[0] == n:
            output_arrays[key] = value[keep_idx]
        else:
            output_arrays[key] = value

    metadata = output_arrays.get("metadata")
    if metadata is not None and metadata.shape == ():
        item = metadata.item()
        if isinstance(item, dict):
            updated = dict(item)
            updated.update(
                {
                    "deduplicated": True,
                    "deduplicate_key": ["session", "start_samples"],
                    "source_rows": n,
                    "deduplicated_rows": int(len(keep_idx)),
                    "removed_duplicate_rows": int(len(duplicate_pairs)),
                }
            )
            output_arrays["metadata"] = np.asarray(updated, dtype=object)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temp_path, **output_arrays)
    os.replace(temp_path, args.output)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "source_rows": n,
        "output_rows": int(len(keep_idx)),
        "removed_rows": int(len(duplicate_pairs)),
        "removed_fraction": float(len(duplicate_pairs) / max(1, n)),
        "deduplicate_key": ["session", "start_samples"],
        "duplicate_label_max_abs_delta": max_embedding_delta,
        "embedding_atol": args.atol,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
