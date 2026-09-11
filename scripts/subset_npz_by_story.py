#!/usr/bin/env python
"""Subset row-aligned NPZ arrays (and optionally a parallel NPY) by story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--story", required=True)
    parser.add_argument("--story-key", default="story")
    parser.add_argument("--parallel-npy", type=Path)
    parser.add_argument("--output-parallel-npy", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_npz.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (pass --overwrite): {args.output_npz}")
    if (args.parallel_npy is None) != (args.output_parallel_npy is None):
        raise SystemExit("--parallel-npy and --output-parallel-npy must be supplied together")

    with np.load(args.input_npz, allow_pickle=True) as source:
        if args.story_key not in source.files:
            raise SystemExit(f"Missing story key {args.story_key!r}: {args.input_npz}")
        stories = source[args.story_key].astype(str)
        indices = np.flatnonzero(stories == args.story)
        if indices.size == 0:
            raise SystemExit(f"Story {args.story!r} has no rows in {args.input_npz}")

        row_count = len(stories)
        output: dict[str, np.ndarray] = {}
        subset_keys: list[str] = []
        preserved_keys: list[str] = []
        for key in source.files:
            value = source[key]
            if value.ndim >= 1 and value.shape[0] == row_count:
                output[key] = value[indices]
                subset_keys.append(key)
            else:
                output[key] = value
                preserved_keys.append(key)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output)

    if args.parallel_npy is not None and args.output_parallel_npy is not None:
        parallel = np.load(args.parallel_npy, mmap_mode="r")
        if parallel.ndim < 1 or parallel.shape[0] != row_count:
            raise SystemExit(
                f"Parallel NPY first dimension {parallel.shape[:1]} does not match NPZ rows {row_count}"
            )
        args.output_parallel_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_parallel_npy, np.asarray(parallel[indices]))

    summary = {
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "story": args.story,
        "source_rows": row_count,
        "selected_rows": int(indices.size),
        "indices": indices.tolist(),
        "subset_keys": subset_keys,
        "preserved_keys": preserved_keys,
        "parallel_npy": str(args.parallel_npy) if args.parallel_npy else None,
        "output_parallel_npy": str(args.output_parallel_npy) if args.output_parallel_npy else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
