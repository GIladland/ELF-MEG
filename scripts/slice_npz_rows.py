#!/usr/bin/env python
"""Write a row slice of an NPZ while preserving scalar metadata fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=True) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    row_counts = [len(value) for value in arrays.values() if value.ndim > 0]
    if not row_counts:
        raise ValueError(f"No row-shaped arrays in {args.input}")
    total = max(set(row_counts), key=row_counts.count)
    start, stop, _ = slice(args.start, args.stop).indices(total)
    payload = {
        key: value[start:stop] if value.ndim > 0 and len(value) == total else value
        for key, value in arrays.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Wrote {stop - start}/{total} rows to {args.output}")


if __name__ == "__main__":
    main()
