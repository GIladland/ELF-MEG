#!/usr/bin/env python
"""Reorder an NPZ so a chosen contiguous block becomes the final validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-start", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help="Use only the first N examples from the input before reordering. Default uses all.",
    )
    parser.add_argument("--schema-key", default="schema_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    example_n = None
    for key in data.files:
        value = data[key]
        if value.ndim > 0:
            example_n = int(value.shape[0])
            break
    if example_n is None:
        raise ValueError(f"No example arrays found in {args.input_npz}")

    total_n = example_n if args.num_examples <= 0 else min(args.num_examples, example_n)
    if args.val_start < 0 or args.val_count <= 0:
        raise ValueError("--val-start must be non-negative and --val-count must be positive.")
    val_end = args.val_start + args.val_count
    if val_end > total_n:
        raise ValueError(f"Validation block [{args.val_start}, {val_end}) exceeds total_n={total_n}.")

    val_indices = np.arange(args.val_start, val_end, dtype=np.int64)
    train_indices = np.concatenate(
        [
            np.arange(0, args.val_start, dtype=np.int64),
            np.arange(val_end, total_n, dtype=np.int64),
        ]
    )
    order = np.concatenate([train_indices, val_indices])

    arrays = {}
    for key in data.files:
        value = data[key]
        if value.ndim > 0 and int(value.shape[0]) >= total_n:
            arrays[key] = value[:total_n][order]
        else:
            arrays[key] = value
    arrays["fixed_validation_source_index"] = order

    source_schema = None
    if args.schema_key in data.files:
        try:
            source_schema = json.loads(str(data[args.schema_key].tolist()))
        except json.JSONDecodeError:
            source_schema = str(data[args.schema_key].tolist())

    summary = {
        "output": args.output,
        "input_npz": args.input_npz,
        "num_examples": int(total_n),
        "train_examples": int(total_n - args.val_count),
        "val_examples": int(args.val_count),
        "val_source_start": int(args.val_start),
        "val_source_end": int(val_end),
        "source_schema": source_schema,
    }
    arrays[args.schema_key] = np.asarray(json.dumps(summary))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
