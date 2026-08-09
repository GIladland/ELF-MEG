#!/usr/bin/env python
"""Create a shuffled-embedding NPZ negative control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val-num-examples",
        type=int,
        default=0,
        help="If set, derange train and validation blocks separately.",
    )
    parser.add_argument(
        "--preserve-val",
        action="store_true",
        help="With --val-num-examples, derange only the train block and leave validation embeddings aligned.",
    )
    return parser.parse_args()


def derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= 1:
        raise ValueError("Need at least two examples per shuffled block.")
    for _ in range(10_000):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        if np.all(shuffled != indices):
            return shuffled
    raise RuntimeError(f"Could not build a derangement for block of size {len(indices)}")


def blockwise_derangement(n: int, val_n: int, rng: np.random.Generator) -> np.ndarray:
    if val_n < 0:
        raise ValueError("--val-num-examples must be non-negative.")
    if val_n == 0:
        return derangement(np.arange(n), rng)
    if val_n >= n:
        raise ValueError(f"Validation block must be smaller than dataset; got n={n}, val_n={val_n}.")
    train_n = n - val_n
    perm = np.empty(n, dtype=np.int64)
    perm[:train_n] = derangement(np.arange(train_n), rng)
    perm[train_n:] = derangement(np.arange(train_n, n), rng)
    return perm


def train_only_derangement(n: int, val_n: int, rng: np.random.Generator) -> np.ndarray:
    if val_n <= 0:
        raise ValueError("--preserve-val requires --val-num-examples > 0.")
    if val_n >= n:
        raise ValueError(f"Validation block must be smaller than dataset; got n={n}, val_n={val_n}.")
    train_n = n - val_n
    perm = np.arange(n, dtype=np.int64)
    perm[:train_n] = derangement(np.arange(train_n), rng)
    return perm


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    if args.embedding_key not in data.files:
        raise KeyError(f"{args.embedding_key!r} not found in {args.input_npz}")
    embeddings = data[args.embedding_key]
    n = int(embeddings.shape[0])
    rng = np.random.default_rng(args.seed)
    if args.preserve_val:
        source_index_for_sentence = train_only_derangement(n, args.val_num_examples, rng)
    else:
        source_index_for_sentence = blockwise_derangement(n, args.val_num_examples, rng)

    arrays = {key: data[key] for key in data.files if key != args.schema_key}
    arrays[args.embedding_key] = embeddings[source_index_for_sentence]
    arrays["shuffled_embedding_source_index"] = source_index_for_sentence

    source_schema = None
    if args.schema_key in data.files:
        try:
            source_schema = json.loads(str(data[args.schema_key].tolist()))
        except json.JSONDecodeError:
            source_schema = str(data[args.schema_key].tolist())

    summary = {
        "output": args.output,
        "input_npz": args.input_npz,
        "embedding_key": args.embedding_key,
        "sentence_key": args.sentence_key,
        "count": n,
        "seed": args.seed,
        "val_num_examples": args.val_num_examples,
        "preserve_val": bool(args.preserve_val),
        "shuffle": (
            "train_only_derangement"
            if args.preserve_val
            else "blockwise_derangement"
            if args.val_num_examples
            else "global_derangement"
        ),
        "fixed_points": int(np.sum(source_index_for_sentence == np.arange(n))),
        "source_schema": source_schema,
    }
    if args.sentence_key in data.files:
        sentences = _strings(data[args.sentence_key])
        summary["sample_pairs"] = [
            {
                "target_index": i,
                "target_sentence": sentences[i],
                "embedding_source_index": int(source_index_for_sentence[i]),
                "embedding_source_sentence": sentences[int(source_index_for_sentence[i])],
            }
            for i in range(min(5, n))
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays[args.schema_key] = np.asarray(json.dumps(summary))
    np.savez_compressed(output, **arrays)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
