#!/usr/bin/env python3
"""Create a leakage-safe noisy semantic training corpus from exact embeddings.

Each source row is retained unchanged and is followed by normalized noisy copies.
Noise is sampled in the tangent space of the unit-normalized source vector so it
changes direction without introducing a systematic radial/scale shortcut.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument(
        "--noise-norms",
        default="0.10,0.20,0.35,0.50,0.75,1.00",
        help="Comma-separated tangent-noise L2 norms before renormalization.",
    )
    parser.add_argument("--copies-per-norm", type=int, default=4)
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def main() -> None:
    args = parse_args()
    if args.copies_per_norm < 1:
        raise ValueError("--copies-per-norm must be positive")
    noise_norms = [float(value) for value in args.noise_norms.split(",") if value]
    if not noise_norms or any(value <= 0.0 for value in noise_norms):
        raise ValueError("--noise-norms must contain positive values")

    source = np.load(args.input, allow_pickle=True)
    exact = normalize(np.asarray(source[args.input_key], dtype=np.float32))
    sentences = np.asarray(source[args.sentence_key])
    if exact.ndim != 2 or sentences.shape[0] != exact.shape[0]:
        raise ValueError("Expected aligned [rows, dim] embeddings and sentences")

    rng = np.random.default_rng(args.seed)
    blocks = [exact]
    source_rows = [np.arange(exact.shape[0], dtype=np.int64)]
    applied_norms = [np.zeros(exact.shape[0], dtype=np.float32)]
    for noise_norm in noise_norms:
        for _ in range(args.copies_per_norm):
            noise = rng.standard_normal(exact.shape, dtype=np.float32)
            noise -= np.sum(noise * exact, axis=1, keepdims=True) * exact
            noise = normalize(noise) * np.float32(noise_norm)
            blocks.append(normalize(exact + noise).astype(np.float32))
            source_rows.append(np.arange(exact.shape[0], dtype=np.int64))
            applied_norms.append(np.full(exact.shape[0], noise_norm, dtype=np.float32))

    embeddings = np.concatenate(blocks, axis=0)
    row_indices = np.concatenate(source_rows, axis=0)
    noise_values = np.concatenate(applied_norms, axis=0)
    payload: dict[str, np.ndarray] = {
        args.input_key: embeddings,
        args.sentence_key: sentences[row_indices],
        "source_row": row_indices,
        "semantic_noise_norm": noise_values,
        "augmentation_seed": np.asarray(args.seed),
    }
    for key in source.files:
        if key in payload or key in {args.input_key, args.sentence_key}:
            continue
        value = np.asarray(source[key])
        if value.ndim >= 1 and value.shape[0] == exact.shape[0]:
            payload[key] = value[row_indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    cosines = np.sum(embeddings * exact[row_indices], axis=1)
    print(
        f"saved={args.output} rows={embeddings.shape[0]} dim={embeddings.shape[1]} "
        f"cosine_min={cosines.min():.4f} cosine_mean={cosines.mean():.4f} "
        f"cosine_max={cosines.max():.4f}"
    )


if __name__ == "__main__":
    main()
