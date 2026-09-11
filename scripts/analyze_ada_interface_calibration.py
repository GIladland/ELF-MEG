#!/usr/bin/env python
"""Audit simple leakage-safe calibrations between predicted and exact ADA vectors.

The input NPZ files must contain identically ordered rows.  The last
``--val-size`` rows are held out from every fitted calibration.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted-npz", required=True)
    parser.add_argument("--exact-npz", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--val-size", type=int, default=110)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def row_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def score(predicted: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    predicted_norm = row_normalize(predicted)
    exact_norm = row_normalize(exact)
    paired = np.sum(predicted_norm * exact_norm, axis=1)
    similarities = predicted_norm @ exact_norm.T
    ranks = 1 + np.sum(similarities > np.diag(similarities)[:, None], axis=1)
    return {
        "cosine_mean": float(paired.mean()),
        "cosine_median": float(np.median(paired)),
        "retrieval_top1": float(np.mean(ranks <= 1)),
        "retrieval_top5": float(np.mean(ranks <= 5)),
        "retrieval_mean_rank": float(ranks.mean()),
        "retrieval_median_rank": float(np.median(ranks)),
    }


def fit_diagonal_affine(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> np.ndarray:
    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean(axis=0)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    slopes = np.sum(centered_x * centered_y, axis=0) / np.maximum(
        np.sum(centered_x * centered_x, axis=0), 1e-12
    )
    return (val_x - x_mean) * slopes + y_mean


def block_ridge_candidates(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    *,
    num_blocks: int,
    ridge_fractions: tuple[float, ...],
) -> dict[str, np.ndarray]:
    dim = train_x.shape[1]
    if dim % num_blocks:
        raise ValueError(f"Dimension {dim} is not divisible by {num_blocks} blocks")
    block_dim = dim // num_blocks
    train_x_blocks = train_x.reshape(-1, num_blocks, block_dim)
    train_y_blocks = train_y.reshape(-1, num_blocks, block_dim)
    val_x_blocks = val_x.reshape(-1, num_blocks, block_dim)
    candidates = {fraction: [] for fraction in ridge_fractions}
    for block in range(num_blocks):
        x = train_x_blocks[:, block]
        y = train_y_blocks[:, block]
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        xc = x - x_mean
        yc = y - y_mean
        gram = xc.T @ xc
        cross = xc.T @ yc
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected_cross = eigenvectors.T @ cross
        scale = max(float(np.trace(gram) / block_dim), 1e-12)
        for fraction in ridge_fractions:
            weights = eigenvectors @ (projected_cross / (eigenvalues[:, None] + fraction * scale))
            candidates[fraction].append((val_x_blocks[:, block] - x_mean) @ weights + y_mean)
    return {
        f"block_ridge_fraction_{fraction:g}": np.stack(blocks, axis=1).reshape(len(val_x), dim)
        for fraction, blocks in candidates.items()
    }


def main() -> None:
    args = parse_args()
    predicted_npz = np.load(args.predicted_npz, allow_pickle=True)
    exact_npz = np.load(args.exact_npz, allow_pickle=True)
    predicted = np.asarray(predicted_npz[args.input_key], dtype=np.float64)
    exact = np.asarray(exact_npz[args.input_key], dtype=np.float64)
    if predicted.shape != exact.shape:
        raise ValueError(f"Shape mismatch: predicted={predicted.shape}, exact={exact.shape}")
    if not 0 < args.val_size < len(predicted):
        raise ValueError(f"Invalid --val-size={args.val_size} for {len(predicted)} rows")
    train_x, val_x = predicted[: -args.val_size], predicted[-args.val_size :]
    train_y, val_y = exact[: -args.val_size], exact[-args.val_size :]

    candidates: dict[str, np.ndarray] = {
        "raw": val_x,
        "row_l2_normalized": row_normalize(val_x),
        "mean_shift": val_x - train_x.mean(axis=0) + train_y.mean(axis=0),
        "diagonal_affine": fit_diagonal_affine(train_x, train_y, val_x),
    }
    candidates.update(
        block_ridge_candidates(
            train_x,
            train_y,
            val_x,
            num_blocks=args.num_blocks,
            ridge_fractions=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        )
    )

    block_dim = predicted.shape[1] // args.num_blocks
    predicted_blocks = val_x.reshape(-1, args.num_blocks, block_dim)
    exact_blocks = val_y.reshape(-1, args.num_blocks, block_dim)
    permutation_scores = []
    for permutation in itertools.permutations(range(args.num_blocks)):
        candidate = predicted_blocks[:, permutation].reshape(val_x.shape)
        permutation_scores.append({"permutation": permutation, **score(candidate, val_y)})

    results = {
        "predicted_npz": args.predicted_npz,
        "exact_npz": args.exact_npz,
        "train_size": len(train_x),
        "val_size": len(val_x),
        "shape": list(predicted.shape),
        "candidates": {name: score(values, val_y) for name, values in candidates.items()},
        "block_permutations": sorted(
            permutation_scores, key=lambda item: item["cosine_mean"], reverse=True
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
