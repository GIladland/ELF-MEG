#!/usr/bin/env python3
"""Compute leakage-safe semantic mean components from one packed train split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--embedding-key", default="embeddings_minilm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_npz).expanduser().resolve()
    output_path = Path(args.output_npz).expanduser().resolve()

    with np.load(input_path, allow_pickle=True) as data:
        if args.embedding_key not in data.files:
            raise KeyError(
                f"{input_path} does not contain {args.embedding_key!r}; keys={data.files}"
            )
        embeddings = np.asarray(data[args.embedding_key], dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(f"Expected [N, D] embeddings with N >= 2, got {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embeddings contain zero-norm rows")
    normalized = embeddings / norms
    train_mean = normalized.mean(axis=0, dtype=np.float64).astype(np.float32)
    residual_norms = np.linalg.norm(normalized - train_mean[None, :], axis=1)
    stats = {
        "schema": "semantic_train_mean_v1",
        "source_split": "train_only",
        "source_npz": str(input_path),
        "embedding_key": args.embedding_key,
        "rows": int(normalized.shape[0]),
        "embedding_dim": int(normalized.shape[1]),
        "input_norm_mean": float(norms.mean()),
        "train_mean_norm": float(np.linalg.norm(train_mean)),
        "residual_norm_mean": float(residual_norms.mean()),
        "residual_norm_median": float(np.median(residual_norms)),
        "residual_norm_p05": float(np.percentile(residual_norms, 5)),
        "residual_norm_p95": float(np.percentile(residual_norms, 95)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_mean=train_mean,
        stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
    )
    print(json.dumps({"output_npz": str(output_path), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
