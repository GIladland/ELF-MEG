#!/usr/bin/env python3
"""Build aligned predicted-to-exact semantic interpolation packs for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--exact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, bytearray)) else str(value) for value in values],
        dtype=object,
    )


def alpha_slug(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    alphas = [float(value) for value in args.alphas.split(",") if value]
    if not alphas or any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("--alphas must be comma-separated values in [0, 1]")

    with np.load(args.predicted, allow_pickle=True) as pred_source, np.load(
        args.exact, allow_pickle=True
    ) as exact_source:
        predicted = normalize(pred_source["input_embeddings"])
        exact = normalize(exact_source["input_embeddings"])
        pred_sentence = strings(pred_source["sentence"])
        exact_sentence = strings(exact_source["sentence"])
        if predicted.shape != exact.shape:
            raise ValueError(f"Embedding shape mismatch: {predicted.shape} != {exact.shape}")
        if not np.array_equal(pred_sentence, exact_sentence):
            raise ValueError("Predicted and exact sentence rows are not aligned")
        metadata = {
            key: np.asarray(pred_source[key])
            for key in pred_source.files
            if key not in {"input_embeddings", "source_vectors", "schema_json"}
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for alpha in alphas:
        interpolated = normalize((1.0 - alpha) * predicted + alpha * exact)
        matched_cosine = np.sum(interpolated * exact, axis=1)
        output = args.output_dir / f"qc4wyals_val_interp_alpha{alpha_slug(alpha)}_ada002.npz"
        schema = {
            "schema": "qc4wyals_predicted_exact_interpolation_v1",
            "alpha": alpha,
            "formula": "normalize((1-alpha)*predicted + alpha*exact)",
            "evaluation_only": True,
        }
        np.savez_compressed(
            output,
            input_embeddings=interpolated.astype(np.float32),
            source_vectors=exact.astype(np.float32),
            schema_json=np.asarray(json.dumps(schema, sort_keys=True)),
            **metadata,
        )
        summary.append(
            {
                "alpha": alpha,
                "path": str(output),
                "matched_cosine_mean": float(matched_cosine.mean()),
                "matched_cosine_median": float(np.median(matched_cosine)),
            }
        )
        print(f"alpha={alpha:g} rows={len(interpolated)} cosine={matched_cosine.mean():.6f} {output}")

    (args.output_dir / "interpolation_ladder.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
