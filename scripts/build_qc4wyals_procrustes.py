#!/usr/bin/env python3
"""Fit variance-preserving train-only qc4wyals ADA alignments.

The mapper is an orthogonal Procrustes rotation learned from the 2,652 paired
training predictions/exact ADA vectors.  Alpha interpolation is evaluated on
val110, while protected test MEG vectors are never loaded by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def semantic_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = normalize(predicted)
    target = normalize(target)
    similarity = predicted @ target.T
    paired = np.diag(similarity)
    ranks = 1 + np.sum(similarity > paired[:, None], axis=1)
    return {
        "matched_cosine": float(paired.mean()),
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "ndcg": float(np.mean(1.0 / np.log2(ranks + 1.0))),
        "prediction_dim_variance": float(np.var(predicted, axis=0).mean()),
    }


def write_like(
    path: Path,
    template: dict[str, np.ndarray],
    embeddings: np.ndarray,
    source_vectors: np.ndarray,
    schema: dict[str, object],
) -> None:
    excluded = {"input_embeddings", "source_vectors", "schema_json"}
    payload = {key: value for key, value in template.items() if key not in excluded}
    payload["input_embeddings"] = normalize(embeddings)
    payload["source_vectors"] = normalize(source_vectors)
    payload["schema_json"] = np.asarray(json.dumps(schema, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    train = load(args.interface_dir / "qc4wyals_train_predicted_ada002.npz")
    val = load(args.interface_dir / "qc4wyals_val_predicted_ada002.npz")
    template = load(args.interface_dir / "qc4wyals_predicted_train_valtail_ada002.npz")

    train_prediction = normalize(train["input_embeddings"])
    train_target = normalize(train["source_vectors"])
    val_prediction = normalize(val["input_embeddings"])
    val_target = normalize(val["source_vectors"])
    x_mean = train_prediction.mean(axis=0, keepdims=True)
    y_mean = train_target.mean(axis=0, keepdims=True)
    cross_covariance = (train_prediction - x_mean).T @ (train_target - y_mean)
    left, singular_values, right_t = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = (left @ right_t).astype(np.float32)

    def mapped(values: np.ndarray) -> np.ndarray:
        return normalize((normalize(values) - x_mean) @ rotation + y_mean)

    mapped_train = mapped(train_prediction)
    mapped_val = mapped(val_prediction)
    rows: list[dict[str, object]] = []
    for alpha in (0.25, 0.5, 0.75, 1.0):
        transformed_train = normalize((1.0 - alpha) * train_prediction + alpha * mapped_train)
        transformed_val = normalize((1.0 - alpha) * val_prediction + alpha * mapped_val)
        name = f"procrustes_{alpha_tag(alpha)}"
        validation = semantic_metrics(transformed_val, val_target)
        schema = {
            "schema": "qc4wyals_train_only_orthogonal_procrustes_v1",
            "name": name,
            "alpha": alpha,
            "fit_rows": int(len(train_prediction)),
            "selection_rows": int(len(val_prediction)),
            "protected_test_meg_loaded": False,
            "validation": validation,
        }
        write_like(
            args.output_dir / f"qc4wyals_{name}_train_valtail_ada002.npz",
            template,
            np.concatenate([transformed_train, transformed_val]),
            np.concatenate([train_target, val_target]),
            schema,
        )
        write_like(
            args.output_dir / f"qc4wyals_{name}_val_ada002.npz",
            val,
            transformed_val,
            val_target,
            {**schema, "split": "validation"},
        )
        rows.append(schema)

    report = {
        "contract": {
            "fit": "train2652 only",
            "selection": "val110 only",
            "protected_test_meg_loaded": False,
        },
        "raw_validation": semantic_metrics(val_prediction, val_target),
        "singular_value_sum": float(singular_values.sum()),
        "candidates": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "procrustes_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
