#!/usr/bin/env python3
"""Calibrate a Tang-to-MiniLM mapper on held-out MRI2SEM validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from train_semantic_mapper import SemanticMapper, predict, retrieval_metrics, strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper-checkpoint", type=Path, required=True)
    parser.add_argument("--predicted-val-npz", type=Path, required=True)
    parser.add_argument("--predicted-test-npz", type=Path, required=True)
    parser.add_argument("--target-train-val-npz", type=Path, required=True)
    parser.add_argument("--target-test-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def assert_rows_aligned(
    predicted: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    target_indices: np.ndarray,
    label: str,
) -> None:
    if len(predicted["pred"]) != len(target_indices):
        raise ValueError(f"{label} row mismatch: {len(predicted['pred'])} != {len(target_indices)}")
    for key in ("story", "start_tr", "stop_tr"):
        if key not in predicted or key not in target:
            raise KeyError(f"{label} alignment requires {key!r}")
        left = predicted[key]
        right = target[key][target_indices]
        if left.dtype.kind in {"O", "S", "U"}:
            left = np.asarray(strings(left), dtype=object)
            right = np.asarray(strings(right), dtype=object)
        if not np.array_equal(left, right):
            raise ValueError(f"{label} rows do not align at {key!r}")


def load_mapper(path: Path, device: torch.device) -> SemanticMapper:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    model = SemanticMapper(
        int(checkpoint["source_dim"]),
        int(checkpoint["target_dim"]),
        int(checkpoint_args["hidden_dim"]),
        float(checkpoint_args["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> Callable[[np.ndarray], np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered @ x_centered.T
    dual = np.linalg.solve(gram + l2 * np.eye(len(x), dtype=np.float32), y_centered)
    weights = x_centered.T @ dual

    def transform(values: np.ndarray) -> np.ndarray:
        return (values - x_mean) @ weights + y_mean

    return transform


def fit_procrustes(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd((x - x_mean).T @ (y - y_mean), full_matrices=False)
    rotation = u @ vt

    def transform(values: np.ndarray) -> np.ndarray:
        return (values - x_mean) @ rotation + y_mean

    return transform


def fit_mean_shift(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    shift = y.mean(axis=0, keepdims=True) - x.mean(axis=0, keepdims=True)
    return lambda values: values + shift


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"kind": "identity", "alpha": 0.0}]
    specs.extend({"kind": "mean_shift", "alpha": alpha} for alpha in (0.25, 0.5, 0.75, 1.0))
    specs.extend({"kind": "procrustes", "alpha": alpha} for alpha in (0.25, 0.5, 0.75, 1.0))
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        specs.extend(
            {"kind": "ridge", "l2": l2, "alpha": alpha}
            for alpha in (0.25, 0.5, 0.75, 1.0)
        )
    return specs


def fit_spec(spec: dict[str, Any], x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    kind = spec["kind"]
    if kind == "identity":
        calibrated = lambda values: values
    elif kind == "mean_shift":
        calibrated = fit_mean_shift(x, y)
    elif kind == "procrustes":
        calibrated = fit_procrustes(x, y)
    elif kind == "ridge":
        calibrated = fit_ridge(x, y, float(spec["l2"]))
    else:  # pragma: no cover - guarded by candidate_specs
        raise ValueError(kind)
    alpha = float(spec["alpha"])

    def transform(values: np.ndarray) -> np.ndarray:
        return normalize((1.0 - alpha) * values + alpha * calibrated(values)).astype(np.float32)

    return transform


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    sentences: list[str],
    folds: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    fold_indices = [part for part in np.array_split(indices, folds) if len(part)]
    results: list[dict[str, Any]] = []
    for spec in candidate_specs():
        fold_metrics = []
        for held_out in fold_indices:
            train_indices = np.setdiff1d(np.arange(len(x)), held_out, assume_unique=False)
            transform = fit_spec(spec, x[train_indices], y[train_indices])
            metrics = retrieval_metrics(
                transform(x[held_out]),
                y[held_out],
                [sentences[index] for index in held_out],
            )
            fold_metrics.append(metrics)
        summary = {
            **spec,
            "mean_percentile": float(
                np.mean([metric["index_retrieval"]["mean_percentile"] for metric in fold_metrics])
            ),
            "top1": float(np.mean([metric["index_retrieval"]["top1"] for metric in fold_metrics])),
            "top5": float(np.mean([metric["index_retrieval"]["top5"] for metric in fold_metrics])),
            "matched_cosine": float(np.mean([metric["matched_cosine_mean"] for metric in fold_metrics])),
        }
        results.append(summary)
    results.sort(key=lambda row: (row["mean_percentile"], row["top1"], row["matched_cosine"]), reverse=True)
    return results[0], results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    predicted_val = load_npz(args.predicted_val_npz)
    predicted_test = load_npz(args.predicted_test_npz)
    target_train_val = load_npz(args.target_train_val_npz)
    target_test = load_npz(args.target_test_npz)
    val_indices = np.flatnonzero(np.asarray(strings(target_train_val["split"])) == "val")
    test_indices = np.arange(len(target_test["input_embeddings"]))
    assert_rows_aligned(predicted_val, target_train_val, val_indices, "validation")
    assert_rows_aligned(predicted_test, target_test, test_indices, "test")

    mapper = load_mapper(args.mapper_checkpoint, device)
    mapped_val = predict(mapper, predicted_val["pred"].astype(np.float32), 256, device)
    mapped_test = predict(mapper, predicted_test["pred"].astype(np.float32), 256, device)
    target_val = normalize(target_train_val["input_embeddings"][val_indices].astype(np.float32))
    target_test_vectors = normalize(target_test["input_embeddings"].astype(np.float32))
    val_sentences = [strings(target_train_val["sentence"])[index] for index in val_indices]
    test_sentences = strings(target_test["sentence"])

    best_spec, cv_results = cross_validate(mapped_val, target_val, val_sentences, args.folds, args.seed)
    transform = fit_spec(best_spec, mapped_val, target_val)
    calibrated_val = transform(mapped_val)
    calibrated_test = transform(mapped_test)
    metrics = {
        "best_spec": best_spec,
        "validation_uncalibrated": retrieval_metrics(mapped_val, target_val, val_sentences),
        "validation_calibrated_in_sample": retrieval_metrics(calibrated_val, target_val, val_sentences),
        "test_uncalibrated": retrieval_metrics(mapped_test, target_test_vectors, test_sentences),
        "test_calibrated": retrieval_metrics(calibrated_test, target_test_vectors, test_sentences),
        "cross_validation": cv_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = dict(target_test)
    output["input_embeddings"] = calibrated_test.astype(np.float32)
    output["schema_json"] = np.asarray(
        json.dumps(
            {
                "role": "mri2sem_predicted_test_minilm384_validation_calibrated",
                "mapper_checkpoint": str(args.mapper_checkpoint),
                "predicted_validation_npz": str(args.predicted_val_npz),
                "best_spec": best_spec,
                "test_labels_used_for_selection": False,
            },
            indent=2,
        ),
        dtype=object,
    )
    np.savez(args.output_dir / "mapped_mri2sem_test_minilm384_valcal.npz", **output)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    concise = {
        key: value["index_retrieval"] if isinstance(value, dict) and "index_retrieval" in value else value
        for key, value in metrics.items()
        if key != "cross_validation"
    }
    print(json.dumps(concise, indent=2), flush=True)


if __name__ == "__main__":
    main()
