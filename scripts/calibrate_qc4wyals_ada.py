#!/usr/bin/env python3
"""Validation-select lightweight qc4wyals ADA calibrations without test fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    predicted = normalize(predicted)
    target = normalize(target)
    similarity = predicted @ target.T
    paired = np.diag(similarity)
    ranks = 1 + np.sum(similarity > paired[:, None], axis=1)
    return {
        "n": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= min(5, len(ranks)))),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "ndcg": float(np.mean(1.0 / np.log2(ranks + 1.0))),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / max(1, len(ranks) - 1))),
        "matched_cosine": float(np.mean(paired)),
        "pred_dim_var_mean": float(np.mean(np.var(predicted, axis=0))),
    }


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = values - np.max(values, axis=axis, keepdims=True)
    result = np.exp(values)
    return result / np.maximum(np.sum(result, axis=axis, keepdims=True), 1e-12)


def knn_residual(
    query: np.ndarray,
    train_prediction: np.ndarray,
    train_target: np.ndarray,
    *,
    k: int,
    alpha: float,
    temperature: float | None,
) -> np.ndarray:
    query_n = normalize(query)
    train_n = normalize(train_prediction)
    similarity = query_n @ train_n.T
    indices = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    selected_similarity = np.take_along_axis(similarity, indices, axis=1)
    if temperature is None:
        weights = np.full(selected_similarity.shape, 1.0 / k, dtype=np.float32)
    else:
        weights = softmax(selected_similarity / temperature).astype(np.float32)
    residual = normalize(train_target) - train_n
    correction = np.sum(residual[indices] * weights[:, :, None], axis=1)
    return normalize(query_n + np.float32(alpha) * correction)


def diagonal_transform(
    values: np.ndarray,
    train_prediction: np.ndarray,
    train_target: np.ndarray,
    *,
    shrinkage: float,
    alpha: float,
) -> np.ndarray:
    x = normalize(train_prediction)
    y = normalize(train_target)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    variance = np.mean(x_centered * x_centered, axis=0, keepdims=True)
    covariance = np.mean(x_centered * y_centered, axis=0, keepdims=True)
    scale = covariance / np.maximum(variance + shrinkage * variance.mean(), 1e-8)
    scale = np.clip(scale, 0.25, 4.0)
    calibrated = (normalize(values) - x_mean) * scale + y_mean
    return normalize((1.0 - alpha) * normalize(values) + alpha * calibrated)


def mean_shift(
    values: np.ndarray,
    train_prediction: np.ndarray,
    train_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x = normalize(train_prediction)
    y = normalize(train_target)
    shifted = normalize(values) - x.mean(axis=0, keepdims=True) + y.mean(axis=0, keepdims=True)
    return normalize((1.0 - alpha) * normalize(values) + alpha * shifted)


def apply_candidate(
    spec: dict[str, object],
    values: np.ndarray,
    train_prediction: np.ndarray,
    train_target: np.ndarray,
) -> np.ndarray:
    kind = str(spec["kind"])
    if kind == "identity":
        return normalize(values)
    if kind == "mean_shift":
        return mean_shift(values, train_prediction, train_target, float(spec["alpha"]))
    if kind == "diagonal":
        return diagonal_transform(
            values,
            train_prediction,
            train_target,
            shrinkage=float(spec["shrinkage"]),
            alpha=float(spec["alpha"]),
        )
    if kind == "knn_residual":
        temperature = spec.get("temperature")
        return knn_residual(
            values,
            train_prediction,
            train_target,
            k=int(spec["k"]),
            alpha=float(spec["alpha"]),
            temperature=None if temperature is None else float(temperature),
        )
    raise ValueError(f"Unknown candidate kind: {kind}")


def save_like(path: Path, template: dict[str, np.ndarray], embeddings: np.ndarray, schema: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in template.items()
        if key not in {"input_embeddings", "schema_json"}
    }
    payload["input_embeddings"] = normalize(embeddings)
    payload["schema_json"] = np.asarray(json.dumps(schema, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    args = parse_args()
    train = load(args.interface_dir / "qc4wyals_train_predicted_ada002.npz")
    val = load(args.interface_dir / "qc4wyals_val_predicted_ada002.npz")
    test = load(args.interface_dir / "qc4wyals_test_predicted_ada002.npz")
    train_prediction = normalize(train["input_embeddings"])
    train_target = normalize(train["source_vectors"])
    val_prediction = normalize(val["input_embeddings"])
    val_target = normalize(val["source_vectors"])
    test_prediction = normalize(test["input_embeddings"])
    test_target = normalize(test["source_vectors"])

    specs: list[dict[str, object]] = [{"name": "identity", "kind": "identity"}]
    for alpha in (0.25, 0.5, 0.75, 1.0):
        specs.append({"name": f"mean_shift_a{alpha:g}", "kind": "mean_shift", "alpha": alpha})
    for shrinkage in (0.01, 0.1, 1.0):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            specs.append(
                {
                    "name": f"diagonal_s{shrinkage:g}_a{alpha:g}",
                    "kind": "diagonal",
                    "shrinkage": shrinkage,
                    "alpha": alpha,
                }
            )
    for k in (1, 4, 16, 64):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            for temperature in (None, 0.02):
                suffix = "uniform" if temperature is None else "t0p02"
                specs.append(
                    {
                        "name": f"knn{k}_{suffix}_a{alpha:g}",
                        "kind": "knn_residual",
                        "k": k,
                        "alpha": alpha,
                        "temperature": temperature,
                    }
                )

    candidates: list[dict[str, object]] = []
    for spec in specs:
        calibrated = apply_candidate(spec, val_prediction, train_prediction, train_target)
        candidates.append({**spec, "validation": metrics(calibrated, val_target)})
    selectors = {
        "ndcg": lambda row: (
            row["validation"]["ndcg"],
            row["validation"]["matched_cosine"],
        ),
        "cosine": lambda row: (
            row["validation"]["matched_cosine"],
            row["validation"]["ndcg"],
        ),
        "top1": lambda row: (
            row["validation"]["top1"],
            row["validation"]["ndcg"],
            row["validation"]["matched_cosine"],
        ),
        # Preserve two interpretable controls instead of selecting them on the
        # protected test split.  Both choices are fixed before text decoding.
        "mean_shift": lambda row: (row["name"] == "mean_shift_a1",),
        "diagonal_balanced": lambda row: (row["name"] == "diagonal_s0.1_a0.5",),
    }
    selected: dict[str, object] = {}
    for selector_name, key in selectors.items():
        winner = max(candidates, key=key)
        calibrated_train = apply_candidate(winner, train_prediction, train_prediction, train_target)
        calibrated_val = apply_candidate(winner, val_prediction, train_prediction, train_target)
        calibrated_test = apply_candidate(winner, test_prediction, train_prediction, train_target)
        train_val_template = load(
            args.interface_dir / "qc4wyals_predicted_train_valtail_ada002.npz"
        )
        train_val_calibrated = np.concatenate([calibrated_train, calibrated_val], axis=0)
        record = {
            "selector": selector_name,
            "candidate": winner,
            "test": metrics(calibrated_test, test_target),
            "test_used_for_selection": False,
        }
        selected[selector_name] = record
        save_like(
            args.output_dir / f"qc4wyals_calibrated_{selector_name}_train_valtail_ada002.npz",
            train_val_template,
            train_val_calibrated,
            record,
        )
        save_like(
            args.output_dir / f"qc4wyals_calibrated_{selector_name}_val_ada002.npz",
            val,
            calibrated_val,
            record,
        )
        save_like(
            args.output_dir / f"qc4wyals_calibrated_{selector_name}_test_ada002.npz",
            test,
            calibrated_test,
            record,
        )

    report = {
        "contract": {
            "fit": "qc4wyals train2652 predictions and exact ADA",
            "selection": "qc4wyals validation110 only",
            "final_test26_used_for_selection": False,
        },
        "baseline_validation": metrics(val_prediction, val_target),
        "baseline_test": metrics(test_prediction, test_target),
        "selected": selected,
        "candidates": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
