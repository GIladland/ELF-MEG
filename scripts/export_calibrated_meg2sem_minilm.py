#!/usr/bin/env python3
"""Export MEG2SEM predictions after low-capacity MiniLM-space calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from modules.meg2sem_bridge import load_meg2sem_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--input-key", default="meg")
    parser.add_argument("--target-key", default="input_embeddings")
    parser.add_argument("--lengths-key", default="meg_lengths")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--test-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


def retrieval_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = normalize(prediction)
    target = normalize(target)
    similarity = prediction @ target.T
    diagonal = similarity[np.arange(len(similarity)), np.arange(len(similarity))]
    ranks = 1 + np.sum(similarity > diagonal[:, None], axis=1)
    n = len(ranks)
    return {
        "num_examples": int(n),
        "paired_cosine_mean": float(diagonal.mean()),
        "paired_cosine_std": float(diagonal.std()),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= min(5, n))),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / max(1, n - 1))),
    }


def fit_mean_shift(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    shift = y.mean(axis=0, keepdims=True) - x.mean(axis=0, keepdims=True)
    return lambda values: values + shift


def fit_procrustes(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd((x - x_mean).T @ (y - y_mean), full_matrices=False)
    rotation = u @ vt
    return lambda values: (values - x_mean) @ rotation + y_mean


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> Callable[[np.ndarray], np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    cross = x_centered.T @ y_centered
    identity = np.eye(gram.shape[0], dtype=np.float32)
    weights = np.linalg.solve(gram + float(l2) * identity, cross)
    return lambda values: (values - x_mean) @ weights + y_mean


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"kind": "identity", "alpha": 0.0}]
    for alpha in (0.25, 0.5, 0.75, 1.0):
        specs.append({"kind": "mean_shift", "alpha": alpha})
        specs.append({"kind": "procrustes", "alpha": alpha})
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            specs.append({"kind": "ridge", "l2": l2, "alpha": alpha})
    return specs


def fit_spec(spec: dict[str, Any], x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    kind = str(spec["kind"])
    if kind == "identity":
        base = lambda values: values
    elif kind == "mean_shift":
        base = fit_mean_shift(x, y)
    elif kind == "procrustes":
        base = fit_procrustes(x, y)
    elif kind == "ridge":
        base = fit_ridge(x, y, float(spec["l2"]))
    else:  # pragma: no cover - candidate_specs controls this
        raise ValueError(kind)
    alpha = float(spec["alpha"])
    return lambda values: normalize((1.0 - alpha) * values + alpha * base(values))


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    meg: torch.Tensor,
    lengths: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(meg), batch_size):
        end = min(start + batch_size, len(meg))
        batch = meg[start:end].to(device=device, dtype=torch.float32)
        batch_lengths = lengths[start:end].to(device=device, dtype=torch.long)
        subjects = torch.zeros(end - start, dtype=torch.long, device=device)
        prediction = model(batch, meg_lengths=batch_lengths, subjects=subjects)
        output.append(prediction.detach().float().cpu().numpy())
        if end % 1024 < batch_size or end == len(meg):
            print(f"predicted {end}/{len(meg)}", flush=True)
    return np.vstack(output).astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    passthrough: dict[str, np.ndarray] = {}
    with np.load(args.npz, allow_pickle=True) as data:
        target = normalize(np.asarray(data[args.target_key], dtype=np.float32))
        meg = torch.as_tensor(np.asarray(data[args.input_key]), dtype=torch.float16)
        if args.lengths_key in data.files:
            lengths = torch.as_tensor(np.asarray(data[args.lengths_key]), dtype=torch.long)
        else:
            lengths = torch.full((len(meg),), int(meg.shape[-1]), dtype=torch.long)
        for key in (
            args.sentence_key,
            "text",
            "subject",
            "session",
            "task",
            "run",
            "source_index",
            "event_path",
            "start_times",
            "end_times",
        ):
            if key in data.files:
                passthrough[key] = np.asarray(data[key])

    n = len(target)
    val_n = int(args.validation_examples)
    test_n = int(args.test_examples)
    train_n = n - val_n - test_n
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError(f"Invalid calibration split: total={n} train={train_n} val={val_n} test={test_n}")
    print(
        f"Loaded total={n} train={train_n} val={val_n} test={test_n} "
        f"meg={tuple(meg.shape)} target={tuple(target.shape)}",
        flush=True,
    )

    model, load_info = load_meg2sem_model(
        args.checkpoint,
        output_normalization="never",
        device=device,
    )
    raw_prediction = predict(
        model,
        meg,
        lengths,
        batch_size=max(1, args.batch_size),
        device=device,
    )
    raw_prediction = normalize(raw_prediction)
    train_slice = slice(0, train_n)
    val_slice = slice(train_n, train_n + val_n)
    test_slice = slice(train_n + val_n, n)

    validation_results: list[dict[str, Any]] = []
    for spec in candidate_specs():
        transform = fit_spec(spec, raw_prediction[train_slice], target[train_slice])
        metrics = retrieval_metrics(transform(raw_prediction[val_slice]), target[val_slice])
        validation_results.append({**spec, **metrics})
    validation_results.sort(
        key=lambda row: (row["mean_percentile"], row["top1"], row["top5"], row["paired_cosine_mean"]),
        reverse=True,
    )
    best_spec = {
        key: validation_results[0][key]
        for key in ("kind", "l2", "alpha")
        if key in validation_results[0]
    }
    best_transform = fit_spec(best_spec, raw_prediction[train_slice], target[train_slice])
    calibrated = best_transform(raw_prediction).astype(np.float32)
    metrics = {
        "checkpoint": args.checkpoint,
        "checkpoint_load_info": {
            "format": load_info.checkpoint_format,
            "embedding_dim": load_info.embedding_dim,
            "input_dim": load_info.input_dim,
            "missing_keys": load_info.missing_keys,
            "unexpected_keys": load_info.unexpected_keys,
        },
        "split": {"total": n, "train": train_n, "validation": val_n, "test": test_n},
        "best_spec": best_spec,
        "raw_validation": retrieval_metrics(raw_prediction[val_slice], target[val_slice]),
        "calibrated_validation": retrieval_metrics(calibrated[val_slice], target[val_slice]),
        "raw_test": retrieval_metrics(raw_prediction[test_slice], target[test_slice]),
        "calibrated_test": retrieval_metrics(calibrated[test_slice], target[test_slice]),
        "validation_candidates": validation_results,
    }

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    split = np.asarray(["mapper_train"] * train_n + ["mapper_val"] * val_n + ["mapper_test"] * test_n)
    schema = {
        "role": "meg2sem_minilm_prediction_with_train_only_low_capacity_calibration",
        "source_npz": args.npz,
        "meg2sem_checkpoint": args.checkpoint,
        "best_spec_selected_on_validation_only": best_spec,
        "test_rows_used_for_selection": False,
        "l2_normalized": True,
        "split": metrics["split"],
    }
    np.savez_compressed(
        output_npz,
        input_embeddings=calibrated,
        meg2sem_raw=raw_prediction,
        split=split,
        schema_json=np.asarray(json.dumps(schema, indent=2)),
        **passthrough,
    )
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "validation_candidates"}, indent=2))
    print(f"Wrote {output_npz} and {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
