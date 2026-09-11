#!/usr/bin/env python3
"""Apply a trained semantic mapper to aligned MRI2SEM split predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_semantic_mapper import SemanticMapper, predict, retrieval_metrics, strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper-checkpoint", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--target-template-npz", type=Path, required=True)
    parser.add_argument("--target-split", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-key", default="pred")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def select_template(
    arrays: dict[str, np.ndarray], target_split: str | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = len(arrays["input_embeddings"])
    indices = np.arange(n)
    if target_split is not None:
        if "split" not in arrays:
            raise KeyError("--target-split requires a split array in the target template")
        split_values = np.asarray(strings(arrays["split"]), dtype=object)
        indices = np.flatnonzero(split_values == target_split)
        if not len(indices):
            raise ValueError(f"Target template contains no {target_split!r} rows")
    selected = {
        key: value[indices] if value.ndim > 0 and len(value) == n else value
        for key, value in arrays.items()
        if key != "schema_json"
    }
    return selected, indices


def assert_metadata_aligned(
    predictions: dict[str, np.ndarray], template: dict[str, np.ndarray]
) -> None:
    prediction_n = len(predictions["story"])
    target_n = len(template["input_embeddings"])
    if prediction_n != target_n:
        raise ValueError(f"Prediction/template row mismatch: {prediction_n} != {target_n}")
    for key in ("story", "start_tr", "stop_tr"):
        if key not in predictions or key not in template:
            raise KeyError(f"Alignment requires {key!r} in both inputs")
        left = predictions[key]
        right = template[key]
        if left.dtype.kind in {"O", "S", "U"}:
            left = np.asarray(strings(left), dtype=object)
            right = np.asarray(strings(right), dtype=object)
        if not np.array_equal(left, right):
            mismatch = int(np.sum(left != right))
            raise ValueError(f"Metadata mismatch for {key!r}: {mismatch} rows")


def load_mapper(path: Path, device: torch.device) -> SemanticMapper:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    model = SemanticMapper(
        int(checkpoint["source_dim"]),
        int(checkpoint["target_dim"]),
        int(checkpoint_args["hidden_dim"]),
        float(checkpoint_args["dropout"]),
        residual=bool(checkpoint_args.get("residual", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    predictions = load_npz(args.predictions_npz)
    if args.prediction_key not in predictions:
        raise KeyError(f"{args.predictions_npz} has no {args.prediction_key!r}")
    template_all = load_npz(args.target_template_npz)
    template, template_indices = select_template(template_all, args.target_split)
    assert_metadata_aligned(predictions, template)

    source_vectors = np.asarray(predictions[args.prediction_key], dtype=np.float32)
    target_vectors = np.asarray(template["input_embeddings"], dtype=np.float32)
    model = load_mapper(args.mapper_checkpoint, device)
    mapped = predict(model, source_vectors, args.batch_size, device)
    if mapped.shape != target_vectors.shape:
        raise ValueError(f"Mapped/target shape mismatch: {mapped.shape} != {target_vectors.shape}")

    sentence_values = strings(template["sentence"])
    metrics = retrieval_metrics(mapped, target_vectors, sentence_values)
    output = dict(template)
    output["source_vectors"] = target_vectors
    output["input_embeddings"] = mapped.astype(np.float32)
    output["schema_json"] = np.asarray(
        json.dumps(
            {
                "role": "mapped_mri2sem_split_predictions",
                "mapper_checkpoint": str(args.mapper_checkpoint),
                "source_predictions_npz": str(args.predictions_npz),
                "target_template_npz": str(args.target_template_npz),
                "target_split": args.target_split,
                "target_template_indices": [
                    int(template_indices[0]),
                    int(template_indices[-1]),
                ],
                "row_alignment": ["story", "start_tr", "stop_tr"],
                "sentence_source": "target_template",
                "test_labels_used_for_mapping": False,
                "metrics": metrics,
            },
            indent=2,
        ),
        dtype=object,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    print(json.dumps({"output": str(args.output), "shape": list(mapped.shape), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
