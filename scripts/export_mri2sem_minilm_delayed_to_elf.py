#!/usr/bin/env python
"""Export delayed MiniLM MRI2SEM segments as ELF oracle/prediction NPZs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
ROW_KEYS = ("input_embeddings", "sentence", "story", "start_tr", "stop_tr", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prefix",
        default="uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536",
    )
    parser.add_argument(
        "--predicted-filename",
        default="uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test.npz",
    )
    parser.add_argument("--expected-dim", type=int, default=1536)
    parser.add_argument("--delay-dim", type=int, default=384)
    parser.add_argument("--expected-train", type=int, default=11725)
    parser.add_argument("--expected-val", type=int, default=266)
    parser.add_argument("--expected-test", type=int, default=107)
    return parser.parse_args()


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()],
        dtype=object,
    )


def load_split(
    source: np.lib.npyio.NpzFile,
    split: str,
    expected_n: int,
    expected_dim: int,
) -> dict[str, np.ndarray]:
    mapping = {
        "input_embeddings": f"{split}_y",
        "sentence": f"{split}_text",
        "story": f"{split}_story",
        "start_tr": f"{split}_start_tr",
        "stop_tr": f"{split}_stop_tr",
    }
    missing = [source_key for source_key in mapping.values() if source_key not in source.files]
    if missing:
        raise KeyError(f"Source dataset is missing {split} keys: {missing}")
    arrays = {key: np.asarray(source[source_key]) for key, source_key in mapping.items()}
    embeddings = np.asarray(arrays["input_embeddings"], dtype=np.float32)
    if embeddings.shape != (expected_n, expected_dim):
        raise ValueError(f"Expected {split} embeddings {(expected_n, expected_dim)}, got {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{split} embeddings contain non-finite values")
    for key in ("sentence", "story", "start_tr", "stop_tr"):
        if arrays[key].shape != (expected_n,):
            raise ValueError(f"Expected {split} {key} shape {(expected_n,)}, got {arrays[key].shape}")
    segment_lengths = np.unique(arrays["stop_tr"] - arrays["start_tr"] + 1).tolist()
    if segment_lengths != [10]:
        raise ValueError(f"Expected only 10-TR {split} rows, got lengths {segment_lengths}")
    return {
        "input_embeddings": embeddings,
        "sentence": strings(arrays["sentence"]),
        "story": strings(arrays["story"]),
        "start_tr": np.asarray(arrays["start_tr"], dtype=np.int64),
        "stop_tr": np.asarray(arrays["stop_tr"], dtype=np.int64),
        "split": np.full(expected_n, split, dtype=object),
    }


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in ROW_KEYS}


def vector_summary(vectors: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "finite": bool(np.isfinite(vectors).all()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }


def schema(
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    split_order: list[str],
    arrays: dict[str, np.ndarray],
    representation: str,
) -> dict[str, Any]:
    return {
        "source_dataset_npz": str(Path(args.dataset_npz)),
        "source_metadata_json": str(Path(args.metadata_json)),
        "source_predictions_npz": str(Path(args.predictions_npz)),
        "representation": representation,
        "embedding_family": metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        "embedding_dim": args.expected_dim,
        "delay_embedding_dim": args.delay_dim,
        "delay_count": args.expected_dim // args.delay_dim,
        "delay_order": metadata.get("delay_block_order", ["delay1", "delay2", "delay3", "delay4"]),
        "stimulus_delays": metadata.get("stim_delays"),
        "concatenation": "np.hstack([delay1, delay2, delay3, delay4])",
        "segment_trs": metadata.get("segment_trs"),
        "tr_seconds": (metadata.get("args") or {}).get("tr", 2.0),
        "target_pooling": metadata.get("target_pooling"),
        "split_order": split_order,
        "shape": list(arrays["input_embeddings"].shape),
        "input_key": "input_embeddings",
        "sentence_key": "sentence",
        "normalized": False,
    }


def save(path: Path, arrays: dict[str, np.ndarray], schema_value: dict[str, Any], **extra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        **extra,
        schema_json=np.asarray(json.dumps(schema_value, indent=2), dtype=object),
    )


def main() -> None:
    args = parse_args()
    if args.expected_dim != 4 * args.delay_dim:
        raise ValueError("This exporter requires four equally sized delayed MiniLM blocks")
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    checks = {
        "stim_delays": (metadata.get("stim_delays"), [1, 2, 3, 4]),
        "delay_block_order": (
            metadata.get("delay_block_order"),
            ["delay1", "delay2", "delay3", "delay4"],
        ),
        "segment_trs": (metadata.get("segment_trs"), 10),
        "target_pooling": (metadata.get("target_pooling"), "mean"),
        "tr": ((metadata.get("args") or {}).get("tr"), 2.0),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Source is not the delayed MiniLM 10-TR contract: {mismatches}")

    expected_counts = {
        "train": args.expected_train,
        "val": args.expected_val,
        "test": args.expected_test,
    }
    with np.load(args.dataset_npz, allow_pickle=True) as source:
        split_arrays = {
            split: load_split(source, split, expected_counts[split], args.expected_dim)
            for split in SPLITS
        }

    output_dir = Path(args.output_dir)
    created: list[str] = []
    for split, arrays in split_arrays.items():
        path = output_dir / f"{args.prefix}_{split}.npz"
        save(
            path,
            arrays,
            schema(
                args=args,
                metadata=metadata,
                split_order=[split],
                arrays=arrays,
                representation="oracle_delayed_minilm_segment_mean",
            ),
        )
        created.append(str(path))

    train_val = concatenate([split_arrays["train"], split_arrays["val"]])
    train_val_schema = schema(
        args=args,
        metadata=metadata,
        split_order=["train", "val"],
        arrays=train_val,
        representation="oracle_delayed_minilm_segment_mean",
    )
    train_val_schema["elf_training_contract"] = {
        "train_rows": args.expected_train,
        "validation_rows": args.expected_val,
        "validation_is_tail": True,
        "trainer_argument": f"--val-num-examples {args.expected_val}",
        "test_excluded": True,
    }
    train_val_path = output_dir / f"{args.prefix}_train_val.npz"
    save(train_val_path, train_val, train_val_schema)
    created.append(str(train_val_path))

    oracle_test = split_arrays["test"]
    with np.load(args.predictions_npz, allow_pickle=True) as predictions:
        if "pred" not in predictions.files or "target" not in predictions.files:
            raise KeyError("predictions NPZ must contain pred and target")
        predicted_vectors = np.asarray(predictions["pred"], dtype=np.float32)
        predicted_targets = np.asarray(predictions["target"], dtype=np.float32)
        expected_shape = (args.expected_test, args.expected_dim)
        if predicted_vectors.shape != expected_shape or predicted_targets.shape != expected_shape:
            raise ValueError(
                f"Expected predicted/target shapes {expected_shape}, got "
                f"{predicted_vectors.shape}/{predicted_targets.shape}"
            )
        if not np.isfinite(predicted_vectors).all():
            raise ValueError("Predicted vectors contain non-finite values")
        max_target_error = float(np.max(np.abs(predicted_targets - oracle_test["input_embeddings"])))
        if max_target_error > 1e-5:
            raise ValueError(f"Prediction targets do not align with oracle test rows; max error={max_target_error}")
        ranks = (
            np.asarray(predictions["ranks"], dtype=np.int64)
            if "ranks" in predictions.files
            else np.asarray([], dtype=np.int64)
        )

    predicted_arrays = dict(oracle_test)
    predicted_arrays["input_embeddings"] = predicted_vectors
    predicted_path = output_dir / args.predicted_filename
    save(
        predicted_path,
        predicted_arrays,
        schema(
            args=args,
            metadata=metadata,
            split_order=["test"],
            arrays=predicted_arrays,
            representation="mri2sem_predicted_delayed_minilm_segment_mean",
        ),
        source_vectors=oracle_test["input_embeddings"],
        rank=ranks,
    )
    created.append(str(predicted_path))

    summary = {
        "created": created,
        "contract": checks,
        "oracle": {split: vector_summary(arrays["input_embeddings"]) for split, arrays in split_arrays.items()},
        "predicted_test": vector_summary(predicted_vectors),
        "predicted_target_max_abs_error": max_target_error,
        "train_val_shape": list(train_val["input_embeddings"].shape),
    }
    summary_path = output_dir / f"{args.prefix}_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
