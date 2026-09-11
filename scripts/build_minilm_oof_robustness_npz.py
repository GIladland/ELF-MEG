#!/usr/bin/env python
"""Build an ELF robustness dataset from oracle and story-level OOF MiniLM inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-train-npz", required=True)
    parser.add_argument("--oracle-val-npz", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--expected-train", type=int, default=11725)
    parser.add_argument("--expected-val", type=int, default=266)
    parser.add_argument("--expected-dim", type=int, default=1536)
    return parser.parse_args()


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()],
        dtype=object,
    )


def load_row_npz(path: Path, expected_n: int, expected_dim: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as source:
        arrays = {
            key: np.asarray(source[key])
            for key in source.files
            if key != "schema_json" and np.asarray(source[key]).ndim >= 1 and np.asarray(source[key]).shape[0] == expected_n
        }
        raw_schema = str(source["schema_json"].tolist()) if "schema_json" in source.files else "{}"
    if "input_embeddings" not in arrays or arrays["input_embeddings"].shape != (expected_n, expected_dim):
        raise ValueError(f"Unexpected semantic shape in {path}: {arrays.get('input_embeddings', np.empty(0)).shape}")
    required = ("sentence", "story", "start_tr", "stop_tr", "split")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{path} is missing row keys {missing}")
    try:
        schema = json.loads(raw_schema)
    except json.JSONDecodeError:
        schema = {"raw_schema": raw_schema}
    arrays["input_embeddings"] = np.asarray(arrays["input_embeddings"], dtype=np.float32)
    arrays["story"] = strings(arrays["story"])
    arrays["sentence"] = strings(arrays["sentence"])
    return arrays, schema


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred_norm = l2_normalize(pred.astype(np.float32, copy=False))
    target_norm = l2_normalize(target.astype(np.float32, copy=False))
    scores = pred_norm @ target_norm.T
    matched = np.diag(scores)
    ranks = 1 + np.sum(scores > matched[:, None], axis=1)
    mismatch = scores[~np.eye(len(scores), dtype=bool)] if len(scores) > 1 else np.asarray([])
    return {
        "n": int(len(ranks)),
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= min(5, len(ranks)))),
        "top10": float(np.mean(ranks <= min(10, len(ranks)))),
        "mean_rank": float(np.mean(ranks)),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / max(len(ranks) - 1, 1))),
        "matched_cosine": float(matched.mean()),
        "mismatch_cosine": float(mismatch.mean()) if len(mismatch) else float("nan"),
    }


def vector_summary(values: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(values, axis=1)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }


def main() -> None:
    args = parse_args()
    train_path = Path(args.oracle_train_npz)
    val_path = Path(args.oracle_val_npz)
    fold_dir = Path(args.fold_dir)
    output_path = Path(args.output_npz)
    train, train_schema = load_row_npz(train_path, args.expected_train, args.expected_dim)
    val, val_schema = load_row_npz(val_path, args.expected_val, args.expected_dim)

    oof_pred = np.empty((args.expected_train, args.expected_dim), dtype=np.float32)
    coverage = np.zeros(args.expected_train, dtype=np.int16)
    val_predictions: list[np.ndarray] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(args.num_folds):
        fold_path = fold_dir / f"fold_{fold}.npz"
        summary_path = fold_dir / f"fold_{fold}.json"
        with np.load(fold_path, allow_pickle=True) as payload:
            indices = np.asarray(payload["holdout_indices"], dtype=np.int64)
            predicted = np.asarray(payload["oof_pred"], dtype=np.float32)
            target = np.asarray(payload["oof_target"], dtype=np.float32)
            fold_story = strings(payload["oof_story"])
            val_pred = np.asarray(payload["val_pred"], dtype=np.float32)
            val_target = np.asarray(payload["val_target"], dtype=np.float32)
            val_story = strings(payload["val_story"])
            fit_stories = set(strings(payload["fit_stories"]).tolist())
            holdout_stories = set(strings(payload["holdout_stories"]).tolist())
        if fit_stories & holdout_stories:
            raise ValueError(f"Fold {fold} has story leakage")
        if predicted.shape != (len(indices), args.expected_dim):
            raise ValueError(f"Fold {fold} OOF shape mismatch: {predicted.shape}")
        if val_pred.shape != (args.expected_val, args.expected_dim):
            raise ValueError(f"Fold {fold} validation prediction shape mismatch: {val_pred.shape}")
        if not np.allclose(target, train["input_embeddings"][indices], atol=1e-6, rtol=1e-5):
            raise ValueError(f"Fold {fold} targets do not match oracle train indices")
        if not np.array_equal(fold_story, strings(train["story"])[indices]):
            raise ValueError(f"Fold {fold} story rows are misaligned")
        if not np.allclose(val_target, val["input_embeddings"], atol=1e-6, rtol=1e-5):
            raise ValueError(f"Fold {fold} validation targets are misaligned")
        if not np.array_equal(val_story, strings(val["story"])):
            raise ValueError(f"Fold {fold} validation stories are misaligned")
        oof_pred[indices] = predicted
        coverage[indices] += 1
        val_predictions.append(val_pred)
        fold_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if not np.all(coverage == 1):
        missing = int(np.sum(coverage == 0))
        duplicated = int(np.sum(coverage > 1))
        raise ValueError(f"OOF coverage is not exactly once: missing={missing}, duplicated={duplicated}")
    if not np.isfinite(oof_pred).all():
        raise ValueError("OOF predictions contain non-finite values")
    oof_pred = l2_normalize(oof_pred).astype(np.float32, copy=False)
    val_pred_ensemble = l2_normalize(np.mean(np.stack(val_predictions), axis=0)).astype(np.float32, copy=False)

    common_keys = sorted(set(train) & set(val))
    output: dict[str, np.ndarray] = {}
    for key in common_keys:
        if key == "input_embeddings":
            output[key] = np.concatenate(
                [train[key], oof_pred, val_pred_ensemble], axis=0
            ).astype(np.float32, copy=False)
        elif key == "split":
            output[key] = np.concatenate(
                [
                    np.full(args.expected_train, "train_oracle", dtype=object),
                    np.full(args.expected_train, "train_oof", dtype=object),
                    np.full(args.expected_val, "val_pred", dtype=object),
                ]
            )
        else:
            output[key] = np.concatenate([train[key], train[key], val[key]], axis=0)
    output["condition_source"] = np.concatenate(
        [
            np.full(args.expected_train, "oracle", dtype=object),
            np.full(args.expected_train, "oof_prediction", dtype=object),
            np.full(args.expected_val, "validation_prediction_ensemble", dtype=object),
        ]
    )

    expected_rows = 2 * args.expected_train + args.expected_val
    if output["input_embeddings"].shape != (expected_rows, args.expected_dim):
        raise AssertionError("Unexpected final robustness dataset shape")
    for key, values in output.items():
        if values.shape[0] != expected_rows:
            raise AssertionError(f"Output row key {key} has shape {values.shape}, expected {expected_rows} rows")

    oof_matched = np.sum(oof_pred * l2_normalize(train["input_embeddings"]), axis=1)
    summary = {
        "oracle_train_npz": str(train_path),
        "oracle_val_npz": str(val_path),
        "fold_dir": str(fold_dir),
        "output_npz": str(output_path),
        "num_folds": args.num_folds,
        "story_level_oof": True,
        "oof_coverage_exactly_once": True,
        "test_rows_used": 0,
        "row_order": ["train_oracle", "train_oof", "val_pred"],
        "elf_training_contract": {
            "total_rows": expected_rows,
            "training_rows": 2 * args.expected_train,
            "validation_rows": args.expected_val,
            "validation_is_tail": True,
            "trainer_argument": f"--val-num-examples {args.expected_val}",
            "checkpoint_selection_input": "five-model validation prediction ensemble",
        },
        "oracle_train": vector_summary(train["input_embeddings"]),
        "oof_train": vector_summary(oof_pred),
        "predicted_val_ensemble": vector_summary(val_pred_ensemble),
        "oof_matched_cosine": {
            "mean": float(oof_matched.mean()),
            "median": float(np.median(oof_matched)),
            "min": float(oof_matched.min()),
            "max": float(oof_matched.max()),
        },
        "validation_ensemble_retrieval": retrieval_metrics(val_pred_ensemble, val["input_embeddings"]),
        "fold_metrics": [
            {
                "fold": int(fold_summary["args"]["fold_index"]),
                "fit_rows": int(fold_summary["fit_rows"]),
                "holdout_rows": int(fold_summary["holdout_rows"]),
                "story_leakage": bool(fold_summary["story_leakage"]),
                "oof_metrics": fold_summary["oof_metrics"],
                "val_metrics": fold_summary["val_metrics"],
            }
            for fold_summary in fold_summaries
        ],
        "source_schemas": {"train": train_schema, "val": val_schema},
    }
    schema = {
        "representation": "oracle_plus_story_oof_delayed_minilm1536_for_elf_robustness",
        "input_key": "input_embeddings",
        "sentence_key": "sentence",
        "embedding_dim": args.expected_dim,
        "delay_blocks": 4,
        "delay_dim": 384,
        "delay_order": [1, 2, 3, 4],
        "row_order": summary["row_order"],
        "elf_training_contract": summary["elf_training_contract"],
        "test_excluded": True,
        "summary_json": str(output_path.with_suffix(".json")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output, schema_json=np.asarray(json.dumps(schema, indent=2), dtype=object))
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
