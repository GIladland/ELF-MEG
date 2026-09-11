#!/usr/bin/env python3
"""Assemble nested session-OOF MEG predictions into an ELF train/val-tail pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-npz", required=True, type=Path)
    parser.add_argument("--fold-prediction", action="append", required=True, type=Path)
    parser.add_argument("--val-prediction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train-rows", default=2652, type=int)
    parser.add_argument("--expected-val-rows", default=110, type=int)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a duplicate-free subset of source rows for an explicitly interim OOF pack.",
    )
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
            for item in np.asarray(values).tolist()
        ],
        dtype=object,
    )


def retrieval_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = normalize(prediction)
    target = normalize(target)
    similarity = prediction @ target.T
    diagonal = np.diag(similarity)
    ranks = 1 + np.sum(similarity > diagonal[:, None], axis=1)
    return {
        "n": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top1_count": int(np.sum(ranks <= 1)),
        "top5_count": int(np.sum(ranks <= 5)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / max(len(ranks) - 1, 1))),
        "matched_cosine_mean": float(np.mean(diagonal)),
        "matched_cosine_median": float(np.median(diagonal)),
    }


def vector_summary(normalized: np.ndarray, raw: np.ndarray) -> dict[str, float | list[int]]:
    return {
        "shape": list(normalized.shape),
        "raw_norm_mean": float(np.linalg.norm(raw, axis=1).mean()),
        "raw_norm_std": float(np.linalg.norm(raw, axis=1).std()),
        "normalized_dim_variance_mean": float(np.var(normalized, axis=0).mean()),
    }


def story_from_session(session: np.ndarray) -> np.ndarray:
    named = {"6": "thatthingonmyarm", "22": "leavingbaghdad", "24": "howtodraw", "19": "birthofanation"}
    return np.asarray([named.get(value, f"session_{value}") for value in session], dtype=object)


def write_pack(
    path: Path,
    embeddings: np.ndarray,
    exact: np.ndarray,
    sentence: np.ndarray,
    session: np.ndarray,
    run: np.ndarray,
    start_samples: np.ndarray,
    condition: np.ndarray,
    schema: dict[str, object],
) -> None:
    story = story_from_session(session)
    np.savez_compressed(
        path,
        input_embeddings=normalize(embeddings),
        source_vectors=normalize(exact),
        sentence=sentence,
        session=session,
        run=run,
        start_samples=start_samples,
        story=story,
        start_tr=start_samples,
        stop_tr=start_samples + 1,
        split=np.asarray(["val" if value == "qc4wyals_val_prediction" else "train" for value in condition], dtype=object),
        source_split=np.asarray(["val" if value == "qc4wyals_val_prediction" else "train" for value in condition], dtype=object),
        condition_source=condition,
        schema_json=np.asarray(json.dumps(schema, sort_keys=True)),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = args.expected_train_rows
    prediction = np.empty((n, 1536), dtype=np.float32)
    raw_prediction = np.empty((n, 1536), dtype=np.float32)
    exact = np.empty((n, 1536), dtype=np.float32)
    sentence = np.empty(n, dtype=object)
    session = np.empty(n, dtype=object)
    run = np.empty(n, dtype=object)
    start_samples = np.empty(n, dtype=np.int64)
    coverage = np.zeros(n, dtype=np.int16)
    fold_rows: list[dict[str, object]] = []

    with np.load(args.source_train_npz, allow_pickle=True) as source:
        source_sentence = strings(source["sentence"])
        source_session = strings(source["session"])
        source_run = strings(source["run"])
        source_start = np.asarray(source["start_samples"], dtype=np.int64)
        if len(source_sentence) != n:
            raise ValueError(f"Expected {n} source rows, found {len(source_sentence)}")

    for path in args.fold_prediction:
        with np.load(path, allow_pickle=True) as fold:
            required = {
                "input_embeddings", "predicted_embeddings_raw", "target_embeddings",
                "sentence", "session", "run", "start_samples", "source_original_index",
            }
            missing = required.difference(fold.files)
            if missing:
                raise KeyError(f"{path} missing keys {sorted(missing)}")
            index = np.asarray(fold["source_original_index"], dtype=np.int64)
            if np.any(index < 0) or np.any(index >= n):
                raise ValueError(f"Out-of-range original indices in {path}")
            fold_sentence = strings(fold["sentence"])
            fold_session = strings(fold["session"])
            fold_run = strings(fold["run"])
            fold_start = np.asarray(fold["start_samples"], dtype=np.int64)
            if not (
                np.array_equal(fold_sentence, source_sentence[index])
                and np.array_equal(fold_session, source_session[index])
                and np.array_equal(fold_run, source_run[index])
                and np.array_equal(fold_start, source_start[index])
            ):
                raise ValueError(f"Row alignment audit failed for {path}")
            coverage[index] += 1
            prediction[index] = normalize(fold["input_embeddings"])
            raw_prediction[index] = np.asarray(fold["predicted_embeddings_raw"], dtype=np.float32)
            exact[index] = normalize(fold["target_embeddings"])
            sentence[index] = fold_sentence
            session[index] = fold_session
            run[index] = fold_run
            start_samples[index] = fold_start
            fold_rows.append({
                "path": str(path),
                "rows": int(len(index)),
                "outer_fold": int(np.asarray(fold["oof_outer_fold"]).reshape(-1)[0])
                if "oof_outer_fold" in fold.files else None,
            })

    missing_rows = int(np.sum(coverage == 0))
    duplicated_rows = int(np.sum(coverage > 1))
    if duplicated_rows:
        raise ValueError(f"OOF rows must not be duplicated: duplicated={duplicated_rows}")
    if missing_rows and not args.allow_partial:
        raise ValueError(f"OOF coverage must be exactly once: missing={missing_rows}")
    included_index = np.flatnonzero(coverage == 1)
    if not len(included_index):
        raise ValueError("No OOF rows were supplied")
    prediction = prediction[included_index]
    raw_prediction = raw_prediction[included_index]
    exact = exact[included_index]
    sentence = sentence[included_index]
    session = session[included_index]
    run = run[included_index]
    start_samples = start_samples[included_index]
    train_rows = int(len(included_index))

    with np.load(args.val_prediction, allow_pickle=True) as val:
        val_prediction = normalize(val["input_embeddings"])
        val_raw = np.asarray(
            val["predicted_embeddings_raw"]
            if "predicted_embeddings_raw" in val.files
            else val["input_embeddings"],
            dtype=np.float32,
        )
        val_exact = normalize(
            val["target_embeddings"]
            if "target_embeddings" in val.files
            else val["source_vectors"]
        )
        val_sentence = strings(val["sentence"])
        val_session = strings(val["session"])
        val_run = strings(val["run"])
        val_start = np.asarray(val["start_samples"], dtype=np.int64)
    if len(val_sentence) != args.expected_val_rows:
        raise ValueError(f"Expected {args.expected_val_rows} validation rows, found {len(val_sentence)}")

    oof_metrics = retrieval_metrics(prediction, exact)
    val_metrics = retrieval_metrics(val_prediction, val_exact)
    schema = {
        "schema": "nested_session_oof_meg_to_elf_train_valtail_v1",
        "source_train_rows": n,
        "train_rows": train_rows,
        "validation_tail_rows": args.expected_val_rows,
        "oof_coverage_exactly_once": missing_rows == 0 and duplicated_rows == 0,
        "partial_oof": missing_rows > 0,
        "missing_source_rows": missing_rows,
        "duplicated_source_rows": duplicated_rows,
        "outer_predictions_used_for_model_selection": False,
        "protected_test_rows_included": 0,
    }
    train_condition = np.full(train_rows, "qc4wyals_train_oof_prediction", dtype=object)
    val_condition = np.full(args.expected_val_rows, "qc4wyals_val_prediction", dtype=object)
    combined = {
        "embeddings": np.concatenate([prediction, val_prediction]),
        "exact": np.concatenate([exact, val_exact]),
        "sentence": np.concatenate([sentence, val_sentence]),
        "session": np.concatenate([session, val_session]),
        "run": np.concatenate([run, val_run]),
        "start": np.concatenate([start_samples, val_start]),
        "condition": np.concatenate([train_condition, val_condition]),
    }
    source_path = args.output_dir / "qc4wyals_oof_predicted_train_valtail_ada002.npz"
    target_path = args.output_dir / "qc4wyals_oof_exact_alignment_train_valtail_ada002.npz"
    write_pack(
        source_path,
        combined["embeddings"],
        combined["exact"],
        combined["sentence"],
        combined["session"],
        combined["run"],
        combined["start"],
        combined["condition"],
        {**schema, "role": "ELF_interface_input"},
    )
    write_pack(
        target_path,
        combined["exact"],
        combined["exact"],
        combined["sentence"],
        combined["session"],
        combined["run"],
        combined["start"],
        combined["condition"],
        {**schema, "role": "semantic_alignment_target"},
    )
    np.savez_compressed(
        args.output_dir / "qc4wyals_oof_train_predicted_ada002.npz",
        input_embeddings=normalize(prediction),
        predicted_embeddings_raw=raw_prediction,
        target_embeddings=normalize(exact),
        sentence=sentence,
        session=session,
        run=run,
        start_samples=start_samples,
        source_original_index=included_index,
        condition_source=train_condition,
    )
    summary = {
        **schema,
        "folds": sorted(fold_rows, key=lambda row: (row["outer_fold"] is None, row["outer_fold"])),
        "oof_train_retrieval": oof_metrics,
        "oof_train_retrieval_2652_way": oof_metrics if train_rows == 2652 else None,
        "deployment_val_retrieval_110_way": val_metrics,
        "oof_train_vectors": vector_summary(normalize(prediction), raw_prediction),
        "deployment_val_vectors": vector_summary(val_prediction, val_raw),
        "source_pack": str(source_path),
        "alignment_target_pack": str(target_path),
    }
    summary_path = args.output_dir / "qc4wyals_nested_oof_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
