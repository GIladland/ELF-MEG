#!/usr/bin/env python3
"""Project qc4wyals predictions toward the known Tang/Apples ADA manifold.

This is a deliberately labelled closed-corpus diagnostic.  It may use all
known exact text/ADA rows as an unlabeled semantic bank, but it never uses the
protected test MEG vectors during validation selection.  The resulting packs
make it possible to measure whether ELF mainly needs an on-manifold condition,
rather than a more expressive neural adapter.
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
    parser.add_argument("--knowntext-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=128)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def top_neighbors(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    max_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    query = normalize(query)
    bank = normalize(bank)
    max_k = min(max_k, len(bank))
    all_indices = []
    all_scores = []
    for start in range(0, len(query), batch_size):
        similarity = query[start : start + batch_size] @ bank.T
        indices = np.argpartition(-similarity, kth=max_k - 1, axis=1)[:, :max_k]
        scores = np.take_along_axis(similarity, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        all_indices.append(np.take_along_axis(indices, order, axis=1))
        all_scores.append(np.take_along_axis(scores, order, axis=1))
    return np.concatenate(all_indices), np.concatenate(all_scores)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    result = np.exp(values)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def anchored(
    raw: np.ndarray,
    bank: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    alpha: float,
    temperature: float | None,
) -> np.ndarray:
    selected = bank[indices[:, :k]]
    if temperature is None:
        weights = np.full((len(raw), k), 1.0 / k, dtype=np.float32)
    else:
        weights = softmax(scores[:, :k] / np.float32(temperature)).astype(np.float32)
    anchor = normalize(np.sum(selected * weights[:, :, None], axis=1))
    return normalize((1.0 - np.float32(alpha)) * normalize(raw) + np.float32(alpha) * anchor)


def semantic_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = normalize(predicted)
    target = normalize(target)
    similarity = predicted @ target.T
    paired = np.diag(similarity)
    ranks = 1 + np.sum(similarity > paired[:, None], axis=1)
    return {
        "matched_cosine": float(paired.mean()),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= min(5, len(ranks)))),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "ndcg": float(np.mean(1.0 / np.log2(ranks + 1.0))),
        "prediction_dim_variance": float(np.var(predicted, axis=0).mean()),
    }


def candidate_name(k: int, alpha: float, temperature: float | None) -> str:
    weighting = "uniform" if temperature is None else f"t{temperature:g}".replace(".", "p")
    return f"knn{k}_{weighting}_a{alpha:g}".replace(".", "p")


def write_like(
    path: Path,
    template: dict[str, np.ndarray],
    embeddings: np.ndarray,
    *,
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
        with np.load(temporary_path, allow_pickle=True) as audit:
            if audit["input_embeddings"].shape != embeddings.shape:
                raise RuntimeError(f"Corrupt temporary archive for {path}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    train = load(args.interface_dir / "qc4wyals_train_predicted_ada002.npz")
    val = load(args.interface_dir / "qc4wyals_val_predicted_ada002.npz")
    known = load(args.knowntext_npz)
    train_raw = normalize(train["input_embeddings"])
    val_raw = normalize(val["input_embeddings"])
    train_target = normalize(train["source_vectors"])
    val_target = normalize(val["source_vectors"])
    bank = normalize(known["input_embeddings"])

    val_indices, val_scores = top_neighbors(
        val_raw,
        bank,
        max_k=args.max_k,
        batch_size=args.query_batch_size,
    )
    specs: list[dict[str, object]] = []
    for k in (1, 4, 16, 64):
        if k > args.max_k:
            continue
        temperatures: tuple[float | None, ...] = (None,) if k == 1 else (None, 0.01, 0.02, 0.05)
        for temperature in temperatures:
            for alpha in (0.25, 0.5, 0.75, 1.0):
                transformed = anchored(
                    val_raw,
                    bank,
                    val_indices,
                    val_scores,
                    k=k,
                    alpha=alpha,
                    temperature=temperature,
                )
                specs.append(
                    {
                        "name": candidate_name(k, alpha, temperature),
                        "k": k,
                        "alpha": alpha,
                        "temperature": temperature,
                        "validation": semantic_metrics(transformed, val_target),
                    }
                )

    selectors = {
        "ndcg": lambda item: (item["validation"]["ndcg"], item["validation"]["matched_cosine"]),
        "cosine": lambda item: (item["validation"]["matched_cosine"], item["validation"]["ndcg"]),
        "top1": lambda item: (item["validation"]["top1"], item["validation"]["ndcg"]),
        "balanced": lambda item: (
            item["validation"]["ndcg"] + 0.25 * item["validation"]["matched_cosine"],
            item["validation"]["top5"],
        ),
        # A fixed closed-corpus retrieval control.  It is intentionally not
        # presented as generative MEG decoding: the condition is replaced by
        # the exact ADA vector of the nearest known sentence before ELF runs.
        "nearest_exact_control": lambda item: (item["name"] == "knn1_uniform_a1",),
    }
    selected = {name: max(specs, key=key) for name, key in selectors.items()}
    unique_selected = {item["name"]: item for item in selected.values()}

    train_indices, train_scores = top_neighbors(
        train_raw,
        bank,
        max_k=args.max_k,
        batch_size=args.query_batch_size,
    )
    train_val_template = load(args.interface_dir / "qc4wyals_predicted_train_valtail_ada002.npz")
    for name, spec in unique_selected.items():
        kwargs = {
            "k": int(spec["k"]),
            "alpha": float(spec["alpha"]),
            "temperature": spec["temperature"],
        }
        mapped_train = anchored(train_raw, bank, train_indices, train_scores, **kwargs)
        mapped_val = anchored(val_raw, bank, val_indices, val_scores, **kwargs)
        schema = {
            "schema": "qc4wyals_knowntext_manifold_anchor_v1",
            "candidate": spec,
            "knowntext_bank_rows": int(len(bank)),
            "knowntext_bank_may_include_validation_and_test_text": True,
            "validation_meg_used_for_fit": False,
            "protected_test_meg_used": False,
            "interpretation": "closed-corpus known-text manifold diagnostic",
        }
        write_like(
            args.output_dir / f"qc4wyals_manifold_{name}_train_valtail_ada002.npz",
            train_val_template,
            np.concatenate([mapped_train, mapped_val]),
            source_vectors=np.concatenate([train_target, val_target]),
            schema=schema,
        )
        write_like(
            args.output_dir / f"qc4wyals_manifold_{name}_val_ada002.npz",
            val,
            mapped_val,
            source_vectors=val_target,
            schema={**schema, "split": "validation"},
        )

    report = {
        "contract": {
            "selection_split": "validation110",
            "protected_test_meg_used": False,
            "knowntext_bank": str(args.knowntext_npz),
            "closed_corpus": True,
        },
        "raw_validation": semantic_metrics(val_raw, val_target),
        "selectors": selected,
        "candidates": specs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifold_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
