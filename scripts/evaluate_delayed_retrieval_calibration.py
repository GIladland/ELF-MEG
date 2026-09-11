#!/usr/bin/env python3
"""Select a delayed-semantic retrieval calibration on held-out validation stories."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_closed_set_semantic_text import tokenize, word_edit_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--delay-count", type=int, default=4)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


def cosine_similarity(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return normalize_rows(prediction) @ normalize_rows(target).T


def delay_similarity(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: tuple[float, ...],
) -> np.ndarray:
    block_dim = prediction.shape[1] // len(weights)
    similarity = np.zeros((len(prediction), len(target)), dtype=np.float32)
    for delay, weight in enumerate(weights):
        start = delay * block_dim
        stop = start + block_dim
        similarity += weight * cosine_similarity(
            prediction[:, start:stop], target[:, start:stop]
        )
    return similarity


def retrieval_summary(similarity: np.ndarray) -> dict[str, Any]:
    paired = np.diag(similarity)
    ranks = 1 + np.sum(similarity > paired[:, None], axis=1)
    n = len(ranks)
    mismatch = ~np.eye(n, dtype=bool)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= min(5, n))),
        "top10": float(np.mean(ranks <= min(10, n))),
        "top1_hits": int(np.sum(ranks <= 1)),
        "top5_hits": int(np.sum(ranks <= min(5, n))),
        "top10_hits": int(np.sum(ranks <= min(10, n))),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_percentile": (
            float(np.mean(1.0 - (ranks - 1) / (n - 1))) if n > 1 else 1.0
        ),
        "matched_cosine": float(paired.mean()),
        "mismatch_cosine": float(similarity[mismatch].mean()) if n > 1 else None,
        "ranks": ranks,
    }


def story_metrics(similarity: np.ndarray, stories: np.ndarray) -> dict[str, Any]:
    paired = np.diag(similarity)
    ranks = 1 + np.sum(similarity > paired[:, None], axis=1)
    result = {}
    for story in sorted(set(str(value) for value in stories)):
        indices = np.flatnonzero(np.asarray(stories, dtype=str) == story)
        result[story] = {
            "rows": int(len(indices)),
            "global_top1": float(np.mean(ranks[indices] == 1)),
            "global_top1_hits": int(np.sum(ranks[indices] == 1)),
            "mean_percentile": float(
                np.mean(1.0 - (ranks[indices] - 1) / (len(ranks) - 1))
            ),
        }
    return result


def fit_base(kind: str, l2: float | None, x: np.ndarray, y: np.ndarray):
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    if kind == "mean_shift":
        return lambda values: values - x_mean + y_mean
    if kind == "diagonal":
        x_std = x.std(axis=0, keepdims=True).clip(min=1e-5)
        y_std = y.std(axis=0, keepdims=True).clip(min=1e-5)
        return lambda values: (values - x_mean) * (y_std / x_std) + y_mean
    if kind == "ridge":
        x_centered = x - x_mean
        y_centered = y - y_mean
        gram = x_centered @ x_centered.T
        ridge_scale = float(np.trace(gram) / max(1, len(x)))
        penalty = float(l2) * max(ridge_scale, 1e-8)
        dual = np.linalg.solve(
            gram + penalty * np.eye(len(x), dtype=np.float32), y_centered
        )
        return lambda values: (values - x_mean) @ x_centered.T @ dual + y_mean
    raise ValueError(f"Unsupported calibration kind: {kind}")


def blend(values: np.ndarray, calibrated: np.ndarray, alpha: float) -> np.ndarray:
    return normalize_rows((1.0 - alpha) * values + alpha * calibrated)


def simplex_weights(delay_count: int, units: int = 10):
    for cuts in itertools.combinations(range(units + delay_count - 1), delay_count - 1):
        boundaries = (-1, *cuts, units + delay_count - 1)
        counts = tuple(
            boundaries[index + 1] - boundaries[index] - 1
            for index in range(delay_count)
        )
        yield tuple(count / units for count in counts)


def grouped_oof_calibration(
    kind: str,
    l2: float | None,
    prediction: np.ndarray,
    target: np.ndarray,
    stories: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float32)
    story_values = np.asarray(stories, dtype=str)
    for story in sorted(set(story_values)):
        held_out = story_values == story
        train = ~held_out
        transform = fit_base(kind, l2, prediction[train], target[train])
        output[held_out] = transform(prediction[held_out]).astype(np.float32)
    return output


def text_metrics(
    similarity: np.ndarray, sentences: list[str]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    retrieved = np.argmax(similarity, axis=1)
    references = [tokenize(sentence) for sentence in sentences]
    errors = np.asarray(
        [
            word_edit_distance(references[index], references[candidate])
            for index, candidate in enumerate(retrieved)
        ],
        dtype=np.int64,
    )
    words = np.asarray([len(value) for value in references], dtype=np.int64)
    return (
        {
            "corpus_wer": float(errors.sum() / max(1, words.sum())),
            "edit_errors": int(errors.sum()),
            "reference_words": int(words.sum()),
            "exact_text_fraction": float(np.mean(retrieved == np.arange(len(retrieved)))),
        },
        retrieved,
        errors,
    )


def public_summary(similarity: np.ndarray) -> dict[str, Any]:
    summary = retrieval_summary(similarity)
    summary.pop("ranks")
    return summary


def main() -> None:
    args = parse_args()
    validation = load_npz(args.validation_npz)
    test = load_npz(args.test_npz)
    val_prediction = validation["pred"].astype(np.float32)
    val_target = validation["target"].astype(np.float32)
    val_stories = validation["story"]
    test_prediction = test["input_embeddings"].astype(np.float32)
    test_target = test["source_vectors"].astype(np.float32)
    test_sentences = [str(value) for value in test["sentence"]]
    test_stories = test["story"]

    if val_prediction.shape != val_target.shape:
        raise ValueError("Validation prediction/target shapes differ")
    if test_prediction.shape != test_target.shape:
        raise ValueError("Test prediction/target shapes differ")
    if val_prediction.shape[1] % args.delay_count:
        raise ValueError("Embedding width is not divisible by delay count")

    candidates: list[dict[str, Any]] = []
    val_similarity_by_name: dict[str, np.ndarray] = {}

    identity_similarity = cosine_similarity(val_prediction, val_target)
    val_similarity_by_name["identity"] = identity_similarity
    candidates.append({"name": "identity", "kind": "identity", **public_summary(identity_similarity)})

    for weights in simplex_weights(args.delay_count):
        name = "delay_weights_" + "_".join(f"{weight:.1f}" for weight in weights)
        similarity = delay_similarity(val_prediction, val_target, weights)
        val_similarity_by_name[name] = similarity
        candidates.append(
            {
                "name": name,
                "kind": "delay_weights",
                "weights": weights,
                **public_summary(similarity),
            }
        )

    base_specs: list[tuple[str, float | None]] = [
        ("mean_shift", None),
        ("diagonal", None),
        *[("ridge", l2) for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)],
    ]
    for kind, l2 in base_specs:
        calibrated = grouped_oof_calibration(
            kind, l2, val_prediction, val_target, val_stories
        )
        for alpha in (0.25, 0.5, 0.75, 1.0):
            transformed = blend(val_prediction, calibrated, alpha)
            name = f"{kind}_l2{l2}_alpha{alpha}" if l2 is not None else f"{kind}_alpha{alpha}"
            similarity = cosine_similarity(transformed, val_target)
            val_similarity_by_name[name] = similarity
            candidates.append(
                {
                    "name": name,
                    "kind": kind,
                    "l2": l2,
                    "alpha": alpha,
                    **public_summary(similarity),
                }
            )

    candidates.sort(
        key=lambda row: (row["top1"], row["mean_percentile"], row["top5"]),
        reverse=True,
    )
    selected = candidates[0]
    selected_val_similarity = val_similarity_by_name[selected["name"]]

    if selected["kind"] == "identity":
        test_similarity = cosine_similarity(test_prediction, test_target)
    elif selected["kind"] == "delay_weights":
        test_similarity = delay_similarity(
            test_prediction, test_target, tuple(selected["weights"])
        )
    else:
        transform = fit_base(
            selected["kind"], selected.get("l2"), val_prediction, val_target
        )
        calibrated_test = blend(
            test_prediction, transform(test_prediction), float(selected["alpha"])
        )
        test_similarity = cosine_similarity(calibrated_test, test_target)

    baseline_test_similarity = cosine_similarity(test_prediction, test_target)
    selected_text, retrieved, errors = text_metrics(test_similarity, test_sentences)
    baseline_text, _, _ = text_metrics(baseline_test_similarity, test_sentences)
    selected_test = public_summary(test_similarity)
    baseline_test = public_summary(baseline_test_similarity)
    result = {
        "contract": {
            "brain_decoder": "fixed leakage-safe MRI2SEM delayed MiniLM1536",
            "selection": "leave-one-story-out validation predictions only",
            "validation_stories": sorted(set(str(value) for value in val_stories)),
            "test_rows_used_for_selection": False,
            "test_candidate_bank": "paired 107 exact delayed MiniLM1536 targets",
        },
        "selected": selected,
        "validation_selected": public_summary(selected_val_similarity),
        "validation_selected_per_story": story_metrics(selected_val_similarity, val_stories),
        "test_baseline": {**baseline_test, "text": baseline_text},
        "test_selected": {**selected_test, "text": selected_text},
        "test_selected_per_story": story_metrics(test_similarity, test_stories),
        "candidate_count": len(candidates),
        "candidate_results": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    ranks = retrieval_summary(test_similarity)["ranks"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "story",
                "target_rank",
                "retrieved_index",
                "top1_correct",
                "word_edit_distance",
                "target",
                "retrieved",
            ],
        )
        writer.writeheader()
        for index, candidate in enumerate(retrieved):
            writer.writerow(
                {
                    "index": index,
                    "story": str(test_stories[index]),
                    "target_rank": int(ranks[index]),
                    "retrieved_index": int(candidate),
                    "top1_correct": bool(candidate == index),
                    "word_edit_distance": int(errors[index]),
                    "target": test_sentences[index],
                    "retrieved": test_sentences[candidate],
                }
            )

    concise = {
        "selected": selected,
        "validation_selected": result["validation_selected"],
        "test_baseline": result["test_baseline"],
        "test_selected": result["test_selected"],
    }
    print(json.dumps(concise, indent=2), flush=True)


if __name__ == "__main__":
    main()
