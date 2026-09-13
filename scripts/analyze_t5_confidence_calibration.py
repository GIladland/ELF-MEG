#!/usr/bin/env python3
"""Diagnose T5 word-confidence variants on the fixed validation source."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(((comparisons > 0).mean() + (comparisons == 0).mean() / 2.0))


def calibration_bins(
    scores: np.ndarray, labels: np.ndarray, bins: int
) -> tuple[list[dict[str, Any]], float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (scores >= lower) & (
            scores <= upper if index == bins - 1 else scores < upper
        )
        count = int(selected.sum())
        if not count:
            continue
        confidence = float(scores[selected].mean())
        accuracy = float(labels[selected].mean())
        weighted_gap += count * abs(confidence - accuracy)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": confidence,
                "target_membership_rate": accuracy,
            }
        )
    return rows, weighted_gap / len(scores)


def aligned_labels(payload: dict[str, Any]) -> tuple[list[bool], list[bool]]:
    bag_labels: list[bool] = []
    position_labels: list[bool] = []
    for target, generated, confidence in zip(
        payload["targets"], payload["generated"], payload["word_confidence"]
    ):
        target_words = [word.lower() for word in WORD_RE.findall(target)]
        target_set = set(target_words)
        lexical_index = 0
        for raw_group, _ in zip(generated.split(), confidence):
            matches = WORD_RE.findall(raw_group)
            if not matches:
                continue
            word = matches[0].lower()
            bag_labels.append(word in target_set)
            position_labels.append(
                lexical_index < len(target_words) and word == target_words[lexical_index]
            )
            lexical_index += 1
    return bag_labels, position_labels


def aligned_scores(payload: dict[str, Any], rows: list[list[float]]) -> list[float]:
    scores: list[float] = []
    for generated, confidence in zip(payload["generated"], rows):
        for raw_group, score in zip(generated.split(), confidence):
            if WORD_RE.search(raw_group):
                scores.append(float(score))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    bag_raw, position_raw = aligned_labels(payload)
    bag = np.asarray(bag_raw, dtype=bool)
    position = np.asarray(position_raw, dtype=bool)
    variants = {"transition_probability": payload["word_confidence"]}
    variants.update(payload.get("word_confidence_variants", {}))
    result_variants: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {}
    quantile_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9, 1.0]
    for name, rows in variants.items():
        scores = np.asarray(aligned_scores(payload, rows), dtype=np.float64)
        if len(scores) != len(bag):
            raise ValueError(f"{name} produced {len(scores)} scores for {len(bag)} words")
        bins, ece = calibration_bins(scores, bag, args.bins)
        score_arrays[name] = scores
        result_variants[name] = {
            "word_count": len(scores),
            "mean": float(scores.mean()),
            "quantiles": {
                f"{level:g}": float(np.quantile(scores, level))
                for level in quantile_levels
            },
            "bag_membership_auc": auc(scores, bag),
            "position_exact_auc": auc(scores, position),
            "bag_membership_brier": float(np.mean((scores - bag.astype(float)) ** 2)),
            "bag_membership_ece": ece,
            "calibration_bins": bins,
        }
    correlations = {
        left: {
            right: float(np.corrcoef(left_scores, right_scores)[0, 1])
            for right, right_scores in score_arrays.items()
        }
        for left, left_scores in score_arrays.items()
    }
    result = {
        "scientific_scope": (
            "validation-only global confidence calibration diagnostic; target labels "
            "are used only for reporting and global hyperparameter selection"
        ),
        "input": str(args.input),
        "word_count": len(bag),
        "bag_membership_rate": float(bag.mean()),
        "position_exact_rate": float(position.mean()),
        "variants": result_variants,
        "pearson_correlations": correlations,
        "protected_test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for name, values in result_variants.items():
        print(
            f"{name}: bag_auc={values['bag_membership_auc']:.6f} "
            f"position_auc={values['position_exact_auc']:.6f} "
            f"ece={values['bag_membership_ece']:.6f}"
        )


if __name__ == "__main__":
    main()
