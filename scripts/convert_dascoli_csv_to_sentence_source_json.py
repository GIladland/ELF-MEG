#!/usr/bin/env python3
"""Convert DaScoli CSV handoff exports into sentence-source JSON proposals.

The DaScoli CSV format stores one row per token/position with a confidence value
that may be on an arbitrary numeric scale.  This utility can keep those values
as-is or apply a deterministic re-scaling step before emitting the sentence-source
JSON that `train_npz_semantic_to_elf.py` expects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Literal

Calibration = Literal["identity", "minmax_global", "minmax_row", "rank_global"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    calibration_choices = [
        "identity",
        "minmax_global",
        "minmax_row",
        "rank_global",
    ]
    parser.add_argument(
        "--confidence-calibration",
        default="identity",
        choices=calibration_choices,
        help=(
            "How to rescale confidence values. "
            "'identity' keeps raw values. "
            "'minmax_global' rescales all rows to [0,1] via global min/max. "
            "'minmax_row' rescales each row to [0,1]. "
            "'rank_global' maps each score to empirical global rank in [0,1]."
        ),
    )
    parser.add_argument(
        "--confidence-epsilon",
        type=float,
        default=1e-8,
        help="Small numeric guard for degenerate denominator cases.",
    )
    parser.add_argument(
        "--expected-words",
        type=int,
        default=10,
        help="Expected lexical word count per row in the output contract.",
    )
    return parser.parse_args()


def read_rows(input_csv: Path) -> list[dict]:
    rows = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "row": int(row["row"]),
                    "position": int(row["position"]),
                    "generated": row["generated"].strip(),
                    "confidence": float(row["confidence"]),
                    "reference": row.get("reference", "").strip(),
                    "token_id": int(row["token_id"]) if row.get("token_id", "").strip() else None,
                    "row_id": row.get("row_id", "").strip(),
                }
            )
    return rows


def transform_confidences(
    rows_by_row: dict[int, list[dict]],
    mode: Calibration,
    epsilon: float,
) -> tuple[dict[int, list[float]], dict[str, float]]:
    all_conf = [value for items in rows_by_row.values() for value in (item["confidence"] for item in items)]
    if not all_conf:
        raise ValueError("No confidence values found.")
    transformed: dict[int, list[float]] = {}
    metadata = {
        "source_mode": mode,
        "source_num_scores": float(len(all_conf)),
        "source_raw_min": float(min(all_conf)),
        "source_raw_max": float(max(all_conf)),
        "source_raw_mean": float(sum(all_conf) / len(all_conf)),
    }
    if mode == "identity":
        for key, items in rows_by_row.items():
            transformed[key] = [float(item["confidence"]) for item in items]
        return transformed, metadata
    if mode == "minmax_global":
        min_v = min(all_conf)
        max_v = max(all_conf)
        span = max(max_v - min_v, epsilon)
        metadata.update(
            {"source_scale_min": float(min_v), "source_scale_max": float(max_v), "source_scale_span": float(span)}
        )
        for key, items in rows_by_row.items():
            transformed[key] = [(item["confidence"] - min_v) / span for item in items]
        return transformed, metadata
    if mode == "rank_global":
        sorted_vals = sorted(all_conf)
        n = len(sorted_vals)
        if n <= 1:
            for key, items in rows_by_row.items():
                transformed[key] = [1.0 for _ in items]
            return transformed, metadata
        denom = n - 1
        # Rank with average tie handling by midpoint rank
        first_occurrence: dict[float, int] = {}
        last_occurrence: dict[float, int] = {}
        for index, value in enumerate(sorted_vals):
            first_occurrence.setdefault(value, index)
            last_occurrence[value] = index
        transformed = {}
        for key, items in rows_by_row.items():
            scaled = []
            for item in items:
                value = item["confidence"]
                start = first_occurrence[value]
                end = last_occurrence[value]
                midpoint = 0.5 * (start + end)
                scaled.append(midpoint / denom)
            transformed[key] = scaled
        return transformed, metadata
    if mode == "minmax_row":
        for key, items in rows_by_row.items():
            vals = [item["confidence"] for item in items]
            min_v = min(vals)
            max_v = max(vals)
            span = max(max_v - min_v, epsilon)
            transformed[key] = [(item["confidence"] - min_v) / span for item in items]
        return transformed, metadata
    raise ValueError(f"Unknown confidence calibration mode: {mode}")


def clip01(values: list[float], epsilon: float) -> list[float]:
    low = 0.0
    high = 1.0
    return [min(high, max(low, val)) for val in values]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    if not rows:
        raise ValueError("CSV contains no rows.")
    rows_by_row: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_row.setdefault(row["row"], []).append(row)
    expected_rows = len(rows_by_row)
    if expected_rows != 110:
        print(
            f"WARNING: found {expected_rows} rows in CSV; expected 110 in this contract."
        )

    for key, items in rows_by_row.items():
        if len(items) != args.expected_words:
            raise ValueError(
                f"row {key} has {len(items)} words, expected {args.expected_words}"
            )
        items.sort(key=lambda item: item["position"])
        if [item["position"] for item in items] != list(range(args.expected_words)):
            raise ValueError(f"row {key} has non-contiguous positions")

    transformed, metadata = transform_confidences(
        rows_by_row, args.confidence_calibration, args.confidence_epsilon
    )
    generated: list[str] = []
    confidence: list[list[float]] = []
    row_ids: list[str] = []
    row_indices: list[int] = []
    for row_id in sorted(rows_by_row):
        items = rows_by_row[row_id]
        generated_sentence = " ".join(item["generated"] for item in items)
        row_conf = clip01(transformed[row_id], args.confidence_epsilon)
        generated.append(generated_sentence)
        confidence.append(row_conf)
        row_ids.append(items[0]["row_id"])
        row_indices.append(row_id)

    payload = {
        "source": str(args.input_csv),
        "calibration": args.confidence_calibration,
        "calibration_summary": metadata,
        "generated": generated,
        "word_confidence": confidence,
        "expected_words": args.expected_words,
        "row_id": row_ids,
        "row_index": row_indices,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat_conf = [value for row in confidence for value in row]
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "n_rows": len(generated),
                "n_scores": len(flat_conf),
                "calibration": args.confidence_calibration,
                "confidence_min": min(flat_conf),
                "confidence_max": max(flat_conf),
                "confidence_mean": sum(flat_conf) / max(1, len(flat_conf)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
