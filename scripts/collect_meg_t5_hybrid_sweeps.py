#!/usr/bin/env python3
"""Collect T5-to-ELF-B hybrid grids and mark their coarse Pareto frontier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PARETO_FIELDS = (
    ("content_f1", True),
    ("word_f1", True),
    ("wer", False),
    ("top1", True),
    ("top5", True),
)


def number(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    return float(default if value is None else value)


def load_row(metrics_path: Path, baseline_generated: list[str]) -> dict[str, Any]:
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    quality = metrics["generation_quality"]
    source = metrics.get("sentence_source", {})
    retrieval = metrics.get("generation_t5_retrieval", {})
    generated = metrics.get("generated", [])
    if len(generated) != len(baseline_generated):
        raise ValueError(
            f"{metrics_path} has {len(generated)} generations; "
            f"baseline has {len(baseline_generated)}"
        )
    content_f1 = number(quality, "content_words_overlap")
    word_f1 = number(quality, "words_overlap")
    generation_sha256 = hashlib.sha256(
        "".join(f"{index}\t{text}\n" for index, text in enumerate(generated)).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "run": metrics_path.parent.name,
        "metrics_path": str(metrics_path),
        "copy_threshold": number(source, "source_copy_confidence_threshold", -1.0),
        "confidence_field": source.get("external_eval_confidence_field", "word_confidence"),
        "copy_function_words": bool(source.get("source_copy_function_words", False)),
        "semantic_scale": number(source, "eval_semantic_scale", 1.0),
        "flow_end": number(metrics, "source_flow_end_time", 1.0),
        "edit_latent_preservation": number(
            metrics, "source_edit_latent_preservation", 0.0
        ),
        "copy_fraction": number(metrics.get("source_token_copy", {}), "copy_fraction"),
        "content_f1": content_f1,
        "word_f1": word_f1,
        "sum": content_f1 + word_f1,
        "wer": number(quality, "word_error_rate"),
        "wer_errors": int(
            quality.get("word_error_insertions", 0)
            + quality.get("word_error_deletions", 0)
            + quality.get("word_error_substitutions", 0)
        ),
        "top1": number(retrieval, "top1"),
        "top5": number(retrieval, "top5"),
        "mean_rank": number(retrieval, "mean_rank", float("inf")),
        "changed_rows_vs_baseline": sum(
            generated_text != baseline_text
            for generated_text, baseline_text in zip(generated, baseline_generated)
        ),
        "generation_sha256": generation_sha256,
    }


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = True
    strictly_better = False
    for field, higher_is_better in PARETO_FIELDS:
        left_value = float(left[field])
        right_value = float(right[field])
        if higher_is_better:
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
    return bool(no_worse and strictly_better)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", action="append", required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.baseline_json.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    baseline_generated = baseline["generated"]
    baseline_quality = baseline["generation_quality"]
    baseline_content = number(baseline_quality, "content_words_overlap")
    baseline_word = number(baseline_quality, "words_overlap")
    baseline_sum = baseline_content + baseline_word
    baseline_wer = number(baseline_quality, "word_error_rate")
    baseline_retrieval = baseline.get("generation_t5_retrieval", {})
    baseline_pareto_row = {
        "content_f1": baseline_content,
        "word_f1": baseline_word,
        "wer": baseline_wer,
        "top1": number(baseline_retrieval, "top1"),
        "top5": number(baseline_retrieval, "top5"),
    }

    metrics_paths: list[Path] = []
    for prefix in args.run_prefix:
        metrics_paths.extend(args.runs_root.glob(f"{prefix}*/best_metrics.json"))
    metrics_paths = sorted(set(metrics_paths))
    if not metrics_paths:
        raise FileNotFoundError("No best_metrics.json files matched the requested prefixes")
    rows = [load_row(path, baseline_generated) for path in metrics_paths]
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        representatives.setdefault(row["generation_sha256"], row)
    unique_rows = list(representatives.values())
    comparison_rows = [*unique_rows, baseline_pareto_row]
    for row in rows:
        representative = representatives[row["generation_sha256"]]
        row["duplicate_of"] = "" if representative is row else representative["run"]
        row["unique_generation"] = representative is row
    for row in rows:
        row["pareto"] = not any(
            dominates(other, row) for other in comparison_rows if other is not row
        ) if row["unique_generation"] else False
        row["dominated_by_current_hybrid"] = dominates(baseline_pareto_row, row)
        row["exceeds_current_sum"] = row["sum"] > baseline_sum
        row["lowers_current_wer"] = row["wer"] < baseline_wer
        row["extended_audit_candidate"] = bool(
            row["unique_generation"]
            and (row["pareto"] or row["exceeds_current_sum"] or row["lowers_current_wer"])
        )
    rows.sort(
        key=lambda row: (
            not row["extended_audit_candidate"],
            -row["sum"],
            row["wer"],
            -row["top5"],
            -row["top1"],
            row["run"],
        )
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shortlist = [row for row in rows if row["extended_audit_candidate"]]
    summary = {
        "contract": {
            "coarse_pareto_metrics": [field for field, _ in PARETO_FIELDS],
            "extended_audit_rule": (
                "coarse Pareto frontier OR overlap sum above current hybrid "
                "OR WER below current hybrid"
            ),
            "protected_test_opened": False,
        },
        "baseline": {
            "content_f1": baseline_content,
            "word_f1": baseline_word,
            "sum": baseline_sum,
            "wer": baseline_wer,
            "top1": baseline_pareto_row["top1"],
            "top5": baseline_pareto_row["top5"],
        },
        "matched_run_count": len(rows),
        "unique_generation_count": len(unique_rows),
        "pareto_count": sum(bool(row["pareto"]) for row in rows),
        "extended_audit_candidate_count": len(shortlist),
        "top_by_sum": sorted(rows, key=lambda row: (-row["sum"], row["wer"]))[:20],
        "top_by_wer": sorted(rows, key=lambda row: (row["wer"], -row["sum"]))[:20],
        "shortlist": shortlist,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"collected={len(rows)} pareto={summary['pareto_count']} "
        f"extended_audit={len(shortlist)}"
    )


if __name__ == "__main__":
    main()
