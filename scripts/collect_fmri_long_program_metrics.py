#!/usr/bin/env python
"""Collect overlap, retrieval, semantic-interface, and BERTScore metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROGRAM_PREFIXES = (
    "fmri_minilm_long100_",
    "fmri_minilm_rawbrain_e2e_projector_long100_",
    "fmri_minilm_stage2_",
    "fmri_knowntext_",
    "fmri_rawbrain_e2e_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="/data/engs-pnpl/glandau/elf-runs")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--exclude-known-text-oracle", action="store_true")
    return parser.parse_args()


def nested(metrics: dict, section: str, key: str):
    value = metrics.get(section) or {}
    return value.get(key)


def scope_for_run(run_name: str, num_examples: int | None) -> str:
    if run_name.startswith("fmri_knowntext_"):
        return "known_text_oracle"
    if num_examples == 266:
        return "brain_validation_266"
    return "other"


def extract_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "best_metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    generated = metrics.get("generated") or []
    num_examples = int(metrics.get("eval_num_examples") or len(generated) or 0)
    bert_path = run_dir / "best_metrics.bertscore.json"
    bert = json.loads(bert_path.read_text(encoding="utf-8")) if bert_path.exists() else {}
    bert_summary = bert.get("summary") or {}
    row = {
        "run": run_dir.name,
        "scope": scope_for_run(run_dir.name, num_examples),
        "step": metrics.get("step"),
        "epoch": metrics.get("epoch"),
        "num_examples": num_examples,
        "word_error_rate": nested(metrics, "generation_quality", "word_error_rate"),
        "word_overlap_f1": nested(metrics, "generation_quality", "words_overlap"),
        "content_overlap_f1": nested(metrics, "generation_quality", "content_words_overlap"),
        "well_structured": nested(metrics, "generation_quality", "well_structured_sentence"),
        "degen_fraction": nested(metrics, "generation_quality", "degen_fraction"),
        "generated_top1": nested(metrics, "generation_t5_retrieval", "top1"),
        "generated_top5": nested(metrics, "generation_t5_retrieval", "top5"),
        "generated_mean_rank": nested(metrics, "generation_t5_retrieval", "mean_rank"),
        "generated_median_rank": nested(metrics, "generation_t5_retrieval", "median_rank"),
        "semantic_cosine": nested(metrics, "semantic_interface", "matched_cosine_mean"),
        "semantic_top1": nested(metrics, "semantic_interface", "top1"),
        "semantic_top5": nested(metrics, "semantic_interface", "top5"),
        "bertscore_precision": bert_summary.get("bertscore_precision"),
        "bertscore_recall": bert_summary.get("bertscore_recall"),
        "bertscore_f1": bert_summary.get("bertscore_f1"),
        "metrics_path": str(metrics_path),
        "bertscore_path": str(bert_path) if bert_path.exists() else None,
    }
    return row


def sort_value(value, *, default: float) -> float:
    return default if value is None else float(value)


def dominates(left: dict, right: dict) -> bool:
    maximize = ("content_overlap_f1", "word_overlap_f1", "generated_top5", "bertscore_f1")
    minimize = ("generated_mean_rank",)
    comparable = False
    strictly_better = False
    for key in maximize:
        if left.get(key) is None or right.get(key) is None:
            continue
        comparable = True
        if float(left[key]) < float(right[key]):
            return False
        strictly_better |= float(left[key]) > float(right[key])
    for key in minimize:
        if left.get(key) is None or right.get(key) is None:
            continue
        comparable = True
        if float(left[key]) > float(right[key]):
            return False
        strictly_better |= float(left[key]) < float(right[key])
    return comparable and strictly_better


def mark_pareto(rows: list[dict]) -> None:
    for row in rows:
        if row["scope"] != "brain_validation_266":
            row["pareto"] = False
            continue
        competitors = [candidate for candidate in rows if candidate["scope"] == row["scope"]]
        row["pareto"] = not any(dominates(candidate, row) for candidate in competitors if candidate is not row)


def markdown(rows: list[dict]) -> str:
    columns = (
        "run",
        "scope",
        "content_overlap_f1",
        "word_overlap_f1",
        "generated_top5",
        "generated_mean_rank",
        "bertscore_f1",
        "pareto",
    )
    lines = ["# fMRI long-program results", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Known-text oracle rows are decoder/interface ceilings and are not compared on the brain-validation Pareto frontier.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    rows = [
        row
        for run_dir in sorted(runs_root.iterdir())
        if run_dir.is_dir() and run_dir.name.startswith(PROGRAM_PREFIXES)
        for row in [extract_run(run_dir)]
        if row is not None
    ]
    if args.exclude_known_text_oracle:
        rows = [row for row in rows if row["scope"] != "known_text_oracle"]
    rows.sort(
        key=lambda row: (
            row["scope"],
            -sort_value(row["content_overlap_f1"], default=-1.0),
            -sort_value(row["word_overlap_f1"], default=-1.0),
            sort_value(row["generated_mean_rank"], default=float("inf")),
        )
    )
    mark_pareto(rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"runs": rows}, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0]) if rows else ["run"]
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_markdown).write_text(markdown(rows), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "output_json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
