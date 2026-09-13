#!/usr/bin/env python
"""Collect D'Ascoli sentence-source validation runs into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_generation_permutation import content_words, counted_f1, edit_distance, words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--glob", default="qc4wyals_elfb_dascoli_eval_*")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def number(mapping: dict, key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def fmt(value: float | None, digits: int = 5) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    rows: list[dict] = []
    for run_dir in sorted(runs_root.glob(args.glob)):
        metrics_path = run_dir / "best_metrics.json"
        config_path = run_dir / "run_config.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        quality = metrics.get("generation_quality") or {}
        retrieval = metrics.get("generation_t5_retrieval") or {}
        source = metrics.get("dascoli_simulation") or {}
        word_f1 = number(quality, "words_overlap")
        content_f1 = number(quality, "content_words_overlap")
        source_sentences = source.get("source_sentences") or []
        targets = metrics.get("targets") or []
        direct_source_word_f1 = None
        direct_source_content_f1 = None
        direct_source_wer = None
        if len(source_sentences) == len(targets) and targets:
            direct_source_word_f1 = sum(
                counted_f1(words(candidate), words(target))[0]
                for candidate, target in zip(source_sentences, targets)
            ) / len(targets)
            direct_source_content_f1 = sum(
                counted_f1(content_words(candidate), content_words(target))[0]
                for candidate, target in zip(source_sentences, targets)
            ) / len(targets)
            reference_words = sum(len(words(target)) for target in targets)
            direct_source_wer = sum(
                edit_distance(words(target), words(candidate))
                for candidate, target in zip(source_sentences, targets)
            ) / max(1, reference_words)
        rows.append(
            {
                "run": run_dir.name,
                "checkpoint": config.get("init_e2e_checkpoint", ""),
                "mode": source.get("eval_mode", config.get("dascoli_eval_mode", "")),
                "semantic_scale": number(source, "eval_semantic_scale"),
                "confidence_floor": number(source, "confidence_floor"),
                "nominal_accuracy": number(source, "eval_accuracy_probability"),
                "realized_accuracy": number(source, "realized_eval_position_accuracy"),
                "paired_source_accuracy": number(source, "paired_source_position_accuracy"),
                "source_word_f1": number(source, "paired_source_word_f1"),
                "direct_source_content_f1": direct_source_content_f1,
                "direct_source_word_f1": direct_source_word_f1,
                "direct_source_word_content_sum": (
                    direct_source_content_f1 + direct_source_word_f1
                    if direct_source_content_f1 is not None
                    and direct_source_word_f1 is not None
                    else None
                ),
                "direct_source_wer": direct_source_wer,
                "mean_confidence": number(source, "mean_eval_word_confidence"),
                "content_f1": content_f1,
                "word_f1": word_f1,
                "word_content_sum": (
                    word_f1 + content_f1
                    if word_f1 is not None and content_f1 is not None
                    else None
                ),
                "wer": number(quality, "word_error_rate"),
                "well_structured": number(quality, "well_structured_sentence"),
                "top1": number(retrieval, "top1"),
                "top5": number(retrieval, "top5"),
                "metrics_path": str(metrics_path),
            }
        )

    if not rows:
        raise SystemExit(f"No completed runs matched {runs_root / args.glob}")

    semantic_only_by_checkpoint = {
        str(row["checkpoint"]): row
        for row in rows
        if row["mode"] == "semantic_only"
    }
    for row in rows:
        baseline = semantic_only_by_checkpoint.get(str(row["checkpoint"]))
        for metric in (
            "content_f1",
            "word_f1",
            "word_content_sum",
            "top1",
            "top5",
        ):
            candidate_value = row[metric]
            baseline_value = baseline[metric] if baseline is not None else None
            row[f"delta_{metric}_vs_semantic_only"] = (
                candidate_value - baseline_value
                if candidate_value is not None and baseline_value is not None
                else None
            )
        candidate_wer = row["wer"]
        baseline_wer = baseline["wer"] if baseline is not None else None
        row["favorable_delta_wer_vs_semantic_only"] = (
            baseline_wer - candidate_wer
            if candidate_wer is not None and baseline_wer is not None
            else None
        )
    rows.sort(
        key=lambda row: (
            str(row["checkpoint"]),
            str(row["mode"]),
            -1.0 if row["nominal_accuracy"] is None else row["nominal_accuracy"],
        )
    )
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Confidence-weighted D'Ascoli sentence-source flow results",
        "",
        "Target-derived D'Ascoli simulations on the established 110-row validation tail. "
        "These are sensitivity/ceiling experiments, not measured brain-decoding results. "
        "Protected test26 is absent.",
        "",
        "| Run | Mode | Sem. scale | Conf. floor | Nominal acc. | Paired source acc. | Direct source sum | Output content F1 | Output word F1 | Sum | Δ sum vs semantic-only | WER ↓ | Top-1 | Top-5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {mode} | {semantic_scale} | {confidence_floor} | {acc} | "
            "{paired} | {source_sum} | {content} | "
            "{word} | {total} | {delta_total} | {wer} | {top1} | {top5} |".format(
                run=row["run"],
                mode=row["mode"],
                semantic_scale=fmt(row["semantic_scale"], 3),
                confidence_floor=fmt(row["confidence_floor"], 2),
                acc=fmt(row["nominal_accuracy"], 2),
                paired=fmt(row["paired_source_accuracy"]),
                source_sum=fmt(row["direct_source_word_content_sum"]),
                content=fmt(row["content_f1"]),
                word=fmt(row["word_f1"]),
                total=fmt(row["word_content_sum"]),
                delta_total=fmt(row["delta_word_content_sum_vs_semantic_only"]),
                wer=fmt(row["wer"]),
                top1=fmt(row["top1"]),
                top5=fmt(row["top5"]),
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} runs to {output_csv} and {output_md}")


if __name__ == "__main__":
    main()
