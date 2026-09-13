#!/usr/bin/env python
"""Collect the validation-only oracle known-word layout matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LAYOUTS = ("none", "center1", "center2", "center4", "dispersed2", "dispersed4", "all")
RUN_TEMPLATE = "meg_elfm_lexwinner_oracleword_eval_{layout}_seed49_20260911"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/data/engs-pnpl/glandau/elf-runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_row(runs_root: Path, layout: str) -> dict[str, object]:
    path = runs_root / RUN_TEMPLATE.format(layout=layout) / "best_metrics.json"
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    adherence = metrics.get("oracle_known_word_adherence") or {}
    known = metrics.get("oracle_known_words") or {}
    word_f1 = float(quality.get("words_overlap", 0.0))
    content_f1 = float(quality.get("content_words_overlap", 0.0))
    return {
        "layout": layout,
        "known_words_mean": known.get("eval_words_per_row_mean"),
        "content_f1": content_f1,
        "word_f1": word_f1,
        "word_content_sum": word_f1 + content_f1,
        "wer": quality.get("word_error_rate"),
        "known_word_recall": adherence.get("known_word_recall"),
        "rows_all_known_words": adherence.get("rows_all_supplied_fraction"),
        "t5_top1": retrieval.get("top1"),
        "t5_top5": retrieval.get("top5"),
        "t5_mean_rank": retrieval.get("mean_rank"),
        "metrics_path": str(path),
    }


def format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = [load_row(args.runs_root, layout) for layout in LAYOUTS]
    baseline = rows[0]
    for row in rows:
        row["delta_word_f1"] = float(row["word_f1"]) - float(baseline["word_f1"])
        row["delta_content_f1"] = float(row["content_f1"]) - float(baseline["content_f1"])
        row["delta_sum"] = float(row["word_content_sum"]) - float(baseline["word_content_sum"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "oracle_word_layouts.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    columns = list(rows[0])
    with (args.output_dir / "oracle_word_layouts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    table_columns = (
        "layout",
        "known_words_mean",
        "content_f1",
        "word_f1",
        "word_content_sum",
        "delta_sum",
        "wer",
        "known_word_recall",
        "t5_top1",
        "t5_top5",
    )
    lines = [
        "# MEG + oracle known-word flow-matching ceiling",
        "",
        "> **Leakage warning:** exact validation target words are supplied as conditions. "
        "These are oracle ceilings, not brain-conditioned D'Ascoli results.",
        "",
        "| " + " | ".join(table_columns) + " |",
        "|" + "|".join("---" if column == "layout" else "---:" for column in table_columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[column]) for column in table_columns) + " |")
    lines.extend(
        [
            "",
            "All layouts use the same frozen lexical-composite winner, trained word-fusion "
            "checkpoint, 110 validation rows, generation seed, and 110 retrieval candidates. "
            "Protected test26 is absent.",
            "",
        ]
    )
    (args.output_dir / "MEG_ORACLE_WORD_CEILING_2026-09-11.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
