#!/usr/bin/env python
"""Collect the D'Ascoli semantic-strength/source-trust grid as heatmap tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--glob",
        default="qc4wyals_elfb_dascoli_weightgrid_*/best_metrics.json",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def metric_matrix(
    rows: list[dict],
    *,
    key: str,
    title: str,
    semantic_scales: list[float],
    confidence_floors: list[float],
) -> list[str]:
    indexed = {
        (row["semantic_scale"], row["confidence_floor"]): row[key]
        for row in rows
    }
    lines = [
        f"### {title}",
        "",
        "| Semantic scale \\ confidence floor | "
        + " | ".join(f"{value:g}" for value in confidence_floors)
        + " |",
        "|---:|" + "---:|" * len(confidence_floors),
    ]
    for semantic_scale in semantic_scales:
        values = [indexed.get((semantic_scale, floor)) for floor in confidence_floors]
        lines.append(
            f"| {semantic_scale:g} | "
            + " | ".join("" if value is None else f"{value:.5f}" for value in values)
            + " |"
        )
    return lines


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.runs_root.glob(args.glob)):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        quality = metrics["generation_quality"]
        retrieval = metrics["generation_t5_retrieval"]
        simulation = metrics["dascoli_simulation"]
        rows.append(
            {
                "semantic_scale": float(simulation["eval_semantic_scale"]),
                "confidence_floor": float(simulation["confidence_floor"]),
                "content_f1": float(quality["content_words_overlap"]),
                "word_f1": float(quality["words_overlap"]),
                "word_content_sum": float(quality["content_words_overlap"])
                + float(quality["words_overlap"]),
                "wer": float(quality["word_error_rate"]),
                "top1": float(retrieval["top1"]),
                "top5": float(retrieval["top5"]),
                "run": metrics_path.parent.name,
                "metrics_path": str(metrics_path),
            }
        )
    if not rows:
        raise SystemExit(f"No completed grid cells matched {args.runs_root / args.glob}")
    rows.sort(key=lambda row: (row["semantic_scale"], row["confidence_floor"]))
    coordinates = [
        (row["semantic_scale"], row["confidence_floor"]) for row in rows
    ]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("Duplicate semantic-scale/confidence-floor grid cells")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    semantic_scales = sorted({row["semantic_scale"] for row in rows})
    confidence_floors = sorted({row["confidence_floor"] for row in rows})
    best_sum = max(rows, key=lambda row: row["word_content_sum"])
    best_wer = min(rows, key=lambda row: row["wer"])
    best_top1 = max(rows, key=lambda row: row["top1"])
    best_top5 = max(rows, key=lambda row: row["top5"])
    lines = [
        "# D'Ascoli sentence/semantic weight grid",
        "",
        "Target-derived 49.82%-accurate simulated full sentences on the fixed 110-row "
        "validation contract. Protected test26 is absent.",
        "",
        f"- Best word+content sum: `{best_sum['word_content_sum']:.5f}` at semantic "
        f"scale `{best_sum['semantic_scale']:g}`, confidence floor "
        f"`{best_sum['confidence_floor']:g}`.",
        f"- Best WER: `{best_wer['wer']:.5f}` at semantic scale "
        f"`{best_wer['semantic_scale']:g}`, confidence floor "
        f"`{best_wer['confidence_floor']:g}`.",
        f"- Best Top-1: `{best_top1['top1']:.5f}` at semantic scale "
        f"`{best_top1['semantic_scale']:g}`, confidence floor "
        f"`{best_top1['confidence_floor']:g}`.",
        f"- Best Top-5: `{best_top5['top5']:.5f}` at semantic scale "
        f"`{best_top5['semantic_scale']:g}`, confidence floor "
        f"`{best_top5['confidence_floor']:g}`.",
        "",
    ]
    for key, title in (
        ("word_content_sum", "Word F1 + content F1 (higher is better)"),
        ("wer", "WER (lower is better)"),
        ("top1", "Generated-text retrieval Top-1 (higher is better)"),
        ("top5", "Generated-text retrieval Top-5 (higher is better)"),
    ):
        lines.extend(
            metric_matrix(
                rows,
                key=key,
                title=title,
                semantic_scales=semantic_scales,
                confidence_floors=confidence_floors,
            )
        )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} grid cells to {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
