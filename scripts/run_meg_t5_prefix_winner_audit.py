#!/usr/bin/env python3
"""Run the standard permutation audit for the selected MEG T5-prefix arm."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from analyze_generation_permutation import main as permutation_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    winner_name = summary["winner"]
    if not winner_name:
        raise ValueError("Summary does not contain a selected brain-conditioned winner.")
    row = next(item for item in summary["rows"] if item["name"] == winner_name)
    metrics = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_csv = args.output_dir / "meg_t5_prefix_winner_val110.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "target", "generated"])
        writer.writeheader()
        for index, (target, generated) in enumerate(
            zip(metrics["targets"], metrics["generated"])
        ):
            writer.writerow({"index": index, "target": target, "generated": generated})

    command = [
        "analyze_generation_permutation.py",
        "--input-csv", str(input_csv),
        "--run-tag", winner_name,
        "--checkpoint", row["checkpoint"],
        "--permutations", "100000",
        "--seed", "20260907",
        "--output-json", str(args.output_dir / "meg_t5_prefix_winner_significance.json"),
        "--output-csv", str(args.output_dir / "meg_t5_prefix_winner_ranked.csv"),
        "--bertscore",
        "--bertscore-model", "roberta-large",
        "--bertscore-layers", "17",
        "--bertscore-device", "cuda",
        "--bertscore-batch-size", "64",
    ]
    generation_retrieval = row.get("generated_retrieval") or {}
    semantic_retrieval = row.get("semantic_retrieval") or {}
    for option, value in (
        ("--generated-top1", generation_retrieval.get("top1")),
        ("--generated-top5", generation_retrieval.get("top5")),
        ("--semantic-top1", semantic_retrieval.get("top1")),
        ("--semantic-top5", semantic_retrieval.get("top5")),
    ):
        if value is not None:
            command.extend([option, str(value)])
    sys.argv = command
    permutation_main()


if __name__ == "__main__":
    main()
