#!/usr/bin/env python3
"""Collect leakage-safe ELF cross-attention validation leaders."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-glob", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--baseline-content-f1", type=float, default=0.0437433181)
    parser.add_argument("--expected-eval-rows", type=int, default=266)
    return parser.parse_args()


def scalar_metrics(run_dir: Path, payload: dict) -> dict:
    quality = payload.get("generation_quality", {})
    retrieval = payload.get("generation_t5_retrieval", {})
    checkpoint = run_dir / "best.pt"
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "split": payload.get("split"),
        "eval_rows": payload.get("eval_num_examples", payload.get("num_eval_examples")),
        "content_f1": quality.get("content_words_overlap"),
        "content_precision": quality.get("content_words_overlap_precision"),
        "content_recall": quality.get("content_words_overlap_recall"),
        "word_f1": quality.get("words_overlap"),
        "word_precision": quality.get("words_overlap_precision"),
        "word_recall": quality.get("words_overlap_recall"),
        "wer": quality.get("word_error_rate"),
        "top1": retrieval.get("top1"),
        "top5": retrieval.get("top5"),
        "mean_rank": retrieval.get("mean_rank"),
        "median_rank": retrieval.get("median_rank"),
    }


def main() -> None:
    args = parse_args()
    run_dirs = sorted(
        {
            Path(path)
            for pattern in args.run_glob
            for path in glob.glob(pattern)
            if Path(path).is_dir()
        }
    )
    records = []
    rejected = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "best_metrics.json"
        if not metrics_path.is_file():
            rejected.append({"run": run_dir.name, "reason": "missing best_metrics.json"})
            continue
        payload = json.loads(metrics_path.read_text())
        record = scalar_metrics(run_dir, payload)
        if record["split"] != "val":
            rejected.append({"run": run_dir.name, "reason": f"split={record['split']!r}"})
            continue
        if record["eval_rows"] != args.expected_eval_rows:
            rejected.append(
                {
                    "run": run_dir.name,
                    "reason": f"eval_rows={record['eval_rows']!r}",
                }
            )
            continue
        if record["content_f1"] is None:
            rejected.append({"run": run_dir.name, "reason": "missing content_f1"})
            continue
        if record["word_f1"] is None or record["wer"] is None:
            rejected.append({"run": run_dir.name, "reason": "missing word_f1/wer"})
            continue
        records.append(record)

    records.sort(key=lambda row: (-float(row["content_f1"]), float(row["wer"])))
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
        record["content_delta_vs_t5"] = (
            float(record["content_f1"]) - args.baseline_content_f1
        )
        record["beats_t5_content"] = bool(
            float(record["content_f1"]) > args.baseline_content_f1
        )

    report = {
        "baseline": {
            "name": "frozen_t5_prefix_val266",
            "content_f1": args.baseline_content_f1,
        },
        "complete_runs": len(records),
        "leader": records[0] if records else None,
        "any_content_win": any(row["beats_t5_content"] for row in records),
        "records": records,
        "rejected": rejected,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0]) if records else ["run", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report["leader"], indent=2))


if __name__ == "__main__":
    main()
