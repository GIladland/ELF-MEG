#!/usr/bin/env python3
"""Aggregate repeated ELF generation evaluations across fixed random seeds."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_globs",
        nargs="+",
        help="One or more run-directory glob patterns containing best_metrics.json.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def optional_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def scalar_metrics(run_dir: Path) -> dict[str, float]:
    metrics = optional_json(run_dir / "best_metrics.json")
    if metrics is None:
        raise FileNotFoundError(run_dir / "best_metrics.json")
    quality = metrics["generation_quality"]
    generated_retrieval = metrics.get("generation_t5_retrieval", {})
    values = {
        "exact_match": float(metrics["exact_match"]),
        "word_precision": float(quality["words_overlap_precision"]),
        "word_recall": float(quality["words_overlap_recall"]),
        "word_f1": float(quality["words_overlap"]),
        "content_precision": float(quality["content_words_overlap_precision"]),
        "content_recall": float(quality["content_words_overlap_recall"]),
        "content_f1": float(quality["content_words_overlap"]),
        "wer": float(quality["word_error_rate"]),
        "generated_top1": float(generated_retrieval["top1"]),
        "generated_top5": float(generated_retrieval["top5"]),
        "generated_mean_rank": float(generated_retrieval["mean_rank"]),
        "generated_median_rank": float(generated_retrieval["median_rank"]),
    }
    retrieval_files = sorted(run_dir.glob("retrieval_step_*.json"))
    if retrieval_files:
        retrieval = optional_json(retrieval_files[-1])
        combined = retrieval.get("combined", {}) if retrieval else {}
        for source, destination in (
            ("top1", "latent_top1"),
            ("top5", "latent_top5"),
            ("mean_rank", "latent_mean_rank"),
            ("median_rank", "latent_median_rank"),
        ):
            if source in combined:
                values[destination] = float(combined[source])
    for bert_path in (
        run_dir / "best_metrics.bertscore.json",
        run_dir / "bertscore_roberta_large_rescaled.json",
    ):
        bert = optional_json(bert_path)
        if bert is not None:
            summary = bert["summary"]
            values.update(
                bertscore_precision=float(summary["bertscore_precision"]),
                bertscore_recall=float(summary["bertscore_recall"]),
                bertscore_f1=float(summary["bertscore_f1"]),
            )
            break
    return values


def main() -> None:
    args = parse_args()
    run_dirs = sorted(
        {
            Path(match)
            for pattern in args.run_globs
            for match in glob.glob(pattern)
            if Path(match).is_dir()
        }
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched: {args.run_globs}")
    rows = [{"run": str(run_dir), **scalar_metrics(run_dir)} for run_dir in run_dirs]
    metric_names = sorted(set.intersection(*(set(row) - {"run"} for row in rows)))
    aggregate = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    result = {"num_runs": len(rows), "runs": rows, "aggregate": aggregate}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
