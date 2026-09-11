#!/usr/bin/env python3
"""Select a completed ELF run using a generation metric.

The script is intentionally small so dependent ARC jobs can promote a winner
without consulting W&B or silently falling back to an older checkpoint.  Each
candidate is supplied as ``LABEL,MODEL,RUN_DIR``.  Missing/incomplete candidates
are skipped, but selection fails if none has both metrics and a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL,MODEL,RUN_DIR",
        help="Candidate label, ELF model name, and run directory.",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "word_content_mean",
            "word_overlap",
            "content_word_overlap",
            "negative_wer",
            "generation_t5_top5",
        ],
        default="word_content_mean",
    )
    parser.add_argument(
        "--checkpoint-name",
        action="append",
        default=[],
        help="Accepted checkpoint basename, in preference order.",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        dest="output_format",
    )
    return parser.parse_args()


def generation_summary(metrics: dict) -> dict:
    word_overlap = metrics.get("word_overlap") or {}
    summary = word_overlap.get("summary") or {}
    quality = metrics.get("generation_quality") or {}
    return {**summary, **quality}


def metric_value(metrics: dict, metric: str) -> float:
    summary = generation_summary(metrics)
    if metric == "word_content_mean":
        word = float(summary.get("words_overlap", 0.0))
        content = float(summary.get("content_words_overlap", 0.0))
        return 0.5 * (word + content)
    if metric == "word_overlap":
        return float(summary.get("words_overlap", 0.0))
    if metric == "content_word_overlap":
        return float(summary.get("content_words_overlap", 0.0))
    if metric == "negative_wer":
        if "word_error_rate" not in summary:
            raise KeyError("word_error_rate")
        return -float(summary["word_error_rate"])
    if metric == "generation_t5_top5":
        retrieval = metrics.get("generation_t5_retrieval") or {}
        return float(retrieval.get("top5", 0.0))
    raise ValueError(f"Unsupported metric: {metric}")


def checkpoint_path(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    checkpoint_names = args.checkpoint_name or ["best.pt", "best_adapter.pt"]
    valid: list[dict] = []
    rejected: list[str] = []

    for encoded in args.candidate:
        parts = encoded.split(",", 2)
        if len(parts) != 3 or not all(parts):
            raise SystemExit(f"Invalid --candidate {encoded!r}; expected LABEL,MODEL,RUN_DIR")
        label, model, run_dir_text = parts
        run_dir = Path(run_dir_text).expanduser().resolve()
        metrics_path = run_dir / "best_metrics.json"
        checkpoint = checkpoint_path(run_dir, checkpoint_names)
        if not metrics_path.is_file() or checkpoint is None:
            rejected.append(
                f"{label}: metrics={metrics_path.is_file()} checkpoint={checkpoint is not None}"
            )
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
            score = metric_value(metrics, args.metric)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            rejected.append(f"{label}: invalid metrics ({exc})")
            continue
        summary = generation_summary(metrics)
        valid.append(
            {
                "label": label,
                "model": model,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "metric": args.metric,
                "score": score,
                "word_overlap": float(summary.get("words_overlap", 0.0)),
                "content_word_overlap": float(summary.get("content_words_overlap", 0.0)),
                "word_error_rate": float(summary.get("word_error_rate", float("inf"))),
            }
        )

    if not valid:
        details = "; ".join(rejected) or "no candidates"
        raise SystemExit(f"No completed candidate is promotable: {details}")

    winner = max(
        valid,
        key=lambda row: (
            row["score"],
            row["content_word_overlap"],
            row["word_overlap"],
            -row["word_error_rate"],
        ),
    )
    winner["num_valid_candidates"] = len(valid)
    winner["rejected_candidates"] = rejected

    if args.output_format == "json":
        print(json.dumps(winner, indent=2, sort_keys=True))
    else:
        fields = [
            winner["label"],
            winner["model"],
            winner["run_dir"],
            winner["checkpoint"],
            f"{winner['score']:.12g}",
        ]
        print("\t".join(fields))


if __name__ == "__main__":
    main()
