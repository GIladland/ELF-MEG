#!/usr/bin/env python3
"""Freeze the best exact-ADA T5-prefix oracle before brain-stage training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = []
    for metrics_path in sorted(args.root.glob("oracle_*/oracle_best_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        matched = payload["best"]["matched"]
        word_f1 = float(matched["words_overlap"])
        content_f1 = float(matched["content_words_overlap"])
        candidates.append(
            {
                "run_dir": str(metrics_path.parent),
                "metrics": str(metrics_path),
                "checkpoint": str(metrics_path.parent / "oracle_best.pt"),
                "word_f1": word_f1,
                "content_f1": content_f1,
                "wer": float(matched["word_error_rate"]),
                "score": 0.5 * (word_f1 + content_f1),
                "epoch": int(payload["best"]["epoch"]),
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No completed oracle metrics found below {args.root}")
    candidates.sort(key=lambda row: (row["score"], -row["wer"]), reverse=True)
    winner = candidates[0]
    checkpoint = Path(winner["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, args.output_dir / "oracle_best.pt")
    shutil.copy2(Path(winner["metrics"]), args.output_dir / "oracle_best_metrics.json")
    result = {
        "contract": (
            "Exact ADA oracle selected on val110; its training corpus is recorded "
            "in winner.metrics args.oracle_augmentation_npz. No predicted test26 "
            "vector or test26 target was loaded."
        ),
        "selection": "maximize 0.5 * (word_f1 + content_f1), then minimize WER",
        "winner": winner,
        "candidates": candidates,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
