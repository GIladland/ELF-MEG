#!/usr/bin/env python3
"""Select and publish the best completed context-distillation checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = []
    for run_dir in args.run_dir:
        metrics_path = run_dir / "best_metrics.json"
        checkpoint_path = run_dir / "best_adapter.pt"
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            print(f"skip incomplete run: {run_dir}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        teacher = metrics.get("teacher_context") or {}
        cosine = teacher.get("teacher_context_cosine")
        if cosine is None:
            print(f"skip run without teacher-context cosine: {run_dir}")
            continue
        quality = metrics.get("generation_quality") or {}
        candidates.append(
            {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "teacher_context_cosine": float(cosine),
                "word_f1": float(quality.get("words_overlap", 0.0)),
                "content_f1": float(quality.get("content_words_overlap", 0.0)),
            }
        )
    if not candidates:
        raise RuntimeError("No completed context-distillation checkpoints were found")
    candidates.sort(
        key=lambda row: (
            row["teacher_context_cosine"], row["word_f1"] + row["content_f1"]
        ),
        reverse=True,
    )
    winner = candidates[0]
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["checkpoint"], args.output_checkpoint)
    result = {"selection_metric": "teacher_context_cosine", "winner": winner, "candidates": candidates}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
