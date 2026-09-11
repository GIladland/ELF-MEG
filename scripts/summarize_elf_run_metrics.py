#!/usr/bin/env python
"""Summarize best validation checkpoints from one or more ELF run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(metrics: dict, path: Path) -> dict:
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    return {
        "path": str(path),
        "step": int(metrics.get("step") or 0),
        "word_f1": quality.get("words_overlap"),
        "content_f1": quality.get("content_words_overlap"),
        "wer": quality.get("word_error_rate"),
        "generated_top1": retrieval.get("top1"),
        "generated_top5": retrieval.get("top5"),
        "generated_mean_rank": retrieval.get("mean_rank"),
        "generated_median_rank": retrieval.get("median_rank"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    output = []
    for run_dir in args.run_dirs:
        rows = []
        for path in sorted(run_dir.glob("eval_step_*.json")):
            rows.append(compact(json.loads(path.read_text(encoding="utf-8")), path))
        if not rows and (run_dir / "best_metrics.json").exists():
            path = run_dir / "best_metrics.json"
            rows.append(compact(json.loads(path.read_text(encoding="utf-8")), path))
        if not rows:
            continue
        valid_content = [row for row in rows if row["content_f1"] is not None]
        valid_word = [row for row in rows if row["word_f1"] is not None]
        valid_wer = [row for row in rows if row["wer"] is not None]
        output.append(
            {
                "run": run_dir.name,
                "num_evaluations": len(rows),
                "latest": rows[-1],
                "best_content": max(valid_content, key=lambda row: row["content_f1"]),
                "best_word": max(valid_word, key=lambda row: row["word_f1"]),
                "best_wer": min(valid_wer, key=lambda row: row["wer"]) if valid_wer else None,
            }
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
