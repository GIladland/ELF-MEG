#!/usr/bin/env python
"""Rank ordered/content ELF diffusion screen outputs with fixed word caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cap_generated_text_words import edit_distance, overlap_summary, words


BASELINE = {
    "content_words_overlap": 0.043743318085423345,
    "words_overlap": 0.08267770173628086,
    "word_error_rate": 0.987218045112782,
}


def evaluate(generated: list[str], targets: list[str]) -> dict[str, float | int]:
    generated_tokens = [words(text) for text in generated]
    target_tokens = [words(text) for text in targets]
    overlap = overlap_summary(generated_tokens, target_tokens)
    errors = sum(
        edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(target_tokens, generated_tokens)
    )
    overlap["word_error_rate"] = errors / max(
        1, sum(len(tokens) for tokens in target_tokens)
    )
    overlap["edit_errors"] = int(errors)
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="fmri_minilm_elf_ordered_content_*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument(
        "--baseline-content", type=float,
        default=BASELINE["content_words_overlap"],
    )
    parser.add_argument(
        "--baseline-word", type=float,
        default=BASELINE["words_overlap"],
    )
    parser.add_argument(
        "--baseline-wer", type=float,
        default=BASELINE["word_error_rate"],
    )
    args = parser.parse_args()
    baseline = {
        "content_words_overlap": args.baseline_content,
        "words_overlap": args.baseline_word,
        "word_error_rate": args.baseline_wer,
    }

    rows: list[dict] = []
    for run_dir in sorted(args.root.glob(args.pattern)):
        metrics_path = run_dir / "eval_step_000000.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        targets = [str(value) for value in payload["targets"]]
        original = [str(value) for value in payload["generated"]]
        for cap in [0, *args.caps]:
            generated = original
            if cap:
                generated = [" ".join(words(text)[:cap]) for text in original]
            metrics = evaluate(generated, targets)
            clears = {
                "content": metrics["content_words_overlap"]
                > baseline["content_words_overlap"],
                "word": metrics["words_overlap"] > baseline["words_overlap"],
                "wer": metrics["word_error_rate"] < baseline["word_error_rate"],
            }
            rows.append(
                {
                    "run": run_dir.name,
                    "metrics_json": str(metrics_path),
                    "word_cap": cap,
                    **metrics,
                    "clears": clears,
                    "clears_all_lexical_wer": all(clears.values()),
                }
            )

    rows.sort(
        key=lambda row: (
            not row["clears_all_lexical_wer"],
            -sum(row["clears"].values()),
            -float(row["content_words_overlap"]),
            -float(row["words_overlap"]),
            float(row["word_error_rate"]),
        )
    )
    output = {
        "contract": "val64 screen only; test107 absent",
        "baseline_thresholds": baseline,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows[:20], indent=2))


if __name__ == "__main__":
    main()
