#!/usr/bin/env python3
"""Apply the fixed ten-word cap and summarize the pure ELF-M decode sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.cap_generated_text_words import (
    cap_text,
    edit_distance,
    overlap_summary,
    words,
)


CONFIGS = [
    (steps, cfg, cfg_tag)
    for cfg, cfg_tag in ((0.5, "0p5"), (1.0, "1p0"), (1.5, "1p5"))
    for steps in (16, 32, 64)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def cap_payload(source: dict, source_path: Path) -> dict:
    targets = [str(value) for value in source["targets"]]
    original = [str(value) for value in source["generated"]]
    capped = [cap_text(value, 10) for value in original]
    target_tokens = [words(value) for value in targets]
    capped_tokens = [words(value) for value in capped]
    errors = [
        edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(target_tokens, capped_tokens)
    ]
    reference_words = sum(len(value) for value in target_tokens)
    overlap = overlap_summary(capped_tokens, target_tokens)
    wer = sum(errors) / max(1, reference_words)
    return {
        "source": str(source_path),
        "contract": "fixed evaluator-lexical-token generated-word cap",
        "max_words": 10,
        "num_examples": len(targets),
        "reference_labels_used_to_choose_or_apply_cap": False,
        "word_error_rate": wer,
        "edit_errors": int(sum(errors)),
        "reference_words": int(reference_words),
        "mean_original_words": sum(len(words(value)) for value in original) / len(original),
        "mean_capped_words": sum(len(value) for value in capped_tokens) / len(capped_tokens),
        "generation_quality": {"word_error_rate": wer, **overlap},
        "targets": targets,
        "generated": capped,
        "original_generated": original,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for steps, cfg, cfg_tag in CONFIGS:
        run_tag = f"meg_elfm_puresem_decode_s{steps}_cfg{cfg_tag}_seed49_20260912"
        run_dir = args.runs_root / run_tag
        metrics_path = run_dir / "best_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        capped = cap_payload(raw, metrics_path)
        capped_path = run_dir / "best_metrics.cap10.json"
        capped_path.write_text(json.dumps(capped, indent=2) + "\n", encoding="utf-8")
        raw_quality = raw["generation_quality"]
        cap_quality = capped["generation_quality"]
        rows.append(
            {
                "steps": steps,
                "cfg": cfg,
                "raw_word_f1": float(raw_quality["words_overlap"]),
                "raw_content_f1": float(raw_quality["content_words_overlap"]),
                "raw_sum": float(
                    raw_quality["words_overlap"] + raw_quality["content_words_overlap"]
                ),
                "raw_wer": float(raw_quality["word_error_rate"]),
                "cap10_word_f1": float(cap_quality["words_overlap"]),
                "cap10_content_f1": float(cap_quality["content_words_overlap"]),
                "cap10_sum": float(
                    cap_quality["words_overlap"] + cap_quality["content_words_overlap"]
                ),
                "cap10_wer": float(cap_quality["word_error_rate"]),
                "mean_capped_words": float(capped["mean_capped_words"]),
                "run_tag": run_tag,
                "metrics": str(metrics_path),
                "cap10_metrics": str(capped_path),
            }
        )
    rows.sort(key=lambda row: (row["cap10_sum"], -row["cap10_wer"]), reverse=True)
    control = next(row for row in rows if row["steps"] == 32 and row["cfg"] == 1.0)
    result = {
        "contract": {
            "condition": "frozen qc4wyals predicted ADA only",
            "start": "noise",
            "source_sentence": False,
            "validation_rows": 110,
            "protected_test26_loaded": False,
            "max_words": 10,
        },
        "t5_target": {"word_content_sum": 0.12493, "wer": 0.97727},
        "published_control": {
            "word_f1": 0.08071,
            "content_f1": 0.02481,
            "sum": 0.10551,
            "wer": 1.03273,
        },
        "observed_control": control,
        "winner": rows[0],
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"winner": rows[0], "control": control}, indent=2))


if __name__ == "__main__":
    main()
