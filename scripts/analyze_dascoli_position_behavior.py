#!/usr/bin/env python
"""Measure whether ELF retains correct or repairs wrong D'Ascoli word positions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument(
        "--generation",
        action="append",
        required=True,
        help="LABEL=ranked_generations.csv; may be repeated.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def main() -> None:
    args = parse_args()
    source_payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    sources = list(map(str, source_payload["generated"]))
    targets = list(map(str, source_payload["targets"]))
    if len(sources) != len(targets):
        raise ValueError("Source and target row counts differ")

    results = []
    for specification in args.generation:
        if "=" not in specification:
            raise ValueError(f"Expected LABEL=CSV, got {specification!r}")
        label, path_text = specification.split("=", 1)
        path = Path(path_text)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_index = {int(row["index"]): row for row in rows}
        if set(by_index) != set(range(len(targets))):
            raise ValueError(f"{path} does not contain each row index exactly once")

        source_correct = 0
        source_wrong = 0
        retained_correct = 0
        corrected_wrong = 0
        copied_wrong = 0
        generated_position_correct = 0
        generated_lengths = []
        for index, (source_text, target_text) in enumerate(zip(sources, targets)):
            source_tokens = words(source_text)
            target_tokens = words(target_text)
            generated_tokens = words(by_index[index]["generated"])
            if words(by_index[index]["target"]) != target_tokens:
                raise ValueError(f"Target mismatch at row {index} in {path}")
            generated_lengths.append(len(generated_tokens))
            for position, target_token in enumerate(target_tokens):
                source_token = (
                    source_tokens[position] if position < len(source_tokens) else None
                )
                generated_token = (
                    generated_tokens[position]
                    if position < len(generated_tokens)
                    else None
                )
                generated_position_correct += generated_token == target_token
                if source_token == target_token:
                    source_correct += 1
                    retained_correct += generated_token == target_token
                else:
                    source_wrong += 1
                    corrected_wrong += generated_token == target_token
                    copied_wrong += generated_token == source_token

        total_positions = source_correct + source_wrong
        results.append(
            {
                "label": label,
                "generation_csv": str(path),
                "num_rows": len(targets),
                "num_positions": total_positions,
                "source_correct_positions": source_correct,
                "source_wrong_positions": source_wrong,
                "generated_exact_position_accuracy": generated_position_correct
                / max(1, total_positions),
                "correct_source_position_retention": retained_correct
                / max(1, source_correct),
                "wrong_source_position_correction": corrected_wrong
                / max(1, source_wrong),
                "wrong_source_position_copy": copied_wrong / max(1, source_wrong),
                "mean_generated_word_count": sum(generated_lengths)
                / max(1, len(generated_lengths)),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"contract": "exact position diagnostic; no test data", "rows": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
