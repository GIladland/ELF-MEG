#!/usr/bin/env python
"""Materialize the simulated D'Ascoli sentence itself as a generation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_generation_permutation import edit_distance, words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-metrics-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_metrics_json.read_text(encoding="utf-8"))
    simulation = payload.get("dascoli_simulation") or {}
    generated = simulation.get("source_sentences")
    targets = payload.get("targets")
    if not isinstance(generated, list) or not isinstance(targets, list):
        raise ValueError("Input must contain targets and dascoli_simulation.source_sentences")
    if len(generated) != len(targets):
        raise ValueError("D'Ascoli source and target row counts differ")
    reference_words = sum(len(words(target)) for target in targets)
    word_error_rate = sum(
        edit_distance(words(target), words(candidate))
        for candidate, target in zip(generated, targets)
    ) / max(1, reference_words)
    output = {
        "contract": {
            "role": "target-derived simulated D'Ascoli sentence before ELF refinement",
            "parent_metrics_json": str(args.input_metrics_json),
            "n": len(targets),
            "protected_test26": "absent",
        },
        "generated": generated,
        "targets": targets,
        "generation_quality": {"word_error_rate": word_error_rate},
        "dascoli_simulation": simulation,
    }
    if "generation_t5_retrieval" in payload:
        output["generation_t5_retrieval"] = payload["generation_t5_retrieval"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(targets)} direct source rows to {args.output_json}")


if __name__ == "__main__":
    main()
