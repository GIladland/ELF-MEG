#!/usr/bin/env python
"""Compare ordinary and train-IDF-weighted content overlap for generations.

This is a reporting audit, not a replacement selection metric.  IDF and the
high-frequency exclusion set are estimated from train11725 only; validation
targets never determine word weights.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from rank_generation_examples import content_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--metrics", action="append", required=True, help="NAME=JSON")
    parser.add_argument("--common-frequency", type=float, default=0.02)
    parser.add_argument(
        "--scientific-scope",
        default="train11725-derived IDF; val266 reporting; test107 absent",
        help="Explicit reporting scope written to the output artifact.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def weighted_overlap(
    hypothesis: list[str],
    reference: list[str],
    weights: dict[str, float],
) -> tuple[float, float, float]:
    hypothesis_count = Counter(hypothesis)
    reference_count = Counter(reference)
    overlap = sum(
        min(hypothesis_count[word], reference_count[word]) * weights.get(word, 1.0)
        for word in hypothesis_count.keys() & reference_count.keys()
    )
    hypothesis_total = sum(count * weights.get(word, 1.0) for word, count in hypothesis_count.items())
    reference_total = sum(count * weights.get(word, 1.0) for word, count in reference_count.items())
    precision = overlap / hypothesis_total if hypothesis_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize(
    generated: list[str],
    targets: list[str],
    idf: dict[str, float],
    common: set[str],
) -> dict:
    idf_rows = []
    rare_rows = []
    generated_frequency: Counter[str] = Counter()
    for hypothesis_text, target_text in zip(generated, targets):
        hypothesis = content_tokens(hypothesis_text)
        reference = content_tokens(target_text)
        generated_frequency.update(hypothesis)
        idf_rows.append(weighted_overlap(hypothesis, reference, idf))
        rare_rows.append(weighted_overlap(
            [word for word in hypothesis if word not in common],
            [word for word in reference if word not in common],
            {word: 1.0 for word in idf},
        ))
    idf_array = np.asarray(idf_rows, dtype=np.float64)
    rare_array = np.asarray(rare_rows, dtype=np.float64)
    return {
        "idf_content_precision": float(idf_array[:, 0].mean()),
        "idf_content_recall": float(idf_array[:, 1].mean()),
        "idf_content_f1": float(idf_array[:, 2].mean()),
        "noncommon_content_precision": float(rare_array[:, 0].mean()),
        "noncommon_content_recall": float(rare_array[:, 1].mean()),
        "noncommon_content_f1": float(rare_array[:, 2].mean()),
        "most_common_generated_content": generated_frequency.most_common(25),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.text_npz, allow_pickle=True) as archive:
        sentences = [str(value) for value in archive["sentence"].tolist()]
    train = sentences[: args.train_count]
    document_frequency: Counter[str] = Counter()
    for sentence in train:
        document_frequency.update(set(content_tokens(sentence)))
    idf = {
        word: math.log((len(train) + 1) / (frequency + 1)) + 1.0
        for word, frequency in document_frequency.items()
    }
    common = {
        word for word, frequency in document_frequency.items()
        if frequency / len(train) >= args.common_frequency
    }
    reports = {}
    target_contract = None
    for specification in args.metrics:
        name, path = specification.split("=", 1)
        payload = json.loads(Path(path).read_text())
        generated = [str(value) for value in payload["generated"]]
        targets = [str(value) for value in payload["targets"]]
        if target_contract is None:
            target_contract = targets
        elif targets != target_contract:
            raise ValueError(f"Target rows for {name!r} do not match the first system.")
        reports[name] = {
            "source": path,
            **summarize(generated, targets, idf, common),
        }
    output = {
        "scientific_scope": args.scientific_scope,
        "common_frequency": args.common_frequency,
        "common_words": sorted(common),
        "systems": reports,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
