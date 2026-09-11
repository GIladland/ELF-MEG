#!/usr/bin/env python
"""Build a leakage-safe overnight fMRI-to-text validation leaderboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from rank_generation_examples import content_tokens
from report_informative_content_overlap import summarize


LOCKED_CONTENT_F1 = 0.023116269572483635


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=METRICS_JSON:BERTSCORE_JSON",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.text_npz, allow_pickle=True) as archive:
        sentences = [str(value) for value in archive["sentence"].tolist()]
    train = sentences[:11725]
    document_frequency: Counter[str] = Counter()
    for sentence in train:
        document_frequency.update(set(content_tokens(sentence)))
    idf = {
        word: math.log((len(train) + 1) / (frequency + 1)) + 1.0
        for word, frequency in document_frequency.items()
    }
    common = {
        word for word, frequency in document_frequency.items()
        if frequency / len(train) >= 0.02
    }

    rows = []
    missing = []
    target_contract = None
    for specification in args.candidate:
        name, paths = specification.split("=", 1)
        metrics_name, bert_name = paths.split(":", 1)
        metrics_path = Path(metrics_name)
        bert_path = Path(bert_name)
        if not metrics_path.exists() or not bert_path.exists():
            missing.append({
                "name": name,
                "metrics": str(metrics_path),
                "bertscore": str(bert_path),
            })
            if args.allow_missing:
                continue
            raise FileNotFoundError(missing[-1])
        metrics = json.loads(metrics_path.read_text())
        bert = json.loads(bert_path.read_text())
        generation_quality = metrics.get("generation_quality", {})
        word_overlap_summary = metrics.get("word_overlap", {}).get("summary", {})

        def metric(name: str, default=None):
            """Read both extracted-sweep and legacy generation metric schemas."""
            if name in metrics:
                return metrics[name]
            if name in generation_quality:
                return generation_quality[name]
            return word_overlap_summary.get(name, default)

        targets = [str(value) for value in metrics["targets"]]
        generated = [str(value) for value in metrics["generated"]]
        if target_contract is None:
            target_contract = targets
        elif targets != target_contract:
            raise ValueError(f"Target contract differs for {name}.")
        informative = summarize(generated, targets, idf, common)
        bert_summary = bert.get("summary", bert)
        margin = metrics.get("conditional_content_margin")
        beats = metrics.get("beats_all_derangements")
        content_f1 = float(metric("content_words_overlap"))
        eligible = (
            content_f1 > LOCKED_CONTENT_F1
            and beats is True
            and margin is not None
            and float(margin) > 0.0
        )
        rows.append({
            "name": name,
            "promotion_eligible": eligible,
            "content_f1": content_f1,
            "content_precision": metric("content_words_overlap_precision"),
            "content_recall": metric("content_words_overlap_recall"),
            "word_f1": float(metric("words_overlap")),
            "wer": float(metric("word_error_rate")),
            "bertscore_f1": float(bert_summary["bertscore_f1"]),
            "idf_content_f1": informative["idf_content_f1"],
            "noncommon_content_f1": informative["noncommon_content_f1"],
            "conditional_content_margin": margin,
            "deranged_content_f1_max": metrics.get("deranged_content_f1_max"),
            "beats_all_derangements": beats,
            "checkpoint": metrics.get("checkpoint"),
            "decoding_config": metrics.get("decoding_config"),
            "metrics_json": str(metrics_path),
            "bertscore_json": str(bert_path),
        })

    content_first = sorted(
        rows,
        key=lambda row: (
            not row["promotion_eligible"],
            -row["content_f1"],
            -row["bertscore_f1"],
            row["wer"],
        ),
    )
    noncommon_first = sorted(
        rows,
        key=lambda row: (
            not row["promotion_eligible"],
            -row["noncommon_content_f1"],
            -row["idf_content_f1"],
            row["wer"],
        ),
    )
    output = {
        "scientific_scope": "train11725-derived IDF; val266 only; test107 absent",
        "locked_content_f1": LOCKED_CONTENT_F1,
        "selection_priority": "eligible, content F1, BERTScore F1, lower WER",
        "common_words": sorted(common),
        "missing_candidates": missing,
        "content_first": content_first,
        "noncommon_first": noncommon_first,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in content_first[0] if key != "decoding_config"] if content_first else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fields}
            for row in content_first
        )
    print(json.dumps({
        "output_json": str(output_path),
        "output_csv": str(csv_path),
        "num_candidates": len(rows),
        "missing": len(missing),
        "content_leader": content_first[0] if content_first else None,
        "noncommon_leader": noncommon_first[0] if noncommon_first else None,
    }, indent=2))


if __name__ == "__main__":
    main()
