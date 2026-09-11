#!/usr/bin/env python
"""Rank a saved target-free decoding sweep by informative content metrics.

Word weights and the high-frequency exclusion list are estimated from the
train11725 text only.  Validation targets are used solely for reporting the
already-generated sweep records; test107 is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from rank_generation_examples import content_tokens
from report_informative_content_overlap import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--common-frequency", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def compact_record(index: int, record: dict, informative: dict) -> dict:
    matched = record["matched"]
    return {
        "record_index": index,
        "config": record["config"],
        "content_words_overlap": matched["content_words_overlap"],
        "words_overlap": matched["words_overlap"],
        "word_error_rate": matched["word_error_rate"],
        "idf_content_f1": informative["idf_content_f1"],
        "idf_content_precision": informative["idf_content_precision"],
        "idf_content_recall": informative["idf_content_recall"],
        "noncommon_content_f1": informative["noncommon_content_f1"],
        "noncommon_content_precision": informative["noncommon_content_precision"],
        "noncommon_content_recall": informative["noncommon_content_recall"],
        "conditional_content_margin": record.get("conditional_content_margin"),
        "deranged_content_f1_max": record.get("deranged_content_f1_max"),
        "beats_all_derangements": record.get("beats_all_derangements"),
        "most_common_generated_content": informative["most_common_generated_content"],
    }


def main() -> None:
    args = parse_args()
    sweep = json.loads(Path(args.sweep_json).read_text())
    targets = [str(value) for value in sweep["targets"]]
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
    records = []
    for index, record in enumerate(sweep["records"]):
        informative = summarize(
            [str(value) for value in record["generated"]], targets, idf, common
        )
        records.append(compact_record(index, record, informative))

    top_k = min(args.top_k, len(records))
    output = {
        "scientific_scope": "train11725-derived IDF; saved val266 sweep; test107 absent",
        "source_sweep": args.sweep_json,
        "common_frequency": args.common_frequency,
        "common_words": sorted(common),
        "top_by_idf_content_f1": sorted(
            records, key=lambda row: row["idf_content_f1"], reverse=True
        )[:top_k],
        "top_by_noncommon_content_f1": sorted(
            records, key=lambda row: row["noncommon_content_f1"], reverse=True
        )[:top_k],
        "top_by_standard_content_f1": sorted(
            records, key=lambda row: row["content_words_overlap"], reverse=True
        )[:top_k],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(destination),
        "num_records": len(records),
        "best_idf": output["top_by_idf_content_f1"][0],
        "best_noncommon": output["top_by_noncommon_content_f1"][0],
    }, indent=2))


if __name__ == "__main__":
    main()
