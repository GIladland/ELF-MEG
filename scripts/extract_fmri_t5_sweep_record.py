#!/usr/bin/env python
"""Export one deterministic decoding-sweep record as a standalone metrics JSON.

The exported file is the contract used by reporting tools and by the final
single-checkpoint packager.  It contains one generation per validation brain
row; it is not a sampled pool or an oracle reranking result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.sweep_json)
    sweep = json.loads(source.read_text())
    records = sweep.get("records", [])
    if not 0 <= args.record_index < len(records):
        raise IndexError(
            f"record index {args.record_index} is outside [0, {len(records)})"
        )
    record = records[args.record_index]
    targets = [str(value) for value in sweep["targets"]]
    generated = [str(value) for value in record["generated"]]
    if len(generated) != len(targets):
        raise ValueError(
            f"generation/target length mismatch: {len(generated)} != {len(targets)}"
        )
    output = {
        "scientific_scope": sweep.get("scientific_scope"),
        "system_kind": (
            "single-checkpoint deterministic decoding with a fixed train-only "
            "dual lexical head; no candidate reranking"
            if record["config"].get("secondary_content_head_checkpoint")
            else "single-checkpoint deterministic decoding; no candidate reranking"
        ),
        "source_sweep_json": str(source),
        "record_index": args.record_index,
        "checkpoint": sweep.get("checkpoint"),
        "checkpoint_best": sweep.get("checkpoint_best"),
        "decoding_config": record["config"],
        **record["matched"],
        "deranged_content_f1_mean": record.get("deranged_content_f1_mean"),
        "deranged_content_f1_max": record.get("deranged_content_f1_max"),
        "conditional_content_margin": record.get("conditional_content_margin"),
        "beats_all_derangements": record.get("beats_all_derangements"),
        "targets": targets,
        "generated": generated,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "system_kind",
        "checkpoint",
        "decoding_config",
        "content_words_overlap",
        "words_overlap",
        "word_error_rate",
        "conditional_content_margin",
        "beats_all_derangements",
    )}, indent=2))


if __name__ == "__main__":
    main()
