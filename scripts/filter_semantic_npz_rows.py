#!/usr/bin/env python
"""Filter semantic-vector NPZ rows by regex-matching row metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-row-regex",
        action="append",
        default=[],
        help="Keep rows where any regex matches the serialized row metadata.",
    )
    parser.add_argument(
        "--include-field-regex",
        action="append",
        default=[],
        metavar="KEY=REGEX",
        help="Keep rows where a metadata field matches. Multiple entries are OR'ed.",
    )
    parser.add_argument(
        "--exclude-row-regex",
        action="append",
        default=[],
        help="Drop rows where any regex matches the serialized row metadata.",
    )
    parser.add_argument(
        "--exclude-field-regex",
        action="append",
        default=[],
        metavar="KEY=REGEX",
        help="Drop rows where a metadata field matches. Multiple entries are OR'ed.",
    )
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def row_text(row: Any) -> str:
    if isinstance(row, dict):
        try:
            return json.dumps(row, sort_keys=True, default=str)
        except TypeError:
            return str(row)
    return str(row)


def parse_field_patterns(values: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    patterns = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=REGEX for field pattern, got {value!r}")
        key, pattern = value.split("=", 1)
        if not key:
            raise ValueError(f"Field pattern has empty key: {value!r}")
        patterns.append((key, re.compile(pattern, re.IGNORECASE)))
    return patterns


def field_match(row: Any, patterns: list[tuple[str, re.Pattern[str]]]) -> bool:
    if not patterns or not isinstance(row, dict):
        return False
    for key, pattern in patterns:
        if pattern.search(str(row.get(key, "") or "")):
            return True
    return False


def group_for_row(row: Any) -> str:
    if isinstance(row, dict):
        for key in ("corpus", "dataset", "task", "source", "book"):
            value = str(row.get(key, "") or "")
            if value:
                return "Sherlock" if value.startswith("Sherlock") else value
    text = row_text(row)
    return "Sherlock" if "Sherlock" in text else "unknown"


def load_schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    value = str(data[schema_key].tolist())
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    args = parse_args()
    include_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.include_row_regex]
    exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.exclude_row_regex]
    include_field_patterns = parse_field_patterns(args.include_field_regex)
    exclude_field_patterns = parse_field_patterns(args.exclude_field_regex)

    data = np.load(args.input, allow_pickle=True)
    if args.sentence_key not in data.files:
        raise KeyError(f"{args.input} has no sentence key {args.sentence_key!r}")
    if args.rows_key not in data.files:
        raise KeyError(f"{args.input} has no row metadata key {args.rows_key!r}")

    sentences = _strings(data[args.sentence_key])
    rows = data[args.rows_key]
    n = len(sentences)
    if len(rows) != n:
        raise ValueError(f"Row count {len(rows)} does not match sentence count {n}.")
    if args.input_key in data.files and len(data[args.input_key]) != n:
        raise ValueError(f"Input embedding count does not match sentence count for {args.input_key!r}.")

    keep = []
    for idx, row in enumerate(rows.tolist()):
        text = row_text(row)
        include = (
            not include_patterns
            and not include_field_patterns
            or any(pattern.search(text) for pattern in include_patterns)
            or field_match(row, include_field_patterns)
        )
        exclude = any(pattern.search(text) for pattern in exclude_patterns) or field_match(row, exclude_field_patterns)
        if include and not exclude:
            keep.append(idx)

    keep_idx = np.asarray(keep, dtype=np.int64)
    output_arrays = {}
    for key in data.files:
        array = data[key]
        if array.shape[:1] == (n,):
            output_arrays[key] = array[keep_idx]
        else:
            output_arrays[key] = array

    kept_rows = rows[keep_idx]
    kept_sentences = np.asarray(sentences, dtype=object)[keep_idx]
    summary = {
        "input": args.input,
        "output": args.output,
        "include_row_regex": args.include_row_regex,
        "include_field_regex": args.include_field_regex,
        "exclude_row_regex": args.exclude_row_regex,
        "exclude_field_regex": args.exclude_field_regex,
        "input_examples": int(n),
        "kept_examples": int(len(keep_idx)),
        "dropped_examples": int(n - len(keep_idx)),
        "group_counts": dict(Counter(group_for_row(row) for row in kept_rows.tolist())),
        "sample_sentences": kept_sentences[: args.max_examples].tolist(),
        "input_schema": load_schema(data, args.schema_key),
    }
    output_arrays[args.schema_key] = np.asarray(json.dumps(summary))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **output_arrays)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
