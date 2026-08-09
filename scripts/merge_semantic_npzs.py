#!/usr/bin/env python
"""Merge semantic-vector NPZ files into one candidate NPZ."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--dedupe", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str):
    if schema_key not in data.files:
        return None
    try:
        return json.loads(str(data[schema_key].tolist()))
    except json.JSONDecodeError:
        return str(data[schema_key].tolist())


def normalize_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip()).casefold()


def main() -> None:
    args = parse_args()
    embeddings = []
    sentences = []
    rows = []
    source_npz = []
    source_summaries = []
    seen: set[str] = set()
    duplicate_count = 0

    for path_str in args.input_npz:
        path = Path(path_str)
        if not path.exists():
            print(f"Skipping missing input: {path}", flush=True)
            continue
        data = np.load(path, allow_pickle=True)
        source_sentences = np.asarray(_strings(data[args.sentence_key]), dtype=object)
        source_embeddings = data[args.input_key]
        source_rows = (
            data["rows"]
            if "rows" in data.files
            else np.asarray([{} for _ in range(len(source_sentences))], dtype=object)
        )
        keep = []
        local_duplicates = 0
        for idx, sentence in enumerate(source_sentences.tolist()):
            key = normalize_key(str(sentence))
            if args.dedupe and key in seen:
                duplicate_count += 1
                local_duplicates += 1
                continue
            seen.add(key)
            keep.append(idx)
        keep_idx = np.asarray(keep, dtype=np.int64)
        if len(keep_idx) == 0:
            continue
        embeddings.append(source_embeddings[keep_idx])
        sentences.append(source_sentences[keep_idx])
        rows.append(source_rows[keep_idx])
        source_npz.extend([str(path)] * len(keep_idx))
        source_summaries.append(
            {
                "path": str(path),
                "input_count": int(len(source_sentences)),
                "kept_count": int(len(keep_idx)),
                "dropped_duplicates": int(local_duplicates),
                "schema": _schema(data, args.schema_key),
            }
        )

    if not embeddings:
        raise RuntimeError("No input rows were merged.")
    embeddings_arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    sentences_arr = np.concatenate(sentences, axis=0)
    rows_arr = np.concatenate(rows, axis=0)
    source_arr = np.asarray(source_npz, dtype=object)
    split_arr = np.asarray(["train"] * len(sentences_arr), dtype=object)

    summary = {
        "output": args.output,
        "input_npz": args.input_npz,
        "source_summaries": source_summaries,
        "total_examples": int(len(sentences_arr)),
        "dedupe": args.dedupe,
        "dropped_duplicates": int(duplicate_count),
        "embedding_shape": list(embeddings_arr.shape),
        "sample": sentences_arr[:20].tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        input_embeddings=embeddings_arr,
        sentence=sentences_arr,
        rows=rows_arr,
        source_npz=source_arr,
        split=split_arr,
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
