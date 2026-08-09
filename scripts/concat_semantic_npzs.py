#!/usr/bin/env python
"""Concatenate semantic-vector NPZs into train-first, validation-last order."""

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
    parser.add_argument("--train-npz", action="append", required=True)
    parser.add_argument(
        "--train-limit",
        action="append",
        type=int,
        default=[],
        help="Optional per-train-NPZ limit. Use 0 for all.",
    )
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--dedupe-train", action="store_true")
    parser.add_argument("--dedupe-val", action="store_true")
    parser.add_argument("--drop-train-overlap-with-val", action="store_true")
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


def normalize_sentence_key(sentence: str) -> str:
    return " ".join(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(sentence).lower()))


def load_npz(path: Path, *, input_key: str, sentence_key: str, schema_key: str, limit: int = 0) -> dict:
    data = np.load(path, allow_pickle=True)
    embeddings = data[input_key]
    sentences = np.asarray(_strings(data[sentence_key]), dtype=object)
    rows = data["rows"] if "rows" in data.files else np.asarray([{} for _ in range(len(sentences))], dtype=object)
    n = len(sentences) if limit <= 0 else min(limit, len(sentences))
    return {
        "path": str(path),
        "embeddings": embeddings[:n],
        "sentences": sentences[:n],
        "rows": rows[:n],
        "schema": _schema(data, schema_key),
    }


def keep_indices(sentences: np.ndarray, *, dedupe: bool, blocked: set[str] | None = None) -> tuple[np.ndarray, int, int]:
    seen: set[str] = set()
    keep = []
    dropped_duplicate = 0
    dropped_blocked = 0
    blocked = blocked or set()
    for idx, sentence in enumerate(sentences.tolist()):
        key = normalize_sentence_key(sentence)
        if key in blocked:
            dropped_blocked += 1
            continue
        if dedupe and key in seen:
            dropped_duplicate += 1
            continue
        seen.add(key)
        keep.append(idx)
    return np.asarray(keep, dtype=np.int64), dropped_duplicate, dropped_blocked


def main() -> None:
    args = parse_args()
    limits = list(args.train_limit)
    while len(limits) < len(args.train_npz):
        limits.append(0)
    if len(limits) > len(args.train_npz):
        raise ValueError("More --train-limit values than --train-npz values.")

    val = load_npz(
        Path(args.val_npz),
        input_key=args.input_key,
        sentence_key=args.sentence_key,
        schema_key=args.schema_key,
    )
    val_keep, val_dropped_duplicates, _ = keep_indices(
        val["sentences"],
        dedupe=args.dedupe_val,
    )
    val_embeddings = val["embeddings"][val_keep]
    val_sentences = val["sentences"][val_keep]
    val_rows = val["rows"][val_keep]
    val_set = set(normalize_sentence_key(sentence) for sentence in val_sentences.tolist())

    train_embeddings = []
    train_sentences = []
    train_rows = []
    train_sources = []
    source_summaries = []
    dropped_train_duplicates = 0
    dropped_train_val_overlap = 0
    global_seen: set[str] = set()
    for train_path, limit in zip(args.train_npz, limits):
        item = load_npz(
            Path(train_path),
            input_key=args.input_key,
            sentence_key=args.sentence_key,
            schema_key=args.schema_key,
            limit=limit,
        )
        blocked = val_set if args.drop_train_overlap_with_val else set()
        keep_local = []
        local_duplicate = 0
        local_blocked = 0
        for idx, sentence in enumerate(item["sentences"].tolist()):
            key = normalize_sentence_key(sentence)
            if key in blocked:
                local_blocked += 1
                continue
            if args.dedupe_train and key in global_seen:
                local_duplicate += 1
                continue
            global_seen.add(key)
            keep_local.append(idx)
        keep = np.asarray(keep_local, dtype=np.int64)
        train_embeddings.append(item["embeddings"][keep])
        train_sentences.append(item["sentences"][keep])
        train_rows.append(item["rows"][keep])
        train_sources.extend([str(train_path)] * len(keep))
        dropped_train_duplicates += local_duplicate
        dropped_train_val_overlap += local_blocked
        source_summaries.append(
            {
                "path": item["path"],
                "limit": limit,
                "input_count": int(len(item["sentences"])),
                "kept_count": int(len(keep)),
                "dropped_duplicates": int(local_duplicate),
                "dropped_train_val_overlap": int(local_blocked),
                "schema": item["schema"],
            }
        )

    if not train_embeddings:
        raise RuntimeError("No train arrays loaded.")
    train_embeddings_arr = np.concatenate(train_embeddings, axis=0)
    train_sentences_arr = np.concatenate(train_sentences, axis=0)
    train_rows_arr = np.concatenate(train_rows, axis=0)
    source_arr = np.asarray(train_sources + [args.val_npz] * len(val_sentences), dtype=object)
    split_arr = np.asarray(["train"] * len(train_sentences_arr) + ["val"] * len(val_sentences), dtype=object)

    embeddings = np.concatenate([train_embeddings_arr, val_embeddings], axis=0)
    sentences = np.concatenate([train_sentences_arr, val_sentences], axis=0)
    rows = np.concatenate([train_rows_arr, val_rows], axis=0)
    summary = {
        "output": args.output,
        "train_examples": int(len(train_sentences_arr)),
        "val_examples": int(len(val_sentences)),
        "total_examples": int(len(sentences)),
        "val_npz": args.val_npz,
        "train_sources": source_summaries,
        "dedupe_train": args.dedupe_train,
        "dedupe_val": args.dedupe_val,
        "drop_train_overlap_with_val": args.drop_train_overlap_with_val,
        "dropped_train_duplicates": int(dropped_train_duplicates),
        "dropped_train_val_overlap": int(dropped_train_val_overlap),
        "dropped_val_duplicates": int(val_dropped_duplicates),
        "embedding_shape": list(embeddings.shape),
        "sample_train": train_sentences_arr[:5].tolist(),
        "sample_val": val_sentences[:5].tolist(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=sentences,
        rows=rows,
        source_npz=source_arr,
        split=split_arr,
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
