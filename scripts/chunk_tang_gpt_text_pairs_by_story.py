#!/usr/bin/env python
"""Chunk Tang-GPT podcast rows across story boundaries.

Unlike chunk_tang_gpt_text_pairs.py, this treats each story as a continuous
stream. That avoids creating targets that inherit arbitrary 10-TR segment
edges, such as chunks ending with an unfinished phrase from the source window.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

from scripts.chunk_tang_gpt_text_pairs import (  # noqa: E402
    alpha_fraction,
    content_words,
    split_word_spans,
    valid_sequence_length,
)
from scripts.filter_semantic_npz_sentences import DEFAULT_META_RE, NOISE_RE, WORD_RE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--sequence-key", default="input_embedding_sequence")
    parser.add_argument("--sequence-mask-key", default="input_sequence_mask")
    parser.add_argument("--group-key", default="story")
    parser.add_argument("--sort-key", default="start_tr")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--target-words", type=int, default=12)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.75)
    parser.add_argument("--min-content-words", type=int, default=2)
    parser.add_argument("--exclude-regex", default=DEFAULT_META_RE)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compression", choices=["compressed", "none"], default="none")
    parser.add_argument("--sample-count", type=int, default=12)
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    value = str(data[schema_key].tolist())
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def group_rows(data: np.lib.npyio.NpzFile, group_key: str, sort_key: str | None, n_rows: int) -> list[list[int]]:
    if group_key not in data.files:
        return [list(range(n_rows))]
    groups: dict[str, list[int]] = {}
    group_values = _strings(data[group_key])
    for row_idx, group_value in enumerate(group_values):
        groups.setdefault(group_value, []).append(row_idx)

    grouped_rows = []
    for _, rows in groups.items():
        if sort_key and sort_key in data.files:
            sort_values = data[sort_key]
            rows = sorted(rows, key=lambda idx: _scalar(sort_values[idx]))
        grouped_rows.append(rows)
    return grouped_rows


def weighted_embedding_for_positions(
    sequence: np.ndarray,
    *,
    start_pos: float,
    end_pos: float,
    normalize: bool,
) -> tuple[np.ndarray, float]:
    t_count = sequence.shape[0]
    start_pos = max(0.0, min(float(t_count), start_pos))
    end_pos = max(start_pos, min(float(t_count), end_pos))
    weights = np.zeros(t_count, dtype=np.float32)
    for idx in range(t_count):
        weights[idx] = max(0.0, min(end_pos, idx + 1.0) - max(start_pos, float(idx)))
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        center = min(t_count - 1, max(0, int(round((start_pos + end_pos) / 2.0))))
        weights[center] = 1.0
        weight_sum = 1.0
    vector = (sequence.astype(np.float32) * weights[:, None]).sum(axis=0) / weight_sum
    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
    return vector.astype(np.float32), weight_sum


def estimate_tr(
    data: np.lib.npyio.NpzFile,
    *,
    row_idx: int,
    local_word_index: int,
    local_word_count: int,
    is_end: bool,
) -> float:
    if "start_tr" not in data.files or "stop_tr" not in data.files or local_word_count <= 0:
        return float("nan")
    start_tr = float(_scalar(data["start_tr"][row_idx]))
    stop_tr = float(_scalar(data["stop_tr"][row_idx]))
    offset = local_word_index + (1 if is_end else 0)
    return start_tr + (offset / local_word_count) * (stop_tr - start_tr)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_npz)
    output_path = Path(args.output)
    data = np.load(input_path, allow_pickle=True)
    if args.sentence_key not in data.files:
        raise KeyError(f"{input_path} has no sentence key {args.sentence_key!r}")
    if args.sequence_key not in data.files:
        raise KeyError(f"{input_path} has no sequence key {args.sequence_key!r}")

    sentences = _strings(data[args.sentence_key])
    sequences = np.asarray(data[args.sequence_key])
    masks = np.asarray(data[args.sequence_mask_key]) if args.sequence_mask_key in data.files else None
    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE) if args.exclude_regex else None

    chunk_vectors: list[np.ndarray] = []
    chunk_sentences: list[str] = []
    source_sentences: list[str] = []
    source_row_starts: list[int] = []
    source_row_ends: list[int] = []
    source_rows_joined: list[str] = []
    group_values_out: list[str] = []
    split_values_out: list[str] = []
    word_starts: list[int] = []
    word_ends: list[int] = []
    word_counts: list[int] = []
    sequence_starts: list[float] = []
    sequence_ends: list[float] = []
    sequence_weight_sums: list[float] = []
    start_tr_est: list[float] = []
    stop_tr_est: list[float] = []
    seen: set[str] = set()
    drop_reasons: dict[str, int] = {}
    groups_with_chunks = 0

    for rows in group_rows(data, args.group_key, args.sort_key, len(sentences)):
        words: list[str] = []
        word_rows: list[int] = []
        word_local_indices: list[int] = []
        word_local_counts: list[int] = []
        word_seq_starts: list[float] = []
        word_seq_ends: list[float] = []
        sequence_parts: list[np.ndarray] = []
        seq_cursor = 0.0
        kept_rows: list[int] = []

        for row_idx in rows:
            sentence = sentences[row_idx]
            if exclude_re and exclude_re.search(sentence):
                drop_reasons["source_excluded_regex"] = drop_reasons.get("source_excluded_regex", 0) + 1
                continue
            row_words = WORD_RE.findall(sentence)
            if not row_words:
                drop_reasons["source_no_words"] = drop_reasons.get("source_no_words", 0) + 1
                continue
            valid_t = valid_sequence_length(None if masks is None else masks[row_idx], sequences.shape[1])
            if valid_t <= 0:
                drop_reasons["source_no_valid_sequence"] = drop_reasons.get("source_no_valid_sequence", 0) + 1
                continue

            sequence_parts.append(sequences[row_idx, :valid_t])
            kept_rows.append(row_idx)
            for local_idx, word in enumerate(row_words):
                words.append(word)
                word_rows.append(row_idx)
                word_local_indices.append(local_idx)
                word_local_counts.append(len(row_words))
                word_seq_starts.append(seq_cursor + (local_idx / len(row_words)) * valid_t)
                word_seq_ends.append(seq_cursor + ((local_idx + 1) / len(row_words)) * valid_t)
            seq_cursor += valid_t

        if not words or not sequence_parts:
            continue
        sequence = np.concatenate(sequence_parts, axis=0)
        spans = split_word_spans(
            words,
            min_words=args.min_words,
            target_words=args.target_words,
            max_words=args.max_words,
        )
        kept_for_group = 0
        group_value = str(_scalar(data[args.group_key][kept_rows[0]])) if args.group_key in data.files else ""
        split_value = str(_scalar(data[args.split_key][kept_rows[0]])) if args.split_key in data.files else ""

        for start, end in spans:
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            reason = "kept"
            if NOISE_RE.search(chunk):
                reason = "noise"
            elif alpha_fraction(chunk) < args.min_alpha_fraction:
                reason = "low_alpha_fraction"
            elif len(content_words(chunk_words)) < args.min_content_words:
                reason = "too_few_content_words"
            elif args.dedupe:
                dedupe_key = re.sub(r"\s+", " ", chunk).casefold()
                if dedupe_key in seen:
                    reason = "duplicate"
                seen.add(dedupe_key)
            if reason != "kept":
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue

            seq_start = word_seq_starts[start]
            seq_end = word_seq_ends[end - 1]
            vector, weight_sum = weighted_embedding_for_positions(
                sequence,
                start_pos=seq_start,
                end_pos=seq_end,
                normalize=args.normalize,
            )
            source_start = word_rows[start]
            source_end = word_rows[end - 1]
            rows_for_chunk = list(range(source_start, source_end + 1))
            source_text = " ".join(sentences[row_idx] for row_idx in rows_for_chunk)
            chunk_vectors.append(vector)
            chunk_sentences.append(chunk)
            source_sentences.append(source_text)
            source_row_starts.append(source_start)
            source_row_ends.append(source_end)
            source_rows_joined.append(",".join(str(row_idx) for row_idx in rows_for_chunk))
            group_values_out.append(group_value)
            split_values_out.append(split_value)
            word_starts.append(start)
            word_ends.append(end)
            word_counts.append(end - start)
            sequence_starts.append(seq_start)
            sequence_ends.append(seq_end)
            sequence_weight_sums.append(weight_sum)
            start_tr_est.append(
                estimate_tr(
                    data,
                    row_idx=source_start,
                    local_word_index=word_local_indices[start],
                    local_word_count=word_local_counts[start],
                    is_end=False,
                )
            )
            stop_tr_est.append(
                estimate_tr(
                    data,
                    row_idx=source_end,
                    local_word_index=word_local_indices[end - 1],
                    local_word_count=word_local_counts[end - 1],
                    is_end=True,
                )
            )
            kept_for_group += 1
        if kept_for_group:
            groups_with_chunks += 1

    if not chunk_vectors:
        raise RuntimeError(f"No chunks produced from {input_path}")

    output: dict[str, np.ndarray] = {
        args.input_key: np.stack(chunk_vectors).astype(np.float32),
        args.sentence_key: np.asarray(chunk_sentences, dtype=object),
        "source_sentence": np.asarray(source_sentences, dtype=object),
        "source_row_start": np.asarray(source_row_starts, dtype=np.int64),
        "source_row_end": np.asarray(source_row_ends, dtype=np.int64),
        "source_rows": np.asarray(source_rows_joined, dtype=object),
        "word_start_in_group": np.asarray(word_starts, dtype=np.int64),
        "word_end_in_group": np.asarray(word_ends, dtype=np.int64),
        "word_count": np.asarray(word_counts, dtype=np.int64),
        "sequence_start_est": np.asarray(sequence_starts, dtype=np.float32),
        "sequence_end_est": np.asarray(sequence_ends, dtype=np.float32),
        "sequence_weight_sum": np.asarray(sequence_weight_sums, dtype=np.float32),
        "start_tr_est": np.asarray(start_tr_est, dtype=np.float32),
        "stop_tr_est": np.asarray(stop_tr_est, dtype=np.float32),
    }
    if args.group_key:
        output[args.group_key] = np.asarray(group_values_out, dtype=object)
    if args.split_key:
        output[args.split_key] = np.asarray(split_values_out, dtype=object)
    for key in data.files:
        value = data[key]
        if getattr(value, "shape", None) and value.shape[:1] == (len(sentences),):
            continue
        if key != args.schema_key:
            output[key] = value

    word_counts_array = output["word_count"]
    summary = {
        "output": str(output_path),
        "input_npz": str(input_path),
        "source_rows": len(sentences),
        "group_key": args.group_key,
        "groups": len(group_rows(data, args.group_key, args.sort_key, len(sentences))),
        "groups_with_chunks": int(groups_with_chunks),
        "chunk_count": int(len(chunk_sentences)),
        "input_key": args.input_key,
        "sentence_key": args.sentence_key,
        "sequence_key": args.sequence_key,
        "sequence_mask_key": args.sequence_mask_key if args.sequence_mask_key in data.files else None,
        "embedding_shape": list(output[args.input_key].shape),
        "min_words": args.min_words,
        "target_words": args.target_words,
        "max_words": args.max_words,
        "word_count_mean": float(word_counts_array.mean()),
        "word_count_median": float(np.median(word_counts_array)),
        "word_count_p10": float(np.percentile(word_counts_array, 10)),
        "word_count_p90": float(np.percentile(word_counts_array, 90)),
        "min_alpha_fraction": args.min_alpha_fraction,
        "min_content_words": args.min_content_words,
        "exclude_regex": args.exclude_regex,
        "dedupe": args.dedupe,
        "normalized": args.normalize,
        "compression": args.compression,
        "drop_reasons": drop_reasons,
        "source_schema": _schema(data, args.schema_key),
        "sample": chunk_sentences[: args.sample_count],
    }
    output[args.schema_key] = np.asarray(json.dumps(summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.compression == "none":
        np.savez(output_path, **output)
    else:
        np.savez_compressed(output_path, **output)
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
