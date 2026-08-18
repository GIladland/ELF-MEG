#!/usr/bin/env python
"""Chunk flowing Tang-GPT text-pair NPZ rows into shorter targets.

The Tang podcast export stores one text target for a 10-TR segment plus
`input_embedding_sequence` with one embedding per TR. This script splits the
text into consecutive word chunks and estimates each chunk's embedding by a
duration-weighted average over the matching part of the TR embedding sequence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

from scripts.filter_semantic_npz_sentences import (  # noqa: E402
    DEFAULT_META_RE,
    NOISE_RE,
    STOP_WORDS,
    WORD_RE,
)

BAD_BOUNDARY_END_WORDS = STOP_WORDS | {
    "also",
    "because",
    "just",
    "like",
    "really",
    "so",
    "then",
    "though",
    "uh",
    "um",
    "used",
    "very",
    "well",
}
BAD_BOUNDARY_START_WORDS = {
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "like",
    "of",
    "on",
    "or",
    "than",
    "then",
    "to",
    "with",
}
GOOD_BOUNDARY_START_WORDS = {
    "after",
    "also",
    "although",
    "because",
    "before",
    "but",
    "eventually",
    "finally",
    "he",
    "here",
    "i",
    "if",
    "it",
    "later",
    "meanwhile",
    "my",
    "now",
    "she",
    "so",
    "that",
    "then",
    "there",
    "they",
    "this",
    "uh",
    "um",
    "we",
    "well",
    "when",
    "while",
    "you",
}
GOOD_BOUNDARY_END_WORDS = {
    "again",
    "anymore",
    "back",
    "before",
    "down",
    "home",
    "inside",
    "now",
    "out",
    "there",
    "together",
    "too",
    "up",
    "yesterday",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--sequence-key", default="input_embedding_sequence")
    parser.add_argument("--sequence-mask-key", default="input_sequence_mask")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--target-words", type=int, default=18)
    parser.add_argument("--max-words", type=int, default=24)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.75)
    parser.add_argument("--min-content-words", type=int, default=2)
    parser.add_argument("--exclude-regex", default=DEFAULT_META_RE)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compression", choices=["compressed", "none"], default="compressed")
    parser.add_argument("--sample-count", type=int, default=10)
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


def content_words(words: list[str]) -> list[str]:
    return [word.lower() for word in words if word.lower() not in STOP_WORDS and len(word) > 2]


def alpha_fraction(text: str) -> float:
    nonspace = sum(1 for char in text if not char.isspace())
    if nonspace == 0:
        return 0.0
    alpha = sum(1 for char in text if char.isalpha())
    return alpha / nonspace


def boundary_score(words: list[str], end: int, *, target_end: int) -> float:
    previous_word = words[end - 1].lower()
    next_word = words[end].lower() if end < len(words) else ""
    score = -0.3 * abs(end - target_end)
    if previous_word in BAD_BOUNDARY_END_WORDS:
        score -= 4.0
    if next_word in BAD_BOUNDARY_START_WORDS:
        score -= 2.0
    if previous_word in GOOD_BOUNDARY_END_WORDS:
        score += 1.0
    if next_word in GOOD_BOUNDARY_START_WORDS:
        score += 2.0
    if previous_word not in STOP_WORDS and len(previous_word) > 3:
        score += 0.5
    if next_word in {"i", "he", "she", "we", "they", "you"}:
        score += 1.0
    return score


def choose_boundary(words: list[str], *, start: int, min_words: int, target_words: int, max_words: int) -> int:
    n_words = len(words)
    remaining = n_words - start
    if remaining <= max_words:
        return n_words
    lower = start + min_words
    upper = min(start + max_words, n_words - min_words)
    if lower > upper:
        return min(n_words, start + min(max_words, remaining))
    target_end = min(upper, max(lower, start + target_words))
    candidates = range(lower, upper + 1)
    return max(candidates, key=lambda end: boundary_score(words, end, target_end=target_end))


def split_word_spans(words: list[str], *, min_words: int, target_words: int, max_words: int) -> list[tuple[int, int]]:
    n_words = len(words)
    if n_words < min_words:
        return []
    if n_words <= max_words:
        return [(0, n_words)]

    spans = []
    start = 0
    while start < n_words:
        remaining = n_words - start
        if remaining <= max_words:
            spans.append((start, n_words))
            break
        end = choose_boundary(
            words,
            start=start,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
        )
        if end - start < min_words:
            if spans:
                prev_start, _ = spans.pop()
                spans.append((prev_start, n_words))
            break
        spans.append((start, end))
        start = end
    return spans


def valid_sequence_length(mask: np.ndarray | None, max_len: int) -> int:
    if mask is None:
        return max_len
    valid = np.flatnonzero(np.asarray(mask) > 0)
    if valid.size == 0:
        return 0
    return int(valid[-1]) + 1


def weighted_chunk_embedding(
    sequence: np.ndarray,
    *,
    word_start: int,
    word_end: int,
    n_words: int,
    normalize: bool,
) -> tuple[np.ndarray, float, float, float]:
    t_count = sequence.shape[0]
    start_pos = (word_start / n_words) * t_count
    end_pos = (word_end / n_words) * t_count
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
    return vector.astype(np.float32), float(start_pos), float(end_pos), weight_sum


def first_dim_keys(data: np.lib.npyio.NpzFile, n_rows: int) -> set[str]:
    keys = set()
    for key in data.files:
        try:
            value = data[key]
        except Exception:
            continue
        if getattr(value, "shape", None) and value.shape[:1] == (n_rows,):
            keys.add(key)
    return keys


def main() -> None:
    args = parse_args()
    if args.min_words <= 0 or args.target_words <= 0 or args.max_words <= 0:
        raise ValueError("word-count arguments must be positive")
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot exceed --max-words")
    if args.target_words > args.max_words:
        raise ValueError("--target-words cannot exceed --max-words")

    input_path = Path(args.input_npz)
    output_path = Path(args.output)
    data = np.load(input_path, allow_pickle=True)
    if args.sequence_key not in data.files:
        raise KeyError(f"{input_path} has no sequence key {args.sequence_key!r}")
    if args.sentence_key not in data.files:
        raise KeyError(f"{input_path} has no sentence key {args.sentence_key!r}")

    sentences = _strings(data[args.sentence_key])
    sequences = np.asarray(data[args.sequence_key])
    masks = np.asarray(data[args.sequence_mask_key]) if args.sequence_mask_key in data.files else None
    if sequences.shape[0] != len(sentences):
        raise ValueError(f"sequence rows {sequences.shape[0]} do not match sentence rows {len(sentences)}")

    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE) if args.exclude_regex else None
    row_keys = first_dim_keys(data, len(sentences))
    metadata_keys = [
        key
        for key in data.files
        if key in row_keys
        and key
        not in {
            args.input_key,
            args.sentence_key,
            args.sequence_key,
            args.sequence_mask_key,
            args.schema_key,
        }
        and data[key].ndim <= 1
    ]

    chunk_vectors: list[np.ndarray] = []
    chunk_sentences: list[str] = []
    source_sentences: list[str] = []
    source_rows: list[int] = []
    chunk_indices: list[int] = []
    word_starts: list[int] = []
    word_ends: list[int] = []
    word_counts: list[int] = []
    sequence_starts: list[float] = []
    sequence_ends: list[float] = []
    sequence_weight_sums: list[float] = []
    metadata_values: dict[str, list[Any]] = {key: [] for key in metadata_keys}
    seen: set[str] = set()
    drop_reasons: dict[str, int] = {}
    source_with_chunks = 0

    for row_idx, sentence in enumerate(sentences):
        words = WORD_RE.findall(sentence)
        if not words:
            drop_reasons["source_no_words"] = drop_reasons.get("source_no_words", 0) + 1
            continue
        if exclude_re and exclude_re.search(sentence):
            drop_reasons["source_excluded_regex"] = drop_reasons.get("source_excluded_regex", 0) + 1
            continue
        valid_t = valid_sequence_length(None if masks is None else masks[row_idx], sequences.shape[1])
        if valid_t <= 0:
            drop_reasons["source_no_valid_sequence"] = drop_reasons.get("source_no_valid_sequence", 0) + 1
            continue

        spans = split_word_spans(
            words,
            min_words=args.min_words,
            target_words=args.target_words,
            max_words=args.max_words,
        )
        kept_for_source = 0
        for chunk_idx, (start, end) in enumerate(spans):
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

            vector, seq_start, seq_end, weight_sum = weighted_chunk_embedding(
                sequences[row_idx, :valid_t],
                word_start=start,
                word_end=end,
                n_words=len(words),
                normalize=args.normalize,
            )
            chunk_vectors.append(vector)
            chunk_sentences.append(chunk)
            source_sentences.append(sentence)
            source_rows.append(row_idx)
            chunk_indices.append(chunk_idx)
            word_starts.append(start)
            word_ends.append(end)
            word_counts.append(end - start)
            sequence_starts.append(seq_start)
            sequence_ends.append(seq_end)
            sequence_weight_sums.append(weight_sum)
            for key in metadata_keys:
                metadata_values[key].append(data[key][row_idx].item() if data[key].shape == () else data[key][row_idx])
            kept_for_source += 1
        if kept_for_source:
            source_with_chunks += 1

    if not chunk_vectors:
        raise RuntimeError(f"No chunks produced from {input_path}")

    output: dict[str, np.ndarray] = {}
    output[args.input_key] = np.stack(chunk_vectors).astype(np.float32)
    output[args.sentence_key] = np.asarray(chunk_sentences, dtype=object)
    output["source_sentence"] = np.asarray(source_sentences, dtype=object)
    output["source_row"] = np.asarray(source_rows, dtype=np.int64)
    output["chunk_index"] = np.asarray(chunk_indices, dtype=np.int64)
    output["word_start"] = np.asarray(word_starts, dtype=np.int64)
    output["word_end"] = np.asarray(word_ends, dtype=np.int64)
    output["word_count"] = np.asarray(word_counts, dtype=np.int64)
    output["sequence_start_est"] = np.asarray(sequence_starts, dtype=np.float32)
    output["sequence_end_est"] = np.asarray(sequence_ends, dtype=np.float32)
    output["sequence_weight_sum"] = np.asarray(sequence_weight_sums, dtype=np.float32)
    for key, values in metadata_values.items():
        output[key] = np.asarray(values, dtype=data[key].dtype if data[key].dtype != object else object)
    for key in data.files:
        if key not in row_keys and key != args.schema_key:
            output[key] = data[key]

    word_counts_array = output["word_count"]
    summary = {
        "output": str(output_path),
        "input_npz": str(input_path),
        "source_rows": len(sentences),
        "source_rows_with_chunks": int(source_with_chunks),
        "chunk_count": int(len(chunk_sentences)),
        "chunks_per_source_with_chunks_mean": float(len(chunk_sentences) / max(1, source_with_chunks)),
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
