#!/usr/bin/env python
"""Export short conversational sentences from HuggingFace datasets with embeddings."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import keep_sentence, words

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SPEAKER_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z0-9_ .'-]{0,28}|#?\w+)\s*:\s+")
TURN_MARKER_RE = re.compile(r"\s*(?:__eou__|<eou>|</s>|<br\s*/?>)\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset path.")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", action="append", default=[])
    parser.add_argument("--infer-text-fields", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--limit-sentences", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.75)
    parser.add_argument("--min-content-words", type=int, default=2)
    parser.add_argument(
        "--filter-mode",
        choices=["short_content", "simple_sv", "simple_sv_no_coord"],
        default="simple_sv_no_coord",
    )
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def field_values(value: Any, parts: list[str]) -> list[Any]:
    if not parts:
        return [value]
    if isinstance(value, dict):
        if parts[0] not in value:
            return []
        return field_values(value[parts[0]], parts[1:])
    if isinstance(value, (list, tuple)):
        values: list[Any] = []
        for item in value:
            values.extend(field_values(item, parts))
        return values
    return []


def flatten_strings(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_strings(item)


def normalize_piece(piece: str) -> str:
    piece = TURN_MARKER_RE.sub("\n", piece)
    piece = piece.replace("\r", "\n")
    piece = re.sub(r"\[[^\]]{0,60}\]", " ", piece)
    piece = re.sub(r"\([^)]{0,60}\)", " ", piece)
    piece = re.sub(r"\s+", " ", piece).strip(" \t\n\r\"'")
    piece = SPEAKER_RE.sub("", piece).strip(" \t\n\r\"'")
    return re.sub(r"\s+", " ", piece)


def split_candidate_sentences(text: str) -> list[str]:
    text = TURN_MARKER_RE.sub("\n", text)
    candidates: list[str] = []
    for line in text.splitlines():
        line = normalize_piece(line)
        if not line:
            continue
        for piece in SENTENCE_SPLIT_RE.split(line):
            piece = normalize_piece(piece)
            if piece:
                candidates.append(piece)
    return candidates


def normalize_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip()).casefold()


def row_texts(row: dict[str, Any], text_fields: list[str], infer_text_fields: bool) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for field in text_fields:
        parts = [part for part in field.split(".") if part]
        for value in field_values(row, parts):
            for text in flatten_strings(value):
                texts.append((field, text))
    if infer_text_fields:
        for key, value in row.items():
            if key in text_fields:
                continue
            if isinstance(value, str) and len(value) >= 8:
                texts.append((str(key), value))
    return texts


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    dataset_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config:
        dataset_kwargs["name"] = args.config
    dataset = load_dataset(**dataset_kwargs)
    print(f"Loaded dataset={args.dataset} config={args.config or '-'} split={args.split}", flush=True)
    if hasattr(dataset, "features"):
        print(f"features={dataset.features}", flush=True)

    sentences: list[str] = []
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    duplicate_count = 0
    input_text_count = 0
    row_count = 0

    for row_idx, row in enumerate(dataset):
        row_count += 1
        if args.limit_rows > 0 and row_idx >= args.limit_rows:
            break
        if not isinstance(row, dict):
            continue
        for field, text in row_texts(row, args.text_field, args.infer_text_fields):
            input_text_count += 1
            for sentence in split_candidate_sentences(text):
                ok, reason = keep_sentence(
                    sentence,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    min_alpha_fraction=args.min_alpha_fraction,
                    min_content_words=args.min_content_words,
                    require_verb=True,
                    filter_mode=args.filter_mode,
                    exclude_re=None,
                )
                if ok and args.dedupe:
                    key = normalize_key(sentence)
                    if key in seen:
                        ok = False
                        reason = "duplicate"
                        duplicate_count += 1
                    seen.add(key)
                reasons[reason] += 1
                if not ok:
                    continue
                sentences.append(sentence)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "config": args.config,
                        "split": args.split,
                        "row_index": int(row_idx),
                        "source_field": field,
                        "sentence": sentence,
                        "word_count": len(words(sentence)),
                    }
                )
                if args.limit_sentences > 0 and len(sentences) >= args.limit_sentences:
                    break
            if args.limit_sentences > 0 and len(sentences) >= args.limit_sentences:
                break
        if row_idx and row_idx % 10000 == 0:
            print(f"scanned rows={row_idx} kept={len(sentences)}", flush=True)
        if args.limit_sentences > 0 and len(sentences) >= args.limit_sentences:
            break

    if not sentences:
        raise RuntimeError(f"No sentences kept from {args.dataset} split={args.split}")

    device = resolve_device(args.device)
    print(f"Using device={device}; encoding {len(sentences)} sentences", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    embeddings = encode_sentences(
        sentences=[f"{args.input_prefix}{sentence}" for sentence in sentences],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        normalize=not args.no_normalize,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    word_counts = np.asarray([len(words(sentence)) for sentence in sentences], dtype=np.int64)
    summary = {
        "output": str(output),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "row_count_scanned": int(row_count),
        "input_text_count": int(input_text_count),
        "kept_count": int(len(sentences)),
        "drop_reasons": dict(reasons),
        "duplicate_count": int(duplicate_count),
        "text_fields": args.text_field,
        "infer_text_fields": args.infer_text_fields,
        "filter_mode": args.filter_mode,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "word_count_mean": float(word_counts.mean()),
        "word_count_median": float(np.median(word_counts)),
        "word_count_p10": float(np.percentile(word_counts, 10)),
        "word_count_p90": float(np.percentile(word_counts, 90)),
        "sample": sentences[: args.sample_count],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(sentences, dtype=object),
        rows=np.asarray(rows, dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
