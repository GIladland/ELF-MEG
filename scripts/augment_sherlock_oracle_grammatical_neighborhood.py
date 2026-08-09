#!/usr/bin/env python
"""Generate grammar-preserving oracle neighborhoods around held-out sentences."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import STOP_WORDS, keep_sentence, words


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


ADJUNCTS = [
    ("again",),
    ("then",),
    ("today",),
    ("there",),
    ("soon",),
    ("at", "once"),
    ("that", "day"),
    ("that", "night"),
    ("in", "silence"),
    ("for", "a", "moment"),
    ("before", "long"),
    ("soon", "afterward"),
]

PREFIXES = [
    ("perhaps",),
    ("indeed",),
    ("then",),
    ("surely",),
    ("therefore",),
]

SAFE_MODIFIERS = {
    "able", "absolute", "actual", "acute", "additional", "admirable", "angry",
    "anxious", "awful", "bad", "black", "blue", "brave", "brief", "bright",
    "brown", "certain", "clear", "close", "cold", "common", "complete",
    "curious", "dark", "dead", "dear", "deep", "different", "double", "dry",
    "dull", "eager", "early", "easy", "empty", "excellent", "extraordinary",
    "fair", "faint", "false", "fine", "first", "fresh", "full", "general",
    "gentle", "good", "grave", "great", "green", "grey", "happy", "hard",
    "heavy", "high", "hot", "huge", "human", "important", "large", "last",
    "late", "little", "long", "loud", "low", "mad", "main", "mere", "new",
    "old", "open", "ordinary", "other", "own", "pale", "plain", "poor",
    "private", "public", "quick", "quiet", "ready", "real", "red", "right",
    "round", "same", "second", "short", "silent", "simple", "single", "small",
    "soft", "strange", "strong", "sudden", "sure", "terrible", "thin", "true",
    "usual", "white", "whole", "wide", "wild", "wrong", "young",
}

SAFE_ADVERBS = {
    "almost", "already", "always", "certainly", "clearly", "closely", "deeply",
    "easily", "entirely", "finally", "hardly", "immediately", "merely", "never",
    "only", "perhaps", "quite", "really", "slowly", "soon", "still", "surely",
    "then", "there", "together", "very",
}

NUMBER_MODIFIERS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve",
}

LOWER_AFTER_PREFIX = {
    "a", "an", "the", "this", "that", "these", "those", "he", "she", "it",
    "we", "they", "you", "his", "her", "my", "our", "their",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants-per-source", type=int, default=64)
    parser.add_argument(
        "--source-limit",
        type=int,
        default=0,
        help="Use only the first N source rows; zero uses the complete source NPZ.",
    )
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--filter-mode", choices=["short_content", "simple_sv", "simple_sv_no_coord"], default="simple_sv_no_coord")
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    try:
        return json.loads(str(data[schema_key].tolist()))
    except json.JSONDecodeError:
        return str(data[schema_key].tolist())


def canonical(sentence: str) -> str:
    return " ".join(token.lower() for token in words(sentence))


def word_metrics(source: str, variant: str) -> dict[str, float]:
    source_counts = Counter(token.lower() for token in words(source))
    variant_counts = Counter(token.lower() for token in words(variant))
    if not source_counts or not variant_counts:
        return {"source_word_recall": 0.0, "source_word_precision": 0.0, "source_word_f1": 0.0}
    overlap = sum(min(count, variant_counts.get(token, 0)) for token, count in source_counts.items())
    recall = overlap / max(1, sum(source_counts.values()))
    precision = overlap / max(1, sum(variant_counts.values()))
    f1 = 0.0 if recall + precision == 0 else 2.0 * recall * precision / (recall + precision)
    return {
        "source_word_recall": float(recall),
        "source_word_precision": float(precision),
        "source_word_f1": float(f1),
    }


def sentence_from_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    output = list(tokens)
    output[0] = output[0][:1].upper() + output[0][1:]
    return re.sub(r"\s+", " ", " ".join(output)).strip()


def lower_for_prefix(token: str) -> str:
    lower = token.lower()
    if lower == "i":
        return "I"
    if lower in LOWER_AFTER_PREFIX:
        return lower
    return token


def add_suffix(tokens: list[str], suffix: tuple[str, ...]) -> list[str]:
    return list(tokens) + list(suffix)


def add_prefix(tokens: list[str], prefix: tuple[str, ...]) -> list[str]:
    if not tokens:
        return list(prefix)
    return list(prefix) + [lower_for_prefix(tokens[0])] + list(tokens[1:])


def modifier_indices(tokens: list[str]) -> list[int]:
    indices = []
    for idx, token in enumerate(tokens):
        lower = token.lower()
        if idx == 0 and lower not in SAFE_ADVERBS:
            continue
        if lower in SAFE_ADVERBS or lower in SAFE_MODIFIERS or lower in NUMBER_MODIFIERS:
            indices.append(idx)
            continue
        if lower.endswith("ly") and len(lower) > 4:
            indices.append(idx)
    return indices


def valid_candidate(sentence: str, seen: set[str], args: argparse.Namespace) -> bool:
    if not sentence or canonical(sentence) in seen:
        return False
    token_list = words(sentence)
    if any(left.lower() == right.lower() for left, right in zip(token_list, token_list[1:])):
        return False
    ok, _reason = keep_sentence(
        sentence,
        min_words=args.min_words,
        max_words=args.max_words,
        min_alpha_fraction=0.75,
        min_content_words=2,
        require_verb=True,
        filter_mode=args.filter_mode,
        exclude_re=None,
    )
    return ok


def blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def generate_for_source(source: str, args: argparse.Namespace) -> list[tuple[str, str]]:
    tokens = words(source)
    raw_variants: list[tuple[list[str], str]] = []

    for suffix in ADJUNCTS:
        raw_variants.append((add_suffix(tokens, suffix), f"suffix:{' '.join(suffix)}"))
    for prefix in PREFIXES:
        raw_variants.append((add_prefix(tokens, prefix), f"prefix:{' '.join(prefix)}"))

    for idx in modifier_indices(tokens):
        deleted = tokens[:idx] + tokens[idx + 1 :]
        raw_variants.append((deleted, f"delete_modifier:{idx}:{tokens[idx]}"))
        for suffix in ADJUNCTS:
            raw_variants.append((add_suffix(deleted, suffix), f"delete_modifier_suffix:{idx}:{tokens[idx]}:{' '.join(suffix)}"))
        for prefix in PREFIXES:
            raw_variants.append((add_prefix(deleted, prefix), f"delete_modifier_prefix:{idx}:{tokens[idx]}:{' '.join(prefix)}"))

    variants: list[tuple[str, str]] = []
    seen_local = set()
    for candidate_tokens, op in raw_variants:
        if len(candidate_tokens) < args.min_words or len(candidate_tokens) > args.max_words:
            continue
        sentence = sentence_from_tokens(candidate_tokens)
        key = canonical(sentence)
        if key in seen_local:
            continue
        seen_local.add(key)
        variants.append((sentence, op))
        if len(variants) >= args.variants_per_source:
            break
    return variants


def generate(source_sentences: list[str], args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]], Counter[str]]:
    seen = {canonical(sentence) for sentence in source_sentences}
    seen.update(blocked_sentences(args.blocked_npz, args.sentence_key))
    variants = []
    rows = []
    reasons: Counter[str] = Counter()

    for source_idx, source in enumerate(source_sentences):
        source_variants = generate_for_source(source, args)
        made = 0
        for candidate, op in source_variants:
            if not valid_candidate(candidate, seen, args):
                reasons["drop_invalid_or_duplicate"] += 1
                continue
            seen.add(canonical(candidate))
            metrics = word_metrics(source, candidate)
            rows.append(
                {
                    "augmentation_mode": "oracle_grammatical_neighborhood",
                    "augmentation_source_index": int(source_idx),
                    "augmentation_source_sentence": source,
                    "augmentation_operation": op,
                    "sentence": candidate,
                    "word_count": len(words(candidate)),
                    **metrics,
                }
            )
            variants.append(candidate)
            made += 1
            if made >= args.variants_per_source:
                break
        if made == 0:
            reasons["source_without_variants"] += 1
        print(f"source_idx={source_idx} made={made} total_variants={len(variants)}", flush=True)
    return variants, rows, reasons


def main() -> None:
    args = parse_args()
    source_data = np.load(args.source_npz, allow_pickle=True)
    source_sentences = _strings(source_data[args.sentence_key])
    if args.source_limit > 0:
        source_sentences = source_sentences[: args.source_limit]
    variants, rows, reasons = generate(source_sentences, args)
    if not variants:
        raise RuntimeError("No grammatical neighborhood variants generated.")

    device = resolve_device(args.device)
    print(f"Using device={device}; encoding {len(variants)} variants", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.embedding_model_name, local_files_only=args.local_files_only).to(device)
    model.eval()
    embeddings = encode_sentences(
        sentences=[f"{args.input_prefix}{sentence}" for sentence in variants],
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
    summary = {
        "output": str(output),
        "source_npz": args.source_npz,
        "blocked_npz": args.blocked_npz,
        "source_count": len(source_sentences),
        "variants_per_source": args.variants_per_source,
        "source_limit": args.source_limit,
        "generated_count": len(variants),
        "drop_reasons": dict(reasons),
        "source_word_recall_mean": float(np.mean([row["source_word_recall"] for row in rows])),
        "source_word_f1_mean": float(np.mean([row["source_word_f1"] for row in rows])),
        "source_schema": _schema(source_data, args.schema_key),
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "sample": rows[:30],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(variants, dtype=object),
        rows=np.asarray(rows, dtype=object),
        source_npz=np.asarray([args.source_npz] * len(variants), dtype=object),
        split=np.asarray(["train"] * len(variants), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
