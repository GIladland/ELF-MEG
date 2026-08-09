#!/usr/bin/env python
"""Generate cleaner oracle augmentations from train phrase templates and heldout lexicon."""

from __future__ import annotations

import argparse
import json
import os
import random
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

from scripts.augment_sherlock_sva_recombinational import ParsedSentence, build_entries, clean_tokens
from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import STOP_WORDS, VERBISH, keep_sentence, words


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


EXTRA_SKIP = {
    "all", "any", "some", "which", "what", "when", "where", "who", "whom", "whose",
    "why", "how", "still", "more", "most", "very", "only", "then", "there", "here",
    "perhaps", "maybe",
}
REPORTING_VERBS = {"said", "asked", "cried", "answered", "returned", "replied", "gasped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--support-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants-per-source", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts-per-source", type=int, default=3000)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--filter-mode", choices=["short_content", "simple_sv", "simple_sv_no_coord"], default="simple_sv_no_coord")
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


def content_counter(sentence: str) -> Counter[str]:
    return Counter(
        token.lower()
        for token in words(sentence)
        if token.lower() not in STOP_WORDS and token.lower() not in EXTRA_SKIP and len(token) > 1
    )


def word_f1(source: str, variant: str) -> float:
    left = Counter(token.lower() for token in words(source))
    right = Counter(token.lower() for token in words(variant))
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right.get(token, 0)) for token, count in left.items())
    precision = overlap / max(1, sum(right.values()))
    recall = overlap / max(1, sum(left.values()))
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def content_recall(source: str, variant: str) -> float:
    left = content_counter(source)
    right = content_counter(variant)
    if not left:
        return 0.0
    overlap = sum(min(count, right.get(token, 0)) for token, count in left.items())
    return overlap / max(1, sum(left.values()))


def content_f1(source: str, variant: str) -> float:
    left = content_counter(source)
    right = content_counter(variant)
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right.get(token, 0)) for token, count in left.items())
    precision = overlap / max(1, sum(right.values()))
    recall = overlap / max(1, sum(left.values()))
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def is_lexical_token(token: str) -> bool:
    lower = token.lower()
    if lower in STOP_WORDS or lower in EXTRA_SKIP or lower in VERBISH or lower in REPORTING_VERBS:
        return False
    return lower.isalpha() and len(lower) > 1


def noun_chunks(sentence: str) -> list[tuple[str, ...]]:
    tokens = words(sentence)
    chunks: list[tuple[str, ...]] = []
    idx = 0
    while idx < len(tokens):
        if not is_lexical_token(tokens[idx]):
            idx += 1
            continue
        start = idx
        while idx < len(tokens) and is_lexical_token(tokens[idx]) and idx - start < 4:
            idx += 1
        chunk = tuple(tokens[start:idx])
        if chunk:
            chunks.append(chunk)
            if len(chunk) > 1:
                chunks.extend((token,) for token in chunk)
    proper = []
    for token in tokens:
        if token[:1].isupper() and token.lower() not in STOP_WORDS and token.lower() not in EXTRA_SKIP:
            proper.append(token)
        else:
            if len(proper) >= 1:
                chunks.append(tuple(proper[:3]))
            proper = []
    if proper:
        chunks.append(tuple(proper[:3]))
    seen = set()
    unique = []
    for chunk in chunks:
        key = tuple(token.lower() for token in chunk)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def normalize_np(chunk: tuple[str, ...], *, position: str) -> tuple[str, ...]:
    if not chunk:
        return chunk
    first = chunk[0]
    lower = first.lower()
    if lower in {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her"}:
        return chunk
    if first[:1].isupper() or lower.isdigit() or lower.endswith("s"):
        return chunk
    if position == "pp":
        return ("the",) + chunk
    return chunk


def safe_template(entry: ParsedSentence) -> bool:
    if not entry.obj and not entry.pps:
        return False
    verb = entry.verb[-1].lower() if entry.verb else ""
    if verb in {"am", "is", "are", "was", "were", "be", "been", "being"}:
        return False
    if len(entry.subject) + len(entry.verb) > 8:
        return False
    return True


def reporting_variant(source: str, chunks: list[tuple[str, ...]], rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    tokens = words(source)
    lowered = [token.lower() for token in tokens]
    report_indices = [idx for idx, token in enumerate(lowered) if token in REPORTING_VERBS]
    if not report_indices:
        return None
    report_idx = rng.choice(report_indices)
    clause = tokens[:report_idx]
    speaker = tokens[report_idx + 1 :] or ["Holmes"]
    verb = tokens[report_idx]
    if len(clause) < 2:
        return None
    patterns = [
        list(speaker[:3]) + [verb] + clause,
        clause + [verb] + list(speaker[:3]),
    ]
    if chunks:
        patterns.append(["Holmes", verb, "that"] + list(normalize_np(rng.choice(chunks), position="obj")))
    return clean_tokens(rng.choice(patterns)), {"augmentation_mode": "oracle_phrase_reporting"}


def phrase_template_variant(source: str, templates: list[ParsedSentence], rng: random.Random) -> tuple[str, dict[str, Any]] | None:
    chunks = noun_chunks(source)
    if not chunks:
        return reporting_variant(source, chunks, rng)
    template = rng.choice(templates)
    tokens = list(template.subject + template.verb)
    used_chunks = []
    if template.obj:
        chunk = rng.choice(chunks)
        tokens.extend(normalize_np(chunk, position="obj"))
        used_chunks.append(" ".join(chunk))
    elif chunks:
        chunk = rng.choice(chunks)
        tokens.extend(normalize_np(chunk, position="obj"))
        used_chunks.append(" ".join(chunk))
    pp_count = min(len(template.pps), max(0, 3 - len(used_chunks)))
    for pp in rng.sample(list(template.pps), k=pp_count) if pp_count else []:
        chunk = rng.choice(chunks)
        tokens.append(pp[0])
        tokens.extend(normalize_np(chunk, position="pp"))
        used_chunks.append(" ".join(chunk))
    return clean_tokens(tokens), {
        "augmentation_mode": "oracle_phrase_template",
        "template_source_index": template.index,
        "used_chunks": used_chunks,
    }


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


def generate(source_sentences: list[str], support_sentences: list[str], args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]], Counter[str]]:
    rng = random.Random(args.seed)
    parsed, _exclude_re = build_entries(support_sentences, args)
    templates = [entry for entry in parsed if safe_template(entry)]
    if len(templates) < 100:
        raise RuntimeError(f"Only found {len(templates)} safe phrase templates.")
    seen = {canonical(sentence) for sentence in source_sentences}
    seen.update(canonical(sentence) for sentence in support_sentences)
    seen.update(blocked_sentences(args.blocked_npz, args.sentence_key))
    variants = []
    rows = []
    reasons: Counter[str] = Counter()
    for source_idx, source in enumerate(source_sentences):
        made = 0
        attempts = 0
        while made < args.variants_per_source and attempts < args.max_attempts_per_source:
            attempts += 1
            if rng.random() < 0.2:
                result = reporting_variant(source, noun_chunks(source), rng)
            else:
                result = phrase_template_variant(source, templates, rng)
            if result is None:
                reasons["no_candidate"] += 1
                continue
            candidate, meta = result
            if not valid_candidate(candidate, seen, args):
                reasons["drop_invalid_or_duplicate"] += 1
                continue
            seen.add(canonical(candidate))
            metrics = {
                "source_word_f1": word_f1(source, candidate),
                "source_content_f1": content_f1(source, candidate),
                "source_content_recall": content_recall(source, candidate),
            }
            variants.append(candidate)
            rows.append(
                {
                    "augmentation_source_index": int(source_idx),
                    "augmentation_source_sentence": source,
                    "sentence": candidate,
                    "word_count": len(words(candidate)),
                    **metrics,
                    **meta,
                }
            )
            made += 1
        if made == 0:
            reasons["source_without_variants"] += 1
        print(f"source_idx={source_idx} made={made} total_variants={len(variants)}", flush=True)
    return variants, rows, reasons


def main() -> None:
    args = parse_args()
    source_data = np.load(args.source_npz, allow_pickle=True)
    support_data = np.load(args.support_npz, allow_pickle=True)
    source_sentences = _strings(source_data[args.sentence_key])
    support_sentences = _strings(support_data[args.sentence_key])
    variants, rows, reasons = generate(source_sentences, support_sentences, args)
    if not variants:
        raise RuntimeError("No phrase-template variants generated.")
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
        "support_npz": args.support_npz,
        "source_count": len(source_sentences),
        "support_count": len(support_sentences),
        "generated_count": len(variants),
        "drop_reasons": dict(reasons),
        "source_word_f1_mean": float(np.mean([row["source_word_f1"] for row in rows])),
        "source_content_f1_mean": float(np.mean([row["source_content_f1"] for row in rows])),
        "source_content_recall_mean": float(np.mean([row["source_content_recall"] for row in rows])),
        "source_schema": _schema(source_data, args.schema_key),
        "support_schema": _schema(support_data, args.schema_key),
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "seed": args.seed,
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
