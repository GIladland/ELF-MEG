#!/usr/bin/env python
"""Generate train-derived sentence augmentations and MiniLM embeddings."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import (
    DEFAULT_META_RE,
    NON_DECLARATIVE_INITIAL,
    STOP_WORDS,
    VERBISH,
    keep_sentence,
    words,
)


PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "during", "for", "from", "in", "inside", "into", "near", "of", "off",
    "on", "onto", "over", "through", "to", "toward", "under", "upon", "with",
    "within", "without",
}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "hundred",
    "thousand", "million",
}
ADJECTIVE_SUFFIXES = (
    "able", "ible", "al", "ant", "ary", "ent", "ful", "ic", "ical", "ive",
    "less", "ous", "some", "y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants-per-sentence", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts-per-sentence", type=int, default=120)
    parser.add_argument("--min-source-word-recall", type=float, default=0.65)
    parser.add_argument("--max-source-word-recall", type=float, default=0.95)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--filter-mode", choices=["short_content", "simple_sv", "simple_sv_no_coord"], default="simple_sv_no_coord")
    parser.add_argument("--quality-mode", choices=["loose", "strict"], default="loose")
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--min-phrase-count", type=int, default=2)
    parser.add_argument("--max-replacement-word-len", type=int, default=14)
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
    return " ".join(word.lower() for word in words(sentence))


def word_recall(source: str, variant: str) -> float:
    source_words = set(word.lower() for word in words(source))
    variant_words = set(word.lower() for word in words(variant))
    if not source_words:
        return 0.0
    return len(source_words & variant_words) / len(source_words)


def apply_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement.lower()


def likely_adjective(tokens: list[str], idx: int) -> bool:
    lower = tokens[idx].lower()
    if lower.endswith(ADJECTIVE_SUFFIXES):
        return True
    if idx > 0 and tokens[idx - 1].lower() in DETERMINERS:
        if idx + 1 < len(tokens):
            next_lower = tokens[idx + 1].lower()
            return next_lower not in STOP_WORDS and next_lower not in VERBISH
    return False


def token_class(tokens: list[str], idx: int) -> str:
    token = tokens[idx]
    lower = token.lower()
    if lower in STOP_WORDS or lower in PREPOSITIONS:
        return "skip"
    if lower in PRONOUNS:
        return "pronoun"
    if lower in NUMBER_WORDS or lower.isdigit():
        return "number"
    if lower in VERBISH or lower.endswith(("ed", "ing")):
        if lower in {"is", "are", "was", "were", "am", "be", "been", "being"}:
            return "verb_be"
        if lower.endswith("ing"):
            return "verb_ing"
        if lower.endswith("ed"):
            return "verb_ed"
        if lower.endswith("s") and len(lower) > 3:
            return "verb_s"
        return "verb_base"
    if likely_adjective(tokens, idx):
        return "adjective"
    if token[:1].isupper() and idx > 0:
        return "proper"
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        return "noun_plural"
    return "noun"


def safe_alpha_token(token: str, *, max_len: int = 14) -> bool:
    lower = token.lower()
    return lower.isalpha() and 2 < len(lower) <= max_len


def strict_token_allowed(tokens: list[str], idx: int, *, max_len: int) -> bool:
    token = tokens[idx]
    lower = token.lower()
    cls = token_class(tokens, idx)
    if idx == 0 or cls in {"skip", "proper", "pronoun", "number", "verb_be", "verb_ing", "verb_ed", "verb_s", "verb_base"}:
        return False
    if not safe_alpha_token(token, max_len=max_len):
        return False
    if lower in STOP_WORDS or lower in PREPOSITIONS or lower in NUMBER_WORDS:
        return False
    if token[:1].isupper() and idx > 0:
        return False
    if cls == "adjective":
        prev_lower = tokens[idx - 1].lower() if idx > 0 else ""
        next_cls = token_class(tokens, idx + 1) if idx + 1 < len(tokens) else ""
        if prev_lower not in DETERMINERS or next_cls not in {"noun", "noun_plural"}:
            return False
        next_next_cls = token_class(tokens, idx + 2) if idx + 2 < len(tokens) else ""
        if next_next_cls in {"noun", "noun_plural"}:
            return False
        if lower.endswith("ly"):
            return False
    if cls in {"noun", "noun_plural"}:
        prev_lower = tokens[idx - 1].lower() if idx > 0 else ""
        prev_cls = token_class(tokens, idx - 1) if idx > 0 else ""
        next_cls = token_class(tokens, idx + 1) if idx + 1 < len(tokens) else ""
        if next_cls in {"noun", "noun_plural"}:
            return False
        if prev_lower not in DETERMINERS and prev_cls != "adjective" and prev_lower not in PREPOSITIONS:
            return False
    return True


def phrase_spans(tokens: list[str], *, strict: bool = False, max_len: int = 14) -> list[tuple[int, int, str]]:
    spans = []
    idx = 0
    while idx < len(tokens):
        lower = tokens[idx].lower()
        if lower not in PREPOSITIONS:
            idx += 1
            continue
        end = idx + 1
        while end < len(tokens) and end - idx < 6 and tokens[end].lower() not in PREPOSITIONS:
            end += 1
        if end - idx >= 2:
            if strict:
                phrase_tokens = tokens[idx:end]
                phrase_classes = [token_class(tokens, phrase_idx) for phrase_idx in range(idx, end)]
                if any(cls in {"proper", "number", "pronoun"} for cls in phrase_classes):
                    idx = end
                    continue
                if any(not safe_alpha_token(token, max_len=max_len) for token in phrase_tokens[1:]):
                    idx = end
                    continue
            spans.append((idx, end, lower))
        idx = end
    return spans


def build_pools(args: argparse.Namespace, sentences: list[str]) -> tuple[dict[str, list[str]], dict[str, list[list[str]]]]:
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_counts: dict[str, Counter[str]] = defaultdict(Counter)
    strict = args.quality_mode == "strict"
    for sentence in sentences:
        tokens = words(sentence)
        if not tokens:
            continue
        for idx, token in enumerate(tokens):
            cls = token_class(tokens, idx)
            if cls == "skip":
                continue
            if strict and not strict_token_allowed(tokens, idx, max_len=args.max_replacement_word_len):
                continue
            class_counts[cls][token.lower()] += 1
        for start, end, prep in phrase_spans(tokens, strict=strict, max_len=args.max_replacement_word_len):
            phrase_counts[prep][" ".join(token.lower() for token in tokens[start:end])] += 1
    class_pools = {
        cls: sorted(
            [token for token, count in counter.items() if count >= args.min_token_count],
            key=lambda token: (-counter[token], token),
        )
        for cls, counter in class_counts.items()
        if len(counter) >= 2
    }
    phrase_pools = {
        prep: [
            phrase.split()
            for phrase, count in counter.most_common()
            if count >= args.min_phrase_count and len(phrase.split()) >= 2
        ]
        for prep, counter in phrase_counts.items()
        if len(counter) >= 2
    }
    class_pools = {cls: pool for cls, pool in class_pools.items() if len(pool) >= 2}
    phrase_pools = {prep: pool for prep, pool in phrase_pools.items() if len(pool) >= 2}
    return class_pools, phrase_pools


def replace_one(
    tokens: list[str],
    class_pools: dict[str, list[str]],
    rng: random.Random,
    *,
    strict: bool = False,
    max_len: int = 14,
) -> tuple[list[str], str] | None:
    positions = [
        idx
        for idx in range(len(tokens))
        if token_class(tokens, idx) in class_pools
        and (not strict or strict_token_allowed(tokens, idx, max_len=max_len))
    ]
    rng.shuffle(positions)
    for idx in positions:
        cls = token_class(tokens, idx)
        pool = class_pools.get(cls, [])
        candidates = [
            token
            for token in rng.sample(pool, min(len(pool), 40))
            if token != tokens[idx].lower() and (not strict or safe_alpha_token(token, max_len=max_len))
        ]
        if not candidates:
            continue
        replacement = apply_case(rng.choice(candidates), tokens[idx])
        new_tokens = list(tokens)
        new_tokens[idx] = replacement
        return new_tokens, f"replace_one:{cls}:{idx}:{tokens[idx]}->{replacement}"
    return None


def replace_two(
    tokens: list[str],
    class_pools: dict[str, list[str]],
    rng: random.Random,
    *,
    strict: bool = False,
    max_len: int = 14,
) -> tuple[list[str], str] | None:
    positions = [
        idx
        for idx in range(len(tokens))
        if token_class(tokens, idx) in class_pools
        and (not strict or strict_token_allowed(tokens, idx, max_len=max_len))
    ]
    if len(positions) < 2:
        return None
    rng.shuffle(positions)
    new_tokens = list(tokens)
    ops = []
    changed = 0
    for idx in positions[:4]:
        cls = token_class(tokens, idx)
        pool = class_pools.get(cls, [])
        candidates = [
            token
            for token in rng.sample(pool, min(len(pool), 40))
            if token != tokens[idx].lower() and (not strict or safe_alpha_token(token, max_len=max_len))
        ]
        if not candidates:
            continue
        replacement = apply_case(rng.choice(candidates), tokens[idx])
        new_tokens[idx] = replacement
        ops.append(f"{cls}:{idx}:{tokens[idx]}->{replacement}")
        changed += 1
        if changed == 2:
            return new_tokens, "replace_two:" + ",".join(ops)
    return None


def replace_phrase(
    tokens: list[str],
    phrase_pools: dict[str, list[list[str]]],
    rng: random.Random,
    *,
    strict: bool = False,
    max_len: int = 14,
) -> tuple[list[str], str] | None:
    spans = phrase_spans(tokens, strict=strict, max_len=max_len)
    rng.shuffle(spans)
    for start, end, prep in spans:
        pool = phrase_pools.get(prep, [])
        if not pool:
            continue
        current = [token.lower() for token in tokens[start:end]]
        candidates = [
            phrase
            for phrase in rng.sample(pool, min(len(pool), 50))
            if phrase != current
            and abs(len(phrase) - len(current)) <= 1
            and (not strict or all(safe_alpha_token(token, max_len=max_len) for token in phrase[1:]))
        ]
        if not candidates:
            continue
        replacement = list(rng.choice(candidates))
        replacement[0] = apply_case(replacement[0], tokens[start])
        new_tokens = tokens[:start] + replacement + tokens[end:]
        return new_tokens, f"replace_phrase:{prep}:{start}:{end}:{' '.join(current)}->{' '.join(replacement)}"
    return None


def delete_adjective(
    tokens: list[str],
    rng: random.Random,
    *,
    strict: bool = False,
    max_len: int = 14,
) -> tuple[list[str], str] | None:
    positions = [
        idx
        for idx in range(1, len(tokens) - 1)
        if token_class(tokens, idx) == "adjective"
        and (not strict or strict_token_allowed(tokens, idx, max_len=max_len))
    ]
    if not positions:
        return None
    idx = rng.choice(positions)
    return tokens[:idx] + tokens[idx + 1 :], f"delete_adjective:{idx}:{tokens[idx]}"


def insert_adjective(
    tokens: list[str],
    class_pools: dict[str, list[str]],
    rng: random.Random,
    *,
    strict: bool = False,
    max_len: int = 14,
) -> tuple[list[str], str] | None:
    adjectives = class_pools.get("adjective", [])
    if not adjectives:
        return None
    positions = []
    for idx, token in enumerate(tokens[:-1]):
        if token.lower() in DETERMINERS:
            positions.append(idx + 1)
    if not positions:
        return None
    idx = rng.choice(positions)
    candidates = [token for token in adjectives if not strict or safe_alpha_token(token, max_len=max_len)]
    if not candidates:
        return None
    adjective = rng.choice(candidates)
    return tokens[:idx] + [apply_case(adjective, tokens[idx])] + tokens[idx:], f"insert_adjective:{idx}:{adjective}"


def cleanup_sentence(tokens: list[str]) -> str:
    sentence = " ".join(tokens)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"\b([Aa]) ([aeiouAEIOU])", lambda m: f"{m.group(1)}n {m.group(2)}", sentence)
    sentence = re.sub(r"\b([Aa])n ([^aeiouAEIOU\\W])", lambda m: f"{m.group(1)} {m.group(2)}", sentence)
    return sentence


def valid_variant(source: str, variant: str, args: argparse.Namespace, blocked: set[str], exclude_re: re.Pattern[str]) -> tuple[bool, str]:
    if not variant or canonical(variant) in blocked:
        return False, "blocked_or_duplicate"
    recall = word_recall(source, variant)
    if recall < args.min_source_word_recall:
        return False, "low_source_recall"
    if recall > args.max_source_word_recall:
        return False, "too_close_to_source"
    if args.quality_mode == "strict":
        variant_words = words(variant)
        if any(len(word) == 1 and word.lower() not in {"a", "i"} for word in variant_words):
            return False, "single_letter_token"
        if any(left.lower() == right.lower() for left, right in zip(variant_words, variant_words[1:])):
            return False, "repeated_adjacent_word"
    ok, reason = keep_sentence(
        variant,
        min_words=args.min_words,
        max_words=args.max_words,
        min_alpha_fraction=0.75,
        min_content_words=2,
        require_verb=True,
        filter_mode=args.filter_mode,
        exclude_re=exclude_re,
    )
    return ok, reason


def generate_variants(args: argparse.Namespace, train_sentences: list[str], blocked: set[str]) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(args.seed)
    class_pools, phrase_pools = build_pools(args, train_sentences)
    exclude_re = re.compile(DEFAULT_META_RE, flags=re.IGNORECASE)
    variants: list[str] = []
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen = set(blocked)
    if args.quality_mode == "strict":
        methods = [replace_one, replace_phrase, delete_adjective]
    else:
        methods = [replace_one, replace_two, replace_phrase, delete_adjective, insert_adjective]

    eligible_sources = []
    for source_idx, source in enumerate(train_sentences):
        ok, reason = keep_sentence(
            source,
            min_words=args.min_words,
            max_words=args.max_words,
            min_alpha_fraction=0.75,
            min_content_words=2,
            require_verb=True,
            filter_mode=args.filter_mode,
            exclude_re=exclude_re,
        )
        if ok and words(source) and words(source)[0].lower() not in NON_DECLARATIVE_INITIAL:
            eligible_sources.append((source_idx, source))
        else:
            reasons[f"source_skipped_{reason}"] += 1

    for source_idx, source in eligible_sources:
        source_tokens = words(source)
        made = 0
        attempts = 0
        while made < args.variants_per_sentence and attempts < args.max_attempts_per_sentence:
            attempts += 1
            method = rng.choice(methods)
            if method in {replace_one, replace_two, insert_adjective}:
                result = method(  # type: ignore[misc]
                    source_tokens,
                    class_pools,
                    rng,
                    strict=args.quality_mode == "strict",
                    max_len=args.max_replacement_word_len,
                )
            elif method is replace_phrase:
                result = method(
                    source_tokens,
                    phrase_pools,
                    rng,
                    strict=args.quality_mode == "strict",
                    max_len=args.max_replacement_word_len,
                )
            else:
                result = method(  # type: ignore[misc]
                    source_tokens,
                    rng,
                    strict=args.quality_mode == "strict",
                    max_len=args.max_replacement_word_len,
                )
            if result is None:
                reasons["no_variant_from_method"] += 1
                continue
            new_tokens, op = result
            variant = cleanup_sentence(new_tokens)
            ok, reason = valid_variant(source, variant, args, seen, exclude_re)
            reasons[reason if ok else f"drop_{reason}"] += 1
            if not ok:
                continue
            seen.add(canonical(variant))
            variants.append(variant)
            rows.append(
                {
                    "augmentation_source_index": int(source_idx),
                    "augmentation_source_sentence": source,
                    "augmentation_operation": op,
                    "sentence": variant,
                    "word_count": len(words(variant)),
                    "source_word_recall": word_recall(source, variant),
                }
            )
            made += 1
        if source_idx and source_idx % 1000 == 0:
            print(f"augmented source_idx={source_idx} variants={len(variants)}", flush=True)
    return variants, rows, dict(reasons)


def blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def main() -> None:
    args = parse_args()
    train_data = np.load(args.train_npz, allow_pickle=True)
    train_sentences = _strings(train_data[args.sentence_key])
    blocked = set(canonical(sentence) for sentence in train_sentences)
    blocked.update(blocked_sentences(args.blocked_npz, args.sentence_key))
    variants, rows, reasons = generate_variants(args, train_sentences, blocked)
    if not variants:
        raise RuntimeError("No variants generated.")

    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    print(f"Using device={device}; encoding {len(variants)} augmented sentences", flush=True)
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
        "train_npz": args.train_npz,
        "blocked_npz": args.blocked_npz,
        "train_count": len(train_sentences),
        "variants_per_sentence": args.variants_per_sentence,
        "quality_mode": args.quality_mode,
        "min_token_count": args.min_token_count,
        "min_phrase_count": args.min_phrase_count,
        "max_replacement_word_len": args.max_replacement_word_len,
        "generated_count": len(variants),
        "drop_reasons": reasons,
        "source_schema": _schema(train_data, args.schema_key),
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "seed": args.seed,
        "sample": [
            {
                "sentence": variants[i],
                "source": rows[i]["augmentation_source_sentence"],
                "operation": rows[i]["augmentation_operation"],
                "source_word_recall": rows[i]["source_word_recall"],
            }
            for i in range(min(20, len(variants)))
        ],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(variants, dtype=object),
        rows=np.asarray(rows, dtype=object),
        source_npz=np.asarray([args.train_npz] * len(variants), dtype=object),
        split=np.asarray(["train"] * len(variants), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
