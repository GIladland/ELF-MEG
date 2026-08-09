#!/usr/bin/env python
"""Generate oracle augmentations that separate sentence structure from lexicon."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.augment_semantic_npz_sentences import PREPOSITIONS, apply_case, cleanup_sentence, token_class
from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import (
    DEFAULT_META_RE,
    NON_DECLARATIVE_INITIAL,
    STOP_WORDS,
    keep_sentence,
    words,
)


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


CONTENT_CLASSES = {
    "adjective",
    "noun",
    "noun_plural",
    "number",
    "proper",
    "verb_base",
    "verb_ed",
    "verb_ing",
    "verb_s",
}


@dataclass(frozen=True)
class SlotTemplate:
    source_index: int
    tokens: tuple[str, ...]
    slots: tuple[tuple[int, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True, help="Oracle source NPZ, normally held-out validation.")
    parser.add_argument("--support-npz", required=True, help="Non-heldout train NPZ used for lexicon/templates.")
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["structure_only", "lexicon_template"],
        required=True,
    )
    parser.add_argument("--variants-per-source", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts-per-source", type=int, default=3000)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--filter-mode", choices=["short_content", "simple_sv", "simple_sv_no_coord"], default="short_content")
    parser.add_argument("--require-verb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-structure-content-f1", type=float, default=0.35)
    parser.add_argument("--min-lexicon-content-recall", type=float, default=0.65)
    parser.add_argument(
        "--strict-same-class",
        action="store_true",
        help="For lexicon_template, only insert validation words into template slots with the same inferred class.",
    )
    parser.add_argument(
        "--no-append-unused",
        action="store_true",
        help="For lexicon_template, do not append leftover validation words to the sentence.",
    )
    parser.add_argument(
        "--template-length-delta",
        type=int,
        default=5,
        help="Preferred max absolute token-length difference between source and strict template.",
    )
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


def broad_class(tokens: list[str], idx: int) -> str | None:
    cls = token_class(tokens, idx)
    return cls if cls in CONTENT_CLASSES else None


def source_content(tokens: list[str]) -> list[tuple[int, str, str]]:
    items = []
    for idx, token in enumerate(tokens):
        lower = token.lower()
        cls = broad_class(tokens, idx)
        if cls is None or lower in STOP_WORDS or lower in PREPOSITIONS:
            continue
        items.append((idx, token, cls))
    return items


def content_counter(sentence: str) -> Counter[str]:
    tokens = words(sentence)
    counter: Counter[str] = Counter()
    for _idx, token, _cls in source_content(tokens):
        counter[token.lower()] += 1
    return counter


def f1_from_counters(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right.get(token, 0)) for token, count in left.items())
    precision = overlap / max(1, sum(right.values()))
    recall = overlap / max(1, sum(left.values()))
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def recall_from_counters(source: Counter[str], variant: Counter[str]) -> float:
    if not source:
        return 0.0
    overlap = sum(min(count, variant.get(token, 0)) for token, count in source.items())
    return overlap / max(1, sum(source.values()))


def word_f1(source: str, variant: str) -> float:
    left = Counter(token.lower() for token in words(source))
    right = Counter(token.lower() for token in words(variant))
    return f1_from_counters(left, right)


def build_lexicon_pools(support_sentences: list[str]) -> dict[str, list[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in support_sentences:
        tokens = words(sentence)
        for idx, token in enumerate(tokens):
            lower = token.lower()
            cls = broad_class(tokens, idx)
            if cls is None or lower in STOP_WORDS or lower in PREPOSITIONS:
                continue
            if not lower.isalpha() or len(lower) <= 1:
                continue
            counts[cls][lower] += 1
    return {
        cls: [token for token, _count in counter.most_common() if len(token) <= 16]
        for cls, counter in counts.items()
        if len(counter) >= 2
    }


def build_templates(support_sentences: list[str], args: argparse.Namespace) -> list[SlotTemplate]:
    exclude_re = re.compile(DEFAULT_META_RE, flags=re.IGNORECASE)
    templates = []
    for source_index, sentence in enumerate(support_sentences):
        ok, _reason = keep_sentence(
            sentence,
            min_words=args.min_words,
            max_words=args.max_words,
            min_alpha_fraction=0.75,
            min_content_words=2,
            require_verb=args.require_verb,
            filter_mode=args.filter_mode,
            exclude_re=exclude_re,
        )
        if not ok:
            continue
        tokens = words(sentence)
        if tokens and tokens[0].lower() in NON_DECLARATIVE_INITIAL:
            continue
        slots = []
        for idx, token in enumerate(tokens):
            lower = token.lower()
            cls = broad_class(tokens, idx)
            if cls is None or lower in STOP_WORDS or lower in PREPOSITIONS:
                continue
            slots.append((idx, cls))
        if len(slots) >= 2:
            templates.append(SlotTemplate(source_index=source_index, tokens=tuple(tokens), slots=tuple(slots)))
    if len(templates) < 100:
        raise RuntimeError(f"Only built {len(templates)} usable support templates.")
    return templates


def strict_template_candidates(
    *,
    source_tokens: list[str],
    content: list[tuple[int, str, str]],
    templates: list[SlotTemplate],
    args: argparse.Namespace,
) -> list[SlotTemplate]:
    source_class_counts = Counter(cls for _idx, _token, cls in content)

    def template_recall(template_: SlotTemplate) -> float:
        template_class_counts = Counter(cls for _idx, cls in template_.slots)
        matched = sum(
            min(count, template_class_counts.get(cls, 0))
            for cls, count in source_class_counts.items()
        )
        return matched / max(1, len(content))

    scored = [
        (template_recall(template_), template_)
        for template_ in templates
        if abs(len(template_.tokens) - len(source_tokens)) <= args.template_length_delta
    ]
    if not scored:
        scored = [(template_recall(template_), template_) for template_ in templates]
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    top_recall = scored[0][0]
    threshold = max(args.min_lexicon_content_recall, top_recall - 0.05)
    return [template_ for score, template_ in scored if score >= threshold]


def structure_only_candidate(
    source_sentence: str,
    pools: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str, dict[str, Any]] | None:
    tokens = words(source_sentence)
    content = source_content(tokens)
    if not content:
        return None
    source_terms = {token.lower() for _idx, token, _cls in content}
    new_tokens = list(tokens)
    replacements = []
    for idx, token, cls in content:
        pool = [
            item
            for item in pools.get(cls, [])
            if item != token.lower() and item not in source_terms
        ]
        if not pool:
            continue
        replacement = apply_case(rng.choice(pool[: min(len(pool), 300)]), token)
        new_tokens[idx] = replacement
        replacements.append(f"{idx}:{cls}:{token}->{replacement}")
    if not replacements:
        return None
    return cleanup_sentence(new_tokens), {"replacement_operations": replacements}


def lexicon_template_candidate(
    source_sentence: str,
    templates: list[SlotTemplate],
    rng: random.Random,
    args: argparse.Namespace,
    strict_templates: list[SlotTemplate] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    source_tokens = words(source_sentence)
    content = source_content(source_tokens)
    if not content:
        return None
    candidates_by_cls: dict[str, list[str]] = defaultdict(list)
    for _idx, token, cls in content:
        candidates_by_cls[cls].append(token)
    if args.strict_same_class:
        viable = strict_templates or []
        if not viable:
            return None
        template = rng.choice(viable[: min(len(viable), 300)])
    else:
        template = rng.choice(templates)
        for _attempt in range(30):
            if len(template.tokens) <= args.max_words:
                break
            template = rng.choice(templates)
    slot_indices = {idx: cls for idx, cls in template.slots}
    unused = list(content)
    if not args.strict_same_class:
        rng.shuffle(unused)
    output_tokens = list(template.tokens)
    used_source_tokens = []

    for idx, cls in template.slots:
        same_cls = [item for item in unused if item[2] == cls]
        if same_cls:
            chosen = rng.choice(same_cls) if not args.strict_same_class else same_cls[0]
            unused.remove(chosen)
        elif unused and not args.strict_same_class:
            chosen = rng.choice(unused)
            unused.remove(chosen)
        elif not args.strict_same_class:
            pool = candidates_by_cls.get(cls) or [item[1] for item in content]
            chosen = (-1, rng.choice(pool), cls)
        else:
            continue
        replacement = apply_case(chosen[1], output_tokens[idx])
        output_tokens[idx] = replacement
        used_source_tokens.append(chosen[1].lower())

    if unused and not args.no_append_unused and len(output_tokens) + len(unused) + 1 <= args.max_words:
        output_tokens.append("with")
        output_tokens.extend(token for _idx, token, _cls in unused)
        used_source_tokens.extend(token.lower() for _idx, token, _cls in unused)

    if not used_source_tokens:
        return None
    return cleanup_sentence(output_tokens), {
        "template_source_index": template.source_index,
        "template_slot_count": len(slot_indices),
        "used_source_tokens": used_source_tokens,
    }


def blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def valid_candidate(
    *,
    source: str,
    candidate: str,
    mode: str,
    seen: set[str],
    args: argparse.Namespace,
    exclude_re: re.Pattern[str],
) -> tuple[bool, str, dict[str, float]]:
    key = canonical(candidate)
    if not candidate or key in seen:
        return False, "blocked_or_duplicate", {}
    token_list = words(candidate)
    if any(left.lower() == right.lower() for left, right in zip(token_list, token_list[1:])):
        return False, "repeated_adjacent_word", {}
    ok, reason = keep_sentence(
        candidate,
        min_words=args.min_words,
        max_words=args.max_words,
        min_alpha_fraction=0.75,
        min_content_words=2,
        require_verb=args.require_verb,
        filter_mode=args.filter_mode,
        exclude_re=exclude_re,
    )
    if not ok:
        return False, reason, {}

    source_counter = content_counter(source)
    candidate_counter = content_counter(candidate)
    metrics = {
        "source_word_f1": word_f1(source, candidate),
        "source_content_f1": f1_from_counters(source_counter, candidate_counter),
        "source_content_recall": recall_from_counters(source_counter, candidate_counter),
    }
    if mode == "structure_only" and metrics["source_content_f1"] > args.max_structure_content_f1:
        return False, "too_much_source_content", metrics
    if mode == "lexicon_template" and metrics["source_content_recall"] < args.min_lexicon_content_recall:
        return False, "too_little_source_content", metrics
    return True, "kept", metrics


def generate_variants(
    *,
    source_sentences: list[str],
    support_sentences: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]], Counter[str]]:
    rng = random.Random(args.seed)
    exclude_re = re.compile(DEFAULT_META_RE, flags=re.IGNORECASE)
    pools = build_lexicon_pools(support_sentences)
    templates = build_templates(support_sentences, args) if args.mode == "lexicon_template" else []
    seen = {canonical(sentence) for sentence in source_sentences}
    seen.update(canonical(sentence) for sentence in support_sentences)
    seen.update(blocked_sentences(args.blocked_npz, args.sentence_key))

    variants: list[str] = []
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for source_idx, source in enumerate(source_sentences):
        source_tokens = words(source)
        if not source_tokens:
            reasons["empty_source"] += 1
            continue
        strict_templates = None
        if args.mode == "lexicon_template" and args.strict_same_class:
            strict_templates = strict_template_candidates(
                source_tokens=source_tokens,
                content=source_content(source_tokens),
                templates=templates,
                args=args,
            )
            if not strict_templates:
                reasons["source_without_strict_templates"] += 1
                print(f"source_idx={source_idx} made=0 total_variants={len(variants)}", flush=True)
                continue
        made = 0
        attempts = 0
        while made < args.variants_per_source and attempts < args.max_attempts_per_source:
            attempts += 1
            if args.mode == "structure_only":
                result = structure_only_candidate(source, pools, rng)
            else:
                result = lexicon_template_candidate(source, templates, rng, args, strict_templates=strict_templates)
            if result is None:
                reasons["no_candidate"] += 1
                continue
            candidate, meta = result
            ok, reason, metrics = valid_candidate(
                source=source,
                candidate=candidate,
                mode=args.mode,
                seen=seen,
                args=args,
                exclude_re=exclude_re,
            )
            reasons[reason if ok else f"drop_{reason}"] += 1
            if not ok:
                continue
            seen.add(canonical(candidate))
            variants.append(candidate)
            rows.append(
                {
                    "oracle_ablation_mode": args.mode,
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
    variants, rows, reasons = generate_variants(
        source_sentences=source_sentences,
        support_sentences=support_sentences,
        args=args,
    )
    if not variants:
        raise RuntimeError("No oracle ablation variants generated.")

    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    print(f"Using device={device}; encoding {len(variants)} variants", flush=True)
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
    source_word_f1 = [float(row["source_word_f1"]) for row in rows]
    source_content_f1 = [float(row["source_content_f1"]) for row in rows]
    source_content_recall = [float(row["source_content_recall"]) for row in rows]
    summary = {
        "output": str(output),
        "mode": args.mode,
        "source_npz": args.source_npz,
        "support_npz": args.support_npz,
        "blocked_npz": args.blocked_npz,
        "source_count": len(source_sentences),
        "support_count": len(support_sentences),
        "variants_per_source": args.variants_per_source,
        "generated_count": len(variants),
        "drop_reasons": dict(reasons),
        "source_word_f1_mean": float(np.mean(source_word_f1)),
        "source_content_f1_mean": float(np.mean(source_content_f1)),
        "source_content_recall_mean": float(np.mean(source_content_recall)),
        "source_schema": _schema(source_data, args.schema_key),
        "support_schema": _schema(support_data, args.schema_key),
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "seed": args.seed,
        "sample": rows[:20],
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
