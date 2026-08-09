#!/usr/bin/env python
"""Generate Sherlock-like sentences from high-frequency train vocabulary."""

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
from scripts.augment_sherlock_sva_recombinational import (
    build_entries,
    clean_tokens,
    phrasebank_candidate,
    slot_candidate,
)
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
NOUN_CLASSES = {"noun", "noun_plural", "proper"}
VERB_CLASSES = {"verb_base", "verb_ed", "verb_ing", "verb_s"}


@dataclass(frozen=True)
class Template:
    source_index: int
    sentence: str
    tokens: tuple[str, ...]
    slots: tuple[tuple[int, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--token-coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts", type=int, default=800000)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--filter-mode", choices=["short_content", "simple_sv", "simple_sv_no_coord"], default="simple_sv_no_coord")
    parser.add_argument("--min-content-slots", type=int, default=2)
    parser.add_argument("--replacement-prob", type=float, default=0.90)
    parser.add_argument("--min-highfreq-content-frac", type=float, default=0.80)
    parser.add_argument("--chunk-mode", choices=["mixed", "slot", "phrasebank", "local"], default="local")
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


def content_items(tokens: list[str]) -> list[tuple[int, str, str]]:
    items = []
    for idx, token in enumerate(tokens):
        lower = token.lower()
        cls = broad_class(tokens, idx)
        if cls is None or lower in STOP_WORDS or lower in PREPOSITIONS:
            continue
        if lower == "s":
            continue
        items.append((idx, token, cls))
    return items


def corpus_counts(sentences: list[str]) -> tuple[Counter[str], Counter[str], dict[str, Counter[str]]]:
    all_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in sentences:
        tokens = words(sentence)
        all_counts.update(token.lower() for token in tokens)
        for idx, token, cls in content_items(tokens):
            lower = token.lower()
            if lower.isalpha():
                content_counts[lower] += 1
                class_counts[cls][lower] += 1
    return all_counts, content_counts, class_counts


def cumulative_vocab(counter: Counter[str], coverage: float) -> tuple[set[str], float]:
    total = max(1, sum(counter.values()))
    running = 0
    vocab = set()
    for token, count in counter.most_common():
        vocab.add(token)
        running += count
        if running / total >= coverage:
            return vocab, running / total
    return vocab, running / total


def coverage_cut(counter: Counter[str], coverage: float) -> dict[str, float]:
    total = max(1, sum(counter.values()))
    running = 0
    for rank, (_token, count) in enumerate(counter.most_common(), start=1):
        running += count
        if running / total >= coverage:
            return {"rank": float(rank), "coverage": float(running / total)}
    return {"rank": float(len(counter)), "coverage": float(running / total)}


def summarize_zipf(all_counts: Counter[str], content_counts: Counter[str]) -> dict[str, Any]:
    return {
        "all_tokens": int(sum(all_counts.values())),
        "all_vocab": int(len(all_counts)),
        "content_tokens": int(sum(content_counts.values())),
        "content_vocab": int(len(content_counts)),
        "coverage_cuts": {
            str(cov): {
                "all": coverage_cut(all_counts, cov),
                "content": coverage_cut(content_counts, cov),
            }
            for cov in (0.50, 0.70, 0.80, 0.90, 0.95)
        },
        "top_all": all_counts.most_common(50),
        "top_content": content_counts.most_common(100),
    }


def build_templates(sentences: list[str], args: argparse.Namespace) -> list[Template]:
    exclude_re = re.compile(DEFAULT_META_RE, flags=re.IGNORECASE)
    templates = []
    for source_index, sentence in enumerate(sentences):
        ok, _reason = keep_sentence(
            sentence,
            min_words=args.min_words,
            max_words=args.max_words,
            min_alpha_fraction=0.75,
            min_content_words=2,
            require_verb=True,
            filter_mode=args.filter_mode,
            exclude_re=exclude_re,
        )
        if not ok:
            continue
        tokens = words(sentence)
        if tokens and tokens[0].lower() in NON_DECLARATIVE_INITIAL:
            continue
        slots = []
        for idx, _token, cls in content_items(tokens):
            slots.append((idx, cls))
        if len(slots) >= args.min_content_slots:
            templates.append(Template(source_index=source_index, sentence=sentence, tokens=tuple(tokens), slots=tuple(slots)))
    if len(templates) < 100:
        raise RuntimeError(f"Only found {len(templates)} usable templates.")
    return templates


def build_class_pools(
    *,
    highfreq_vocab: set[str],
    class_counts: dict[str, Counter[str]],
) -> dict[str, list[str]]:
    pools = {}
    for cls, counter in class_counts.items():
        values = [
            token
            for token, count in counter.most_common()
            if token in highfreq_vocab and count >= 2 and 2 <= len(token) <= 16
        ]
        if values:
            pools[cls] = values
    return pools


def compatible_pool(cls: str, pools: dict[str, list[str]]) -> list[str]:
    exact = pools.get(cls, [])
    if exact:
        return exact
    if cls in NOUN_CLASSES:
        return sum((pools.get(item, []) for item in ("noun", "noun_plural", "proper")), [])
    if cls in VERB_CLASSES:
        return sum((pools.get(item, []) for item in ("verb_base", "verb_ed", "verb_ing", "verb_s")), [])
    return pools.get(cls, [])


def highfreq_fraction(sentence: str, highfreq_vocab: set[str]) -> float:
    tokens = words(sentence)
    content = [token.lower() for _idx, token, _cls in content_items(tokens)]
    if not content:
        return 0.0
    return sum(token in highfreq_vocab for token in content) / len(content)


def chunk_highfreq_fraction(tokens: tuple[str, ...] | list[str], highfreq_vocab: set[str]) -> float:
    content = []
    for token in tokens:
        lower = token.lower()
        if lower in STOP_WORDS or lower in PREPOSITIONS or lower == "s":
            continue
        content.append(lower)
    if not content:
        return 0.0
    return sum(token in highfreq_vocab for token in content) / len(content)


def generate_candidate(
    template: Template,
    pools: dict[str, list[str]],
    highfreq_vocab: set[str],
    rng: random.Random,
    replacement_prob: float,
) -> tuple[str, dict[str, Any]] | None:
    tokens = list(template.tokens)
    replacements = []
    for idx, cls in template.slots:
        if rng.random() > replacement_prob and tokens[idx].lower() in highfreq_vocab:
            continue
        pool = compatible_pool(cls, pools)
        if not pool:
            continue
        original = tokens[idx]
        candidates = [token for token in rng.sample(pool, min(len(pool), 100)) if token != original.lower()]
        if not candidates:
            continue
        replacement = apply_case(rng.choice(candidates), original)
        tokens[idx] = replacement
        replacements.append(f"{idx}:{cls}:{original}->{replacement}")
    if not replacements:
        return None
    return cleanup_sentence(tokens), {
        "template_source_index": template.source_index,
        "template_source_sentence": template.sentence,
        "replacement_operations": replacements,
    }


def generate_chunk_candidate(entries: list[Any], rng: random.Random, mode: str) -> tuple[str, dict[str, Any]]:
    if mode == "slot":
        sentence, row = slot_candidate(entries, rng)
    elif mode == "phrasebank":
        sentence, row = phrasebank_candidate(entries, rng)
    elif mode == "mixed":
        sentence, row = (slot_candidate if rng.random() < 0.5 else phrasebank_candidate)(entries, rng)
        row = {**row, "augmentation_mode": f"highfreq_{row['augmentation_mode']}"}
    else:
        raise ValueError(f"Unsupported chunk mode: {mode}")
    return sentence, row


def build_local_chunk_banks(entries: list[Any], highfreq_vocab: set[str]) -> tuple[list[tuple[int, tuple[str, ...]]], dict[str, list[tuple[int, tuple[str, ...]]]]]:
    objects: list[tuple[int, tuple[str, ...]]] = []
    pps: dict[str, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    for entry in entries:
        if entry.obj and chunk_highfreq_fraction(entry.obj, highfreq_vocab) >= 0.80:
            objects.append((entry.index, entry.obj))
        for pp in entry.pps:
            if pp and chunk_highfreq_fraction(pp, highfreq_vocab) >= 0.80:
                pps[pp[0].lower()].append((entry.index, pp))
    return objects, pps


def generate_local_chunk_candidate(
    *,
    entries: list[Any],
    object_bank: list[tuple[int, tuple[str, ...]]],
    pp_bank: dict[str, list[tuple[int, tuple[str, ...]]]],
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    base = rng.choice(entries)
    tokens = list(base.subject + base.verb)
    changed = False
    object_source_index = None
    pp_source_indices: list[int] = []

    obj = base.obj
    if object_bank and (base.obj and rng.random() < 0.60 or not base.obj):
        choices = [item for item in rng.sample(object_bank, min(len(object_bank), 128)) if item[0] != base.index]
        if choices:
            object_source_index, obj = rng.choice(choices)
            changed = True
    tokens.extend(obj)

    pps: list[tuple[str, ...]] = []
    for pp in base.pps:
        prep = pp[0].lower() if pp else ""
        candidates = pp_bank.get(prep, [])
        if candidates and rng.random() < 0.50:
            choices = [item for item in rng.sample(candidates, min(len(candidates), 128)) if item[0] != base.index]
            if choices:
                pp_source_index, new_pp = rng.choice(choices)
                pps.append(new_pp)
                pp_source_indices.append(int(pp_source_index))
                changed = True
                continue
        pps.append(pp)

    if not pps and pp_bank and rng.random() < 0.35:
        prep = rng.choice(list(pp_bank.keys()))
        pp_source_index, new_pp = rng.choice(pp_bank[prep])
        pps.append(new_pp)
        pp_source_indices.append(int(pp_source_index))
        changed = True

    if not changed:
        if object_bank:
            choices = [item for item in rng.sample(object_bank, min(len(object_bank), 128)) if item[0] != base.index]
            if choices:
                object_source_index, obj = rng.choice(choices)
                tokens = list(base.subject + base.verb + obj)
                changed = True
        if not changed and pp_bank:
            prep = rng.choice(list(pp_bank.keys()))
            pp_source_index, new_pp = rng.choice(pp_bank[prep])
            pps = [new_pp]
            pp_source_indices.append(int(pp_source_index))

    for pp in pps:
        tokens.extend(pp)
    return clean_tokens(tokens), {
        "augmentation_mode": "highfreq_local_chunk",
        "base_source_index": base.index,
        "object_source_index": object_source_index,
        "pp_source_indices": pp_source_indices,
    }


def blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def valid_candidate(
    sentence: str,
    *,
    seen: set[str],
    highfreq_vocab: set[str],
    args: argparse.Namespace,
    exclude_re: re.Pattern[str],
) -> tuple[bool, str, float]:
    if not sentence or canonical(sentence) in seen:
        return False, "blocked_or_duplicate", 0.0
    token_list = words(sentence)
    if any(left.lower() == right.lower() for left, right in zip(token_list, token_list[1:])):
        return False, "repeated_adjacent_word", 0.0
    frac = highfreq_fraction(sentence, highfreq_vocab)
    if frac < args.min_highfreq_content_frac:
        return False, "low_highfreq_content_fraction", frac
    ok, reason = keep_sentence(
        sentence,
        min_words=args.min_words,
        max_words=args.max_words,
        min_alpha_fraction=0.75,
        min_content_words=2,
        require_verb=True,
        filter_mode=args.filter_mode,
        exclude_re=exclude_re,
    )
    return ok, reason, frac


def generate_sentences(
    *,
    train_sentences: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    all_counts, content_counts, class_counts = corpus_counts(train_sentences)
    highfreq_vocab, actual_coverage = cumulative_vocab(content_counts, args.token_coverage)
    entries, exclude_re = build_entries(train_sentences, args)
    object_bank, pp_bank = build_local_chunk_banks(entries, highfreq_vocab)
    seen = {canonical(sentence) for sentence in train_sentences}
    seen.update(blocked_sentences(args.blocked_npz, args.sentence_key))

    generated: list[str] = []
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    attempts = 0
    while len(generated) < args.count and attempts < args.max_attempts:
        attempts += 1
        if args.chunk_mode == "local":
            sentence, row = generate_local_chunk_candidate(
                entries=entries,
                object_bank=object_bank,
                pp_bank=pp_bank,
                rng=rng,
            )
        else:
            sentence, row = generate_chunk_candidate(entries=entries, rng=rng, mode=args.chunk_mode)
        ok, reason, frac = valid_candidate(
            sentence,
            seen=seen,
            highfreq_vocab=highfreq_vocab,
            args=args,
            exclude_re=exclude_re,
        )
        reasons[reason if ok else f"drop_{reason}"] += 1
        if not ok:
            continue
        seen.add(canonical(sentence))
        generated.append(sentence)
        rows.append(
            {
                "augmentation_mode": "highfreq_template",
                "sentence": sentence,
                "word_count": len(words(sentence)),
                "highfreq_content_fraction": float(frac),
                **row,
            }
        )
        if len(generated) % 1000 == 0:
            print(f"generated {len(generated)} attempts={attempts}", flush=True)
    if len(generated) < args.count:
        raise RuntimeError(f"Generated {len(generated)} sentences out of requested {args.count}.")

    stats = {
        "zipf": summarize_zipf(all_counts, content_counts),
        "requested_content_token_coverage": float(args.token_coverage),
        "actual_content_token_coverage": float(actual_coverage),
        "highfreq_content_vocab_size": int(len(highfreq_vocab)),
        "class_pool_sizes": {
            key: len(value)
            for key, value in sorted(build_class_pools(highfreq_vocab=highfreq_vocab, class_counts=class_counts).items())
        },
        "parsed_entry_count": int(len(entries)),
        "object_bank_size": int(len(object_bank)),
        "pp_bank_size": int(sum(len(items) for items in pp_bank.values())),
        "pp_bank_prepositions": {key: len(value) for key, value in sorted(pp_bank.items())},
        "chunk_mode": args.chunk_mode,
        "attempts": int(attempts),
        "drop_reasons": dict(reasons),
    }
    return generated, rows, stats


def main() -> None:
    args = parse_args()
    train_data = np.load(args.train_npz, allow_pickle=True)
    train_sentences = _strings(train_data[args.sentence_key])
    generated, rows, stats = generate_sentences(train_sentences=train_sentences, args=args)

    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    print(f"Using device={device}; encoding {len(generated)} high-frequency sentences", flush=True)
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
        sentences=[f"{args.input_prefix}{sentence}" for sentence in generated],
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
    highfreq_fracs = [float(row["highfreq_content_fraction"]) for row in rows]
    summary = {
        "output": str(output),
        "train_npz": args.train_npz,
        "blocked_npz": args.blocked_npz,
        "count": len(generated),
        "token_coverage": float(args.token_coverage),
        "min_highfreq_content_frac": float(args.min_highfreq_content_frac),
        "replacement_prob": float(args.replacement_prob),
        "chunk_mode": args.chunk_mode,
        "highfreq_content_fraction_mean": float(np.mean(highfreq_fracs)),
        "stats": stats,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "seed": args.seed,
        "source_schema": _schema(train_data, args.schema_key),
        "sample": rows[:30],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(generated, dtype=object),
        rows=np.asarray(rows, dtype=object),
        source_npz=np.asarray([args.train_npz] * len(generated), dtype=object),
        split=np.asarray(["train"] * len(generated), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
