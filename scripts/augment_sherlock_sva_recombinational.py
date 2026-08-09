#!/usr/bin/env python
"""Generate Sherlock-style SVA recombinational augmentations and embeddings."""

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
import torch
from transformers import AutoModel, AutoTokenizer

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


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


@dataclass(frozen=True)
class ParsedSentence:
    index: int
    sentence: str
    tokens: tuple[str, ...]
    subject: tuple[str, ...]
    verb: tuple[str, ...]
    obj: tuple[str, ...]
    pps: tuple[tuple[str, ...], ...]
    tail: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["slot", "phrasebank", "devcover"],
        default="slot",
    )
    parser.add_argument("--target-count", type=int, default=50000)
    parser.add_argument("--candidate-count", type=int, default=160000)
    parser.add_argument("--dev-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts", type=int, default=1200000)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--filter-mode", choices=["simple_sv", "simple_sv_no_coord"], default="simple_sv_no_coord")
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
    parser.add_argument("--selection-batch-size", type=int, default=4096)
    parser.add_argument("--selection-top-k", type=int, default=96)
    parser.add_argument("--cosine-weight", type=float, default=0.50)
    parser.add_argument("--lexical-weight", type=float, default=0.40)
    parser.add_argument("--length-weight", type=float, default=0.10)
    parser.add_argument("--max-per-dev-query", type=int, default=96)
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


def canonical(sentence: str) -> str:
    return " ".join(token.lower() for token in words(sentence))


def normalize_vector_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def content_tokens(sentence_or_tokens: str | tuple[str, ...] | list[str]) -> set[str]:
    tokens = words(sentence_or_tokens) if isinstance(sentence_or_tokens, str) else list(sentence_or_tokens)
    return {token.lower() for token in tokens if token.lower() not in STOP_WORDS and len(token) > 2}


def word_f1(left: str | tuple[str, ...], right: str | tuple[str, ...]) -> float:
    left_tokens = list(left) if isinstance(left, tuple) else words(left)
    right_tokens = list(right) if isinstance(right, tuple) else words(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts = Counter(token.lower() for token in left_tokens)
    right_counts = Counter(token.lower() for token in right_tokens)
    overlap = sum(min(count, right_counts.get(token, 0)) for token, count in left_counts.items())
    precision = overlap / max(1, len(left_tokens))
    recall = overlap / max(1, len(right_tokens))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def length_score(candidate: str, query: str) -> float:
    candidate_len = max(1, len(words(candidate)))
    query_len = max(1, len(words(query)))
    return max(0.0, 1.0 - abs(candidate_len - query_len) / max(candidate_len, query_len))


def first_verb_index(tokens: list[str]) -> int | None:
    for idx, token in enumerate(tokens):
        lower = token.lower()
        if lower not in VERBISH and not lower.endswith(("ed", "ing")):
            continue
        before = [item for item in tokens[:idx] if item.lower() not in STOP_WORDS and len(item) > 2]
        after = [item for item in tokens[idx + 1 :] if item.lower() not in STOP_WORDS and len(item) > 2]
        if before and after:
            return idx
    return None


def split_tail(tail: list[str]) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    first_pp = None
    for idx, token in enumerate(tail):
        if token.lower() in PREPOSITIONS:
            first_pp = idx
            break
    if first_pp is None:
        return tuple(tail), tuple()

    obj = tuple(tail[:first_pp])
    pps: list[tuple[str, ...]] = []
    idx = first_pp
    while idx < len(tail):
        end = idx + 1
        while end < len(tail) and tail[end].lower() not in PREPOSITIONS:
            end += 1
        if end - idx >= 2:
            pps.append(tuple(tail[idx:end]))
        idx = end
    return obj, tuple(pps)


def parse_sentence(index: int, sentence: str, exclude_re: re.Pattern[str], args: argparse.Namespace) -> ParsedSentence | None:
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
        return None
    tokens = words(sentence)
    if not tokens or tokens[0].lower() in NON_DECLARATIVE_INITIAL:
        return None
    verb_idx = first_verb_index(tokens)
    if verb_idx is None or verb_idx == 0 or verb_idx >= len(tokens) - 1:
        return None
    verb_end = verb_idx + 1
    if (
        verb_end < len(tokens)
        and tokens[verb_idx].lower() in {"is", "are", "was", "were", "am", "be", "been", "being", "has", "had", "have"}
        and (tokens[verb_end].lower() in VERBISH or tokens[verb_end].lower().endswith(("ed", "ing")))
    ):
        verb_end += 1
    subject = tuple(tokens[:verb_idx])
    verb = tuple(tokens[verb_idx:verb_end])
    tail = tuple(tokens[verb_end:])
    if not subject or not verb or not tail:
        return None
    obj, pps = split_tail(list(tail))
    return ParsedSentence(
        index=index,
        sentence=sentence,
        tokens=tuple(tokens),
        subject=subject,
        verb=verb,
        obj=obj,
        pps=pps,
        tail=tail,
    )


def clean_tokens(tokens: list[str]) -> str:
    cleaned = [token for token in tokens if token]
    if not cleaned:
        return ""
    cleaned[0] = cleaned[0][:1].upper() + cleaned[0][1:]
    sentence = " ".join(cleaned)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"\b([Aa]) ([aeiouAEIOU])", lambda match: f"{match.group(1)}n {match.group(2)}", sentence)
    sentence = re.sub(r"\b([Aa])n ([^aeiouAEIOU\\W])", lambda match: f"{match.group(1)} {match.group(2)}", sentence)
    return sentence


def valid_candidate(sentence: str, *, seen: set[str], exclude_re: re.Pattern[str], args: argparse.Namespace) -> bool:
    key = canonical(sentence)
    if not sentence or key in seen:
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
        exclude_re=exclude_re,
    )
    return ok


def build_entries(sentences: list[str], args: argparse.Namespace) -> tuple[list[ParsedSentence], re.Pattern[str]]:
    exclude_re = re.compile(DEFAULT_META_RE, flags=re.IGNORECASE)
    entries = []
    for idx, sentence in enumerate(sentences):
        parsed = parse_sentence(idx, sentence, exclude_re, args)
        if parsed is not None:
            entries.append(parsed)
    if len(entries) < 100:
        raise RuntimeError(f"Only parsed {len(entries)} usable Sherlock SVA sentences.")
    return entries, exclude_re


def slot_candidate(entries: list[ParsedSentence], rng: random.Random) -> tuple[str, dict[str, Any]]:
    subject_src = rng.choice(entries)
    verb_src = rng.choice(entries)
    tail_src = rng.choice(entries)
    tokens = list(subject_src.subject + verb_src.verb + tail_src.tail)
    return clean_tokens(tokens), {
        "augmentation_mode": "slot",
        "subject_source_index": subject_src.index,
        "verb_source_index": verb_src.index,
        "tail_source_index": tail_src.index,
    }


def phrasebank_candidate(entries: list[ParsedSentence], rng: random.Random) -> tuple[str, dict[str, Any]]:
    subject_src = rng.choice(entries)
    verb_src = rng.choice(entries)
    object_entries = [entry for entry in entries if entry.obj]
    pp_entries = [entry for entry in entries if entry.pps]
    obj_src = rng.choice(object_entries) if object_entries and rng.random() < 0.75 else None
    pp_count = rng.choice([0, 1, 1, 2])
    pp_sources = rng.sample(pp_entries, k=min(pp_count, len(pp_entries))) if pp_entries and pp_count else []
    tail_tokens: list[str] = []
    if obj_src is not None:
        tail_tokens.extend(obj_src.obj)
    elif not pp_sources:
        tail_src = rng.choice(entries)
        tail_tokens.extend(tail_src.tail)
    for pp_src in pp_sources:
        tail_tokens.extend(rng.choice(pp_src.pps))
    if not tail_tokens:
        tail_tokens.extend(rng.choice(entries).tail)
    tokens = list(subject_src.subject + verb_src.verb + tuple(tail_tokens))
    return clean_tokens(tokens), {
        "augmentation_mode": "phrasebank",
        "subject_source_index": subject_src.index,
        "verb_source_index": verb_src.index,
        "object_source_index": None if obj_src is None else obj_src.index,
        "pp_source_indices": [entry.index for entry in pp_sources],
    }


def generate_candidates(
    *,
    entries: list[ParsedSentence],
    mode: str,
    count: int,
    seen: set[str],
    exclude_re: re.Pattern[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[str], list[dict[str, Any]], Counter[str]]:
    candidates: list[str] = []
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    attempts = 0
    while len(candidates) < count and attempts < args.max_attempts:
        attempts += 1
        if mode == "slot":
            sentence, row = slot_candidate(entries, rng)
        elif mode == "phrasebank":
            sentence, row = phrasebank_candidate(entries, rng)
        elif mode == "mixed":
            sentence, row = (slot_candidate if rng.random() < 0.5 else phrasebank_candidate)(entries, rng)
            row["augmentation_mode"] = f"devcover_{row['augmentation_mode']}"
        else:
            raise ValueError(f"Unsupported generation mode: {mode}")
        if not valid_candidate(sentence, seen=seen, exclude_re=exclude_re, args=args):
            reasons["dropped_invalid_or_duplicate"] += 1
            continue
        seen.add(canonical(sentence))
        row = {
            **row,
            "sentence": sentence,
            "word_count": len(words(sentence)),
        }
        candidates.append(sentence)
        rows.append(row)
    if len(candidates) < count:
        raise RuntimeError(f"Only generated {len(candidates)} valid candidates out of requested {count}.")
    reasons["attempts"] = attempts
    return candidates, rows, reasons


def choose_dev_indices(sentences: list[str], *, count: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    return np.asarray(sorted(indices[: min(count, len(indices))]), dtype=np.int64)


def select_by_internal_dev_coverage(
    *,
    candidates: list[str],
    rows: list[dict[str, Any]],
    candidate_embeddings: np.ndarray,
    train_sentences: list[str],
    train_embeddings: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    dev_indices = choose_dev_indices(train_sentences, count=args.dev_count, seed=args.seed + 1009)
    query_embeddings = normalize_vector_rows(train_embeddings[dev_indices])
    candidate_embeddings = normalize_vector_rows(candidate_embeddings)
    query_tensor = torch.as_tensor(query_embeddings, dtype=torch.float32, device=device)
    selected_pairs: list[dict[str, Any]] = []
    selected_count_by_query: Counter[int] = Counter()

    for start in range(0, candidate_embeddings.shape[0], args.selection_batch_size):
        end = min(start + args.selection_batch_size, candidate_embeddings.shape[0])
        candidate_tensor = torch.as_tensor(candidate_embeddings[start:end], dtype=torch.float32, device=device)
        sims = query_tensor @ candidate_tensor.T
        top_k = min(args.selection_top_k, end - start)
        scores, indices = torch.topk(sims, k=top_k, dim=1)
        scores_np = scores.detach().cpu().numpy()
        indices_np = indices.detach().cpu().numpy()
        for query_local, query_global in enumerate(dev_indices.tolist()):
            query_sentence = train_sentences[query_global]
            for slot in range(top_k):
                candidate_idx = int(start + indices_np[query_local, slot])
                candidate_sentence = candidates[candidate_idx]
                lexical = word_f1(candidate_sentence, query_sentence)
                length = length_score(candidate_sentence, query_sentence)
                score = (
                    args.cosine_weight * float(scores_np[query_local, slot])
                    + args.lexical_weight * lexical
                    + args.length_weight * length
                )
                selected_pairs.append(
                    {
                        "candidate_index": candidate_idx,
                        "dev_query_index": int(query_global),
                        "dev_query_sentence": query_sentence,
                        "cosine_score": float(scores_np[query_local, slot]),
                        "lexical_f1": float(lexical),
                        "length_score": float(length),
                        "selection_score": float(score),
                    }
                )
        print(f"selected candidate scores {end}/{candidate_embeddings.shape[0]}", flush=True)

    selected_pairs.sort(key=lambda item: item["selection_score"], reverse=True)
    selected_indices: list[int] = []
    used: set[int] = set()
    for pair in selected_pairs:
        query_idx = int(pair["dev_query_index"])
        if selected_count_by_query[query_idx] >= args.max_per_dev_query:
            continue
        candidate_idx = int(pair["candidate_index"])
        if candidate_idx in used:
            continue
        used.add(candidate_idx)
        selected_count_by_query[query_idx] += 1
        selected_indices.append(candidate_idx)
        if len(selected_indices) >= args.target_count:
            break
    if len(selected_indices) < args.target_count:
        for candidate_idx in range(len(candidates)):
            if candidate_idx not in used:
                selected_indices.append(candidate_idx)
                used.add(candidate_idx)
                if len(selected_indices) >= args.target_count:
                    break

    selected_sentences = [candidates[idx] for idx in selected_indices[: args.target_count]]
    selected_rows = []
    best_pair_by_candidate: dict[int, dict[str, Any]] = {}
    for pair in selected_pairs:
        candidate_idx = int(pair["candidate_index"])
        if (
            candidate_idx not in best_pair_by_candidate
            or pair["selection_score"] > best_pair_by_candidate[candidate_idx]["selection_score"]
        ):
            best_pair_by_candidate[candidate_idx] = pair
    for idx in selected_indices[: args.target_count]:
        selected_rows.append(
            {
                **rows[idx],
                "augmentation_mode": "devcover",
                "candidate_pool_index": int(idx),
                "selection": best_pair_by_candidate.get(int(idx), {}),
            }
        )
    stats = {
        "dev_count": int(len(dev_indices)),
        "candidate_pool_count": int(len(candidates)),
        "selected_count": int(len(selected_sentences)),
        "selected_dev_queries": int(len(selected_count_by_query)),
        "selection_score_top": selected_pairs[:5],
    }
    return selected_sentences, selected_rows, stats


def load_blocked(paths: list[str], sentence_key: str) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_data = np.load(args.train_npz, allow_pickle=True)
    train_sentences = _strings(train_data[args.sentence_key])
    train_embeddings = np.asarray(train_data[args.input_key], dtype=np.float32)
    entries, exclude_re = build_entries(train_sentences, args)
    seen = {canonical(sentence) for sentence in train_sentences}
    seen.update(load_blocked(args.blocked_npz, args.sentence_key))

    device = resolve_device(args.device)
    print(f"Using device={device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    generation_mode = "mixed" if args.mode == "devcover" else args.mode
    generation_count = args.candidate_count if args.mode == "devcover" else args.target_count
    candidates, rows, reasons = generate_candidates(
        entries=entries,
        mode=generation_mode,
        count=generation_count,
        seen=seen,
        exclude_re=exclude_re,
        args=args,
        rng=rng,
    )
    embedding_inputs = [f"{args.input_prefix}{sentence}" for sentence in candidates]
    embeddings = encode_sentences(
        sentences=embedding_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        normalize=not args.no_normalize,
    )

    selection_stats = None
    if args.mode == "devcover":
        candidates, rows, selection_stats = select_by_internal_dev_coverage(
            candidates=candidates,
            rows=rows,
            candidate_embeddings=embeddings,
            train_sentences=train_sentences,
            train_embeddings=train_embeddings,
            args=args,
            device=device,
        )
        selected_indices = [int(row["candidate_pool_index"]) for row in rows]
        embeddings = embeddings[np.asarray(selected_indices, dtype=np.int64)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "train_npz": args.train_npz,
        "blocked_npz": args.blocked_npz,
        "mode": args.mode,
        "target_count": args.target_count,
        "candidate_count": generation_count,
        "kept_count": len(candidates),
        "parsed_train_sentences": len(entries),
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "seed": args.seed,
        "generation_reasons": dict(reasons),
        "selection_stats": selection_stats,
        "source_schema": _schema(train_data, args.schema_key),
        "sample": candidates[:20],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(candidates, dtype=object),
        rows=np.asarray(rows, dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
