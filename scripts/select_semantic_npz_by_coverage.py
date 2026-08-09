#!/usr/bin/env python
"""Select candidate semantic-vector rows for train-query coverage.

This is stricter than pure embedding-nearest selection: candidates are first
found by MiniLM cosine, then reranked by a mixture of cosine, content-word
overlap with the matched train query, and length match.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOP_WORDS = set(
    """
    the a an and or but if then than that this these those of in on at to for from with
    without by as is are was were be been being am i you he she it we they his her him
    them their my your our me us not no yes do did does have has had will would could
    should can may might must into out up down over under there here
    """.split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-npz", action="append", required=True)
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument(
        "--exclude-candidate-row-regex",
        action="append",
        default=[],
        help="Drop candidates whose serialized row metadata matches any regex.",
    )
    parser.add_argument("--per-query-candidates", type=int, default=32)
    parser.add_argument("--per-query-lexical-candidates", type=int, default=64)
    parser.add_argument("--lexical-token-max-candidates", type=int, default=5000)
    parser.add_argument("--candidate-batch-size", type=int, default=4096)
    parser.add_argument("--query-sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cosine-weight", type=float, default=0.45)
    parser.add_argument("--lexical-weight", type=float, default=0.45)
    parser.add_argument("--length-weight", type=float, default=0.10)
    parser.add_argument("--order-weight", type=float, default=0.0)
    parser.add_argument(
        "--lexical-mode",
        choices=["idf_recall", "idf_f1", "word_f1"],
        default="idf_recall",
    )
    parser.add_argument("--min-lexical-score", type=float, default=0.0)
    parser.add_argument("--max-per-query", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true")
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


def normalize_sentence_key(sentence: str) -> str:
    return " ".join(word.lower() for word in WORD_RE.findall(str(sentence)))


def words(sentence: str) -> list[str]:
    return WORD_RE.findall(str(sentence))


def content_tokens(sentence: str) -> list[str]:
    return [
        word.lower()
        for word in words(sentence)
        if len(word) > 2 and word.lower() not in STOP_WORDS
    ]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def maybe_normalize(array: np.ndarray, *, normalize: bool) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if not normalize:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def load_blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(normalize_sentence_key(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def query_token_stats(query_sentences: list[str]) -> tuple[list[set[str]], list[float], dict[str, float]]:
    token_sets = [set(content_tokens(sentence)) for sentence in query_sentences]
    df: Counter[str] = Counter()
    for token_set in token_sets:
        df.update(token_set)
    n = max(1, len(token_sets))
    idf = {token: math.log((n + 1) / (freq + 1)) + 1.0 for token, freq in df.items()}
    weights = [sum(idf.get(token, 1.0) for token in token_set) for token_set in token_sets]
    return token_sets, weights, idf


def lexical_recall(
    query_tokens: set[str],
    query_weight: float,
    candidate_tokens: set[str],
    idf: dict[str, float],
) -> float:
    if not query_tokens or query_weight <= 0:
        return 0.0
    return float(sum(idf.get(token, 1.0) for token in query_tokens & candidate_tokens) / query_weight)


def lexical_precision(candidate_tokens: set[str], query_tokens: set[str], idf: dict[str, float]) -> float:
    if not candidate_tokens:
        return 0.0
    denominator = sum(idf.get(token, 1.0) for token in candidate_tokens)
    if denominator <= 0:
        return 0.0
    return float(sum(idf.get(token, 1.0) for token in candidate_tokens & query_tokens) / denominator)


def harmonic_mean(left: float, right: float) -> float:
    return 0.0 if left + right <= 0 else float(2.0 * left * right / (left + right))


def set_word_f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return harmonic_mean(overlap / len(left), overlap / len(right))


def ordered_overlap(query_words: list[str], candidate_words: list[str]) -> float:
    if not query_words or not candidate_words:
        return 0.0
    match = SequenceMatcher(None, query_words, candidate_words, autojunk=False).find_longest_match()
    return harmonic_mean(match.size / len(query_words), match.size / len(candidate_words))


def row_text(row: Any) -> str:
    if isinstance(row, dict):
        return json.dumps(row, sort_keys=True, default=str)
    return str(row)


def length_match(query_word_count: int, candidate_word_count: int) -> float:
    denom = max(1, query_word_count, candidate_word_count)
    return max(0.0, 1.0 - abs(query_word_count - candidate_word_count) / denom)


def top_pairs_for_source(
    *,
    source_id: int,
    source_name: str,
    candidate_path: str,
    query_embeddings: np.ndarray,
    query_sentences: list[str],
    query_token_sets: list[set[str]],
    query_token_weights: list[float],
    query_word_counts: list[int],
    idf: dict[str, float],
    blocked: set[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(candidate_path, allow_pickle=True) as data:
        candidate_sentences = _strings(data[args.sentence_key])
        candidate_embeddings = maybe_normalize(
            data[args.input_key],
            normalize=not args.no_normalize,
        )
        blocked_mask = np.asarray(
            [normalize_sentence_key(sentence) in blocked for sentence in candidate_sentences],
            dtype=bool,
        )
        excluded_row_mask = np.zeros((len(candidate_sentences),), dtype=bool)
        if args.exclude_candidate_row_regex:
            if "rows" not in data.files:
                raise KeyError(
                    f"{candidate_path} has no rows metadata required by --exclude-candidate-row-regex"
                )
            patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.exclude_candidate_row_regex]
            excluded_row_mask = np.asarray(
                [any(pattern.search(row_text(row)) for pattern in patterns) for row in data["rows"].tolist()],
                dtype=bool,
            )
        blocked_mask |= excluded_row_mask
        query_tensor = torch.as_tensor(query_embeddings, dtype=torch.float32, device=device)
        n_queries = int(query_tensor.shape[0])
        k = min(args.per_query_candidates, max(1, candidate_embeddings.shape[0]))
        top_scores = torch.full((n_queries, k), -float("inf"), dtype=torch.float32, device=device)
        top_indices = torch.full((n_queries, k), -1, dtype=torch.long, device=device)
        for start in range(0, candidate_embeddings.shape[0], args.candidate_batch_size):
            end = min(start + args.candidate_batch_size, candidate_embeddings.shape[0])
            candidate_tensor = torch.as_tensor(candidate_embeddings[start:end], dtype=torch.float32, device=device)
            sims = query_tensor @ candidate_tensor.T
            local_blocked = blocked_mask[start:end]
            if local_blocked.any():
                blocked_tensor = torch.as_tensor(local_blocked, dtype=torch.bool, device=device)
                sims[:, blocked_tensor] = -float("inf")
            batch_k = min(k, end - start)
            batch_scores, batch_local_indices = torch.topk(sims, k=batch_k, dim=1)
            batch_indices = batch_local_indices.to(torch.long) + start
            merged_scores = torch.cat([top_scores, batch_scores], dim=1)
            merged_indices = torch.cat([top_indices, batch_indices], dim=1)
            top_scores, merged_positions = torch.topk(merged_scores, k=k, dim=1)
            top_indices = torch.gather(merged_indices, 1, merged_positions)
            print(f"{source_name}: cosine_scored {end}/{candidate_embeddings.shape[0]}", flush=True)

        top_scores_np = top_scores.detach().cpu().numpy()
        top_indices_np = top_indices.detach().cpu().numpy()
        best_by_candidate: dict[int, dict[str, Any]] = {}
        candidate_token_cache: dict[int, set[str]] = {}
        candidate_full_token_cache: dict[int, set[str]] = {}
        candidate_ordered_token_cache: dict[int, list[str]] = {}
        candidate_word_count_cache: dict[int, int] = {}

        def make_pair(query_idx: int, candidate_idx: int, cosine_score: float) -> dict[str, Any] | None:
            if candidate_idx < 0 or not np.isfinite(cosine_score):
                return None
            if candidate_idx not in candidate_token_cache:
                sentence = candidate_sentences[candidate_idx]
                candidate_token_cache[candidate_idx] = set(content_tokens(sentence))
                ordered_tokens = [word.lower() for word in words(sentence)]
                candidate_ordered_token_cache[candidate_idx] = ordered_tokens
                candidate_full_token_cache[candidate_idx] = set(ordered_tokens)
                candidate_word_count_cache[candidate_idx] = len(ordered_tokens)
            lex_recall = lexical_recall(
                query_token_sets[query_idx],
                query_token_weights[query_idx],
                candidate_token_cache[candidate_idx],
                idf,
            )
            lex_precision = lexical_precision(
                candidate_token_cache[candidate_idx],
                query_token_sets[query_idx],
                idf,
            )
            lex_f1 = harmonic_mean(lex_recall, lex_precision)
            full_word_f1 = set_word_f1(
                set(word.lower() for word in words(query_sentences[query_idx])),
                candidate_full_token_cache[candidate_idx],
            )
            if args.lexical_mode == "idf_f1":
                lexical_score = lex_f1
            elif args.lexical_mode == "word_f1":
                lexical_score = full_word_f1
            else:
                lexical_score = lex_recall
            if lexical_score < args.min_lexical_score:
                return None
            length_score = length_match(query_word_counts[query_idx], candidate_word_count_cache[candidate_idx])
            order_score = ordered_overlap(
                [word.lower() for word in words(query_sentences[query_idx])],
                candidate_ordered_token_cache[candidate_idx],
            )
            coverage_score = (
                args.cosine_weight * cosine_score
                + args.lexical_weight * lexical_score
                + args.length_weight * length_score
                + args.order_weight * order_score
            )
            return {
                "source_id": source_id,
                "source_name": source_name,
                "source_npz": candidate_path,
                "candidate_index": candidate_idx,
                "query_index": query_idx,
                "cosine_score": cosine_score,
                "lexical_score": lexical_score,
                "lexical_recall": lex_recall,
                "lexical_precision": lex_precision,
                "lexical_f1": lex_f1,
                "word_f1": full_word_f1,
                "length_score": length_score,
                "order_score": order_score,
                "coverage_score": float(coverage_score),
                "candidate_sentence": candidate_sentences[candidate_idx],
                "candidate_text_key": normalize_sentence_key(candidate_sentences[candidate_idx]),
                "query_sentence": query_sentences[query_idx],
            }

        def keep_pair(pair: dict[str, Any] | None) -> None:
            if pair is None:
                return
            prev = best_by_candidate.get(int(pair["candidate_index"]))
            if prev is None or pair["coverage_score"] > prev["coverage_score"]:
                best_by_candidate[int(pair["candidate_index"])] = pair

        for query_idx in range(top_indices_np.shape[0]):
            for slot in range(top_indices_np.shape[1]):
                keep_pair(
                    make_pair(
                        query_idx,
                        int(top_indices_np[query_idx, slot]),
                        float(top_scores_np[query_idx, slot]),
                    )
                )

        if args.per_query_lexical_candidates > 0:
            print(f"{source_name}: building lexical inverted index", flush=True)
            token_to_indices: dict[str, list[int]] = defaultdict(list)
            all_candidate_word_counts = np.zeros((len(candidate_sentences),), dtype=np.int16)
            for candidate_idx, sentence in enumerate(candidate_sentences):
                all_candidate_word_counts[candidate_idx] = len(words(sentence))
                if blocked_mask[candidate_idx]:
                    continue
                for token in set(content_tokens(sentence)):
                    token_to_indices[token].append(candidate_idx)
                if candidate_idx and candidate_idx % 250000 == 0:
                    print(f"{source_name}: indexed {candidate_idx}/{len(candidate_sentences)}", flush=True)
            for query_idx, query_tokens in enumerate(query_token_sets):
                accum: dict[int, float] = {}
                for token in query_tokens:
                    indices = token_to_indices.get(token)
                    if not indices or len(indices) > args.lexical_token_max_candidates:
                        continue
                    weight = idf.get(token, 1.0)
                    for candidate_idx in indices:
                        accum[candidate_idx] = accum.get(candidate_idx, 0.0) + weight
                if not accum:
                    continue
                query_vec = query_embeddings[query_idx]
                lexical_pairs = []
                denom = max(query_token_weights[query_idx], 1e-12)
                for candidate_idx, overlap_weight in accum.items():
                    lex = float(overlap_weight / denom)
                    if lex < args.min_lexical_score:
                        continue
                    length_score = length_match(
                        query_word_counts[query_idx],
                        int(all_candidate_word_counts[candidate_idx]),
                    )
                    approx_score = args.lexical_weight * lex + args.length_weight * length_score
                    lexical_pairs.append((approx_score, candidate_idx))
                lexical_pairs.sort(reverse=True)
                for _approx_score, candidate_idx in lexical_pairs[: args.per_query_lexical_candidates]:
                    cosine_score = float(np.dot(query_vec, candidate_embeddings[candidate_idx]))
                    keep_pair(make_pair(query_idx, candidate_idx, cosine_score))
                if query_idx and query_idx % 1000 == 0:
                    print(f"{source_name}: lexical_scored_queries {query_idx}/{len(query_token_sets)}", flush=True)

        pairs = list(best_by_candidate.values())
        pairs.sort(key=lambda item: item["coverage_score"], reverse=True)
        summary = {
            "source_name": source_name,
            "source_npz": candidate_path,
            "candidate_count": int(len(candidate_sentences)),
            "blocked_count": int(blocked_mask.sum()),
            "excluded_row_count": int(excluded_row_mask.sum()),
            "pair_count": int(len(pairs)),
            "schema": _schema(data, args.schema_key),
            "sample": [
                {
                    "coverage_score": float(item["coverage_score"]),
                    "cosine_score": float(item["cosine_score"]),
                    "lexical_score": float(item["lexical_score"]),
                    "word_f1": float(item["word_f1"]),
                    "order_score": float(item["order_score"]),
                    "sentence": item["candidate_sentence"],
                    "query": item["query_sentence"],
                }
                for item in pairs[:10]
            ],
        }
    return pairs, summary


def round_robin_select(pairs: list[dict[str, Any]], count: int, max_per_query: int) -> list[dict[str, Any]]:
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_query[int(pair["query_index"])].append(pair)
    for items in by_query.values():
        items.sort(key=lambda item: item["coverage_score"], reverse=True)
    query_order = sorted(
        by_query,
        key=lambda query_idx: by_query[query_idx][0]["coverage_score"],
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()
    selected_text_keys: set[str] = set()
    per_query_counts: Counter[int] = Counter()
    cursors = {query_idx: 0 for query_idx in query_order}

    made_progress = True
    while len(selected) < count and made_progress:
        made_progress = False
        for query_idx in query_order:
            if len(selected) >= count:
                break
            if max_per_query > 0 and per_query_counts[query_idx] >= max_per_query:
                continue
            items = by_query[query_idx]
            cursor = cursors[query_idx]
            while cursor < len(items):
                pair = items[cursor]
                cursor += 1
                key = (int(pair["source_id"]), int(pair["candidate_index"]))
                if key in selected_keys:
                    continue
                text_key = str(pair["candidate_text_key"])
                if text_key in selected_text_keys:
                    continue
                selected_keys.add(key)
                selected_text_keys.add(text_key)
                selected.append(pair)
                per_query_counts[query_idx] += 1
                made_progress = True
                break
            cursors[query_idx] = cursor

    if len(selected) < count:
        for pair in sorted(pairs, key=lambda item: item["coverage_score"], reverse=True):
            if len(selected) >= count:
                break
            key = (int(pair["source_id"]), int(pair["candidate_index"]))
            if key in selected_keys:
                continue
            text_key = str(pair["candidate_text_key"])
            if text_key in selected_text_keys:
                continue
            selected_keys.add(key)
            selected_text_keys.add(text_key)
            selected.append(pair)
    return selected


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    candidate_names = list(args.candidate_name)
    while len(candidate_names) < len(args.candidate_npz):
        candidate_names.append(Path(args.candidate_npz[len(candidate_names)]).stem)
    if len(candidate_names) > len(args.candidate_npz):
        raise ValueError("More --candidate-name values than --candidate-npz values.")

    query_data = np.load(args.query_npz, allow_pickle=True)
    query_sentences_all = _strings(query_data[args.sentence_key])
    query_embeddings_all = maybe_normalize(query_data[args.input_key], normalize=not args.no_normalize)
    if args.query_sample_size > 0 and len(query_sentences_all) > args.query_sample_size:
        query_indices = np.sort(rng.choice(len(query_sentences_all), size=args.query_sample_size, replace=False))
        query_sentences = [query_sentences_all[int(index)] for index in query_indices]
        query_embeddings = query_embeddings_all[query_indices]
    else:
        query_indices = np.arange(len(query_sentences_all), dtype=np.int64)
        query_sentences = query_sentences_all
        query_embeddings = query_embeddings_all
    query_token_sets, query_token_weights, idf = query_token_stats(query_sentences)
    query_word_counts = [len(words(sentence)) for sentence in query_sentences]
    blocked = load_blocked_sentences(args.blocked_npz, args.sentence_key)

    all_pairs: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_id, (candidate_path, source_name) in enumerate(zip(args.candidate_npz, candidate_names)):
        pairs, source_summary = top_pairs_for_source(
            source_id=source_id,
            source_name=source_name,
            candidate_path=candidate_path,
            query_embeddings=query_embeddings,
            query_sentences=query_sentences,
            query_token_sets=query_token_sets,
            query_token_weights=query_token_weights,
            query_word_counts=query_word_counts,
            idf=idf,
            blocked=blocked,
            args=args,
            device=device,
        )
        all_pairs.extend(pairs)
        source_summaries.append(source_summary)

    if not all_pairs:
        raise RuntimeError("No candidate/query pairs survived scoring.")
    selected = round_robin_select(all_pairs, args.count, args.max_per_query)
    selected_by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for out_idx, pair in enumerate(selected):
        selected_by_source[int(pair["source_id"])].append((out_idx, int(pair["candidate_index"])))

    out_embeddings: list[np.ndarray] = [None] * len(selected)  # type: ignore[list-item]
    out_sentences: list[str] = [""] * len(selected)
    out_rows: list[dict[str, Any]] = [{} for _ in selected]
    for source_id, selected_items in selected_by_source.items():
        candidate_path = args.candidate_npz[source_id]
        with np.load(candidate_path, allow_pickle=True) as data:
            source_embeddings = data[args.input_key]
            source_sentences = _strings(data[args.sentence_key])
            source_rows = (
                data["rows"]
                if "rows" in data.files
                else np.asarray([{} for _ in range(len(source_sentences))], dtype=object)
            )
            for out_idx, source_idx in selected_items:
                pair = selected[out_idx]
                row_obj = source_rows[source_idx]
                row = dict(row_obj) if isinstance(row_obj, dict) else {"source_row": row_obj}
                row.update(
                    {
                        "coverage_source_name": pair["source_name"],
                        "coverage_source_npz": pair["source_npz"],
                        "coverage_source_index": int(pair["candidate_index"]),
                        "matched_query_index": int(query_indices[int(pair["query_index"])]),
                        "matched_query_sentence": pair["query_sentence"],
                        "coverage_score": float(pair["coverage_score"]),
                        "semantic_cosine_score": float(pair["cosine_score"]),
                        "lexical_score": float(pair["lexical_score"]),
                        "length_score": float(pair["length_score"]),
                    }
                )
                out_embeddings[out_idx] = np.asarray(source_embeddings[source_idx], dtype=np.float32)
                out_sentences[out_idx] = source_sentences[source_idx]
                out_rows[out_idx] = row

    embeddings = np.stack(out_embeddings, axis=0).astype(np.float32)  # type: ignore[arg-type]
    sentences = np.asarray(out_sentences, dtype=object)
    rows = np.asarray(out_rows, dtype=object)
    source_npz = np.asarray([pair["source_npz"] for pair in selected], dtype=object)
    split = np.asarray(["train"] * len(selected), dtype=object)
    coverage_score = np.asarray([pair["coverage_score"] for pair in selected], dtype=np.float32)
    semantic_score = np.asarray([pair["cosine_score"] for pair in selected], dtype=np.float32)
    lexical_score = np.asarray([pair["lexical_score"] for pair in selected], dtype=np.float32)
    length_score = np.asarray([pair["length_score"] for pair in selected], dtype=np.float32)
    order_score = np.asarray([pair["order_score"] for pair in selected], dtype=np.float32)
    word_f1 = np.asarray([pair["word_f1"] for pair in selected], dtype=np.float32)
    matched_query_index = np.asarray([query_indices[int(pair["query_index"])] for pair in selected], dtype=np.int64)
    source_index = np.asarray([pair["candidate_index"] for pair in selected], dtype=np.int64)
    source_name = np.asarray([pair["source_name"] for pair in selected], dtype=object)
    matched_query_sentence = np.asarray([pair["query_sentence"] for pair in selected], dtype=object)

    summary = {
        "output": args.output,
        "candidate_npz": args.candidate_npz,
        "candidate_name": candidate_names,
        "query_npz": args.query_npz,
        "blocked_npz": args.blocked_npz,
        "query_count": int(len(query_sentences_all)),
        "query_sample_size": int(len(query_sentences)),
        "query_sample_indices_sha1": __import__("hashlib").sha1(query_indices.tobytes()).hexdigest(),
        "selected_count": int(len(selected)),
        "target_count": int(args.count),
        "per_query_candidates": int(args.per_query_candidates),
        "candidate_batch_size": int(args.candidate_batch_size),
        "weights": {
            "cosine": float(args.cosine_weight),
            "lexical": float(args.lexical_weight),
            "length": float(args.length_weight),
            "order": float(args.order_weight),
        },
        "lexical_mode": args.lexical_mode,
        "exclude_candidate_row_regex": args.exclude_candidate_row_regex,
        "min_lexical_score": float(args.min_lexical_score),
        "max_per_query": int(args.max_per_query),
        "source_summaries": source_summaries,
        "selected_source_counts": dict(Counter(str(pair["source_name"]) for pair in selected)),
        "coverage_score_min": float(coverage_score.min()),
        "coverage_score_mean": float(coverage_score.mean()),
        "coverage_score_max": float(coverage_score.max()),
        "semantic_score_mean": float(semantic_score.mean()),
        "lexical_score_mean": float(lexical_score.mean()),
        "lexical_score_p50": float(np.percentile(lexical_score, 50)),
        "lexical_score_p90": float(np.percentile(lexical_score, 90)),
        "length_score_mean": float(length_score.mean()),
        "order_score_mean": float(order_score.mean()),
        "word_f1_mean": float(word_f1.mean()),
        "sample": [
            {
                "coverage_score": float(coverage_score[i]),
                "semantic_score": float(semantic_score[i]),
                "lexical_score": float(lexical_score[i]),
                "length_score": float(length_score[i]),
                "order_score": float(order_score[i]),
                "word_f1": float(word_f1[i]),
                "source_name": str(source_name[i]),
                "sentence": str(sentences[i]),
                "matched_query_sentence": str(matched_query_sentence[i]),
            }
            for i in range(min(20, len(sentences)))
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        input_embeddings=embeddings,
        sentence=sentences,
        rows=rows,
        source_npz=source_npz,
        split=split,
        source_name=source_name,
        source_index=source_index,
        matched_query_index=matched_query_index,
        matched_query_sentence=matched_query_sentence,
        coverage_score=coverage_score,
        semantic_score=semantic_score,
        lexical_score=lexical_score,
        length_score=length_score,
        order_score=order_score,
        word_f1=word_f1,
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
