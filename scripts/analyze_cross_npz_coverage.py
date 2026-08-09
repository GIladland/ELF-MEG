#!/usr/bin/env python
"""Measure query coverage against one or more separate candidate NPZ banks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-npz", action="append", required=True)
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--exclude-candidate-row-regex", action="append", default=[])
    parser.add_argument("--exclude-identical-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def normalize_text(sentence: str) -> str:
    return " ".join(word.lower() for word in WORD_RE.findall(str(sentence)))


def word_set(sentence: str) -> set[str]:
    return set(word.lower() for word in WORD_RE.findall(str(sentence)))


def word_metrics(target: str, neighbor: str) -> dict[str, float]:
    target_words = word_set(target)
    neighbor_words = word_set(neighbor)
    overlap = len(target_words & neighbor_words)
    recall = overlap / max(1, len(target_words))
    precision = overlap / max(1, len(neighbor_words))
    f1 = 0.0 if recall + precision == 0 else 2.0 * recall * precision / (recall + precision)
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "jaccard": float(overlap / max(1, len(target_words | neighbor_words))),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ge_0.5": float(np.mean(array >= 0.5)),
        "ge_0.7": float(np.mean(array >= 0.7)),
    }


def row_text(row: Any) -> str:
    if isinstance(row, dict):
        return json.dumps(row, sort_keys=True, default=str)
    return str(row)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    names = list(args.candidate_name)
    while len(names) < len(args.candidate_npz):
        names.append(Path(args.candidate_npz[len(names)]).stem)
    if len(names) != len(args.candidate_npz):
        raise ValueError("More candidate names than candidate NPZs")

    device = resolve_device(args.device)
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.exclude_candidate_row_regex]
    with np.load(args.query_npz, allow_pickle=True) as query_data:
        query_sentences = strings(query_data[args.sentence_key])
        query_embeddings = normalize_rows(query_data[args.input_key])
    query_keys = [normalize_text(sentence) for sentence in query_sentences]
    all_neighbors: list[list[dict[str, Any]]] = [[] for _ in query_sentences]
    source_summaries = []

    for source_name, candidate_path in zip(names, args.candidate_npz):
        with np.load(candidate_path, allow_pickle=True) as data:
            candidate_sentences_all = strings(data[args.sentence_key])
            keep_mask = np.ones((len(candidate_sentences_all),), dtype=bool)
            excluded_rows = 0
            if patterns:
                if args.rows_key not in data.files:
                    raise KeyError(f"{candidate_path} has no {args.rows_key!r} metadata")
                keep_mask = np.asarray(
                    [not any(pattern.search(row_text(row)) for pattern in patterns) for row in data[args.rows_key].tolist()],
                    dtype=bool,
                )
                excluded_rows = int((~keep_mask).sum())
            if args.exclude_identical_text:
                blocked = set(query_keys)
                keep_mask &= np.asarray(
                    [normalize_text(sentence) not in blocked for sentence in candidate_sentences_all],
                    dtype=bool,
                )
            keep_indices = np.flatnonzero(keep_mask)
            candidate_sentences = [candidate_sentences_all[int(index)] for index in keep_indices]
            candidate_embeddings = normalize_rows(data[args.input_key][keep_indices])

        target_tensor = torch.as_tensor(candidate_embeddings, dtype=torch.float32, device=device).T.contiguous()
        source_k = min(args.top_k, len(candidate_sentences))
        for start in range(0, len(query_sentences), args.query_batch_size):
            end = min(start + args.query_batch_size, len(query_sentences))
            query_tensor = torch.as_tensor(query_embeddings[start:end], dtype=torch.float32, device=device)
            scores, indices = torch.topk(query_tensor @ target_tensor, k=source_k, dim=1)
            scores_np = scores.detach().cpu().numpy()
            indices_np = indices.detach().cpu().numpy()
            for local_query in range(end - start):
                query_idx = start + local_query
                for slot in range(source_k):
                    candidate_local = int(indices_np[local_query, slot])
                    all_neighbors[query_idx].append(
                        {
                            "source": source_name,
                            "source_index": int(keep_indices[candidate_local]),
                            "cosine": float(scores_np[local_query, slot]),
                            "sentence": candidate_sentences[candidate_local],
                        }
                    )
            print(f"{source_name}: queries {end}/{len(query_sentences)}", flush=True)
        del target_tensor, candidate_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
        source_summaries.append(
            {
                "name": source_name,
                "path": candidate_path,
                "input_count": len(candidate_sentences_all),
                "kept_count": len(candidate_sentences),
                "excluded_row_count": excluded_rows,
            }
        )

    records = []
    for query_idx, neighbors in enumerate(all_neighbors):
        neighbors.sort(key=lambda item: item["cosine"], reverse=True)
        neighbors = neighbors[: args.top_k]
        for item in neighbors:
            item.update(word_metrics(query_sentences[query_idx], item["sentence"]))
        semantic = neighbors[0]
        oracle = max(neighbors, key=lambda item: (item["f1"], item["cosine"]))
        composite = max(neighbors, key=lambda item: (0.5 * item["cosine"] + 0.5 * item["f1"], item["cosine"]))
        records.append(
            {
                "query_index": query_idx,
                "query": query_sentences[query_idx],
                "semantic_top1": semantic,
                "oracle_topk": oracle,
                "composite_topk": composite,
            }
        )

    output = {
        "query_npz": args.query_npz,
        "query_count": len(query_sentences),
        "top_k": args.top_k,
        "exclude_candidate_row_regex": args.exclude_candidate_row_regex,
        "exclude_identical_text": args.exclude_identical_text,
        "sources": source_summaries,
        "semantic_top1": {
            "cosine": summarize([row["semantic_top1"]["cosine"] for row in records]),
            "word_recall": summarize([row["semantic_top1"]["recall"] for row in records]),
            "word_f1": summarize([row["semantic_top1"]["f1"] for row in records]),
        },
        "oracle_topk": {
            "cosine": summarize([row["oracle_topk"]["cosine"] for row in records]),
            "word_recall": summarize([row["oracle_topk"]["recall"] for row in records]),
            "word_f1": summarize([row["oracle_topk"]["f1"] for row in records]),
        },
        "composite_topk": {
            "cosine": summarize([row["composite_topk"]["cosine"] for row in records]),
            "word_recall": summarize([row["composite_topk"]["recall"] for row in records]),
            "word_f1": summarize([row["composite_topk"]["f1"] for row in records]),
        },
        "high_examples": sorted(records, key=lambda row: row["semantic_top1"]["f1"], reverse=True)[:10],
        "low_examples": sorted(records, key=lambda row: row["semantic_top1"]["f1"])[:10],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
