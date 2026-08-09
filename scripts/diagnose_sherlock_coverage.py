#!/usr/bin/env python
"""Diagnose why held-out Sherlock sentences have low lexical coverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--train-npz", action="append", required=True)
    parser.add_argument("--train-name", action="append", default=[])
    parser.add_argument("--output-json", default="")
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def normalize_text(sentence: str) -> str:
    return " ".join(word.lower() for word in WORD_RE.findall(str(sentence)))


def words(sentence: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(str(sentence))]


def word_set(sentence: str) -> set[str]:
    return set(words(sentence))


def content_set(sentence: str) -> set[str]:
    return {word for word in words(sentence) if len(word) > 2 and word not in STOP_WORDS}


def word_metrics(target_words: set[str], neighbor_words: set[str]) -> dict[str, float]:
    if not target_words and not neighbor_words:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0, "jaccard": 1.0}
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
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ge_0.5": float(np.mean(array >= 0.5)),
        "ge_0.7": float(np.mean(array >= 0.7)),
        "ge_0.8": float(np.mean(array >= 0.8)),
    }


def split_train_indices(data: np.lib.npyio.NpzFile, split_key: str, total_n: int) -> np.ndarray:
    if split_key not in data.files:
        return np.arange(total_n, dtype=np.int64)
    split = strings(data[split_key])
    train = [idx for idx, value in enumerate(split) if value == "train"]
    if not train:
        return np.arange(total_n, dtype=np.int64)
    return np.asarray(train, dtype=np.int64)


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


def semantic_top1(
    *,
    val_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    val_keys: list[str],
    train_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    val_embeddings = normalize_rows(val_embeddings)
    blocked: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(train_keys):
        blocked[key].append(idx)
    best_scores = np.full((val_embeddings.shape[0],), -np.inf, dtype=np.float32)
    best_indices = np.full((val_embeddings.shape[0],), -1, dtype=np.int64)
    chunk_size = 32768
    for start in range(0, train_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, train_embeddings.shape[0])
        chunk = np.asarray(train_embeddings[start:end], dtype=np.float32)
        chunk = chunk / np.clip(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12, None)
        sims = val_embeddings @ chunk.T
        for val_idx, key in enumerate(val_keys):
            for train_idx in blocked.get(key, []):
                if start <= train_idx < end:
                    sims[val_idx, train_idx - start] = -np.inf
        local_indices = np.argmax(sims, axis=1)
        local_scores = sims[np.arange(sims.shape[0]), local_indices]
        improved = local_scores > best_scores
        best_scores[improved] = local_scores[improved]
        best_indices[improved] = local_indices[improved].astype(np.int64) + start
    return best_indices, best_scores


def lexical_oracle(
    *,
    val_sentences: list[str],
    train_sentences: list[str],
    val_keys: list[str],
    train_keys: list[str],
) -> tuple[list[int], list[dict[str, float]]]:
    token_to_train: dict[str, list[int]] = defaultdict(list)
    for train_idx, sentence in enumerate(train_sentences):
        for token in content_set(sentence):
            token_to_train[token].append(train_idx)
        if train_idx and train_idx % 100000 == 0:
            print(f"  lexical index {train_idx}/{len(train_sentences)}", flush=True)

    best_indices: list[int] = []
    best_metrics: list[dict[str, float]] = []
    train_word_cache: dict[int, set[str]] = {}
    for val_sentence, val_key in zip(val_sentences, val_keys):
        target_words = word_set(val_sentence)
        target_content = content_set(val_sentence)
        candidate_indices: set[int] = set()
        for token in target_content:
            candidate_indices.update(token_to_train.get(token, []))
        best_idx = -1
        best = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "jaccard": 0.0}
        for train_idx in candidate_indices:
            if train_keys[train_idx] == val_key:
                continue
            if train_idx not in train_word_cache:
                train_word_cache[train_idx] = word_set(train_sentences[train_idx])
            metrics = word_metrics(target_words, train_word_cache[train_idx])
            if (metrics["f1"], metrics["recall"], metrics["precision"]) > (
                best["f1"],
                best["recall"],
                best["precision"],
            ):
                best_idx = train_idx
                best = metrics
        best_indices.append(best_idx)
        best_metrics.append(best)
    return best_indices, best_metrics


def analyze_train_pool(
    *,
    name: str,
    path: str,
    val_sentences: list[str],
    val_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"Analyzing {name}: {path}", flush=True)
    with np.load(path, allow_pickle=True) as data:
        all_sentences = strings(data[args.sentence_key])
        train_indices = split_train_indices(data, args.split_key, len(all_sentences))
        train_sentences = [all_sentences[int(index)] for index in train_indices]
        train_embeddings = np.asarray(data[args.input_key][train_indices], dtype=np.float32)

    val_keys = [normalize_text(sentence) for sentence in val_sentences]
    train_keys = [normalize_text(sentence) for sentence in train_sentences]
    val_word_sets = [word_set(sentence) for sentence in val_sentences]
    val_content_sets = [content_set(sentence) for sentence in val_sentences]
    train_vocab: set[str] = set()
    train_content_vocab: set[str] = set()
    for idx, sentence in enumerate(train_sentences):
        train_vocab.update(word_set(sentence))
        train_content_vocab.update(content_set(sentence))
        if idx and idx % 100000 == 0:
            print(f"  vocab {idx}/{len(train_sentences)}", flush=True)

    sem_indices, sem_scores = semantic_top1(
        val_embeddings=val_embeddings,
        train_embeddings=train_embeddings,
        val_keys=val_keys,
        train_keys=train_keys,
    )
    lex_indices, lex_metrics = lexical_oracle(
        val_sentences=val_sentences,
        train_sentences=train_sentences,
        val_keys=val_keys,
        train_keys=train_keys,
    )

    records = []
    for val_idx, val_sentence in enumerate(val_sentences):
        semantic_neighbor = train_sentences[int(sem_indices[val_idx])]
        semantic_metrics = word_metrics(val_word_sets[val_idx], word_set(semantic_neighbor))
        lexical_idx = int(lex_indices[val_idx])
        lexical_neighbor = train_sentences[lexical_idx] if lexical_idx >= 0 else ""
        missing_words = sorted(val_word_sets[val_idx] - train_vocab)
        missing_content_words = sorted(val_content_sets[val_idx] - train_content_vocab)
        records.append(
            {
                "index": val_idx,
                "query": val_sentence,
                "word_count": len(words(val_sentence)),
                "content_words": sorted(val_content_sets[val_idx]),
                "missing_words": missing_words,
                "missing_content_words": missing_content_words,
                "vocab_recall": float(1.0 - len(missing_words) / max(1, len(val_word_sets[val_idx]))),
                "content_vocab_recall": float(
                    1.0 - len(missing_content_words) / max(1, len(val_content_sets[val_idx]))
                ),
                "semantic_top1": {
                    "cosine": float(sem_scores[val_idx]),
                    **semantic_metrics,
                    "neighbor": semantic_neighbor,
                },
                "lexical_oracle": {
                    **lex_metrics[val_idx],
                    "neighbor": lexical_neighbor,
                },
                "oracle_minus_semantic_f1": float(lex_metrics[val_idx]["f1"] - semantic_metrics["f1"]),
            }
        )

    all_val_words = Counter(word for sentence in val_sentences for word in word_set(sentence))
    missing_content_counter = Counter(
        token
        for record in records
        for token in record["missing_content_words"]
    )
    return {
        "name": name,
        "path": path,
        "train_n": len(train_sentences),
        "val_n": len(val_sentences),
        "train_vocab_n": len(train_vocab),
        "train_content_vocab_n": len(train_content_vocab),
        "val_vocab_n": len(all_val_words),
        "semantic_top1": {
            "cosine": summarize([record["semantic_top1"]["cosine"] for record in records]),
            "word_f1": summarize([record["semantic_top1"]["f1"] for record in records]),
            "word_recall": summarize([record["semantic_top1"]["recall"] for record in records]),
        },
        "lexical_oracle": {
            "word_f1": summarize([record["lexical_oracle"]["f1"] for record in records]),
            "word_recall": summarize([record["lexical_oracle"]["recall"] for record in records]),
        },
        "vocab_recall": summarize([record["vocab_recall"] for record in records]),
        "content_vocab_recall": summarize([record["content_vocab_recall"] for record in records]),
        "missing_content_words": missing_content_counter.most_common(50),
        "largest_semantic_failures": sorted(records, key=lambda item: item["oracle_minus_semantic_f1"], reverse=True)[
            : args.examples
        ],
        "lowest_oracle_examples": sorted(records, key=lambda item: item["lexical_oracle"]["f1"])[: args.examples],
    }


def main() -> None:
    args = parse_args()
    names = list(args.train_name)
    while len(names) < len(args.train_npz):
        names.append(Path(args.train_npz[len(names)]).stem)
    if len(names) > len(args.train_npz):
        raise ValueError("More train names than train NPZs")

    with np.load(args.val_npz, allow_pickle=True) as val_data:
        val_sentences = strings(val_data[args.sentence_key])
        val_embeddings = np.asarray(val_data[args.input_key], dtype=np.float32)

    results = [
        analyze_train_pool(
            name=name,
            path=path,
            val_sentences=val_sentences,
            val_embeddings=val_embeddings,
            args=args,
        )
        for name, path in zip(names, args.train_npz)
    ]
    output = {"val_npz": args.val_npz, "results": results}
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("name train_n sem_f1 lex_oracle_f1 vocab content_vocab sem_cos")
    for result in results:
        print(
            f"{result['name']} {result['train_n']} "
            f"{result['semantic_top1']['word_f1']['mean']:.3f} "
            f"{result['lexical_oracle']['word_f1']['mean']:.3f} "
            f"{result['vocab_recall']['mean']:.3f} "
            f"{result['content_vocab_recall']['mean']:.3f} "
            f"{result['semantic_top1']['cosine']['mean']:.3f}"
        )
    if args.output_json:
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
