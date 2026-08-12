#!/usr/bin/env python
"""Analyze saved MEG2SEM->ELF generation eval JSONs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
QUALITY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "him", "his", "i", "in", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "she", "so", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "we",
    "were", "what", "when", "which", "who", "with", "you",
}
CONTENT_WORD_STOPWORDS = QUALITY_STOPWORDS | {
    "about", "after", "again", "against", "all", "am", "any", "because",
    "been", "before", "being", "below", "between", "both", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "further",
    "having", "here", "hers", "herself", "himself", "how", "if", "into",
    "itself", "just", "more", "most", "off", "once", "only", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "should", "some",
    "quite", "really", "such", "sure", "than", "then", "these", "those",
    "through", "too", "under", "until", "up", "us", "very", "where", "whom",
    "whose", "why", "will", "would", "your", "yours", "yourself",
    "yourselves", "well", "yes",
}


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_word_tokens(text: str) -> list[str]:
    return [
        token
        for token in word_tokens(text)
        if token not in CONTENT_WORD_STOPWORDS and any(char.isalpha() for char in token)
    ]


def counted_overlap(generated_tokens: Sequence[str], target_tokens: Sequence[str]) -> dict:
    if not generated_tokens or not target_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "counts": {},
            "count": 0,
        }

    generated_counts = Counter(generated_tokens)
    target_counts = Counter(target_tokens)
    overlap_counts = {
        token: min(count, target_counts[token])
        for token, count in generated_counts.items()
        if target_counts[token] > 0
    }
    overlap_count = sum(overlap_counts.values())
    precision = overlap_count / max(1, len(generated_tokens))
    recall = overlap_count / max(1, len(target_tokens))
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
    generated_set = set(generated_tokens)
    target_set = set(target_tokens)
    jaccard = len(generated_set & target_set) / max(1, len(generated_set | target_set))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "counts": dict(sorted(overlap_counts.items())),
        "count": int(overlap_count),
    }


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{token}x{count}" if count > 1 else token for token, count in counts.items())


def load_eval(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def select_best_eval(root: str | Path, criterion: str) -> tuple[Path, dict]:
    root = Path(root)
    paths = sorted(root.glob("eval_step_*.json"))
    if not paths:
        raise FileNotFoundError(f"No eval_step_*.json files found under {root}")

    def score(metrics: dict) -> tuple[float, float, float, float]:
        quality = metrics.get("generation_quality") or {}
        retrieval = metrics.get("generation_t5_retrieval") or {}
        scores = metrics.get("eval_checkpoint_scores") or {}
        content = float(quality.get("content_words_overlap") or 0.0)
        top5 = float(retrieval.get("top5") or 0.0)
        words = float(quality.get("words_overlap") or 0.0)
        structured_rank = float(scores.get("structured_rank") or -1e9)
        if criterion == "structured_rank":
            return structured_rank, content, top5, words
        if criterion == "top5":
            return top5, content, words, structured_rank
        if criterion == "words":
            return words, content, top5, structured_rank
        return content, top5, words, structured_rank

    loaded = [(path, load_eval(path)) for path in paths]
    return max(loaded, key=lambda item: score(item[1]))


def build_rows(metrics: dict) -> list[dict]:
    generated = metrics.get("generated") or []
    targets = metrics.get("targets") or []
    exact = metrics.get("exact") or []
    retrieval = metrics.get("generation_t5_retrieval") or {}
    ranks = retrieval.get("ranks") or []
    top_indices = retrieval.get("top_indices") or []
    rows = []
    for idx, (generated_text, target_text) in enumerate(zip(generated, targets)):
        word = counted_overlap(word_tokens(generated_text), word_tokens(target_text))
        content = counted_overlap(content_word_tokens(generated_text), content_word_tokens(target_text))
        rows.append(
            {
                "index": idx,
                "target": target_text,
                "generated": generated_text,
                "exact": int(exact[idx]) if idx < len(exact) else int(generated_text == target_text),
                "rank": int(ranks[idx]) if idx < len(ranks) else None,
                "top5_hit": int(ranks[idx] <= 5) if idx < len(ranks) else None,
                "top_indices": ",".join(str(item) for item in top_indices[idx]) if idx < len(top_indices) else "",
                "words_f1": word["f1"],
                "words_precision": word["precision"],
                "words_recall": word["recall"],
                "words_jaccard": word["jaccard"],
                "words_overlap_count": word["count"],
                "words_overlap_words": format_counts(word["counts"]),
                "content_f1": content["f1"],
                "content_precision": content["precision"],
                "content_recall": content["recall"],
                "content_jaccard": content["jaccard"],
                "content_overlap_count": content["count"],
                "content_overlap_words": format_counts(content["counts"]),
                "target_content_words": " ".join(content_word_tokens(target_text)),
                "generated_content_words": " ".join(content_word_tokens(generated_text)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["content_overlap_count"],
            row["content_f1"],
            row["words_overlap_count"],
            row["words_f1"],
            -(row["rank"] or 10**9),
        ),
        reverse=True,
    )
    return rows


def mean_overlap_f1(
    generated_tokens: Sequence[Sequence[str]],
    target_tokens: Sequence[Sequence[str]],
) -> float:
    scores = [counted_overlap(gen, target)["f1"] for gen, target in zip(generated_tokens, target_tokens)]
    return float(np.mean(scores)) if scores else 0.0


def mean_content_f1(generated: Sequence[str], targets: Sequence[str]) -> float:
    return mean_overlap_f1(
        [content_word_tokens(text) for text in generated],
        [content_word_tokens(text) for text in targets],
    )


def mean_words_f1(generated: Sequence[str], targets: Sequence[str]) -> float:
    return mean_overlap_f1([word_tokens(text) for text in generated], [word_tokens(text) for text in targets])


def permutation_test(metrics: dict, permutations: int, seed: int) -> dict:
    generated = list(metrics.get("generated") or [])
    targets = list(metrics.get("targets") or [])
    if len(generated) != len(targets):
        raise ValueError(f"Generated/target length mismatch: {len(generated)} vs {len(targets)}")
    generated_content = [content_word_tokens(text) for text in generated]
    target_content = [content_word_tokens(text) for text in targets]
    generated_words = [word_tokens(text) for text in generated]
    target_words = [word_tokens(text) for text in targets]
    observed_content = mean_overlap_f1(generated_content, target_content)
    observed_words = mean_overlap_f1(generated_words, target_words)
    n = len(targets)
    content_matrix = np.zeros((n, n), dtype=np.float32)
    words_matrix = np.zeros((n, n), dtype=np.float32)
    for gen_idx in range(n):
        for target_idx in range(n):
            content_matrix[gen_idx, target_idx] = counted_overlap(
                generated_content[gen_idx],
                target_content[target_idx],
            )["f1"]
            words_matrix[gen_idx, target_idx] = counted_overlap(generated_words[gen_idx], target_words[target_idx])[
                "f1"
            ]
    rng = random.Random(seed)
    null_content = np.zeros(permutations, dtype=np.float32)
    null_words = np.zeros(permutations, dtype=np.float32)
    row_indices = np.arange(n)
    indices = list(range(len(targets)))
    for perm_idx in range(permutations):
        rng.shuffle(indices)
        null_content[perm_idx] = float(content_matrix[row_indices, indices].mean())
        null_words[perm_idx] = float(words_matrix[row_indices, indices].mean())
    null_content_arr = null_content.astype(np.float64, copy=False)
    null_words_arr = null_words.astype(np.float64, copy=False)
    return {
        "permutations": permutations,
        "seed": seed,
        "observed_content_words_overlap": observed_content,
        "observed_words_overlap": observed_words,
        "null_content_mean": float(null_content_arr.mean()) if permutations else None,
        "null_content_std": float(null_content_arr.std(ddof=1)) if permutations > 1 else None,
        "null_content_p_ge_observed": float((np.sum(null_content_arr >= observed_content) + 1) / (permutations + 1))
        if permutations
        else None,
        "null_content_percentile": float(np.mean(null_content_arr <= observed_content)) if permutations else None,
        "null_words_mean": float(null_words_arr.mean()) if permutations else None,
        "null_words_std": float(null_words_arr.std(ddof=1)) if permutations > 1 else None,
        "null_words_p_ge_observed": float((np.sum(null_words_arr >= observed_words) + 1) / (permutations + 1))
        if permutations
        else None,
        "null_words_percentile": float(np.mean(null_words_arr <= observed_words)) if permutations else None,
    }


def write_tsv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="Run root containing eval_step_*.json files.")
    source.add_argument("--eval-json", help="Specific eval JSON to analyze.")
    parser.add_argument(
        "--criterion",
        choices=["content", "top5", "words", "structured_rank"],
        default="content",
        help="Criterion used with --root to choose the best eval JSON.",
    )
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--print-rows", type=int, default=0, help="Print the first N sorted rows to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_json:
        eval_path = Path(args.eval_json)
        metrics = load_eval(eval_path)
    else:
        eval_path, metrics = select_best_eval(args.root, args.criterion)

    root = Path(args.root) if args.root else eval_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    rows = build_rows(metrics)
    perm = permutation_test(metrics, args.permutations, args.seed)
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    oracle = metrics.get("oracle_ada") or {}
    oracle_quality = oracle.get("generation_quality") or {}
    oracle_retrieval = oracle.get("generation_t5_retrieval") or {}
    interface = metrics.get("meg2sem_interface") or {}
    overlap_word_counts = Counter()
    for row in rows:
        for item in row["content_overlap_words"].split(", "):
            if not item:
                continue
            if "x" in item:
                token, count = item.rsplit("x", 1)
                overlap_word_counts[token] += int(count)
            else:
                overlap_word_counts[item] += 1

    summary = {
        "eval_path": str(eval_path),
        "step": metrics.get("step"),
        "epoch": metrics.get("epoch"),
        "eval_num_examples": metrics.get("eval_num_examples"),
        "generation_quality": quality,
        "generation_t5_retrieval": {
            key: retrieval.get(key)
            for key in ("top1", "top5", "mean_rank", "median_rank")
            if key in retrieval
        },
        "oracle_ada_generation_quality": oracle_quality,
        "oracle_ada_generation_t5_retrieval": {
            key: oracle_retrieval.get(key)
            for key in ("top1", "top5", "mean_rank", "median_rank")
            if key in oracle_retrieval
        },
        "meg2sem_interface": interface,
        "permutation_test": perm,
        "content_overlap_word_frequency": dict(overlap_word_counts.most_common()),
        "num_rows": len(rows),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{eval_path.stem}_analysis_summary.json"
    rows_path = out_dir / f"{eval_path.stem}_decoded_sorted.tsv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_tsv(rows, rows_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"rows_tsv={rows_path}")
    if args.print_rows:
        for row in rows[: args.print_rows]:
            print(
                "idx={index} rank={rank} c={content_f1:.3f} words={content_overlap_words!r} "
                "target={target!r} generated={generated!r}".format(**row)
            )


if __name__ == "__main__":
    main()
