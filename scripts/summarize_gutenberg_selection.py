#!/usr/bin/env python
"""Summarize a selected Gutenberg NPZ by source book."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-books", type=int, default=50)
    parser.add_argument("--examples-per-book", type=int, default=5)
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--score-key", default="selection_score")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "item"):
        item = row.item()
        if isinstance(item, dict):
            return item
    return {}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    sentences = _strings(data[args.sentence_key])
    scores = (
        np.asarray(data[args.score_key], dtype=np.float32)
        if args.score_key in data.files
        else np.full((len(sentences),), np.nan, dtype=np.float32)
    )
    rows = data["rows"] if "rows" in data.files else np.asarray([{} for _ in sentences], dtype=object)

    groups: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for sentence, score, row_obj in zip(sentences, scores.tolist(), rows.tolist()):
        row = _row_dict(row_obj)
        book_id = str(row.get("gutenberg_id", row.get("book", "unknown")))
        title = str(row.get("title", row.get("book", "unknown")))
        authors = str(row.get("authors", ""))
        key = f"{book_id}|{title}|{authors}"
        if key not in groups:
            groups[key] = {
                "gutenberg_id": book_id,
                "title": title,
                "authors": authors,
                "count": 0,
                "scores": [],
            }
        groups[key]["count"] += 1
        groups[key]["scores"].append(float(score))
        examples[key].append((float(score), sentence))

    books = []
    for key, item in groups.items():
        book_scores = item.pop("scores")
        item.update(
            {
                "mean_score": float(np.mean(book_scores)),
                "p50_score": percentile(book_scores, 50),
                "p95_score": percentile(book_scores, 95),
                "max_score": float(np.max(book_scores)),
                "examples": [
                    {"score": score, "sentence": sentence}
                    for score, sentence in sorted(examples[key], reverse=True)[: args.examples_per_book]
                ],
            }
        )
        books.append(item)

    books.sort(key=lambda x: (x["count"], x["mean_score"]), reverse=True)
    summary = {
        "npz": args.npz,
        "total_sentences": int(len(sentences)),
        "unique_books": int(len(books)),
        "top_books": books[: args.top_books],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
