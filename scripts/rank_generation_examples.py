#!/usr/bin/env python
"""Rank ELF generations by lexical, content, and optional BERTScore metrics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CONTENT_STOPWORDS = {
    "a", "about", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "no", "not", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "quite", "really", "same", "she", "should", "so",
    "some", "such", "sure", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "us", "very", "was", "we",
    "well", "were", "what", "when", "where", "which", "who", "whom", "whose", "why", "will", "with",
    "would", "yes", "you", "your", "yours", "yourself", "yourselves",
}


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in word_tokens(text)
        if token not in CONTENT_STOPWORDS and any(character.isalpha() for character in token)
    ]


def overlap_f1(hypothesis: Sequence[str], reference: Sequence[str]) -> float:
    if not hypothesis or not reference:
        return 0.0
    overlap = sum((Counter(hypothesis) & Counter(reference)).values())
    precision = overlap / len(hypothesis)
    recall = overlap / len(reference)
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--bertscore-json", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def ranked(records: list[dict], key: str, top_k: int) -> list[dict]:
    return sorted(records, key=lambda row: (-float(row[key]), int(row["index"])))[:top_k]


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    generated = [str(value) for value in metrics["generated"]]
    targets = [str(value) for value in metrics["targets"]]

    bert_f1 = None
    if args.bertscore_json:
        bert = json.loads(Path(args.bertscore_json).read_text(encoding="utf-8"))
        bert_f1 = bert["per_sample"]["f1"]
        if len(bert_f1) != len(generated):
            raise ValueError("BERTScore rows do not match generation rows.")

    records = []
    for index, (target, hypothesis) in enumerate(zip(targets, generated)):
        target_words = word_tokens(target)
        hypothesis_words = word_tokens(hypothesis)
        record = {
            "index": index,
            "target": target,
            "generated": hypothesis,
            "word_f1": overlap_f1(hypothesis_words, target_words),
            "content_f1": overlap_f1(content_tokens(hypothesis), content_tokens(target)),
            "wer": edit_distance(target_words, hypothesis_words) / max(1, len(target_words)),
        }
        if bert_f1 is not None:
            record["bertscore_f1"] = float(bert_f1[index])
        records.append(record)

    output = {
        "source_metrics_json": str(metrics_path),
        "top_word_f1": ranked(records, "word_f1", args.top_k),
        "top_content_f1": ranked(records, "content_f1", args.top_k),
    }
    if bert_f1 is not None:
        output["top_bertscore_f1"] = ranked(records, "bertscore_f1", args.top_k)

    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
