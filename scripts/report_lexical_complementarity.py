#!/usr/bin/env python
"""Backfill exact, morphology-normalized, and function-word generation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean

from nltk.stem import PorterStemmer


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CONTENT_STOPWORDS = {
    "a", "about", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "quite", "really", "same", "she",
    "should", "so", "some", "such", "sure", "than", "that", "the", "their",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "we", "well", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "will", "with", "would", "yes", "you", "your", "yours",
    "yourself", "yourselves",
}
FUNCTION_CONTRACTION_SUFFIXES = {"s", "re", "ve", "d", "ll", "m"}
FUNCTION_NEGATIVE_CONTRACTIONS = {
    "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't",
    "hadn't", "hasn't", "haven't", "isn't", "mustn't", "needn't", "shan't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def counted_f1(left: list[str], right: list[str]) -> tuple[float, list[str]]:
    overlap = Counter(left) & Counter(right)
    count = sum(overlap.values())
    if not left or not right or count == 0:
        return 0.0, []
    precision = count / len(left)
    recall = count / len(right)
    expanded = [token for token, n in sorted(overlap.items()) for _ in range(n)]
    return 2 * precision * recall / (precision + recall), expanded


def is_function_word(token: str) -> bool:
    if token in CONTENT_STOPWORDS or token in FUNCTION_NEGATIVE_CONTRACTIONS:
        return True
    if "'" not in token:
        return False
    base, suffix = token.rsplit("'", 1)
    return base in CONTENT_STOPWORDS and suffix in FUNCTION_CONTRACTION_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Metrics JSON containing targets and generated text.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def content(tokens: list[str]) -> list[str]:
    return [
        token
        for token in tokens
        if token not in CONTENT_STOPWORDS and any(character.isalpha() for character in token)
    ]


def stemmed(tokens: list[str], stemmer) -> list[str]:
    # Match the standard ROUGE-1 Porter contract while retaining the original
    # evaluator's token boundaries and counts.
    return [stemmer.stem(token) if len(token) > 3 else token for token in tokens]


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    generated = [str(value) for value in payload["generated"]]
    targets = [str(value) for value in payload["targets"]]
    if len(generated) != len(targets) or not generated:
        raise ValueError("Input must contain equally sized, non-empty generated and targets lists")

    stemmer = PorterStemmer()
    rows: list[dict] = []
    for index, (hypothesis, reference) in enumerate(zip(generated, targets)):
        hypothesis_words = words(hypothesis)
        reference_words = words(reference)
        hypothesis_content = content(hypothesis_words)
        reference_content = content(reference_words)
        hypothesis_function = [token for token in hypothesis_words if is_function_word(token)]
        reference_function = [token for token in reference_words if is_function_word(token)]

        exact_word_f1, exact_word_matches = counted_f1(hypothesis_words, reference_words)
        exact_content_f1, exact_content_matches = counted_f1(hypothesis_content, reference_content)
        stemmed_word_f1, stemmed_word_matches = counted_f1(
            stemmed(hypothesis_words, stemmer), stemmed(reference_words, stemmer)
        )
        stemmed_content_f1, stemmed_content_matches = counted_f1(
            stemmed(hypothesis_content, stemmer), stemmed(reference_content, stemmer)
        )
        rows.append(
            {
                "index": index,
                "target": reference,
                "generated": hypothesis,
                "target_word_count": len(reference_words),
                "decoded_word_count": len(hypothesis_words),
                "exact_word_f1": exact_word_f1,
                "stemmed_word_f1": stemmed_word_f1,
                "exact_content_word_f1": exact_content_f1,
                "stemmed_content_word_f1": stemmed_content_f1,
                "exact_overlap_words": " ".join(exact_word_matches),
                "stemmed_overlap_forms": " ".join(stemmed_word_matches),
                "exact_content_overlap_words": " ".join(exact_content_matches),
                "stemmed_content_overlap_forms": " ".join(stemmed_content_matches),
                "decoded_function_word_count": len(hypothesis_function),
                "decoded_function_word_fraction": len(hypothesis_function) / max(1, len(hypothesis_words)),
                "decoded_function_words": " ".join(hypothesis_function),
                "target_function_word_count": len(reference_function),
                "target_function_word_fraction": len(reference_function) / max(1, len(reference_words)),
                "target_function_words": " ".join(reference_function),
            }
        )

    total_decoded = sum(row["decoded_word_count"] for row in rows)
    total_decoded_function = sum(row["decoded_function_word_count"] for row in rows)
    total_target = sum(row["target_word_count"] for row in rows)
    total_target_function = sum(row["target_function_word_count"] for row in rows)
    summary = {
        "num_sentences": len(rows),
        "exact_word_f1": fmean(row["exact_word_f1"] for row in rows),
        "stemmed_word_f1": fmean(row["stemmed_word_f1"] for row in rows),
        "exact_content_word_f1": fmean(row["exact_content_word_f1"] for row in rows),
        "stemmed_content_word_f1": fmean(row["stemmed_content_word_f1"] for row in rows),
        "decoded_word_count": total_decoded,
        "decoded_function_word_count": total_decoded_function,
        "decoded_function_word_fraction": total_decoded_function / max(1, total_decoded),
        "target_word_count": total_target,
        "target_function_word_count": total_target_function,
        "target_function_word_fraction": total_target_function / max(1, total_target),
    }
    Path(args.output_json).write_text(json.dumps({"summary": summary, "per_sample": rows}, indent=2) + "\n")
    with Path(args.output_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
