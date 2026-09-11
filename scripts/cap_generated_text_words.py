#!/usr/bin/env python3
"""Apply a fixed word-count cap to saved generations and report corpus WER."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Sequence


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CONTENT_STOPWORDS = {
    "a", "about", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "no", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "quite", "really", "same",
    "she", "should", "so", "some", "such", "sure", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "us", "very", "was", "we", "well", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "why", "will", "with", "would", "yes",
    "you", "your", "yours", "yourself", "yourselves",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-words", type=int, required=True)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def counted_overlap(generated: list[str], target: list[str]) -> tuple[float, float, float]:
    overlap = sum((Counter(generated) & Counter(target)).values())
    precision = overlap / max(1, len(generated))
    recall = overlap / max(1, len(target))
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def overlap_summary(generated: list[list[str]], targets: list[list[str]]) -> dict[str, float]:
    word = [counted_overlap(candidate, target) for candidate, target in zip(generated, targets)]
    content = [
        counted_overlap(
            [token for token in candidate if token not in CONTENT_STOPWORDS and any(c.isalpha() for c in token)],
            [token for token in target if token not in CONTENT_STOPWORDS and any(c.isalpha() for c in token)],
        )
        for candidate, target in zip(generated, targets)
    ]
    return {
        "words_overlap_precision": fmean(row[0] for row in word),
        "words_overlap_recall": fmean(row[1] for row in word),
        "words_overlap": fmean(row[2] for row in word),
        "content_words_overlap_precision": fmean(row[0] for row in content),
        "content_words_overlap_recall": fmean(row[1] for row in content),
        "content_words_overlap": fmean(row[2] for row in content),
    }


def main() -> None:
    args = parse_args()
    if args.max_words <= 0:
        raise ValueError("--max-words must be positive")
    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    targets = [str(value) for value in payload["targets"]]
    original = [str(value) for value in payload["generated"]]
    capped_tokens = [words(text)[: args.max_words] for text in original]
    capped = [" ".join(tokens) for tokens in capped_tokens]
    target_tokens = [words(text) for text in targets]
    errors = [edit_distance(reference, hypothesis) for reference, hypothesis in zip(target_tokens, capped_tokens)]
    reference_words = sum(len(tokens) for tokens in target_tokens)
    result = {
        "contract": "fixed generated-word cap",
        "source_metrics_json": str(args.metrics_json),
        "max_words": args.max_words,
        "reference_labels_used_to_choose_or_apply_cap": False,
        "cap_source": "dataset exact-word-count contract",
        "num_examples": len(targets),
        "word_error_rate": sum(errors) / max(1, reference_words),
        "edit_errors": int(sum(errors)),
        "reference_words": int(reference_words),
        "mean_original_words": sum(len(words(text)) for text in original) / max(1, len(original)),
        "mean_capped_words": sum(len(tokens) for tokens in capped_tokens) / max(1, len(capped_tokens)),
        "per_sample_wer": [error / max(1, len(reference)) for error, reference in zip(errors, target_tokens)],
        "overlap_metrics": overlap_summary(capped_tokens, target_tokens),
        "targets": targets,
        # Retain the standard ELF metrics schema so post-hoc BERTScore and
        # retrieval utilities can evaluate the capped deterministic decode.
        "generated": capped,
        "original_generated": original,
        "capped_generated": capped,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key not in {"targets", "original_generated", "capped_generated", "per_sample_wer"}}, indent=2))


if __name__ == "__main__":
    main()
