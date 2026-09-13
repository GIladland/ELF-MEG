#!/usr/bin/env python3
"""Apply a fixed word-count cap to saved generations and report corpus WER."""

from __future__ import annotations

import argparse
import csv
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics-json", type=Path)
    source.add_argument(
        "--input-csv",
        type=Path,
        help="CSV with index, target, and generated columns.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-words", type=int, required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def cap_text(text: str, max_words: int) -> str:
    """Keep the text prefix ending at the Nth evaluator-visible word.

    This deliberately uses the same lexical contract as WER and overlap
    scoring while retaining the original spelling and punctuation before the
    cutoff.  It therefore avoids the old whitespace-token mismatch without
    otherwise normalizing an already short generation.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    matches = list(WORD_RE.finditer(text))
    if len(matches) <= max_words:
        return text.strip()
    return text[: matches[max_words - 1].end()].strip()


def load_rows(
    *, metrics_json: Path | None, input_csv: Path | None
) -> tuple[list[str], list[str], dict]:
    if metrics_json is not None:
        payload = json.loads(metrics_json.read_text(encoding="utf-8"))
        targets = [str(value) for value in payload["targets"]]
        generated = [str(value) for value in payload["generated"]]
        return targets, generated, payload

    if input_csv is None:
        raise ValueError("one input source is required")
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {input_csv}")
    required = {"index", "target", "generated"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    try:
        rows.sort(key=lambda row: int(row["index"]))
    except ValueError as error:
        raise ValueError("CSV index values must be integers") from error
    indices = [int(row["index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"CSV indices must be exactly 0..{len(rows) - 1}")
    return (
        [str(row["target"]) for row in rows],
        [str(row["generated"]) for row in rows],
        {},
    )


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
    targets, original, source_payload = load_rows(
        metrics_json=args.metrics_json, input_csv=args.input_csv
    )
    if len(targets) != len(original):
        raise ValueError("generated/targets length mismatch")
    capped = [cap_text(text, args.max_words) for text in original]
    capped_tokens = [words(text) for text in capped]
    target_tokens = [words(text) for text in targets]
    errors = [edit_distance(reference, hypothesis) for reference, hypothesis in zip(target_tokens, capped_tokens)]
    reference_words = sum(len(tokens) for tokens in target_tokens)
    overlap_metrics = overlap_summary(capped_tokens, target_tokens)
    word_error_rate = sum(errors) / max(1, reference_words)
    source_path = args.metrics_json if args.metrics_json is not None else args.input_csv
    result = {
        "run_tag": args.run_tag or source_payload.get("run_tag", source_path.stem),
        "checkpoint": args.checkpoint or source_payload.get("checkpoint", ""),
        "contract": "fixed evaluator-lexical-token generated-word cap",
        "source": str(source_path),
        "max_words": args.max_words,
        "word_regex": WORD_RE.pattern,
        "reference_labels_used_to_choose_or_apply_cap": False,
        "cap_source": "dataset exact-word-count contract",
        "num_examples": len(targets),
        "word_error_rate": word_error_rate,
        "edit_errors": int(sum(errors)),
        "reference_words": int(reference_words),
        "mean_original_words": sum(len(words(text)) for text in original) / max(1, len(original)),
        "mean_capped_words": sum(len(tokens) for tokens in capped_tokens) / max(1, len(capped_tokens)),
        "per_sample_wer": [error / max(1, len(reference)) for error, reference in zip(errors, target_tokens)],
        "overlap_metrics": overlap_metrics,
        "generation_quality": {
            "word_error_rate": word_error_rate,
            "word_error_reference_words": int(reference_words),
            **overlap_metrics,
        },
        "word_overlap": {"summary": overlap_metrics},
        "targets": targets,
        # Retain the standard ELF metrics schema so post-hoc BERTScore and
        # retrieval utilities can evaluate the capped deterministic decode.
        "generated": capped,
        "original_generated": original,
        "capped_generated": capped,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key not in {"targets", "generated", "original_generated", "capped_generated", "per_sample_wer"}}, indent=2))


if __name__ == "__main__":
    main()
