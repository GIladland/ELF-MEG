#!/usr/bin/env python
"""Paired-bootstrap comparison of two generation-metrics JSON files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CONTENT_STOPWORDS = set(
    """
    a about after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into
    is it its itself just me more most my no nor not of off on once only or other our ours
    ourselves out over own quite really same she should so some such sure than that the
    their theirs them themselves then there these they this those through to too under
    until up us very was we well were what when where which while who whom whose why will
    with would yes you your yours yourself yourselves
    """.split()
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, reference_token in enumerate(reference, start=1):
        current = [index]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def overlap_f1(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-bertscore", default="")
    parser.add_argument("--candidate-bertscore", default="")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    higher_is_better: bool,
    rng: np.random.Generator,
    samples: int,
) -> dict:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError(f"paired metric shape mismatch: {baseline.shape} != {candidate.shape}")
    delta = candidate - baseline
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    bootstrap = delta[indices].mean(axis=1)
    signed = delta if higher_is_better else -delta
    return {
        "num_rows": int(delta.size),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "candidate_minus_baseline": float(delta.mean()),
        "bootstrap_95ci_candidate_minus_baseline": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_probability_candidate_better": float(
            np.mean(bootstrap > 0.0) if higher_is_better else np.mean(bootstrap < 0.0)
        ),
        "rows_candidate_better": int(np.sum(signed > 0.0)),
        "rows_tied": int(np.sum(signed == 0.0)),
        "rows_candidate_worse": int(np.sum(signed < 0.0)),
    }


def per_sample_text_metrics(payload: dict) -> dict[str, np.ndarray]:
    """Return paired row metrics, reconstructing them for compact reranker JSONs."""
    overlap = payload.get("word_overlap", {})
    required = ("per_sample_content", "per_sample", "per_sample_wer")
    if all(key in overlap for key in required):
        return {key: np.asarray(overlap[key], dtype=np.float64) for key in required}

    generated = payload.get("generated")
    targets = payload.get("targets")
    if generated is None or targets is None or len(generated) != len(targets):
        raise ValueError("metrics JSON lacks both per-row metrics and aligned generated/targets")
    generated_words = [words(str(value)) for value in generated]
    target_words = [words(str(value)) for value in targets]
    return {
        "per_sample_content": np.asarray(
            [
                overlap_f1(
                    [token for token in hypothesis if token not in CONTENT_STOPWORDS],
                    [token for token in reference if token not in CONTENT_STOPWORDS],
                )
                for hypothesis, reference in zip(generated_words, target_words)
            ],
            dtype=np.float64,
        ),
        "per_sample": np.asarray(
            [
                overlap_f1(hypothesis, reference)
                for hypothesis, reference in zip(generated_words, target_words)
            ],
            dtype=np.float64,
        ),
        "per_sample_wer": np.asarray(
            [
                edit_distance(reference, hypothesis) / max(1, len(reference))
                for hypothesis, reference in zip(generated_words, target_words)
            ],
            dtype=np.float64,
        ),
    }


def corpus_wer(payload: dict) -> float:
    if "word_error_rate" in payload:
        return float(payload["word_error_rate"])
    return float(payload["generation_quality"]["word_error_rate"])


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    if baseline.get("targets") != candidate.get("targets"):
        raise ValueError("baseline and candidate targets are not in identical row order")

    baseline_overlap = per_sample_text_metrics(baseline)
    candidate_overlap = per_sample_text_metrics(candidate)
    rng = np.random.default_rng(args.seed)
    result = {
        "contract": "paired validation-row comparison; no test data",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "metrics": {
            "content_f1": paired_summary(
                np.asarray(baseline_overlap["per_sample_content"], dtype=np.float64),
                np.asarray(candidate_overlap["per_sample_content"], dtype=np.float64),
                higher_is_better=True,
                rng=rng,
                samples=args.samples,
            ),
            "word_f1": paired_summary(
                np.asarray(baseline_overlap["per_sample"], dtype=np.float64),
                np.asarray(candidate_overlap["per_sample"], dtype=np.float64),
                higher_is_better=True,
                rng=rng,
                samples=args.samples,
            ),
            "per_row_wer": paired_summary(
                np.asarray(baseline_overlap["per_sample_wer"], dtype=np.float64),
                np.asarray(candidate_overlap["per_sample_wer"], dtype=np.float64),
                higher_is_better=False,
                rng=rng,
                samples=args.samples,
            ),
        },
        "corpus_wer": {
            "baseline": corpus_wer(baseline),
            "candidate": corpus_wer(candidate),
            "candidate_minus_baseline": corpus_wer(candidate) - corpus_wer(baseline),
        },
    }

    if bool(args.baseline_bertscore) != bool(args.candidate_bertscore):
        raise ValueError("provide both BERTScore JSONs or neither")
    if args.baseline_bertscore:
        baseline_bert = np.asarray(
            load_json(args.baseline_bertscore)["per_sample"]["f1"], dtype=np.float64
        )
        candidate_bert = np.asarray(
            load_json(args.candidate_bertscore)["per_sample"]["f1"], dtype=np.float64
        )
        result["metrics"]["bertscore_f1"] = paired_summary(
            baseline_bert,
            candidate_bert,
            higher_is_better=True,
            rng=rng,
            samples=args.samples,
        )

    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
