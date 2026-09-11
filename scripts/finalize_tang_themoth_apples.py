#!/usr/bin/env python
"""Validate and summarize the matched Tang/TheMoth fMRI-versus-MEG evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri-json", type=Path, required=True)
    parser.add_argument("--meg-json", type=Path, required=True)
    parser.add_argument("--meg-test-npz", type=Path, required=True)
    parser.add_argument(
        "--expected-meg-samples",
        type=int,
        default=None,
        help="Require this many MEG samples per row (for example, 2500 for 10 seconds at 250 Hz).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def binomial_tail(n: int, k: int, probability: float) -> float:
    return float(
        sum(
            math.comb(n, hits)
            * probability**hits
            * (1.0 - probability) ** (n - hits)
            for hits in range(k, n + 1)
        )
    )


def require_fraction_count(value: Any, n: int, name: str) -> tuple[float, int]:
    fraction = float(value)
    count = int(round(fraction * n))
    if not math.isclose(fraction, count / n, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name}={fraction} is not an integer count over n={n}")
    return fraction, count


def semantic_metrics(payload: dict[str, Any], modality: str, n: int) -> dict[str, Any]:
    if modality == "fmri":
        source = payload["semantic_interface"]
        top1_key, top5_key = "top1", "top5"
        mean_rank_key, cosine_key = "mean_rank", "matched_cosine_mean"
    else:
        source = payload["meg2sem_interface"]
        top1_key, top5_key = "interface_semantic_top1", "interface_semantic_top5"
        mean_rank_key, cosine_key = "interface_semantic_mean_rank", "interface_semantic_cosine"
    top1, top1_count = require_fraction_count(source[top1_key], n, f"{modality}.semantic_top1")
    top5, top5_count = require_fraction_count(source[top5_key], n, f"{modality}.semantic_top5")
    return {
        "top1": top1,
        "top1_count": top1_count,
        "top5": top5,
        "top5_count": top5_count,
        "mean_rank": float(source[mean_rank_key]),
        "matched_cosine": float(source[cosine_key]),
        "top1_binomial_p": binomial_tail(n, top1_count, 1.0 / n),
        "top5_binomial_p": binomial_tail(n, top5_count, min(5, n) / n),
    }


def generation_metrics(payload: dict[str, Any], modality: str, n: int) -> dict[str, Any]:
    retrieval = payload["generation_t5_retrieval"]
    quality = payload["generation_quality"]
    top1, top1_count = require_fraction_count(retrieval["top1"], n, f"{modality}.generated_top1")
    top5, top5_count = require_fraction_count(retrieval["top5"], n, f"{modality}.generated_top5")
    return {
        "top1": top1,
        "top1_count": top1_count,
        "top5": top5,
        "top5_count": top5_count,
        "mean_rank": float(retrieval["mean_rank"]),
        "wer": float(quality["word_error_rate"]),
        "wer_errors": int(quality["word_error_deletions"])
        + int(quality["word_error_insertions"])
        + int(quality["word_error_substitutions"]),
        "reference_words": int(quality["word_error_reference_words"]),
        "word_overlap": float(quality["words_overlap"]),
        "content_overlap": float(quality["content_words_overlap"]),
        "top1_binomial_p": binomial_tail(n, top1_count, 1.0 / n),
        "top5_binomial_p": binomial_tail(n, top5_count, min(5, n) / n),
    }


def per_sample(payload: dict[str, Any], key: str, n: int, default: Any = None) -> list[Any]:
    value = payload.get("word_overlap", {}).get(key)
    if value is None:
        return [default for _ in range(n)]
    if len(value) != n:
        raise ValueError(f"Per-sample field {key!r} has {len(value)} rows, expected {n}")
    return list(value)


def retrieval_ranks(payload: dict[str, Any], n: int) -> list[Any]:
    ranks = payload.get("generation_t5_retrieval", {}).get("ranks")
    if ranks is None:
        return [None for _ in range(n)]
    if len(ranks) != n:
        raise ValueError(f"Generation ranks have {len(ranks)} rows, expected {n}")
    return list(ranks)


def main() -> None:
    args = parse_args()
    fmri = json.loads(args.fmri_json.read_text(encoding="utf-8"))
    meg = json.loads(args.meg_json.read_text(encoding="utf-8"))

    fmri_targets = [str(value) for value in fmri["targets"]]
    meg_targets = [str(value) for value in meg["targets"]]
    n = len(fmri_targets)
    if n == 0 or len(meg_targets) != n:
        raise ValueError(f"Evaluation sizes differ: fMRI={n}, MEG={len(meg_targets)}")
    if fmri_targets != meg_targets:
        mismatch = next(index for index, pair in enumerate(zip(fmri_targets, meg_targets)) if pair[0] != pair[1])
        raise ValueError(
            f"Ordered fMRI and MEG targets differ at row {mismatch}: "
            f"{fmri_targets[mismatch]!r} != {meg_targets[mismatch]!r}"
        )
    if len(set(fmri_targets)) != n:
        raise ValueError("The matched retrieval bank contains duplicate target strings")

    with np.load(args.meg_test_npz, allow_pickle=True) as packed:
        packed_targets = [str(value) for value in packed["sentence"]]
        if packed_targets != fmri_targets:
            raise ValueError("Packed locked-test sentences do not match ordered evaluation targets")
        if packed["input_embeddings"].shape != (n, 384):
            raise ValueError(f"Unexpected locked-test semantic shape: {packed['input_embeddings'].shape}")
        meg_shape = tuple(int(value) for value in packed["meg"].shape)
        if len(meg_shape) != 3 or meg_shape[0] != n or meg_shape[1] != 306:
            raise ValueError(f"Unexpected locked-test MEG shape: {packed['meg'].shape}")
        if args.expected_meg_samples is not None and meg_shape[2] != args.expected_meg_samples:
            raise ValueError(
                f"Locked-test MEG has {meg_shape[2]} samples per row, "
                f"expected {args.expected_meg_samples}"
            )
        sessions = [str(value) for value in packed["session"]]
        if set(sessions) != {"19"}:
            raise ValueError(f"Locked-test session is not exactly session 19: {sorted(set(sessions))}")
        start_samples = [int(value) for value in packed["start_samples"]]

    if len(fmri["generated"]) != n or len(meg["generated"]) != n:
        raise ValueError("Generation row counts do not match the candidate bank")

    comparison = {
        "status": "PASS",
        "n": n,
        "story": "birthofanation",
        "meg_session": "19",
        "target_words_per_row": 10,
        "semantic_dim": 384,
        "meg_shape": list(meg_shape),
        "chance": {"top1": 1.0 / n, "top5": min(5, n) / n},
        "ordered_target_bank_equal": True,
        "unique_target_count": n,
        "fmri": {
            "semantic": semantic_metrics(fmri, "fmri", n),
            "generation": generation_metrics(fmri, "fmri", n),
            "metrics_json": str(args.fmri_json),
        },
        "meg": {
            "semantic": semantic_metrics(meg, "meg", n),
            "generation": generation_metrics(meg, "meg", n),
            "metrics_json": str(args.meg_json),
        },
        "meg_test_npz": str(args.meg_test_npz),
    }

    fmri_wer = per_sample(fmri, "per_sample_wer", n)
    meg_wer = per_sample(meg, "per_sample_wer", n)
    fmri_word = per_sample(fmri, "per_sample", n)
    meg_word = per_sample(meg, "per_sample", n)
    fmri_content = per_sample(fmri, "per_sample_content", n)
    meg_content = per_sample(meg, "per_sample_content", n)
    fmri_ranks = retrieval_ranks(fmri, n)
    meg_ranks = retrieval_ranks(meg, n)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "story",
                "meg_session",
                "meg_start_sample",
                "target",
                "fmri_generated",
                "meg_generated",
                "fmri_wer",
                "meg_wer",
                "fmri_word_overlap",
                "meg_word_overlap",
                "fmri_content_overlap",
                "meg_content_overlap",
                "fmri_generated_rank",
                "meg_generated_rank",
            ],
        )
        writer.writeheader()
        for index in range(n):
            writer.writerow(
                {
                    "row": index,
                    "story": "birthofanation",
                    "meg_session": sessions[index],
                    "meg_start_sample": start_samples[index],
                    "target": fmri_targets[index],
                    "fmri_generated": fmri["generated"][index],
                    "meg_generated": meg["generated"][index],
                    "fmri_wer": fmri_wer[index],
                    "meg_wer": meg_wer[index],
                    "fmri_word_overlap": fmri_word[index],
                    "meg_word_overlap": meg_word[index],
                    "fmri_content_overlap": fmri_content[index],
                    "meg_content_overlap": meg_content[index],
                    "fmri_generated_rank": fmri_ranks[index],
                    "meg_generated_rank": meg_ranks[index],
                }
            )

    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
