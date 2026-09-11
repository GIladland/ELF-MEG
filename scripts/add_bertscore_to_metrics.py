#!/usr/bin/env python
"""Compute post-hoc BERTScore for an ELF generation metrics JSON.

This is deliberately a reporting utility rather than part of checkpoint
selection.  In particular, test references must not be used to choose a model
or any decoding hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-rescale-with-baseline",
        action="store_true",
        help="Return raw BERTScore rather than the recommended language baseline-rescaled score.",
    )
    return parser.parse_args()


def ranked_examples(
    generated: list[str],
    targets: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    *,
    top_k: int,
) -> list[dict]:
    order = np.argsort(-f1, kind="stable")[: max(0, top_k)]
    return [
        {
            "index": int(index),
            "target": targets[index],
            "generated": generated[index],
            "bertscore_precision": float(precision[index]),
            "bertscore_recall": float(recall[index]),
            "bertscore_f1": float(f1[index]),
        }
        for index in order
    ]


def main() -> None:
    args = parse_args()
    source_path = Path(args.metrics_json)
    with source_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    generated = [str(text) for text in metrics.get("generated", [])]
    targets = [str(text) for text in metrics.get("targets", [])]
    if not generated or not targets:
        raise ValueError("metrics JSON must contain non-empty generated and targets arrays")
    if len(generated) != len(targets):
        raise ValueError(f"generated/targets length mismatch: {len(generated)} != {len(targets)}")

    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise SystemExit(
            "bert-score is required; install it with `python -m pip install bert-score==0.3.13`"
        ) from exc

    scorer_kwargs = {
        "lang": args.lang,
        "rescale_with_baseline": not args.no_rescale_with_baseline,
    }
    if args.model_type:
        scorer_kwargs["model_type"] = args.model_type
    if args.num_layers is not None:
        scorer_kwargs["num_layers"] = args.num_layers
    if args.device:
        scorer_kwargs["device"] = args.device
    scorer = BERTScorer(**scorer_kwargs)
    precision_t, recall_t, f1_t = scorer.score(generated, targets, batch_size=args.batch_size)
    precision = precision_t.detach().cpu().float().numpy()
    recall = recall_t.detach().cpu().float().numpy()
    f1 = f1_t.detach().cpu().float().numpy()

    output = {
        "contract": "post-hoc reporting only; never use test BERTScore for model selection",
        "source_metrics_json": str(source_path),
        "num_examples": len(generated),
        "model_type": scorer.model_type,
        "num_layers": int(scorer.num_layers),
        "lang": args.lang,
        "rescale_with_baseline": not args.no_rescale_with_baseline,
        "summary": {
            "bertscore_precision": float(np.mean(precision)),
            "bertscore_recall": float(np.mean(recall)),
            "bertscore_f1": float(np.mean(f1)),
        },
        "per_sample": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
        },
        "best_examples": ranked_examples(
            generated,
            targets,
            precision,
            recall,
            f1,
            top_k=args.top_k,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({**output["summary"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
