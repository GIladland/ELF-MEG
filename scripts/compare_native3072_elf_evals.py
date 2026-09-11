#!/usr/bin/env python
"""Summarize the native ELF oracle-vs-MRI2SEM 3072 test-input drop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HIGHER_IS_BETTER = {
    "generation/exact_match",
    "generation/words_overlap",
    "generation/content_words_overlap",
    "generation/well_structured_sentence",
    "generation_t5_retrieval/top1",
    "generation_t5_retrieval/top5",
    "condition_target_retrieval/combined/top1",
    "condition_target_retrieval/combined/top5",
    "condition_target_retrieval/denoiser/top1",
    "condition_target_retrieval/denoiser/top5",
    "condition_target_retrieval/decoder/top1",
    "condition_target_retrieval/decoder/top5",
}

LOWER_IS_BETTER = {
    "generation_t5_retrieval/mean_rank",
    "generation_t5_retrieval/median_rank",
    "condition_target_retrieval/combined/mean_rank",
    "condition_target_retrieval/combined/median_rank",
    "condition_target_retrieval/denoiser/mean_rank",
    "condition_target_retrieval/denoiser/median_rank",
    "condition_target_retrieval/decoder/mean_rank",
    "condition_target_retrieval/decoder/median_rank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--predicted-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract(eval_dir: Path) -> tuple[dict[str, float], dict]:
    generation = load_json(eval_dir / "best_metrics.json")
    condition_retrieval = load_json(eval_dir / "retrieval_step_000000.json")
    quality = generation.get("generation_quality") or {}
    t5_retrieval = generation.get("generation_t5_retrieval") or {}
    metrics = {
        "generation/exact_match": float(generation["exact_match"]),
        "generation/words_overlap": float(quality.get("words_overlap", 0.0)),
        "generation/content_words_overlap": float(quality.get("content_words_overlap", 0.0)),
        "generation/well_structured_sentence": float(quality.get("well_structured_sentence", 0.0)),
    }
    for key in ("top1", "top5", "mean_rank", "median_rank"):
        metrics[f"generation_t5_retrieval/{key}"] = float(t5_retrieval[key])
    for branch in ("combined", "denoiser", "decoder"):
        for key in ("top1", "top5", "mean_rank", "median_rank"):
            metrics[f"condition_target_retrieval/{branch}/{key}"] = float(
                condition_retrieval[branch][key]
            )
    return metrics, {
        "generation": generation,
        "condition_retrieval": condition_retrieval,
    }


def nested(metrics: dict[str, float]) -> dict:
    output: dict = {}
    for path, value in metrics.items():
        cursor = output
        parts = path.split("/")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return output


def main() -> None:
    args = parse_args()
    oracle_dir = Path(args.oracle_dir)
    predicted_dir = Path(args.predicted_dir)
    oracle, oracle_raw = extract(oracle_dir)
    predicted, predicted_raw = extract(predicted_dir)

    oracle_targets = oracle_raw["generation"].get("targets") or []
    predicted_targets = predicted_raw["generation"].get("targets") or []
    if oracle_targets != predicted_targets:
        raise ValueError("Oracle and MRI2SEM predicted evaluations do not use identical ordered test targets.")

    comparison = {}
    for path in sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER):
        oracle_value = oracle[path]
        predicted_value = predicted[path]
        row = {
            "oracle": oracle_value,
            "mri2sem_predicted": predicted_value,
        }
        if path in HIGHER_IS_BETTER:
            row["absolute_drop"] = oracle_value - predicted_value
            row["predicted_retention"] = (
                predicted_value / oracle_value if oracle_value != 0.0 else None
            )
        else:
            row["rank_increase"] = predicted_value - oracle_value
        comparison[path] = row

    output = {
        "contract": "oracle Tang-GPT 3072 -> ELF vs fMRI -> MRI2SEM predicted 3072 -> ELF",
        "n_test": len(oracle_targets),
        "paired_targets_identical": True,
        "oracle_eval_dir": str(oracle_dir),
        "predicted_eval_dir": str(predicted_dir),
        "oracle": nested(oracle),
        "mri2sem_predicted": nested(predicted),
        "comparison": comparison,
        "existing_mri2sem_3072_retrieval_baseline": {
            "top1": 0.22429906542056074,
            "top5": 0.4205607476635514,
            "top10": 0.5420560747663551,
            "mean_percentile": 0.8217245635690356,
            "matched_cosine": 0.1738278567790985,
            "mismatch_cosine": -0.010794306173920631,
            "chance_top1": 0.009345794392523364,
            "chance_top5": 0.04672897196261682,
            "chance_top10": 0.09345794392523364,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
