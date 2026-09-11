#!/usr/bin/env python
"""Select a retrieval-preserving fMRI-to-ELF interface on val266 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = [
    "residual_anchor_geometry",
    "residual_rankdistill_pair",
    "residual_exactsemantic_rank_pair",
    "residual_exactsemantic_rank_pair_content",
    "residual_exactsemantic_rank_pair_content_lora4last2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--snapshot-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def metric_row(metrics: dict) -> dict[str, float | int]:
    quality = metrics.get("generation_quality") or {}
    generated = metrics.get("generation_t5_retrieval") or {}
    semantic = metrics.get("semantic_interface") or {}
    return {
        "step": int(metrics.get("step") or 0),
        "n": int(metrics.get("eval_num_examples") or metrics.get("num_eval_examples") or 0),
        "wer": float(quality.get("word_error_rate", float("inf"))),
        "word_f1": float(quality.get("words_overlap", 0.0)),
        "content_f1": float(quality.get("content_words_overlap", 0.0)),
        "generated_top1": float(generated.get("top1", 0.0)),
        "generated_top5": float(generated.get("top5", 0.0)),
        "generated_mean_rank": float(generated.get("mean_rank", float("inf"))),
        "semantic_top1": float(semantic.get("top1", 0.0)),
        "semantic_top5": float(semantic.get("top5", 0.0)),
        "semantic_mean_rank": float(semantic.get("mean_rank", float("inf"))),
        "semantic_cosine": float(semantic.get("matched_cosine_mean", float("nan"))),
    }


def add_gate_checks(row: dict, floors: dict[str, float]) -> dict:
    gate_checks = {
        "generated_top1": row["generated_top1"] >= floors["generated_top1"],
        "generated_top5": row["generated_top5"] >= floors["generated_top5"],
        "semantic_top1": row["semantic_top1"] >= floors["semantic_top1"],
        "semantic_top5": row["semantic_top5"] >= floors["semantic_top5"],
        "generated_mean_rank": row["generated_mean_rank"] <= floors["generated_mean_rank"],
        "semantic_mean_rank": row["semantic_mean_rank"] <= floors["semantic_mean_rank"],
    }
    row["gate_checks"] = gate_checks
    row["passes_retrieval_gate"] = all(gate_checks.values())
    return row


def lexical_score(row: dict) -> tuple[float, ...]:
    return (
        row["content_f1"],
        row["word_f1"],
        row["generated_top1"],
        row["generated_top5"],
        -row["wer"],
    )


def checkpoint_for_step(run_dir: Path, step: int, best_step: int) -> Path | None:
    if step == best_step and (run_dir / "best.pt").exists():
        return run_dir / "best.pt"
    matches = sorted((run_dir / "checkpoints").glob(f"eval_step_{step:08d}_score_*.pt"))
    return matches[-1] if matches else None


def main() -> None:
    args = parse_args()
    baseline_tag = (
        f"fmri_freshmem_semantic_preservation_{args.snapshot_tag}_"
        "raw_residual_val266_seed49_20260901"
    )
    baseline = metric_row(load(args.run_root / baseline_tag / "eval_step_000000.json"))
    n = max(1, int(baseline["n"]))
    one_hit = 1.0 / n
    floors = {
        "generated_top1": max(0.0, float(baseline["generated_top1"]) - one_hit),
        "generated_top5": max(0.0, float(baseline["generated_top5"]) - 2.0 * one_hit),
        "semantic_top1": max(0.0, float(baseline["semantic_top1"]) - one_hit),
        "semantic_top5": max(0.0, float(baseline["semantic_top5"]) - 2.0 * one_hit),
        "generated_mean_rank": float(baseline["generated_mean_rank"]) * 1.10,
        "semantic_mean_rank": float(baseline["semantic_mean_rank"]) * 1.10,
    }

    candidates = []
    trajectories = {}
    for variant in VARIANTS:
        run_tag = (
            f"fmri_freshmem_semantic_preservation_{args.snapshot_tag}_{variant}_"
            "val266_seed49_20260901"
        )
        run_dir = args.run_root / run_tag
        best_metrics = load(run_dir / "best_metrics.json")
        best_step = int(best_metrics.get("step") or 0)
        available = []
        seen = set()
        for metrics_path in sorted(run_dir.glob("eval_step_*.json")):
            metrics = load(metrics_path)
            step = int(metrics.get("step") or 0)
            checkpoint = checkpoint_for_step(run_dir, step, best_step)
            if checkpoint is None or (step, str(checkpoint)) in seen:
                continue
            seen.add((step, str(checkpoint)))
            row = metric_row(metrics)
            row.update(
                {
                    "variant": variant,
                    "run_tag": run_tag,
                    "checkpoint": str(checkpoint),
                }
            )
            available.append(add_gate_checks(row, floors))
        if not available:
            raise FileNotFoundError(f"No saved validation checkpoints found in {run_dir}")
        trajectories[variant] = available
        eligible_for_arm = [row for row in available if row["passes_retrieval_gate"]]
        if eligible_for_arm:
            row = max(eligible_for_arm, key=lexical_score)
        else:
            row = max(
                available,
                key=lambda item: (
                    item["semantic_top5"],
                    item["semantic_top1"],
                    -item["semantic_mean_rank"],
                ),
            )
        candidates.append(row)

    eligible = [row for row in candidates if row["passes_retrieval_gate"]]
    winner = None
    if eligible:
        winner = max(eligible, key=lexical_score)

    payload = {
        "selection_split": "predicted val266",
        "test_rows_loaded_during_selection": 0,
        "selection_rule": (
            "Require generated and semantic retrieval to remain within one Top-1 hit, "
            "two Top-5 hits, and 10% mean-rank of the frozen raw baseline; then maximize "
            "content F1, word F1, generated Top-1/Top-5, and finally minimize WER."
        ),
        "baseline": baseline,
        "retrieval_floors": floors,
        "candidates": candidates,
        "validation_checkpoint_trajectories": trajectories,
        "promoted": winner is not None,
        "winner": winner,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
