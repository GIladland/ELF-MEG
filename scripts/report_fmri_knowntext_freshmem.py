#!/usr/bin/env python
"""Build the exact-semantic, frozen-brain, and adapted-interface report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[a-z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--snapshot-tag", required=True)
    parser.add_argument("--oracle-train-tag", required=True)
    parser.add_argument("--baseline-eval-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def metrics_row(metrics: dict) -> dict[str, float | int | None]:
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    return {
        "step": metrics.get("step"),
        "wer": quality.get("word_error_rate"),
        "word_f1": quality.get("words_overlap"),
        "content_f1": quality.get("content_words_overlap"),
        "exact_match": metrics.get("exact_match"),
        "top1": retrieval.get("top1"),
        "top5": retrieval.get("top5"),
        "mean_rank": retrieval.get("mean_rank"),
    }


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def bag_f1(target: str, generated: str) -> float:
    left = Counter(tokens(target))
    right = Counter(tokens(generated))
    overlap = sum((left & right).values())
    if not left or not right or overlap == 0:
        return 0.0
    precision = overlap / sum(right.values())
    recall = overlap / sum(left.values())
    return 2.0 * precision * recall / (precision + recall)


def export_pairs(metrics: dict, path: Path) -> list[dict]:
    targets = [str(value) for value in metrics.get("targets", [])]
    generated = [str(value) for value in metrics.get("generated", [])]
    if len(targets) != len(generated):
        raise ValueError(f"target/generated length mismatch in {path}: {len(targets)} != {len(generated)}")
    rows = [
        {
            "index": index,
            "target": target,
            "generated": prediction,
            "word_f1": bag_f1(target, prediction),
        }
        for index, (target, prediction) in enumerate(zip(targets, generated))
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "target", "generated", "word_f1"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def fmt(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oracle_dir = args.run_root / args.oracle_train_tag
    snapshot_step = args.snapshot_tag.removeprefix("step")
    snapshot_metrics = load_json(
        oracle_dir / "snapshots" / f"best_through_step_{snapshot_step}_metrics.json"
    )
    exact_eval = load_json(
        args.run_root
        / f"eval_{args.baseline_eval_tag}_exact_minilm384_oracle_seed7_20260901"
        / "eval_step_000000.json"
    )
    frozen_brain_eval = load_json(
        args.run_root
        / f"eval_{args.baseline_eval_tag}_unseen_fmri_delayed1536_meanblocks_seed7_20260901"
        / "eval_step_000000.json"
    )
    selection_dir = args.run_root / f"fmri_knowntext_freshmem_{args.snapshot_tag}_selection_20260901"
    selection = load_json(selection_dir / "selection.json")
    winner = selection["winner"]
    winner_eval_tag = f"{args.snapshot_tag}_{winner['variant']}"
    winner_test = load_json(
        args.run_root
        / f"eval_fmri_knowntext_freshmem_{winner_eval_tag}_unseenbrain_test107_seed49_20260901"
        / "eval_step_000000.json"
    )

    stages = {
        "oracle_snapshot": metrics_row(snapshot_metrics),
        "oracle_replay": metrics_row(exact_eval),
        "fixed_brain_mapper": metrics_row(frozen_brain_eval),
        "adapted_interface_val266": {
            "step": winner.get("step"),
            "wer": winner.get("wer"),
            "word_f1": winner.get("word_f1"),
            "content_f1": winner.get("content_f1"),
            "exact_match": None,
            "top1": winner.get("top1"),
            "top5": winner.get("top5"),
            "mean_rank": None,
        },
        "adapted_interface_test107": metrics_row(winner_test),
    }
    t5_prefix = {
        "val266": {"wer": 0.9872180451, "word_f1": 0.0826777017, "content_f1": 0.0437433181},
        "test107": {"wer": 0.988785, "word_f1": 0.060702, "content_f1": 0.017756},
    }

    frozen_rows = export_pairs(frozen_brain_eval, args.output_dir / "fixed_brain_target_generated.csv")
    winner_rows = export_pairs(winner_test, args.output_dir / "adapted_winner_target_generated.csv")
    oracle_rows = export_pairs(exact_eval, args.output_dir / "oracle_target_generated.csv")
    ranked = {
        "oracle_top_word_f1": sorted(oracle_rows, key=lambda row: row["word_f1"], reverse=True)[:10],
        "fixed_brain_top_word_f1": sorted(frozen_rows, key=lambda row: row["word_f1"], reverse=True)[:10],
        "adapted_winner_top_word_f1": sorted(winner_rows, key=lambda row: row["word_f1"], reverse=True)[:10],
    }

    payload = {
        "protocol": {
            "oracle_diffusion_training": "all 12,098 exact MiniLM384/text pairs, including test text",
            "predicted_test_vectors_in_diffusion_training": 0,
            "interface_training": "exact oracle plus story-level OOF train predictions",
            "interface_selection": "predicted val266",
            "test_access": "one frozen evaluation after validation selection",
        },
        "snapshot_tag": args.snapshot_tag,
        "oracle_train_tag": args.oracle_train_tag,
        "interface_selection": selection,
        "stages": stages,
        "t5_prefix_reference": t5_prefix,
        "ranked_examples": ranked,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# fMRI → memorized ELF-B report",
        "",
        "The diffusion model was trained on every exact MiniLM384/text pair, including the 107 known target texts. It never saw a test107 fMRI-predicted vector during training or checkpoint selection.",
        "",
        "| Stage | WER | Word F1 | Content F1 | Top-1 | Top-5 | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "oracle_snapshot": "1a. Oracle snapshot (training audit)",
        "oracle_replay": "1b. Frozen exact MiniLM384 replay",
        "fixed_brain_mapper": "2. Fixed unseen-fMRI mapped384 input",
        "adapted_interface_val266": "3a. Validation-selected interface winner (val266)",
        "adapted_interface_test107": "3b. Interface winner on test107",
    }
    for key, label in labels.items():
        row = stages[key]
        lines.append(
            f"| {label} | {fmt(row['wer'])} | {fmt(row['word_f1'])} | {fmt(row['content_f1'])} | "
            f"{fmt(row['top1'])} | {fmt(row['top5'])} | {fmt(row['mean_rank'])} |"
        )
    lines.extend(
        [
            "",
            "## T5-prefix reference",
            "",
            "| Split | WER | Word F1 | Content F1 |",
            "|---|---:|---:|---:|",
            f"| val266 | {fmt(t5_prefix['val266']['wer'])} | {fmt(t5_prefix['val266']['word_f1'])} | {fmt(t5_prefix['val266']['content_f1'])} |",
            f"| test107 | {fmt(t5_prefix['test107']['wer'])} | {fmt(t5_prefix['test107']['word_f1'])} | {fmt(t5_prefix['test107']['content_f1'])} |",
            "",
            "## Selected interface",
            "",
            f"Winner: `{winner['variant']}` at step `{winner.get('step')}`.",
            "",
            f"Selection rule: {selection['selection_rule']}.",
            "",
            "## Best adapted test examples by word overlap",
            "",
        ]
    )
    for row in ranked["adapted_winner_top_word_f1"][:5]:
        lines.extend(
            [
                f"- Target: {row['target']}",
                f"  Generated: {row['generated']}",
                f"  Word F1: {row['word_f1']:.4f}",
                "",
            ]
        )
    (args.output_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
