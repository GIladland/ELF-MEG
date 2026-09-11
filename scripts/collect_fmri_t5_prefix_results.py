#!/usr/bin/env python
"""Collect and rank leakage-safe fMRI-to-T5 validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: list[dict] = []
    for history_path in sorted(root.glob("*/history.json")):
        payload = json.loads(history_path.read_text())
        best = payload.get("best")
        if not isinstance(best, dict) or not isinstance(best.get("matched"), dict):
            continue
        matched = best["matched"]
        semantic = best.get("semantic_retrieval") or {}
        saved_args = payload.get("args") or {}
        row = {
            "variant": history_path.parent.name,
            "status": payload.get("status"),
            "stage": best.get("stage"),
            "epoch": best.get("epoch"),
            "content_f1": matched.get("content_words_overlap"),
            "content_precision": matched.get("content_words_overlap_precision"),
            "content_recall": matched.get("content_words_overlap_recall"),
            "word_f1": matched.get("words_overlap"),
            "wer": matched.get("word_error_rate"),
            "conditional_content_margin": best.get("conditional_content_margin"),
            "deranged_content_f1_mean": best.get("deranged_content_f1_mean"),
            "deranged_content_f1_max": best.get("deranged_content_f1_max"),
            "beats_all_derangements": (
                matched.get("content_words_overlap", 0.0)
                > best.get("deranged_content_f1_max", float("inf"))
            ),
            "semantic_cosine": semantic.get("matched_cosine"),
            "semantic_top1": semantic.get("top1"),
            "semantic_top5": semantic.get("top5"),
            "semantic_mean_rank": semantic.get("mean_rank"),
            "projector_kind": saved_args.get("projector_kind"),
            "oof_delay_mode": saved_args.get("oof_delay_mode"),
            "content_bias_topk": saved_args.get("content_bias_topk"),
            "content_bias_strength": saved_args.get("content_bias_strength"),
            "content_keyword_context": saved_args.get("content_keyword_context"),
            "t5_lora_rank": saved_args.get("t5_lora_rank"),
            "best_metrics_json": str(history_path.parent / "best_metrics.json"),
            "checkpoint": str(history_path.parent / "best.pt"),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -(row["content_f1"] or 0.0),
            -(row["conditional_content_margin"] or 0.0),
            row["wer"] if row["wer"] is not None else float("inf"),
        )
    )
    output = {
        "scientific_scope": "validation-only collection; test107 remains sealed",
        "locked_fluent_baseline": {
            "content_f1": 0.02311626957,
            "word_f1": 0.0666108,
            "wer": 1.031578947,
        },
        "num_completed_variants": len(rows),
        "rows": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    csv_path = output_path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"output": str(output_path), "csv": str(csv_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
