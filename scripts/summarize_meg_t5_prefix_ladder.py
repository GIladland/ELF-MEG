#!/usr/bin/env python3
"""Summarize the fixed-val110 MEG/ADA T5-prefix comparison ladder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARM_NAMES = [
    "raw_frozen_oracle",
    "adapter_h1024",
    "adapter_h2048_content_pair",
    "adapter_h1024_lora4",
    "adapter_h1024_lora8",
    "joint_projector_adapter",
    "joint_projector_adapter_lora4_content_pair",
    "direct_projector",
    "lora4_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def load_row(name: str, metrics_path: Path, checkpoint_path: Path) -> dict | None:
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = payload["best"]
    matched = record["matched"]
    retrieval_path = metrics_path.with_suffix(".t5retrieval.json")
    retrieval = None
    if retrieval_path.is_file():
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8")).get(
            "generation_t5_retrieval"
        )
    bert_rescaled_path = metrics_path.with_suffix(".bertscore_rescaled.json")
    bert_raw_path = metrics_path.with_suffix(".bertscore_raw.json")
    bert_rescaled = None
    bert_raw = None
    if bert_rescaled_path.is_file():
        bert_rescaled = json.loads(
            bert_rescaled_path.read_text(encoding="utf-8")
        )["summary"]["bertscore_f1"]
    if bert_raw_path.is_file():
        bert_raw = json.loads(bert_raw_path.read_text(encoding="utf-8"))["summary"][
            "bertscore_f1"
        ]
    word_f1 = float(matched["words_overlap"])
    content_f1 = float(matched["content_words_overlap"])
    return {
        "name": name,
        "metrics_path": str(metrics_path),
        "checkpoint": str(checkpoint_path),
        "stage": record["stage"],
        "epoch": int(record["epoch"]),
        "word_f1": word_f1,
        "content_f1": content_f1,
        "word_content_mean": 0.5 * (word_f1 + content_f1),
        "wer": float(matched["word_error_rate"]),
        "bertscore_rescaled_f1": bert_rescaled,
        "bertscore_raw_f1": bert_raw,
        "generated_retrieval": retrieval,
        "semantic_retrieval": record.get("semantic_retrieval"),
        "deranged_content_mean": float(record["deranged_content_f1_mean"]),
        "deranged_content_max": float(record["deranged_content_f1_max"]),
        "conditional_content_margin": float(record["conditional_content_margin"]),
        "generated": payload["generated"],
        "targets": payload["targets"],
    }


def value(row: dict, key: str, digits: int = 5) -> str:
    item = row.get(key)
    return "—" if item is None else f"{float(item):.{digits}f}"


def main() -> None:
    args = parse_args()
    rows = []
    oracle = load_row(
        "exact_ada_oracle",
        args.root / "selected_oracle" / "oracle_best_metrics.json",
        args.root / "selected_oracle" / "oracle_best.pt",
    )
    if oracle is not None:
        rows.append(oracle)
    for name in ARM_NAMES:
        run_dir = args.root / name
        row = load_row(name, run_dir / "best_metrics.json", run_dir / "best.pt")
        if row is not None:
            rows.append(row)
    brain_rows = [row for row in rows if row["name"] != "exact_ada_oracle"]
    eligible = [row for row in brain_rows if row["conditional_content_margin"] > 0]
    pool = eligible or brain_rows
    winner = max(pool, key=lambda row: row["word_content_mean"]) if pool else None
    compact_rows = [
        {key: val for key, val in row.items() if key not in {"generated", "targets"}}
        for row in rows
    ]
    result = {
        "contract": {
            "modality": "MEG qc4wyals to ADA002 to direct T5-prefix text",
            "split": "train2652 / val110; protected test26 sealed",
            "generation": "one deterministic generation per row; ten-word cap",
            "retrieval_pool": "the same 110 validation targets for every arm",
            "train_prediction_note": (
                "qc4wyals train predictions are in-sample; val110 predictions are held out. "
                "This is evaluation-leakage safe but not an OOF-train noise match."
            ),
        },
        "selection": (
            "Highest 0.5*(word F1 + content F1) among positive matched-minus-deranged "
            "content-margin brain arms."
        ),
        "winner": None if winner is None else winner["name"],
        "winner_metrics_path": None if winner is None else winner["metrics_path"],
        "winner_checkpoint": None if winner is None else winner["checkpoint"],
        "rows": compact_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MEG ADA → T5-prefix baseline ladder",
        "",
        "All brain rows use qc4wyals, the fixed val110 split, one deterministic generation, "
        "and the same 110-way retrieval pool. Protected test26 remains sealed.",
        "",
        "| Arm | Word F1 | Content F1 | Mean(W,C) | WER | BERT raw / rescaled | Gen T1 / T5 | Sem T1 / T5 | Content margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gen = row["generated_retrieval"] or {}
        sem = row["semantic_retrieval"] or {}
        lines.append(
            f"| {row['name']} | {value(row, 'word_f1')} | {value(row, 'content_f1')} | "
            f"{value(row, 'word_content_mean')} | {value(row, 'wer')} | "
            f"{value(row, 'bertscore_raw_f1')} / {value(row, 'bertscore_rescaled_f1')} | "
            f"{value(gen, 'top1')} / {value(gen, 'top5')} | "
            f"{value(sem, 'top1')} / {value(sem, 'top5')} | "
            f"{value(row, 'conditional_content_margin')} |"
        )
    lines.extend(
        [
            "",
            f"Selected brain-conditioned winner: `{result['winner']}`.",
            "",
            "The exact-ADA row is a decoder ceiling on unseen val110 text. The raw row "
            "isolates the exact-to-qc4wyals interface drop; later rows test whether adapter, "
            "LoRA, or joint prefix tuning recovers it.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if winner is not None:
        csv_path = args.output_md.with_name("meg_t5_prefix_winner_val110.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "target", "generated"])
            writer.writeheader()
            for index, (target, generated) in enumerate(
                zip(winner["targets"], winner["generated"])
            ):
                writer.writerow(
                    {"index": index, "target": target, "generated": generated}
                )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
