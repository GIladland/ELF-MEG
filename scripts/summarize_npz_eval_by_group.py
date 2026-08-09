#!/usr/bin/env python
"""Summarize saved NPZ-semantic ELF eval JSONs by validation corpus group."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Combined train+val semantic NPZ used by the run.")
    parser.add_argument("--eval-json", action="append", required=True, help="Saved eval_step_*.json file.")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--split-key", default="split")
    return parser.parse_args()


def words(text: str) -> set[str]:
    return set(WORD_RE.findall(str(text).lower()))


def overlap(target: str, generated: str) -> tuple[float, float, float, float]:
    target_words = words(target)
    generated_words = words(generated)
    if not target_words and not generated_words:
        return 1.0, 1.0, 1.0, 1.0
    intersection = len(target_words & generated_words)
    recall = intersection / len(target_words) if target_words else 0.0
    precision = intersection / len(generated_words) if generated_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(target_words | generated_words)
    jaccard = intersection / union if union else 0.0
    return recall, precision, f1, jaccard


def corpus_label(row: Any) -> str:
    if not isinstance(row, dict):
        return "unknown"
    label = str(
        row.get("corpus")
        or row.get("task")
        or row.get("dataset")
        or row.get("source")
        or row.get("source_name")
        or "unknown"
    )
    if label.startswith("Sherlock"):
        return "Sherlock"
    return label


def summarize(vals: list[tuple[float, float, float, float]]) -> dict[str, float | int]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(vals)),
        "recall": float(arr[:, 0].mean()) if len(vals) else 0.0,
        "precision": float(arr[:, 1].mean()) if len(vals) else 0.0,
        "f1": float(arr[:, 2].mean()) if len(vals) else 0.0,
        "jaccard": float(arr[:, 3].mean()) if len(vals) else 0.0,
    }


def main() -> None:
    args = parse_args()
    npz = np.load(args.npz, allow_pickle=True)

    for eval_path_str in args.eval_json:
        eval_path = Path(eval_path_str)
        with eval_path.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
        targets = eval_data["targets"]
        generated = eval_data["generated"]
        n = len(targets)
        rows = list(npz[args.rows_key][-n:])
        splits = npz[args.split_key][-n:] if args.split_key in npz.files else np.asarray(["val"] * n)
        if len(rows) != n:
            raise ValueError(f"Eval examples ({n}) and NPZ rows ({len(rows)}) are misaligned.")

        groups: dict[str, list[tuple[float, float, float, float]]] = {}
        all_vals = []
        for target, gen, row, split in zip(targets, generated, rows, splits):
            if str(split) != "val":
                raise ValueError("Expected eval examples to align with trailing validation rows.")
            vals = overlap(target, gen)
            all_vals.append(vals)
            groups.setdefault(corpus_label(row), []).append(vals)

        payload = {
            "eval_json": str(eval_path),
            "step": eval_data.get("step"),
            "epoch": eval_data.get("epoch"),
            "overall": summarize(all_vals),
            "logged_words_overlap": eval_data.get("generation_quality", {}).get("words_overlap"),
            "logged_t5_top1": eval_data.get("generation_t5_retrieval", {}).get("top1"),
            "groups": {name: summarize(vals) for name, vals in sorted(groups.items())},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
