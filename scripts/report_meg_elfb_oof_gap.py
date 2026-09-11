#!/usr/bin/env python3
"""Report exact-ADA -> raw MEG -> nested-OOF-adapted ELF-B gap closure."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-summary", required=True, type=Path)
    parser.add_argument("--system", action="append", required=True, help="NAME=RUN_GLOB")
    parser.add_argument("--oracle-name", default="exact_ada")
    parser.add_argument("--raw-name", default="raw_meg")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def one_run(path: Path) -> dict[str, float]:
    payload = json.loads((path / "best_metrics.json").read_text())
    quality = payload["generation_quality"]
    retrieval = payload.get("generation_t5_retrieval", {})
    return {
        "word_f1": float(quality["words_overlap"]),
        "content_f1": float(quality["content_words_overlap"]),
        "word_plus_content": float(quality["words_overlap"] + quality["content_words_overlap"]),
        "wer": float(quality["word_error_rate"]),
        "generated_top1": float(retrieval["top1"]),
        "generated_top5": float(retrieval["top5"]),
        "generated_mean_rank": float(retrieval["mean_rank"]),
        "generated_median_rank": float(retrieval["median_rank"]),
    }


def aggregate(pattern: str) -> dict[str, object]:
    runs = sorted(Path(value) for value in glob.glob(pattern) if (Path(value) / "best_metrics.json").is_file())
    if not runs:
        raise FileNotFoundError(f"No completed runs match {pattern}")
    rows = [one_run(path) for path in runs]
    names = rows[0].keys()
    summary = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {"pattern": pattern, "num_seeds": len(runs), "runs": [str(path) for path in runs], "metrics": summary}


def closure(raw: float, oracle: float, candidate: float, lower_is_better: bool) -> float | None:
    numerator = (raw - candidate) if lower_is_better else (candidate - raw)
    denominator = (raw - oracle) if lower_is_better else (oracle - raw)
    if abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    systems: dict[str, dict[str, object]] = {}
    for specification in args.system:
        name, pattern = specification.split("=", 1)
        systems[name] = aggregate(pattern)
    if args.oracle_name not in systems or args.raw_name not in systems:
        raise KeyError("Oracle and raw system names must both be supplied")
    oracle = systems[args.oracle_name]["metrics"]
    raw = systems[args.raw_name]["metrics"]
    lower = {"wer", "generated_mean_rank", "generated_median_rank"}
    closure_rows: dict[str, dict[str, float | None]] = {}
    for name, system in systems.items():
        if name in (args.oracle_name, args.raw_name):
            continue
        closure_rows[name] = {
            metric: closure(
                float(raw[metric]["mean"]),
                float(oracle[metric]["mean"]),
                float(system["metrics"][metric]["mean"]),
                metric in lower,
            )
            for metric in oracle
        }
    oof = json.loads(args.oof_summary.read_text())
    result = {
        "schema": "meg_elfb_nested_oof_gap_closure_v1",
        "evaluation_contract": "same val110 rows, same 110 candidates, five generation seeds per system",
        "oof_audit": oof,
        "systems": systems,
        "gap_closure_fraction": closure_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    metrics = ["word_f1", "content_f1", "word_plus_content", "wer", "generated_top1", "generated_top5", "generated_mean_rank"]
    lines = [
        "# MEG → ELF-B nested-OOF gap closure",
        "",
        "All systems use the same 110 validation sentences, 110 retrieval candidates, decoding settings, and five generation seeds.",
        "",
        "| System | Word F1 | Content F1 | Sum | WER ↓ | Generated T1 | Generated T5 | Mean rank ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, system in systems.items():
        values = system["metrics"]
        lines.append(
            f"| `{name}` | {values['word_f1']['mean']:.4f} | {values['content_f1']['mean']:.4f} | "
            f"{values['word_plus_content']['mean']:.4f} | {values['wer']['mean']:.4f} | "
            f"{values['generated_top1']['mean']:.4f} | {values['generated_top5']['mean']:.4f} | "
            f"{values['generated_mean_rank']['mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Fraction of the raw-MEG → exact-ADA gap closed",
        "",
        "`0` means no gain over raw MEG; `1` reaches the exact-ADA oracle. Negative values regress.",
        "",
        "| Adapter | Word F1 | Content F1 | Sum | WER | Generated T1 | Generated T5 | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, values in closure_rows.items():
        lines.append("| `" + name + "` | " + " | ".join(fmt(values[metric]) for metric in metrics) + " |")
    oof_retrieval = oof.get("oof_train_retrieval") or oof["oof_train_retrieval_2652_way"]
    val_retrieval = oof["deployment_val_retrieval_110_way"]
    lines.extend([
        "",
        "## OOF audit",
        "",
        f"- OOF coverage: no duplicates for {oof['train_rows']} rows; partial={oof.get('partial_oof', False)}; outer predictions were not used for checkpoint selection.",
        f"- OOF train retrieval ({oof['train_rows']:,}-way): Top-1 `{oof_retrieval['top1']:.4f}`, Top-5 `{oof_retrieval['top5']:.4f}`, cosine `{oof_retrieval['matched_cosine_mean']:.4f}`.",
        f"- Deployment validation retrieval (110-way): Top-1 `{val_retrieval['top1']:.4f}`, Top-5 `{val_retrieval['top5']:.4f}`, cosine `{val_retrieval['matched_cosine_mean']:.4f}`.",
        "- The candidate counts differ in the diagnostic above; the generation table itself is strictly 110-way for every system.",
        "",
    ])
    args.output_md.write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
