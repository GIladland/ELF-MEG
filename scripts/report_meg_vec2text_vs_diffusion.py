#!/usr/bin/env python
"""Create the matched MEG Vec2Text, T5-prefix, and ELF comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    ("content_f1", "Content F1", "higher"),
    ("word_f1", "Word F1", "higher"),
    ("wer", "WER", "lower"),
    ("bertscore_raw_f1", "Raw BERTScore F1", "higher"),
    ("bleu1", "BLEU-1", "higher"),
    ("rouge1_f1", "ROUGE-1 F1", "higher"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-json", type=Path, required=True)
    parser.add_argument("--content-diffusion-json", type=Path)
    parser.add_argument("--retrieval-diffusion-json", type=Path)
    parser.add_argument("--t5-json", type=Path)
    parser.add_argument("--t5-exact-json", type=Path)
    parser.add_argument("--predicted-capped-json", type=Path, required=True)
    parser.add_argument("--predicted-native-json", type=Path, required=True)
    parser.add_argument("--exact-capped-json", type=Path, required=True)
    parser.add_argument("--exact-native-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(label: str, path: Path) -> tuple[list[dict], dict]:
    payload = load(path)
    rows = []
    for key, display, direction in METRICS:
        result = payload["permutation_tests"][key]
        observed = float(result["observed"])
        null_mean = float(result["null_mean"])
        raw_difference = observed - null_mean
        favorable_difference = raw_difference if direction == "higher" else -raw_difference
        rows.append(
            {
                "system": label,
                "metric": display,
                "direction": direction,
                "actual_mean": observed,
                "permutation_mean": null_mean,
                "actual_minus_permutation": raw_difference,
                "favorable_difference": favorable_difference,
                "permutation_ci95_low": float(result["null_ci95"][0]),
                "permutation_ci95_high": float(result["null_ci95"][1]),
                "one_sided_p": float(result["one_sided_p"]),
            }
        )
    retrieval = payload.get("retrieval_chance_tests", {})
    summary = {
        "top1": retrieval.get("generated_top1", {}).get("rate"),
        "top5": retrieval.get("generated_top5", {}).get("rate"),
    }
    return rows, summary


def fmt(value: float | None, places: int = 5) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def main() -> None:
    args = parse_args()
    systems = [
        ("ELF diffusion: lexical-composite winner", args.diffusion_json),
    ]
    if args.content_diffusion_json:
        systems.append(("ELF diffusion: content specialist", args.content_diffusion_json))
    if args.retrieval_diffusion_json:
        systems.append(("ELF diffusion: retrieval specialist", args.retrieval_diffusion_json))
    if args.t5_json:
        systems.append(("T5 prefix: direct-projector winner", args.t5_json))
    systems.extend(
        (
            ("Vec2Text: predicted qc4wyals ADA, 10-word cap", args.predicted_capped_json),
            ("Vec2Text: predicted qc4wyals ADA, native length", args.predicted_native_json),
        )
    )
    if args.t5_exact_json:
        systems.append(("T5 prefix: exact-ADA oracle", args.t5_exact_json))
    systems.extend(
        (
            ("Vec2Text: exact ADA ceiling, 10-word cap", args.exact_capped_json),
            ("Vec2Text: exact ADA ceiling, native length", args.exact_native_json),
        )
    )
    all_rows = []
    retrieval_by_system = {}
    for label, path in systems:
        rows, retrieval = flatten(label, path)
        all_rows.extend(rows)
        retrieval_by_system[label] = retrieval

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    lookup = {(row["system"], row["metric"]): row for row in all_rows}
    lines = [
        "# MEG ADA decoder comparison: Vec2Text, T5 prefix, and ELF diffusion",
        "",
        "## Contract",
        "",
        "- Identical 110 Apple validation rows and 110 retrieval candidates.",
        "- MEG semantic source: frozen `qc4wyals` 1,536-D ADA predictions.",
        "- Diffusion comparators: the lexical-composite winner plus optional content and retrieval specialists.",
        "- T5 comparator: the optional direct-projector winner from the original 2,652-row prefix ladder.",
        "- Rows labelled exact-ADA are semantic-inversion ceilings, not brain-conditioned results.",
        "- Vec2Text: pretrained ADA-002 hypothesizer/corrector, 20 recursive steps, beam width 4.",
        "- Primary comparison: ten-word output cap; native Vec2Text length is retained as a diagnostic.",
        "- Null: 100,000 one-to-one row permutations; lower WER and higher other scores are favorable.",
        "- Protected test26 remains unopened.",
        "",
        "## Actual performance",
        "",
        "| System | Content F1 | Word F1 | Sum | WER ↓ | Raw BERTScore | BLEU-1 | ROUGE-1 | Top-1 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _ in systems:
        get = lambda metric: lookup[(label, metric)]["actual_mean"]
        retrieval = retrieval_by_system[label]
        overlap_sum = get("Content F1") + get("Word F1")
        lines.append(
            f"| {label} | {fmt(get('Content F1'))} | {fmt(get('Word F1'))} | "
            f"{fmt(overlap_sum)} | {fmt(get('WER'))} | {fmt(get('Raw BERTScore F1'))} | "
            f"{fmt(get('BLEU-1'))} | {fmt(get('ROUGE-1 F1'))} | "
            f"{fmt(retrieval['top1'])} | {fmt(retrieval['top5'])} |"
        )

    lines.extend(
        [
            "",
            "## Actual mean versus permutation mean",
            "",
            "| System | Metric | Actual | Permutation mean | Favorable difference | One-sided p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in all_rows:
        lines.append(
            f"| {row['system']} | {row['metric']} | {fmt(row['actual_mean'])} | "
            f"{fmt(row['permutation_mean'])} | {fmt(row['favorable_difference'])} | "
            f"{row['one_sided_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "`Favorable difference` is actual minus null for overlap/BERTScore/BLEU/ROUGE, "
            "and null minus actual for WER.",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_markdown)
    print(args.output_csv)


if __name__ == "__main__":
    main()
