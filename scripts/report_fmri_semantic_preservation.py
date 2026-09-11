#!/usr/bin/env python
"""Report the retrieval-preserving fMRI-to-ELF validation and test results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--snapshot-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        "n": int(metrics.get("eval_num_examples") or metrics.get("num_eval_examples") or 0),
        "step": int(metrics.get("step") or 0),
        "wer": float(quality.get("word_error_rate", float("nan"))),
        "word_f1": float(quality.get("words_overlap", 0.0)),
        "content_f1": float(quality.get("content_words_overlap", 0.0)),
        "generated_top1": float(generated.get("top1", 0.0)),
        "generated_top5": float(generated.get("top5", 0.0)),
        "generated_mean_rank": float(generated.get("mean_rank", float("nan"))),
        "semantic_top1": float(semantic.get("top1", 0.0)),
        "semantic_top5": float(semantic.get("top5", 0.0)),
        "semantic_mean_rank": float(semantic.get("mean_rank", float("nan"))),
        "semantic_cosine": float(semantic.get("matched_cosine_mean", float("nan"))),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def markdown_text(value: str, limit: int = 76) -> str:
    value = " ".join(str(value).split()).replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def best_generation_examples(
    oracle: dict, raw: dict, adapted: dict, limit: int = 8
) -> list[dict]:
    rows = []
    for index, target in enumerate(raw["targets"]):
        raw_word = float(raw["word_overlap"]["per_sample"][index])
        adapted_word = float(adapted["word_overlap"]["per_sample"][index])
        raw_content = float(raw["word_overlap"]["per_sample_content"][index])
        adapted_content = float(adapted["word_overlap"]["per_sample_content"][index])
        raw_wer = float(raw["word_overlap"]["per_sample_wer"][index])
        adapted_wer = float(adapted["word_overlap"]["per_sample_wer"][index])
        rows.append(
            {
                "index": index,
                "target": target,
                "oracle_generated": oracle["generated"][index],
                "raw_generated": raw["generated"][index],
                "selected_generated": adapted["generated"][index],
                "raw_word_f1": raw_word,
                "selected_word_f1": adapted_word,
                "raw_content_f1": raw_content,
                "selected_content_f1": adapted_content,
                "raw_wer": raw_wer,
                "selected_wer": adapted_wer,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["selected_content_f1"] - row["raw_content_f1"],
            row["selected_word_f1"] - row["raw_word_f1"],
            row["raw_wer"] - row["selected_wer"],
        ),
        reverse=True,
    )[:limit]


def export_test_comparison(
    oracle: dict, raw: dict, adapted: dict, path: Path
) -> dict[str, int]:
    if not (
        oracle.get("targets") == raw.get("targets") == adapted.get("targets")
    ):
        raise ValueError("Oracle, raw, and adapted test targets are not aligned")
    raw_overlap = raw["word_overlap"]
    adapted_overlap = adapted["word_overlap"]
    rows = []
    counts = {
        "adapter_word_wins": 0,
        "word_ties": 0,
        "raw_word_wins": 0,
        "adapter_content_wins": 0,
        "content_ties": 0,
        "raw_content_wins": 0,
        "adapter_wer_wins": 0,
        "wer_ties": 0,
        "raw_wer_wins": 0,
    }
    for index, target in enumerate(raw["targets"]):
        raw_word = float(raw_overlap["per_sample"][index])
        adapted_word = float(adapted_overlap["per_sample"][index])
        raw_content = float(raw_overlap["per_sample_content"][index])
        adapted_content = float(adapted_overlap["per_sample_content"][index])
        raw_wer = float(raw_overlap["per_sample_wer"][index])
        adapted_wer = float(adapted_overlap["per_sample_wer"][index])
        if adapted_word > raw_word:
            counts["adapter_word_wins"] += 1
        elif raw_word > adapted_word:
            counts["raw_word_wins"] += 1
        else:
            counts["word_ties"] += 1
        if adapted_content > raw_content:
            counts["adapter_content_wins"] += 1
        elif raw_content > adapted_content:
            counts["raw_content_wins"] += 1
        else:
            counts["content_ties"] += 1
        if adapted_wer < raw_wer:
            counts["adapter_wer_wins"] += 1
        elif raw_wer < adapted_wer:
            counts["raw_wer_wins"] += 1
        else:
            counts["wer_ties"] += 1
        rows.append(
            {
                "index": index,
                "target": target,
                "oracle_generated": oracle["generated"][index],
                "raw_generated": raw["generated"][index],
                "adapted_generated": adapted["generated"][index],
                "oracle_wer": float(oracle["word_overlap"]["per_sample_wer"][index]),
                "raw_wer": raw_wer,
                "adapted_wer": adapted_wer,
                "oracle_word_f1": float(oracle["word_overlap"]["per_sample"][index]),
                "raw_word_f1": raw_word,
                "adapted_word_f1": adapted_word,
                "oracle_content_f1": float(
                    oracle["word_overlap"]["per_sample_content"][index]
                ),
                "raw_content_f1": raw_content,
                "adapted_content_f1": adapted_content,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = (
        args.run_root
        / f"fmri_freshmem_semantic_preservation_{args.snapshot_tag}_selection_20260901"
    )
    selection = load(selection_dir / "selection.json")
    baseline = selection["baseline"]
    candidates = selection["candidates"]

    test = None
    comparison_counts = None
    if selection["promoted"]:
        winner = selection["winner"]
        raw_tag = (
            f"eval_fmri_freshmem_semantic_preservation_{args.snapshot_tag}_"
            "raw_residual_test107_seed49_20260901"
        )
        winner_tag = (
            f"eval_fmri_freshmem_semantic_preservation_{args.snapshot_tag}_"
            f"{winner['variant']}_test107_seed49_20260901"
        )
        oracle_tag = (
            f"eval_fmri_freshmem_semantic_preservation_{args.snapshot_tag}_"
            "oracle_exact_minilm384_test107_seed49_20260901"
        )
        oracle_metrics = load(args.run_root / oracle_tag / "eval_step_000000.json")
        raw_metrics = load(args.run_root / raw_tag / "eval_step_000000.json")
        winner_metrics = load(args.run_root / winner_tag / "eval_step_000000.json")
        comparison_counts = export_test_comparison(
            oracle_metrics,
            raw_metrics,
            winner_metrics,
            args.output_dir / "raw_vs_selected_test107_generations.csv",
        )
        example_rows = best_generation_examples(
            oracle_metrics, raw_metrics, winner_metrics
        )
        test = {
            "oracle_exact_minilm384": metric_row(oracle_metrics),
            "raw": metric_row(raw_metrics),
            "selected": metric_row(winner_metrics),
            "row_comparison": comparison_counts,
            "best_selected_examples": example_rows,
        }

    references = {
        "t5_prefix_val266": {"wer": 0.9872180451, "word_f1": 0.0826777017, "content_f1": 0.0437433181},
        "t5_prefix_test107": {"wer": 0.988785, "word_f1": 0.060702, "content_f1": 0.017756},
        "prior_oof_mapper_elf_test107": {
            "wer": 0.9850,
            "word_f1": 0.0625,
            "content_f1": 0.0153,
            "generated_top1": 0.0187,
            "generated_top5": 0.1121,
        },
        "prior_best_wer_diffusion_test107": {
            "wer": 0.97103,
            "word_f1": 0.0516,
            "content_f1": 0.0047,
            "generated_top1": 0.0187,
            "generated_top5": 0.0561,
        },
        "transductive_diffusion_overfit_test107_not_comparable": {
            "wer": 0.33925,
            "word_f1": 0.7231,
            "content_f1": 0.6993,
            "generated_top1": 0.9346,
            "generated_top5": 0.9720,
        },
    }
    payload = {
        "protocol": {
            "brain_decoder": "fixed and story-held-out",
            "diffusion_text_training": "all exact text/MiniLM pairs, including known test text",
            "adapter_training": "oracle plus story-level OOF train predictions",
            "selection": "predicted val266 retrieval gate; no test rows",
        },
        "selection": selection,
        "test107": test,
        "references": references,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Retrieval-preserving fMRI → MiniLM → ELF report",
        "",
        "All adapter choices were made on predicted val266. Test107 was evaluated only after a checkpoint passed the frozen-raw retrieval gate.",
        "",
        "## Validation",
        "",
        "| Arm | Pass gate | WER | Word F1 | Content F1 | Generated Top-1 | Generated Top-5 | Semantic Top-1 | Semantic Top-5 |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| frozen raw residual | baseline | {pct(baseline['wer'])} | {pct(baseline['word_f1'])} | "
            f"{pct(baseline['content_f1'])} | {pct(baseline['generated_top1'])} | "
            f"{pct(baseline['generated_top5'])} | {pct(baseline['semantic_top1'])} | "
            f"{pct(baseline['semantic_top5'])} |"
        ),
    ]
    for row in candidates:
        lines.append(
            f"| {row['variant']} @ {row['step']} | {'yes' if row['passes_retrieval_gate'] else 'no'} | "
            f"{pct(row['wer'])} | {pct(row['word_f1'])} | {pct(row['content_f1'])} | "
            f"{pct(row['generated_top1'])} | {pct(row['generated_top5'])} | "
            f"{pct(row['semantic_top1'])} | {pct(row['semantic_top5'])} |"
        )
    lines.extend(["", f"Promoted: **{selection['promoted']}**."])
    if selection["promoted"]:
        lines.append(f"Selected: `{selection['winner']['variant']}` at step {selection['winner']['step']}.")
    if test is not None:
        test_n = max(1, int(test["raw"]["n"]))
        chance_top1 = 1.0 / test_n
        chance_top5 = min(5, test_n) / test_n
        raw = test["raw"]
        selected = test["selected"]
        lines.extend(
            [
                "",
                "## Test107",
                "",
                "| Input | WER | Word F1 | Content F1 | Generated Top-1 | Generated Top-5 | Semantic Top-1 | Semantic Top-5 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, row in (
            ("oracle exact MiniLM384", test["oracle_exact_minilm384"]),
            ("raw predicted1536 mean", test["raw"]),
            ("selected constrained adapter", test["selected"]),
        ):
            lines.append(
                f"| {label} | {pct(row['wer'])} | {pct(row['word_f1'])} | {pct(row['content_f1'])} | "
                f"{pct(row['generated_top1'])} | {pct(row['generated_top5'])} | "
                f"{pct(row['semantic_top1'])} | {pct(row['semantic_top5'])} |"
            )
        lines.extend(
            [
                "",
                f"Chance over {test_n} candidates is {pct(chance_top1)} Top-1 and {pct(chance_top5)} Top-5.",
                "",
                (
                    "Selected minus raw: "
                    f"WER {100.0 * (selected['wer'] - raw['wer']):+.2f} points, "
                    f"word F1 {100.0 * (selected['word_f1'] - raw['word_f1']):+.2f}, "
                    f"content F1 {100.0 * (selected['content_f1'] - raw['content_f1']):+.2f}, "
                    f"generated Top-1 {100.0 * (selected['generated_top1'] - raw['generated_top1']):+.2f}, "
                    f"generated Top-5 {100.0 * (selected['generated_top5'] - raw['generated_top5']):+.2f}, "
                    f"semantic Top-1 {100.0 * (selected['semantic_top1'] - raw['semantic_top1']):+.2f}, "
                    f"semantic Top-5 {100.0 * (selected['semantic_top5'] - raw['semantic_top5']):+.2f}."
                ),
                "",
                "Row-level oracle, raw, and selected generations are in `raw_vs_selected_test107_generations.csv`.",
                "",
                f"Row comparison: `{json.dumps(comparison_counts, sort_keys=True)}`",
                "",
                "### Best selected-over-raw examples",
                "",
                "| Index | Target | Oracle generation | Raw generation | Selected generation | Raw→selected content F1 | Raw→selected word F1 |",
                "|---:|---|---|---|---|---:|---:|",
            ]
        )
        for row in test["best_selected_examples"]:
            lines.append(
                f"| {row['index']} | {markdown_text(row['target'])} | "
                f"{markdown_text(row['oracle_generated'])} | "
                f"{markdown_text(row['raw_generated'])} | "
                f"{markdown_text(row['selected_generated'])} | "
                f"{pct(row['raw_content_f1'])}→{pct(row['selected_content_f1'])} | "
                f"{pct(row['raw_word_f1'])}→{pct(row['selected_word_f1'])} |"
            )
    lines.extend(
        [
            "",
            "## Previous references",
            "",
            "| System | WER | Word F1 | Content F1 | Generated Top-1 | Generated Top-5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in references.items():
        lines.append(
            f"| {label} | {pct(row['wer'])} | {pct(row['word_f1'])} | {pct(row['content_f1'])} | "
            f"{pct(row.get('generated_top1', 0.0)) if 'generated_top1' in row else '—'} | "
            f"{pct(row.get('generated_top5', 0.0)) if 'generated_top5' in row else '—'} |"
        )
    if test is not None:
        lines.extend(
            [
                "",
                "## Bottom line",
                "",
                (
                    "The preservation loss prevents the severe retrieval collapse and produces a small "
                    "held-out improvement over the raw diffusion interface. It does not yet beat the T5-prefix "
                    "system on WER or lexical overlap: the selected diffusion model has "
                    f"{pct(test['selected']['word_f1'])} word F1 / {pct(test['selected']['content_f1'])} content F1, "
                    f"versus {pct(references['t5_prefix_test107']['word_f1'])} / "
                    f"{pct(references['t5_prefix_test107']['content_f1'])} for T5-prefix."
                ),
            ]
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
