#!/usr/bin/env python3
"""Report fixed-brain-vector performance as only the ELF corpus scale changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("/data/engs-pnpl/glandau/elf-runs")
DEFAULT_DATA_ROOT = Path("/data/engs-pnpl/glandau/elf-cache/semantic_scaling_20260902")
SPECS = {
    "MRI / MiniLM": {
        "key": "mri_minilm",
        "n": 266,
        "scales": ("x1", "x4", "x8"),
        "x1_rows": 12098,
        "x1_tag": "fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901",
        "bootstrap": "direct_brain_mri_x1_vs_x4_paired_bootstrap_20260903.json",
        "bootstrap_scale": "x4",
    },
    "MEG / ADA": {
        "key": "meg_ada",
        "n": 110,
        "scales": ("x1", "x4", "x16"),
        "x1_rows": 2788,
        "x1_tag": "tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901",
        "bootstrap": "direct_brain_meg_x1_vs_x16_paired_bootstrap_20260903.json",
        "bootstrap_scale": "x16",
    },
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_tag(spec: dict[str, Any], scale: str) -> str:
    if scale == "x1":
        return str(spec["x1_tag"])
    return f"elf_semantic_scaling_{spec['key']}_{scale}_anchorbalanced_compute12k_continuex1_seed7_20260902"


def exact_tag(spec: dict[str, Any], scale: str) -> str:
    if scale == "x1":
        return f"eval_elf_semantic_scaling_{spec['key']}_x1_exact_val{spec['n']}_seed49_20260902"
    return oracle_tag(spec, scale)


def raw_tag(spec: dict[str, Any], scale: str) -> str:
    return f"eval_{oracle_tag(spec, scale)}_rawbrain_seed49"


def corpus_rows(data_root: Path, spec: dict[str, Any], scale: str) -> int | None:
    if scale == "x1":
        return int(spec["x1_rows"])
    path = data_root / spec["key"] / f"{spec['key']}_{scale}_anchorbalanced_auditfirst.summary.json"
    payload = load(path)
    return None if payload is None else int(payload["unique_semantic_text_rows"])


def metrics(run_root: Path, tag: str) -> dict[str, Any]:
    run_dir = run_root / tag
    payload = load(run_dir / "best_metrics.json")
    if payload is None:
        return {"status": "pending", "tag": tag, "run_dir": str(run_dir)}
    quality = payload.get("generation_quality") or {}
    retrieval = payload.get("generation_t5_retrieval") or {}
    semantic = payload.get("semantic_interface") or {}
    bert_payload = load(run_dir / "best_metrics.bertscore.json") or {}
    bert = bert_payload.get("summary") or {}
    return {
        "status": "complete",
        "tag": tag,
        "run_dir": str(run_dir),
        "step": payload.get("step"),
        "word_f1": quality.get("words_overlap"),
        "content_f1": quality.get("content_words_overlap"),
        "wer": quality.get("word_error_rate"),
        "bertscore_f1": bert.get("bertscore_f1"),
        "generated_top1": retrieval.get("top1"),
        "generated_top5": retrieval.get("top5"),
        "generated_mean_rank": retrieval.get("mean_rank"),
        "generated_median_rank": retrieval.get("median_rank"),
        "semantic_top1": semantic.get("top1"),
        "semantic_top5": semantic.get("top5"),
    }


def fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def ratio(value: Any, ceiling: Any) -> float | None:
    if value is None or ceiling in (None, 0):
        return None
    return float(value) / float(ceiling)


def interval(values: Any) -> str:
    if not isinstance(values, list) or len(values) != 2:
        return "—"
    return f"[{float(values[0]):+.4f}, {float(values[1]):+.4f}]"


def main() -> None:
    options = args()
    result: dict[str, Any] = {
        "contract": {
            "only_variable": "ELF semantic-text training corpus scale",
            "brain_vectors": "fixed and untouched within modality",
            "mri_rows_and_candidates": 266,
            "meg_rows_and_candidates": 110,
            "generation_seed": 49,
            "reference_reranking": False,
            "mri_test107_opened": False,
            "meg_test26_opened": False,
        },
        "modalities": {},
    }
    lines = [
        "# Direct brain vectors into scaled ELF diffusion",
        "",
        "Only the ELF semantic-to-text training-corpus scale changes. Brain-predicted vectors, validation rows,",
        "candidate pools, checkpoint seed, and diffusion generation settings are fixed within each modality.",
        "MRI uses the same deterministic untrained delayed-vector input reduction at every scale; MEG passes",
        "the 1536-dimensional predicted ADA vector directly through the checkpoint's flat projection.",
        "",
    ]
    for title, spec in SPECS.items():
        rows = []
        for scale in spec["scales"]:
            row = {
                "scale": scale,
                "unique_pairs": corpus_rows(options.data_root, spec, scale),
                "exact": metrics(options.run_root, exact_tag(spec, scale)),
                "direct_brain": metrics(options.run_root, raw_tag(spec, scale)),
            }
            rows.append(row)
        result["modalities"][spec["key"]] = rows
        bootstrap = load(options.run_root / spec["bootstrap"])
        result["modalities"][spec["key"] + "_paired_bootstrap"] = bootstrap
        lines.extend(
            [
                f"## {title}: fixed {spec['n']}-way evaluation",
                "",
                "| ELF scale | Unique pairs | Exact content F1 | Exact WER | Direct word F1 | Direct content F1 | Direct WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            exact = row["exact"]
            direct = row["direct_brain"]
            lines.append(
                f"| {row['scale']} | {row['unique_pairs'] or '—'} | {fmt(exact.get('content_f1'))} | "
                f"{fmt(exact.get('wer'))} | {fmt(direct.get('word_f1'))} | {fmt(direct.get('content_f1'))} | "
                f"{fmt(direct.get('wer'))} | {fmt(direct.get('bertscore_f1'))} | "
                f"{fmt(direct.get('generated_top1'))} | {fmt(direct.get('generated_top5'))} | "
                f"{fmt(direct.get('generated_mean_rank'), 2)} / {fmt(direct.get('generated_median_rank'), 1)} |"
            )
        lines.extend(
            [
                "",
                "| ELF scale | Content retained vs exact | Δ content F1 vs x1 | Δ WER vs x1 | Δ BERT vs x1 | Δ Top-1 vs x1 | Δ Top-5 vs x1 | Brain semantic Top-1 / Top-5 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        baseline = rows[0]["direct_brain"]
        for row in rows:
            exact = row["exact"]
            direct = row["direct_brain"]
            lines.append(
                f"| {row['scale']} | {fmt(ratio(direct.get('content_f1'), exact.get('content_f1')))} | "
                f"{fmt(delta(direct.get('content_f1'), baseline.get('content_f1')))} | "
                f"{fmt(delta(direct.get('wer'), baseline.get('wer')))} | "
                f"{fmt(delta(direct.get('bertscore_f1'), baseline.get('bertscore_f1')))} | "
                f"{fmt(delta(direct.get('generated_top1'), baseline.get('generated_top1')))} | "
                f"{fmt(delta(direct.get('generated_top5'), baseline.get('generated_top5')))} | "
                f"{fmt(direct.get('semantic_top1'))} / {fmt(direct.get('semantic_top5'))} |"
            )
        lines.append("")
        if bootstrap is not None:
            paired = bootstrap.get("metrics", {})
            content = paired.get("content_f1", {})
            word = paired.get("word_f1", {})
            wer = paired.get("per_row_wer", {})
            bert = paired.get("bertscore_f1", {})
            lines.extend(
                [
                    f"Paired x1 -> {spec['bootstrap_scale']} deltas (20,000 bootstrap samples):",
                    "",
                    "| Metric | Delta | 95% CI | Probability larger ELF is better |",
                    "|---|---:|---:|---:|",
                    f"| Content F1 | {fmt(content.get('candidate_minus_baseline'))} | "
                    f"{interval(content.get('bootstrap_95ci_candidate_minus_baseline'))} | "
                    f"{fmt(content.get('bootstrap_probability_candidate_better'))} |",
                    f"| Word F1 | {fmt(word.get('candidate_minus_baseline'))} | "
                    f"{interval(word.get('bootstrap_95ci_candidate_minus_baseline'))} | "
                    f"{fmt(word.get('bootstrap_probability_candidate_better'))} |",
                    f"| Per-row WER | {fmt(wer.get('candidate_minus_baseline'))} | "
                    f"{interval(wer.get('bootstrap_95ci_candidate_minus_baseline'))} | "
                    f"{fmt(wer.get('bootstrap_probability_candidate_better'))} |",
                    f"| BERTScore F1 | {fmt(bert.get('candidate_minus_baseline'))} | "
                    f"{interval(bert.get('bootstrap_95ci_candidate_minus_baseline'))} | "
                    f"{fmt(bert.get('bootstrap_probability_candidate_better'))} |",
                    "",
                ]
            )

    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    options.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    options.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    options.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(options.output_markdown)


if __name__ == "__main__":
    main()
