#!/usr/bin/env python3
"""Collect the fixed exact/raw/adapted ELF scaling ladders into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/data/engs-pnpl/glandau/elf-runs")
DATA_ROOT = Path("/data/engs-pnpl/glandau/elf-cache/semantic_scaling_20260902")
SPECS = {
    "mri_minilm": {
        "audit_n": 266,
        "scales": ("x1", "x4", "x8"),
        "x1": "fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901",
    },
    "meg_ada": {
        "audit_n": 110,
        "scales": ("x1", "x4", "x16"),
        "x1": "tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_tag(modality: str, scale: str) -> str:
    if scale == "x1":
        return str(SPECS[modality]["x1"])
    return f"elf_semantic_scaling_{modality}_{scale}_anchorbalanced_compute12k_continuex1_seed7_20260902"


def run_tag(modality: str, scale: str, rung: str) -> str:
    audit_n = int(SPECS[modality]["audit_n"])
    if rung == "exact":
        if scale == "x1":
            return f"eval_elf_semantic_scaling_{modality}_x1_exact_val{audit_n}_seed49_20260902"
        return oracle_tag(modality, scale)
    if rung == "raw":
        return f"eval_{oracle_tag(modality, scale)}_rawbrain_seed49"
    if rung == "adapter":
        return f"elf_semantic_scaling_{modality}_{scale}_trained_adapter_val{audit_n}_seed49_20260902"
    raise ValueError(rung)


def extract(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "best_metrics.json"
    metrics = load_json(metrics_path)
    if metrics is None:
        return {"status": "missing", "run_dir": str(run_dir), "metrics": str(metrics_path)}
    quality = metrics.get("generation_quality") or (metrics.get("word_overlap") or {}).get("summary") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    semantic = metrics.get("semantic_interface") or {}
    bert = load_json(run_dir / "best_metrics.bertscore.json") or {}
    bert_summary = bert.get("summary") or {}
    return {
        "status": "complete",
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None,
        "step": metrics.get("step"),
        "epoch": metrics.get("epoch"),
        "word_f1": quality.get("words_overlap"),
        "content_f1": quality.get("content_words_overlap"),
        "wer": quality.get("word_error_rate"),
        "bertscore_f1": bert_summary.get("bertscore_f1"),
        "generated_top1": retrieval.get("top1"),
        "generated_top5": retrieval.get("top5"),
        "generated_mean_rank": retrieval.get("mean_rank"),
        "generated_median_rank": retrieval.get("median_rank"),
        "semantic_top1": semantic.get("top1"),
        "semantic_top5": semantic.get("top5"),
        "semantic_mean_rank": semantic.get("mean_rank"),
        "semantic_median_rank": semantic.get("median_rank"),
        "targets": metrics.get("targets") or [],
        "generated": metrics.get("generated") or [],
        "per_sample_content_f1": (metrics.get("word_overlap") or {}).get("per_sample_content") or [],
    }


def training_curve(run_dir: Path) -> dict[str, Any] | None:
    points = []
    for path in sorted(run_dir.glob("eval_step_*.json")):
        metrics = load_json(path)
        if not metrics:
            continue
        quality = metrics.get("generation_quality") or {}
        wer = quality.get("word_error_rate")
        if wer is None:
            continue
        points.append(
            {
                "step": int(metrics.get("step") or 0),
                "wer": float(wer),
                "word_f1": quality.get("words_overlap"),
                "content_f1": quality.get("content_words_overlap"),
            }
        )
    if not points:
        return None
    best = min(points, key=lambda point: point["wer"])
    tail = points[-3:]
    tail_gain = tail[0]["wer"] - tail[-1]["wer"] if len(tail) >= 2 else 0.0
    continue_training = best["step"] >= tail[0]["step"] and tail_gain >= 0.003
    return {
        "num_evaluations": len(points),
        "first": points[0],
        "best": best,
        "last": points[-1],
        "last_three_wer_gain": tail_gain,
        "decision": "continue" if continue_training else "plateau",
        "points": points,
    }


def corpus_summary(data_root: Path, modality: str, scale: str) -> dict[str, Any]:
    if scale == "x1":
        rows = 12098 if modality == "mri_minilm" else 2788
        return {
            "unique_exact_rows": rows,
            "unique_augmentation_rows": 0,
            "unique_semantic_text_rows": rows,
            "physical_training_rows": rows,
            "augmentation_exact_overlap": 0,
        }
    path = data_root / modality / f"{modality}_{scale}_anchorbalanced_auditfirst.summary.json"
    return load_json(path) or {"status": "missing", "path": str(path)}


def ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "contract": {
            "fixed_rungs": ["exact semantic -> ELF", "raw brain prediction -> ELF", "trained adapter -> ELF"],
            "mri_validation_rows": 266,
            "meg_validation_rows": 110,
            "mri_test107_opened": False,
            "meg_test26_opened": False,
            "known_text_audit_trainable": True,
            "brain_validation_vectors_trainable": False,
            "selection": "one generation per row; no reference-based reranking",
        },
        "modalities": {},
    }
    lines = [
        "# ELF semantic-corpus scaling ladder",
        "",
        "Each scale uses the same validation rows and candidate pool. Exact text/semantic pairs are a deliberately",
        "known-text oracle; validation brain vectors remain evaluation-only. MRI test107 and MEG test26 stay sealed.",
        "",
    ]
    for modality, spec in SPECS.items():
        rows = []
        corpora = {str(scale): corpus_summary(args.data_root, modality, str(scale)) for scale in spec["scales"]}
        for scale in spec["scales"]:
            for rung in ("exact", "raw", "adapter"):
                tag = run_tag(modality, str(scale), rung)
                result = extract(args.run_root / tag)
                result.update({"scale": scale, "rung": rung, "run_tag": tag})
                rows.append(result)
        report["modalities"][modality] = rows
        report.setdefault("corpora", {})[modality] = corpora
        lines.extend(
            [
                f"## {modality}",
                "",
                "### Training corpus",
                "",
                "| Scale | Unique exact | Unique augmented | Unique total | Physical rows | Exact duplicates in augmentation |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for scale in spec["scales"]:
            corpus = corpora[str(scale)]
            lines.append(
                f"| {scale} | {corpus.get('unique_exact_rows', '—')} | "
                f"{corpus.get('unique_augmentation_rows', '—')} | "
                f"{corpus.get('unique_semantic_text_rows', '—')} | "
                f"{corpus.get('physical_training_rows', '—')} | "
                f"{corpus.get('augmentation_exact_overlap', '—')} |"
            )
        lines.extend(
            [
                "",
                "### Fixed-row ladder",
                "",
                "| Scale | Rung | Word F1 | Content F1 | WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank | Sem Top-1 / Top-5 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["status"] != "complete":
                lines.append(f"| {row['scale']} | {row['rung']} | pending | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {row['scale']} | {row['rung']} | {fmt(row['word_f1'])} | {fmt(row['content_f1'])} | "
                f"{fmt(row['wer'])} | {fmt(row['bertscore_f1'])} | {fmt(row['generated_top1'])} | "
                f"{fmt(row['generated_top5'])} | {fmt(row['generated_mean_rank'], 2)} / "
                f"{fmt(row['generated_median_rank'], 1)} | {fmt(row['semantic_top1'])} / {fmt(row['semantic_top5'])} |"
            )
        lines.append("")

        lines.extend(
            [
                "### Information retained from the exact-semantic ceiling",
                "",
                "Ratios compare each brain-conditioned rung with the exact ELF at the same corpus scale.",
                "",
                "| Scale | Rung | Word-F1 retained | Content-F1 retained | Gen Top-1 retained | Gen Top-5 retained |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        by_scale = {
            str(scale): {row["rung"]: row for row in rows if str(row["scale"]) == str(scale)}
            for scale in spec["scales"]
        }
        for scale in spec["scales"]:
            ceiling = by_scale[str(scale)].get("exact", {})
            for rung in ("raw", "adapter"):
                row = by_scale[str(scale)].get(rung, {})
                lines.append(
                    f"| {scale} | {rung} | {fmt(ratio(row.get('word_f1'), ceiling.get('word_f1')))} | "
                    f"{fmt(ratio(row.get('content_f1'), ceiling.get('content_f1')))} | "
                    f"{fmt(ratio(row.get('generated_top1'), ceiling.get('generated_top1')))} | "
                    f"{fmt(ratio(row.get('generated_top5'), ceiling.get('generated_top5')))} |"
                )
        lines.append("")

        curves = {}
        for scale in spec["scales"]:
            if scale == "x1":
                continue
            curve = training_curve(args.run_root / oracle_tag(modality, str(scale)))
            curves[str(scale)] = curve
        report.setdefault("training_curves", {})[modality] = curves
        if curves:
            lines.extend(
                [
                    "### ELF-stage convergence audit",
                    "",
                    "`continue` means the best checkpoint is in the last three evaluations and WER improved by at least 0.003 across that tail.",
                    "",
                    "| Scale | Evaluations | First WER | Best WER @ step | Last WER @ step | Tail gain | Decision |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for scale, curve in curves.items():
                if curve is None:
                    lines.append(f"| {scale} | pending | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| {scale} | {curve['num_evaluations']} | {fmt(curve['first']['wer'])} | "
                    f"{fmt(curve['best']['wer'])} @ {curve['best']['step']} | "
                    f"{fmt(curve['last']['wer'])} @ {curve['last']['step']} | "
                    f"{fmt(curve['last_three_wer_gain'])} | {curve['decision']} |"
                )
            lines.append("")

        complete_adapters = [row for row in rows if row["rung"] == "adapter" and row["status"] == "complete"]
        if complete_adapters:
            winner = max(complete_adapters, key=lambda row: float(row["content_f1"] or -1.0))
            per_sample = winner["per_sample_content_f1"]
            targets = winner["targets"]
            generated = winner["generated"]
            if per_sample and len(per_sample) == len(targets) == len(generated):
                top = sorted(range(len(per_sample)), key=lambda index: per_sample[index], reverse=True)[:10]
                lines.extend(
                    [
                        f"Best adapted scale by content F1: `{winner['scale']}` (`{winner['run_tag']}`).",
                        "",
                        "| Row | Content F1 | Target | Prediction |",
                        "|---:|---:|---|---|",
                    ]
                )
                for index in top:
                    lines.append(
                        f"| {index} | {float(per_sample[index]):.3f} | {str(targets[index]).replace('|', '/')} | "
                        f"{str(generated[index]).replace('|', '/')} |"
                    )
                lines.append("")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_markdown)


if __name__ == "__main__":
    main()
