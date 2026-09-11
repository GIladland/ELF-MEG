#!/usr/bin/env python3
"""Report ELF-B/M/L exact, raw-brain, and adapter ladders at x1 and MEG x16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/data/engs-pnpl/glandau/elf-runs")
SPECS = {
    "mri_minilm": {"audit_n": 266},
    "meg_ada": {"audit_n": 110},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def tags(modality: str, model: str, audit_n: int, scale: str = "x1") -> dict[str, str]:
    if modality == "meg_ada" and scale == "x16":
        if model == "ELF-B":
            oracle = "elf_semantic_scaling_meg_ada_x16_anchorbalanced_compute12k_continuex1_seed7_20260902"
            return {
                "exact": oracle,
                "raw": f"eval_{oracle}_rawbrain_seed49",
                "adapter": "elf_semantic_scaling_meg_ada_x16_trained_adapter_val110_seed49_20260902",
            }
        slug = model.lower().replace("-", "")
        return {
            "exact": f"eval_elf_archscale_meg_ada_x16_{slug}_exact_val110_seed49_20260904",
            "raw": f"eval_elf_archscale_meg_ada_x16_{slug}_rawbrain_seed49_20260904",
            "adapter": f"elf_archscale_meg_ada_x16_{slug}_trained_adapter_continue1_val110_seed49_20260905",
        }
    if scale != "x1":
        raise ValueError(f"Unsupported architecture scale combination: {modality=} {scale=}")
    if model == "ELF-B":
        oracle = (
            "fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901"
            if modality == "mri_minilm"
            else "tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901"
        )
        return {
            "exact": f"eval_elf_semantic_scaling_{modality}_x1_exact_val{audit_n}_seed49_20260902",
            "raw": f"eval_{oracle}_rawbrain_seed49",
            "adapter": f"elf_semantic_scaling_{modality}_x1_trained_adapter_val{audit_n}_seed49_20260902",
        }
    slug = model.lower().replace("-", "")
    return {
        "exact": f"eval_elf_archscale_{modality}_x1_{slug}_exact_val{audit_n}_seed49_20260904",
        "raw": f"eval_elf_archscale_{modality}_x1_{slug}_rawbrain_seed49_20260904",
        "adapter": f"elf_archscale_{modality}_x1_{slug}_trained_adapter_val{audit_n}_seed49_20260904",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def extract(run_root: Path, tag: str) -> dict[str, Any]:
    run_dir = run_root / tag
    metrics = read_json(run_dir / "best_metrics.json")
    if not metrics:
        return {"status": "pending", "run_tag": tag, "run_dir": str(run_dir)}
    quality = metrics.get("generation_quality") or (metrics.get("word_overlap") or {}).get("summary") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    semantic = metrics.get("semantic_interface") or {}
    bert = (read_json(run_dir / "best_metrics.bertscore.json").get("summary") or {})
    return {
        "status": "complete",
        "run_tag": tag,
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None,
        "step": metrics.get("step"),
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


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "axis": "ELF architecture capacity at x1, plus the combined MEG x16 model/data scale",
        "models": {"ELF-B": 105_000_000, "ELF-M": 342_000_000, "ELF-L": 652_000_000},
        "sampling_steps": 32,
        "modalities": {},
    }
    lines = [
        "# ELF architecture and MEG data scaling",
        "",
        "The x1 rows isolate ELF capacity. MEG x16 then tests the interaction between model and corpus size.",
        "Every row uses one generation per fixed validation item, 32 sampling steps, and the same retrieval",
        "candidate count within each modality.",
        "",
    ]
    for modality, spec in SPECS.items():
        audit_n = int(spec["audit_n"])
        rows = []
        scales = ("x1", "x16") if modality == "meg_ada" else ("x1",)
        for scale in scales:
            for model in ("ELF-B", "ELF-M", "ELF-L"):
                for rung, tag in tags(modality, model, audit_n, scale).items():
                    row = extract(args.run_root, tag)
                    row.update({"model": model, "scale": scale, "rung": rung})
                    rows.append(row)
        report["modalities"][modality] = rows
        lines.extend([
            f"## {modality} (N={audit_n})",
            "",
            "| Data | Model | Params | Rung | Word F1 | Content F1 | WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank | Sem Top-1 / Top-5 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            params = report["models"][row["model"]] // 1_000_000
            if row["status"] != "complete":
                lines.append(f"| {row['scale']} | {row['model']} | {params}M | {row['rung']} | pending | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {row['scale']} | {row['model']} | {params}M | {row['rung']} | {fmt(row['word_f1'])} | "
                f"{fmt(row['content_f1'])} | {fmt(row['wer'])} | {fmt(row['bertscore_f1'])} | "
                f"{fmt(row['generated_top1'])} | {fmt(row['generated_top5'])} | "
                f"{fmt(row['generated_mean_rank'], 2)} / {fmt(row['generated_median_rank'], 1)} | "
                f"{fmt(row['semantic_top1'])} / {fmt(row['semantic_top5'])} |"
            )
        lines.append("")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
