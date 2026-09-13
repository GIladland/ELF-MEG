#!/usr/bin/env python3
"""Build conservative target-free consensus editors from hybrid sweep arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.cap_generated_text_words import edit_distance, overlap_summary, words
except ModuleNotFoundError:
    from cap_generated_text_words import edit_distance, overlap_summary, words


def generation_digest(generated: Sequence[str]) -> str:
    payload = "".join(f"{index}\t{text}\n" for index, text in enumerate(generated))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_consensus_generation(
    source: str,
    candidates: Sequence[str],
    *,
    minimum_support: float,
    max_source_edits: int,
) -> tuple[str, int, int]:
    """Select a repeated non-source edit without inspecting reference text."""
    if not 0.0 < minimum_support <= 1.0:
        raise ValueError("minimum_support must be in (0, 1]")
    if max_source_edits < 1:
        raise ValueError("max_source_edits must be positive")
    source_normalized = " ".join(source.split())
    alternatives = [
        " ".join(candidate.split())
        for candidate in candidates
        if " ".join(candidate.split()) != source_normalized
    ]
    if not alternatives:
        return source, 0, 0
    counts = Counter(alternatives)
    best_support = max(counts.values())
    minimum_count = math.ceil(minimum_support * len(candidates))
    best = sorted(text for text, count in counts.items() if count == best_support)
    if best_support < minimum_count or len(best) != 1:
        return source, best_support, minimum_count
    winner = best[0]
    if edit_distance(words(source), words(winner)) > max_source_edits:
        return source, best_support, minimum_count
    return winner, best_support, minimum_count


def load_generation_json(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    targets = [str(value) for value in payload["targets"]]
    generated = [str(value) for value in payload["generated"]]
    if len(targets) != len(generated):
        raise ValueError(f"row mismatch in {path}")
    return targets, generated


def quality(generated: list[str], targets: list[str]) -> dict[str, Any]:
    generated_words = [words(text) for text in generated]
    target_words = [words(text) for text in targets]
    overlap = overlap_summary(generated_words, target_words)
    errors = [
        edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(target_words, generated_words)
    ]
    reference_words = sum(map(len, target_words))
    return {
        **overlap,
        "word_error_rate": sum(errors) / reference_words,
        "word_error_reference_words": reference_words,
        "word_error_total": sum(errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", action="append", required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--minimum-support", type=float, nargs="+", required=True)
    parser.add_argument("--max-source-edits", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets, source = load_generation_json(args.source_json)
    paths: list[Path] = []
    for prefix in args.run_prefix:
        paths.extend(args.runs_root.glob(f"{prefix}*/best_metrics.json"))
    paths = sorted(set(paths))
    candidate_sets: dict[str, tuple[Path, list[str]]] = {}
    for path in paths:
        candidate_targets, generated = load_generation_json(path)
        if candidate_targets != targets:
            raise ValueError(f"target rows do not align in {path}")
        candidate_sets.setdefault(generation_digest(generated), (path, generated))
    if not candidate_sets:
        raise FileNotFoundError("no candidate best_metrics.json files matched")
    candidates = [generated for _, generated in candidate_sets.values()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard: list[dict[str, Any]] = []

    # Construct every output using source/candidates only. Reference labels are
    # consulted afterward solely to report global validation metrics.
    for minimum_support in args.minimum_support:
        for max_source_edits in args.max_source_edits:
            generated: list[str] = []
            row_support: list[int] = []
            row_minimum: list[int] = []
            for row_index, source_text in enumerate(source):
                selected, support, minimum = select_consensus_generation(
                    source_text,
                    [candidate[row_index] for candidate in candidates],
                    minimum_support=minimum_support,
                    max_source_edits=max_source_edits,
                )
                generated.append(selected)
                row_support.append(support)
                row_minimum.append(minimum)
            metrics = quality(generated, targets)
            tag = f"support{minimum_support:g}_maxedits{max_source_edits}"
            payload = {
                "run_tag": f"meg_t5_elfb_consensus_{tag}",
                "scientific_scope": (
                    "target-free per-row consensus over globally validation-selected "
                    "T5-to-ELF-B configurations; protected test26 not loaded"
                ),
                "selection_contract": {
                    "source": str(args.source_json),
                    "candidate_run_prefixes": args.run_prefix,
                    "candidate_run_count": len(paths),
                    "unique_candidate_generation_sets": len(candidates),
                    "minimum_support_fraction": minimum_support,
                    "minimum_support_count": math.ceil(minimum_support * len(candidates)),
                    "max_source_word_edits": max_source_edits,
                    "reference_text_used_for_row_selection": False,
                    "ties_return_source": True,
                },
                "targets": targets,
                "generated": generated,
                "generation_quality": metrics,
                "changed_rows_vs_source": sum(
                    generated_text != source_text
                    for generated_text, source_text in zip(generated, source)
                ),
                "row_winning_support": row_support,
                "row_minimum_support": row_minimum,
            }
            output = args.output_dir / f"{tag}.json"
            with output.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            leaderboard.append(
                {
                    "run_tag": payload["run_tag"],
                    "minimum_support": minimum_support,
                    "max_source_edits": max_source_edits,
                    "unique_candidate_generation_sets": len(candidates),
                    "changed_rows_vs_source": payload["changed_rows_vs_source"],
                    "content_f1": metrics["content_words_overlap"],
                    "word_f1": metrics["words_overlap"],
                    "sum": metrics["content_words_overlap"] + metrics["words_overlap"],
                    "wer": metrics["word_error_rate"],
                    "wer_errors": metrics["word_error_total"],
                    "output": str(output),
                }
            )
    leaderboard.sort(key=lambda row: (-row["sum"], row["wer"], row["run_tag"]))
    with (args.output_dir / "leaderboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(leaderboard)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_run_count": len(paths),
                "unique_candidate_generation_sets": len(candidates),
                "leaderboard": leaderboard,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(json.dumps(leaderboard[:10], indent=2))


if __name__ == "__main__":
    main()
