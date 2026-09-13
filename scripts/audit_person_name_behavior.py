#!/usr/bin/env python3
"""Audit person-name behavior for a manually reviewed generation set.

The MEG validation text is lowercased, which makes generic named-entity
recognition unreliable.  This audit therefore consumes explicit, full-set
person-name annotations and binds them to the exact target/generation pairs
with a SHA-256 digest.  A changed generation file cannot silently reuse an
older manual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def normalized_phrase(text: str) -> str:
    return " ".join(token.lower() for token in WORD_RE.findall(text))


def load_pairs(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"index", "target", "generated"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"CSV must contain columns {sorted(required)}")
        ordered = sorted(rows, key=lambda row: int(row["index"]))
        indices = [int(row["index"]) for row in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError("CSV indices must be unique and contiguous from zero")
        return [(row["target"], row["generated"]) for row in ordered]

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    targets = payload.get("targets")
    generated = payload.get("generated")
    if not isinstance(targets, list) or not isinstance(generated, list):
        raise ValueError("JSON must contain list fields 'targets' and 'generated'")
    if len(targets) != len(generated):
        raise ValueError("JSON target and generation row counts differ")
    return list(zip(targets, generated))


def paired_text_sha256(pairs: list[tuple[str, str]]) -> str:
    payload = "".join(
        f"{index}\t{target}\t{generated}\n"
        for index, (target, generated) in enumerate(pairs)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def name_map(raw: Any, *, field: str, row_count: int) -> dict[int, list[str]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object mapping row indices to names")
    parsed: dict[int, list[str]] = {}
    for raw_index, raw_names in raw.items():
        index = int(raw_index)
        if not 0 <= index < row_count:
            raise ValueError(f"{field} row {index} is outside 0..{row_count - 1}")
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(f"{field} row {index} must have a non-empty name list")
        names = [normalized_phrase(str(name)) for name in raw_names]
        if any(not name for name in names):
            raise ValueError(f"{field} row {index} contains an empty name")
        parsed[index] = sorted(set(names))
    return parsed


def require_annotated_names_in_text(
    pairs: list[tuple[str, str]],
    annotations: dict[int, list[str]],
    *,
    side: int,
    field: str,
) -> None:
    for index, names in annotations.items():
        text = f" {normalized_phrase(pairs[index][side])} "
        for name in names:
            if f" {name} " not in text:
                raise ValueError(
                    f"Annotated name {name!r} is absent from {field} row {index}: "
                    f"{pairs[index][side]!r}"
                )


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def audit(
    pairs: list[tuple[str, str]],
    target_names: dict[int, list[str]],
    generated_names: dict[int, list[str]],
    *,
    system: str,
    digest: str,
) -> dict[str, Any]:
    named_target_rows = sorted(target_names)
    unnamed_target_rows = sorted(set(range(len(pairs))) - set(named_target_rows))
    any_name_rows: list[int] = []
    exact_rows: list[int] = []
    substitution_rows: list[int] = []
    deletion_rows: list[int] = []
    false_insertion_rows: list[int] = []
    per_row: list[dict[str, Any]] = []

    for index, (target, generated) in enumerate(pairs):
        expected = set(target_names.get(index, []))
        observed = set(generated_names.get(index, []))
        exact = bool(expected & observed)
        if expected:
            if observed:
                any_name_rows.append(index)
                if exact:
                    exact_rows.append(index)
                else:
                    substitution_rows.append(index)
            else:
                deletion_rows.append(index)
        elif observed:
            false_insertion_rows.append(index)
        if expected or observed:
            per_row.append(
                {
                    "index": index,
                    "target": target,
                    "generated": generated,
                    "target_person_names": sorted(expected),
                    "generated_person_names": sorted(observed),
                    "category": (
                        "exact_name_recovery"
                        if exact
                        else "name_substitution"
                        if expected and observed
                        else "name_deletion"
                        if expected
                        else "false_name_insertion"
                    ),
                }
            )

    named_count = len(named_target_rows)
    unnamed_count = len(unnamed_target_rows)
    return {
        "schema_version": 1,
        "system": system,
        "row_count": len(pairs),
        "paired_text_sha256": digest,
        "contract": {
            "unit": "validation row",
            "entity_scope": "person names only",
            "annotations": "manual full-set review bound to paired_text_sha256",
            "categories_on_named_targets": (
                "exact recovery, wrong-name substitution, and deletion are mutually exclusive"
            ),
        },
        "counts": {
            "named_target_rows": named_count,
            "unnamed_target_rows": unnamed_count,
            "named_target_to_any_name": len(any_name_rows),
            "exact_name_recovery": len(exact_rows),
            "name_substitution": len(substitution_rows),
            "name_deletion": len(deletion_rows),
            "false_name_insertion": len(false_insertion_rows),
        },
        "rates": {
            "named_target_to_any_name": safe_rate(len(any_name_rows), named_count),
            "exact_name_recovery": safe_rate(len(exact_rows), named_count),
            "name_substitution": safe_rate(len(substitution_rows), named_count),
            "name_deletion": safe_rate(len(deletion_rows), named_count),
            "false_name_insertion": safe_rate(len(false_insertion_rows), unnamed_count),
        },
        "row_indices": {
            "named_target_to_any_name": any_name_rows,
            "exact_name_recovery": exact_rows,
            "name_substitution": substitution_rows,
            "name_deletion": deletion_rows,
            "false_name_insertion": false_insertion_rows,
        },
        "annotated_rows": per_row,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.input)
    digest = paired_text_sha256(pairs)
    with args.annotations.open(encoding="utf-8") as handle:
        annotation_payload = json.load(handle)
    row_count = int(annotation_payload.get("row_count", -1))
    if row_count != len(pairs):
        raise ValueError(
            f"Annotation row_count={row_count} does not match input rows={len(pairs)}"
        )
    target_names = name_map(
        annotation_payload.get("target_person_names", {}),
        field="target_person_names",
        row_count=len(pairs),
    )
    systems = annotation_payload.get("systems", {})
    if args.system not in systems:
        raise ValueError(
            f"Unknown system {args.system!r}; available: {sorted(systems)}"
        )
    system_annotation = systems[args.system]
    expected_digest = system_annotation.get("paired_text_sha256")
    if expected_digest != digest:
        raise ValueError(
            "Generation set does not match its reviewed annotation: "
            f"expected {expected_digest}, observed {digest}"
        )
    generated_names = name_map(
        system_annotation.get("generated_person_names", {}),
        field=f"systems.{args.system}.generated_person_names",
        row_count=len(pairs),
    )
    require_annotated_names_in_text(
        pairs, target_names, side=0, field="target text"
    )
    require_annotated_names_in_text(
        pairs, generated_names, side=1, field="generated text"
    )
    result = audit(
        pairs,
        target_names,
        generated_names,
        system=args.system,
        digest=digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    rates = result["rates"]
    print(
        f"{args.system}: any-name={rates['named_target_to_any_name']:.6f} "
        f"exact={rates['exact_name_recovery']:.6f} "
        f"substitution={rates['name_substitution']:.6f} "
        f"deletion={rates['name_deletion']:.6f} "
        f"false-insertion={rates['false_name_insertion']:.6f}"
    )


if __name__ == "__main__":
    main()
