#!/usr/bin/env python3
"""Materialize one nested, session-grouped MEG OOF split.

The outer sessions are never loaded by the MEG2SEM trainer.  One additional
session is reserved for checkpoint selection, so the outer predictions are not
indirectly selected with their own labels.  All remaining sessions form the
training pack.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-inner-val", required=True, type=Path)
    parser.add_argument("--output-outer", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--num-folds", default=5, type=int)
    parser.add_argument("--group-key", default="session")
    parser.add_argument("--inner-target-rows", default=110, type=int)
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Compress output packs. The default stored ZIP is much faster for 4 GB MEG arrays.",
    )
    return parser.parse_args()


def as_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
            for item in np.asarray(values).tolist()
        ],
        dtype=object,
    )


def balanced_group_folds(groups: np.ndarray, num_folds: int) -> tuple[dict[str, int], list[int]]:
    counts = Counter(groups.tolist())
    fold_sizes = [0] * num_folds
    assignment: dict[str, int] = {}
    # Largest-first greedy bin packing gives deterministic, row-balanced folds.
    for group, size in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(num_folds), key=lambda index: (fold_sizes[index], index))
        assignment[group] = fold
        fold_sizes[fold] += size
    return assignment, fold_sizes


def nested_masks(
    groups: np.ndarray,
    assignment: dict[str, int],
    outer_fold: int,
    num_folds: int,
    inner_target_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    outer_groups = {group for group, fold in assignment.items() if fold == outer_fold}
    inner_pool_fold = (outer_fold + 1) % num_folds
    inner_candidates = [group for group, fold in assignment.items() if fold == inner_pool_fold]
    counts = Counter(groups.tolist())
    inner_group = min(
        inner_candidates,
        key=lambda group: (abs(counts[group] - inner_target_rows), group),
    )
    outer = np.isin(groups, sorted(outer_groups))
    inner = groups == inner_group
    train = ~(outer | inner)
    if np.any(train & outer) or np.any(train & inner) or np.any(outer & inner):
        raise RuntimeError("Nested OOF masks overlap")
    if not np.all(train | inner | outer):
        raise RuntimeError("Nested OOF masks do not cover every source row")
    return train, inner, outer, inner_group


def write_pack(path: Path, payload: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **payload)


def main() -> None:
    args = parse_args()
    if not 0 <= args.outer_fold < args.num_folds:
        raise ValueError(f"outer fold must be in [0, {args.num_folds})")

    with np.load(args.input_npz, allow_pickle=True) as source:
        if args.group_key not in source.files:
            raise KeyError(f"Missing group key {args.group_key!r} in {args.input_npz}")
        groups = as_strings(source[args.group_key])
        row_count = len(groups)
        assignment, fold_sizes = balanced_group_folds(groups, args.num_folds)
        train_mask, inner_mask, outer_mask, inner_group = nested_masks(
            groups,
            assignment,
            args.outer_fold,
            args.num_folds,
            args.inner_target_rows,
        )
        indices = {
            "train": np.flatnonzero(train_mask),
            "inner_val": np.flatnonzero(inner_mask),
            "outer": np.flatnonzero(outer_mask),
        }
        payloads: dict[str, dict[str, np.ndarray]] = {name: {} for name in indices}
        for key in source.files:
            value = np.asarray(source[key])
            if value.ndim >= 1 and value.shape[0] == row_count:
                for name, selected in indices.items():
                    payloads[name][key] = value[selected]
            else:
                for payload in payloads.values():
                    payload[key] = value

    outer_groups = sorted(group for group, fold in assignment.items() if fold == args.outer_fold)
    split_metadata = {
        "schema": "nested_session_grouped_meg_oof_split_v1",
        "source_npz": str(args.input_npz),
        "group_key": args.group_key,
        "num_folds": args.num_folds,
        "outer_fold": args.outer_fold,
        "fold_row_counts": fold_sizes,
        "outer_groups": outer_groups,
        "inner_selection_group": inner_group,
        "train_rows": int(train_mask.sum()),
        "inner_selection_rows": int(inner_mask.sum()),
        "outer_rows": int(outer_mask.sum()),
        "outer_loaded_by_trainer": False,
    }
    for name, selected in indices.items():
        payloads[name]["source_original_index"] = selected.astype(np.int64)
        payloads[name]["oof_outer_fold"] = np.full(len(selected), args.outer_fold, dtype=np.int16)
        payloads[name]["oof_role"] = np.full(len(selected), name, dtype=object)
        payloads[name]["oof_split_json"] = np.asarray(json.dumps(split_metadata, sort_keys=True))

    write_pack(args.output_train, payloads["train"], args.compressed)
    write_pack(args.output_inner_val, payloads["inner_val"], args.compressed)
    write_pack(args.output_outer, payloads["outer"], args.compressed)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(split_metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(split_metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
