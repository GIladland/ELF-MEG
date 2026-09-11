#!/usr/bin/env python
"""Verify train/validation temporal segment reconstruction without reading test arrays."""

from __future__ import annotations

import argparse
import json

import numpy as np


def strings(array: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in array.tolist()]


def build(stories, trs, segment_stories, starts, stops, length):
    lookup = {(story, int(tr)): row for row, (story, tr) in enumerate(zip(strings(stories), trs))}
    indices = np.empty((len(starts), length), dtype=np.int64)
    for row, (story, start, stop) in enumerate(zip(strings(segment_stories), starts, stops)):
        # MRI2SEM segment metadata stores stop_tr as the last included TR.
        # A ten-TR segment therefore has stop_tr - start_tr + 1 == 10.
        inclusive_length = int(stop) - int(start) + 1
        if inclusive_length != length:
            raise ValueError(f"row {row}: inclusive_length={inclusive_length}")
        indices[row] = [lookup[(story, tr)] for tr in range(int(start), int(stop) + 1)]
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tr-npz", required=True)
    parser.add_argument("--segment-npz", required=True)
    args = parser.parse_args()
    per_tr = np.load(args.per_tr_npz, allow_pickle=True)
    segments = np.load(args.segment_npz, allow_pickle=True)
    train = build(
        per_tr["train_story"], per_tr["train_tr"], segments["train_story"],
        segments["train_start_tr"], segments["train_stop_tr"], 10,
    )
    val = build(
        per_tr["val_story"], per_tr["val_tr"], segments["val_story"],
        segments["val_start_tr"], segments["val_stop_tr"], 10,
    )
    print(json.dumps({
        "status": "complete",
        "scientific_scope": "only train/val metadata keys indexed; no test key read",
        "train_shape": list(train.shape),
        "val_shape": list(val.shape),
        "train_row_minmax": [int(train.min()), int(train.max())],
        "val_row_minmax": [int(val.min()), int(val.max())],
        "train_consecutive": bool(np.all(np.diff(train, axis=1) == 1)),
        "val_consecutive": bool(np.all(np.diff(val, axis=1) == 1)),
    }, indent=2))


if __name__ == "__main__":
    main()
