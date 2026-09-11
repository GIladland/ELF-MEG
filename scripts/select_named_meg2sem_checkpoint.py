#!/usr/bin/env python
"""Select the validation-best checkpoint for a named local MEG2SEM run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meg2sem-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--metric", default="val_nDCG_epoch")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--output", choices=["path", "json"], default="path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(
        re.escape(args.run_name)
        + r"-epoch=(?P<epoch>\d+)-"
        + re.escape(args.metric)
        + r"=(?P<score>[-+0-9.eE]+)\.ckpt$"
    )
    candidates = []
    for path in args.meg2sem_root.glob(f"wandb/*/checkpoints/{args.run_name}-*.ckpt"):
        match = pattern.search(path.name)
        if match is None:
            continue
        candidates.append(
            {
                "path": str(path),
                "epoch": int(match.group("epoch")),
                "score": float(match.group("score")),
                "mtime": path.stat().st_mtime,
            }
        )
    if not candidates:
        raise SystemExit(
            f"No checkpoints matched run={args.run_name!r}, metric={args.metric!r} under {args.meg2sem_root}/wandb"
        )
    reverse = args.mode == "max"
    candidates.sort(key=lambda row: (row["score"], row["mtime"], row["epoch"]), reverse=reverse)
    selected = candidates[0]
    if args.output == "path":
        print(selected["path"])
    else:
        print(json.dumps({"selected": selected, "candidates": candidates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
