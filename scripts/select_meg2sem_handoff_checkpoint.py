#!/usr/bin/env python
"""Select a MEG2SEM checkpoint for BrainDiffusion handoff from W&B metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable


CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>\d+)-val_loss=(?P<loss>[0-9.]+)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-run", required=True, help="W&B run path, e.g. pnpl/b2s2t/dt8z7o55.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=["priority", "ndcg_dominant", "balanced", "loss"],
        default="priority",
        help=(
            "priority sorts by nDCG desc, cosine desc, procrustes asc, loss asc. "
            "ndcg_dominant and balanced use rank-percentile weighted scores."
        ),
    )
    parser.add_argument("--ndcg-key", default="val_nDCG_epoch")
    parser.add_argument("--cosine-key", default="val_cosine_mean_epoch")
    parser.add_argument("--procrustes-key", default="val_procrustes_epoch")
    parser.add_argument("--loss-key", default="val_loss")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        choices=["path", "json", "table"],
        default="table",
        help="path prints only the selected checkpoint path for shell use.",
    )
    return parser.parse_args()


def finite_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def checkpoint_by_epoch(checkpoint_dir: Path) -> dict[int, dict[str, object]]:
    checkpoints: dict[int, dict[str, object]] = {}
    for path in checkpoint_dir.glob("*.ckpt"):
        match = CHECKPOINT_RE.search(path.name)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        checkpoints[epoch] = {
            "checkpoint": str(path),
            "checkpoint_val_loss": float(match.group("loss")),
        }
    return checkpoints


def percentile_scores(values: Iterable[float], *, high_is_good: bool) -> dict[float, float]:
    ordered = sorted(values, reverse=high_is_good)
    n = len(ordered)
    if n == 0:
        return {}
    scores: dict[float, float] = {}
    idx = 0
    while idx < n:
        end = idx + 1
        while end < n and ordered[end] == ordered[idx]:
            end += 1
        # Average rank percentile, where best value is 1.0 and worst is 1 / n.
        avg_rank = (idx + 1 + end) / 2.0
        scores[ordered[idx]] = 1.0 - ((avg_rank - 1.0) / max(1.0, n - 1.0))
        idx = end
    return scores


def weighted_score(records: list[dict[str, object]], record: dict[str, object], strategy: str) -> float:
    weights = {
        "ndcg_dominant": {
            "ndcg": 0.80,
            "cosine": 0.10,
            "procrustes": 0.07,
            "loss": 0.03,
        },
        "balanced": {
            "ndcg": 0.65,
            "cosine": 0.20,
            "procrustes": 0.10,
            "loss": 0.05,
        },
    }[strategy]
    percentiles = {
        "ndcg": percentile_scores([float(r["ndcg"]) for r in records], high_is_good=True),
        "cosine": percentile_scores([float(r["cosine"]) for r in records], high_is_good=True),
        "procrustes": percentile_scores([float(r["procrustes"]) for r in records], high_is_good=False),
        "loss": percentile_scores([float(r["loss"]) for r in records], high_is_good=False),
    }
    return sum(weights[key] * percentiles[key][float(record[key])] for key in weights)


def load_metric_records(args: argparse.Namespace) -> list[dict[str, object]]:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required to select by W&B metrics.") from exc

    checkpoints = checkpoint_by_epoch(args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit(f"No epoch checkpoints found under {args.checkpoint_dir}")

    api = wandb.Api()
    run = api.run(args.wandb_run)
    keys = ["epoch", args.ndcg_key, args.cosine_key, args.procrustes_key, args.loss_key]
    records_by_epoch: dict[int, dict[str, object]] = {}
    for row in run.scan_history(keys=keys, page_size=1000):
        epoch_value = finite_float(row.get("epoch"))
        if epoch_value is None:
            continue
        epoch = int(round(epoch_value))
        if epoch not in checkpoints:
            continue
        ndcg = finite_float(row.get(args.ndcg_key))
        cosine = finite_float(row.get(args.cosine_key))
        procrustes = finite_float(row.get(args.procrustes_key))
        loss = finite_float(row.get(args.loss_key))
        if None in (ndcg, cosine, procrustes, loss):
            continue
        records_by_epoch[epoch] = {
            "epoch": epoch,
            "ndcg": ndcg,
            "cosine": cosine,
            "procrustes": procrustes,
            "loss": loss,
            **checkpoints[epoch],
        }

    records = list(records_by_epoch.values())
    if not records:
        raise SystemExit(
            "No checkpoints had complete W&B metric rows for "
            f"{args.ndcg_key}, {args.cosine_key}, {args.procrustes_key}, and {args.loss_key}."
        )
    if args.strategy in {"ndcg_dominant", "balanced"}:
        for record in records:
            record["selection_score"] = weighted_score(records, record, args.strategy)
    else:
        for record in records:
            record["selection_score"] = None
    return records


def sort_records(records: list[dict[str, object]], strategy: str) -> list[dict[str, object]]:
    if strategy == "priority":
        return sorted(
            records,
            key=lambda r: (float(r["ndcg"]), float(r["cosine"]), -float(r["procrustes"]), -float(r["loss"])),
            reverse=True,
        )
    if strategy == "loss":
        return sorted(records, key=lambda r: (float(r["loss"]), -float(r["ndcg"])))
    return sorted(
        records,
        key=lambda r: (float(r["selection_score"]), float(r["ndcg"]), float(r["cosine"])),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    records = sort_records(load_metric_records(args), args.strategy)
    selected = records[0]
    if args.output == "path":
        print(selected["checkpoint"])
        return
    if args.output == "json":
        print(json.dumps({"selected": selected, "top": records[: args.top_k]}, indent=2, sort_keys=True))
        return

    columns = ["rank", "epoch", "selection_score", "ndcg", "cosine", "procrustes", "loss", "checkpoint"]
    print("\t".join(columns))
    for rank, record in enumerate(records[: args.top_k], start=1):
        values = {
            "rank": rank,
            "epoch": record["epoch"],
            "selection_score": "" if record["selection_score"] is None else f"{float(record['selection_score']):.6f}",
            "ndcg": f"{float(record['ndcg']):.6f}",
            "cosine": f"{float(record['cosine']):.6f}",
            "procrustes": f"{float(record['procrustes']):.6f}",
            "loss": f"{float(record['loss']):.6f}",
            "checkpoint": record["checkpoint"],
        }
        print("\t".join(str(values[col]) for col in columns))


if __name__ == "__main__":
    main()
