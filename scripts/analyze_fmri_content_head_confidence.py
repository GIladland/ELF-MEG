#!/usr/bin/env python
"""Audit whether a bundled lexical head's confidence predicts held-out hits.

Only val266 is read.  The report is diagnostic: it bins prior-corrected lexical
scores and rank gaps, allowing a later global confidence gate.  It never makes
row-wise choices using target text and never loads test107.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from probe_fmri_supervised_content_head import target_matrix
from probe_fmri_supervised_content_head import ContentHead


def strings(values: np.ndarray) -> list[str]:
    return [
        str(value.decode("utf-8") if isinstance(value, bytes) else value)
        for value in values.tolist()
    ]


def infer_hidden(state: dict[str, torch.Tensor], vocabulary_size: int) -> int:
    first = state.get("net.1.weight")
    if first is None:
        raise ValueError("Unrecognized content-head state dict.")
    return 0 if first.shape[0] == vocabulary_size else int(first.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oof-semantic-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--prior-subtraction", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def quantile_bins(values: np.ndarray, outcomes: np.ndarray, bins: int) -> list[dict]:
    order = np.argsort(values, kind="stable")
    result = []
    for indices in np.array_split(order, bins):
        if len(indices) == 0:
            continue
        result.append({
            "count": int(len(indices)),
            "score_min": float(values[indices].min()),
            "score_mean": float(values[indices].mean()),
            "score_max": float(values[indices].max()),
            "hit_rate": float(outcomes[indices].mean()),
        })
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = payload["args"]
    vocabulary = [str(value) for value in payload["content_vocabulary"]]
    state = payload["content_head_state_dict"]
    content_head = ContentHead(
        384, infer_hidden(state, len(vocabulary)), len(vocabulary)
    ).to(device)
    content_head.load_state_dict(state, strict=True)
    content_head.eval()
    prior_logit = torch.as_tensor(payload["content_prior_logit"], dtype=torch.float32)
    with np.load(args.oof_semantic_npz, allow_pickle=True, mmap_mode="r") as archive:
        condition = np.asarray(strings(archive["condition_source"]))
        rows = np.flatnonzero(condition == "validation_prediction_ensemble")
        delayed = np.asarray(archive["input_embeddings"][rows], dtype=np.float32)
    if len(rows) != 266 or delayed.shape != (266, 1536):
        raise ValueError(f"Expected val266 delayed embeddings, got {delayed.shape}.")
    mean_semantic = F.normalize(
        torch.as_tensor(delayed).reshape(len(delayed), 4, 384).mean(dim=1),
        p=2,
        dim=-1,
    ).to(device)
    with np.load(args.text_npz, allow_pickle=True) as archive:
        targets = strings(archive["sentence"])[-266:]
    target = target_matrix(targets, vocabulary)
    with torch.no_grad():
        logits = content_head(mean_semantic).cpu()
    adjusted = logits - args.prior_subtraction * prior_logit.detach().cpu()[None, :]
    top_values, top_indices = adjusted.topk(k=5, dim=1)
    hits = target.gather(1, top_indices).numpy().astype(np.float64)
    values = top_values.numpy().astype(np.float64)
    top1_gap = values[:, 0] - values[:, 1]
    any_top5 = (hits.sum(axis=1) > 0).astype(np.float64)
    report = {
        "scientific_scope": "val266 confidence audit; test107 absent",
        "checkpoint": args.checkpoint,
        "prior_subtraction": args.prior_subtraction,
        "saved_oof_delay_mode": saved.get("oof_delay_mode"),
        "rank_summary": [
            {
                "rank": rank + 1,
                "score_quantiles": np.quantile(
                    values[:, rank], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                ).tolist(),
                "hit_rate": float(hits[:, rank].mean()),
                "score_bins": quantile_bins(values[:, rank], hits[:, rank], args.bins),
            }
            for rank in range(5)
        ],
        "top1_gap_quantiles": np.quantile(
            top1_gap, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        ).tolist(),
        "top1_gap_bins_for_any_top5_hit": quantile_bins(top1_gap, any_top5, args.bins),
        "any_top5_hit_rate": float(any_top5.mean()),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
