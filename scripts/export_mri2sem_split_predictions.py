#!/usr/bin/env python3
"""Export MRI2SEM predictions for selected dataset splits from a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri2sem-root", type=Path, required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
        outputs.append(model(batch).cpu().numpy())
    return np.vstack(outputs).astype(np.float32, copy=False)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = l2_normalize(pred.astype(np.float32, copy=False))
    target = l2_normalize(target.astype(np.float32, copy=False))
    scores = pred @ target.T
    matched = np.diag(scores)
    ranks = 1 + np.sum(scores > matched[:, None], axis=1)
    n = len(ranks)
    mismatch = scores[~np.eye(n, dtype=bool)] if n > 1 else np.array([], dtype=np.float32)
    return {
        "n": n,
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= min(5, n))),
        "top10": float(np.mean(ranks <= min(10, n))),
        "mean_rank": float(np.mean(ranks)),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / max(n - 1, 1))),
        "matched_cosine": float(matched.mean()),
        "mismatch_cosine": float(mismatch.mean()) if len(mismatch) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.mri2sem_root / "src"))
    from mri2sem.models import MindEyeStyleMLP  # noqa: PLC0415

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    model = MindEyeStyleMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(checkpoint_args.get("hidden_dim", 2048)),
        res_blocks=int(checkpoint_args.get("res_blocks", 4)),
        dropout=float(checkpoint_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "mri2sem_root": str(args.mri2sem_root),
        "dataset_npz": str(args.dataset_npz),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "splits": {},
    }
    with np.load(args.dataset_npz, allow_pickle=True) as dataset:
        for split in args.splits:
            x = dataset[f"{split}_x"].astype(np.float32, copy=False)
            target = dataset[f"{split}_y"].astype(np.float32, copy=False)
            if x.shape[1] != input_dim or target.shape[1] != output_dim:
                raise ValueError(
                    f"{split} dimensions {x.shape}/{target.shape} do not match checkpoint "
                    f"{input_dim}/{output_dim}"
                )
            pred = predict(model, x, args.batch_size, device)
            metrics = retrieval_metrics(pred, target)
            output_path = args.output_dir / f"predictions_{split}.npz"
            payload: dict[str, np.ndarray] = {"pred": pred, "target": target}
            for field in ("story", "start_tr", "stop_tr", "text"):
                key = f"{split}_{field}"
                if key in dataset:
                    payload[field] = dataset[key]
            np.savez_compressed(output_path, **payload)
            summary["splits"][split] = {
                "output_path": str(output_path),
                "input_shape": list(x.shape),
                "prediction_shape": list(pred.shape),
                "metrics": metrics,
            }
            print(json.dumps({"split": split, **metrics}, sort_keys=True), flush=True)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
