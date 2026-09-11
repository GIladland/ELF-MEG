#!/usr/bin/env python3
"""Compare MEG2SEM checkpoints directly in their semantic target space.

This deliberately stops before the ELF projector.  It is intended to answer
whether a weak brain-conditioned diffusion result is caused by the semantic
decoder itself or by the semantic-to-ELF bridge.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from modules.meg2sem_bridge import load_meg2sem_model


EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="meg")
    parser.add_argument("--target-key", default="input_embeddings")
    parser.add_argument("--lengths-key", default="meg_lengths")
    parser.add_argument("--head-examples", type=int, default=512)
    parser.add_argument("--tail-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def retrieval_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = F.normalize(prediction.float(), p=2, dim=-1)
    target = F.normalize(target.float(), p=2, dim=-1)
    similarity = prediction @ target.T
    diagonal = similarity.diagonal()
    ranks = 1 + (similarity > diagonal[:, None]).sum(dim=1)
    n = int(ranks.numel())
    if n <= 1:
        mean_percentile = 1.0
    else:
        mean_percentile = float(((n - ranks).float() / (n - 1)).mean().cpu())
    return {
        "num_examples": n,
        "paired_cosine_mean": float(diagonal.mean().cpu()),
        "paired_cosine_std": float(diagonal.std(unbiased=False).cpu()),
        "top1": float((ranks <= 1).float().mean().cpu()),
        "top5": float((ranks <= min(5, n)).float().mean().cpu()),
        "mean_rank": float(ranks.float().mean().cpu()),
        "median_rank": float(ranks.float().median().cpu()),
        "mean_percentile": mean_percentile,
    }


def tensor_summary(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target_norm = target.float().norm(dim=-1)
    prediction_norm = prediction.float().norm(dim=-1)
    return {
        "prediction_norm_mean": float(prediction_norm.mean().cpu()),
        "prediction_norm_std": float(prediction_norm.std(unbiased=False).cpu()),
        "target_norm_mean": float(target_norm.mean().cpu()),
        "target_norm_std": float(target_norm.std(unbiased=False).cpu()),
        "raw_mse": float(F.mse_loss(prediction.float(), target.float()).cpu()),
    }


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    meg: torch.Tensor,
    lengths: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    model.eval()
    for start in range(0, len(meg), batch_size):
        end = min(start + batch_size, len(meg))
        batch = meg[start:end].to(device=device, dtype=torch.float32)
        batch_lengths = lengths[start:end].to(device=device, dtype=torch.long)
        subjects = torch.zeros(end - start, dtype=torch.long, device=device)
        outputs.append(
            model(batch, meg_lengths=batch_lengths, subjects=subjects).detach().float().cpu()
        )
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    with np.load(args.npz, allow_pickle=True) as data:
        n = int(data[args.target_key].shape[0])
        head_n = min(max(0, args.head_examples), n)
        tail_n = min(max(0, args.tail_examples), n - head_n)
        head_indices = np.arange(head_n, dtype=np.int64)
        tail_indices = np.arange(n - tail_n, n, dtype=np.int64)
        indices = np.concatenate([head_indices, tail_indices])
        meg = torch.as_tensor(np.asarray(data[args.input_key][indices]), dtype=torch.float16)
        target = torch.as_tensor(np.asarray(data[args.target_key][indices]), dtype=torch.float32)
        if args.lengths_key in data.files:
            lengths = torch.as_tensor(np.asarray(data[args.lengths_key][indices]), dtype=torch.long)
        else:
            lengths = torch.full((len(indices),), int(meg.shape[-1]), dtype=torch.long)

    print(
        f"Loaded {len(indices)}/{n} rows: meg={tuple(meg.shape)} "
        f"target={tuple(target.shape)} device={device}",
        flush=True,
    )
    split_slices = {
        "head": slice(0, head_n),
        "tail": slice(head_n, head_n + tail_n),
        "combined": slice(0, head_n + tail_n),
    }
    results: dict[str, object] = {
        "npz": args.npz,
        "total_examples": n,
        "head_examples": head_n,
        "tail_examples": tail_n,
        "checkpoints": [],
    }

    for checkpoint in args.checkpoint:
        model, load_info = load_meg2sem_model(
            checkpoint,
            output_normalization="never",
            device=device,
        )
        prediction = predict(
            model,
            meg,
            lengths,
            batch_size=max(1, args.batch_size),
            device=device,
        )
        checkpoint_results: dict[str, object] = {
            "path": checkpoint,
            "epoch": int(EPOCH_RE.search(checkpoint).group(1)) if EPOCH_RE.search(checkpoint) else None,
            "load_info": {
                "format": load_info.checkpoint_format,
                "embedding_dim": load_info.embedding_dim,
                "input_dim": load_info.input_dim,
                "n_subjects": load_info.n_subjects,
                "missing_keys": load_info.missing_keys,
                "unexpected_keys": load_info.unexpected_keys,
            },
            "splits": {},
        }
        for split_name, split_slice in split_slices.items():
            split_prediction = prediction[split_slice]
            split_target = target[split_slice]
            if len(split_prediction) == 0:
                continue
            split_result = tensor_summary(split_prediction, split_target)
            split_result.update(retrieval_metrics(split_prediction, split_target))
            checkpoint_results["splits"][split_name] = split_result
        results["checkpoints"].append(checkpoint_results)
        tail = checkpoint_results["splits"].get("tail", {})
        print(
            f"epoch={checkpoint_results['epoch']} tail_cos={tail.get('paired_cosine_mean')} "
            f"tail_top1={tail.get('top1')} tail_top5={tail.get('top5')} "
            f"tail_mean_rank={tail.get('mean_rank')}",
            flush=True,
        )
        del model, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
