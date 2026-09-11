#!/usr/bin/env python
"""Add closed-set T5-small retrieval metrics to an existing generation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded, return_dict=True).last_hidden_state
        mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        rows.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def retrieval_metrics(query: np.ndarray, targets: np.ndarray) -> dict:
    query = np.asarray(query, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if query.shape != targets.shape or query.ndim != 2:
        raise ValueError(f"query/target shape mismatch: {query.shape} != {targets.shape}")
    similarity = query @ targets.T
    diagonal = np.diag(similarity)
    ranks = 1 + np.sum(similarity > diagonal[:, None], axis=1)
    top_indices = np.argsort(-similarity, axis=1)[:, : min(5, len(targets))]
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= min(5, len(targets)))),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "ranks": ranks.astype(int).tolist(),
        "top_indices": top_indices.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    generated = [str(value) for value in payload["generated"]]
    targets = [str(value) for value in payload["targets"]]
    if len(generated) != len(targets):
        raise ValueError("generated and target rows must align")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    model = T5EncoderModel.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    ).to(device).eval()
    generated_embeddings = encode_texts(
        [value if value.strip() else "." for value in generated],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    target_embeddings = encode_texts(
        targets,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    payload["generation_t5_retrieval"] = retrieval_metrics(
        generated_embeddings, target_embeddings
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["generation_t5_retrieval"], indent=2))


if __name__ == "__main__":
    main()
