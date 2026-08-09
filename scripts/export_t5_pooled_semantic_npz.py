#!/usr/bin/env python
"""Convert a synthetic T5-latent NPZ into one pooled T5 hidden-state vector per sentence.

The source latents are T5 encoder last hidden states after ELF's affine latent
normalization. With the default L2 normalization, this is equivalent to masked
mean pooling the raw T5 encoder hidden states and normalizing the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-key", default="target_t5_latents")
    parser.add_argument("--mask-key", default="t5_attention_mask")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def masked_mean_pool(latents: torch.Tensor, mask: torch.Tensor, normalize: bool) -> torch.Tensor:
    mask = mask.to(dtype=latents.dtype).unsqueeze(-1)
    pooled = (latents * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pooled = pooled.float()
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
    return pooled


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    latents = torch.as_tensor(data[args.latent_key], dtype=torch.float32)
    mask = torch.as_tensor(data[args.mask_key], dtype=torch.float32)
    sentences = _strings(data[args.sentence_key])
    embeddings = masked_mean_pool(latents, mask, normalize=not args.no_normalize).cpu().numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "input_npz": args.input_npz,
        "latent_key": args.latent_key,
        "mask_key": args.mask_key,
        "sentence_key": args.sentence_key,
        "count": len(sentences),
        "embedding_model_name": "t5-small-encoder-hidden-mean-pool",
        "embedding_shape": list(embeddings.shape),
        "normalized": not args.no_normalize,
        "sample": sentences[:5],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(sentences, dtype=str),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
