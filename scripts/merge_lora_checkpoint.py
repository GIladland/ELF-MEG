#!/usr/bin/env python
"""Merge ELF LoRA tensors into ordinary linear weights for inference/model soup."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.0)
    return parser.parse_args()


def merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor], *, rank: int, alpha: float
) -> dict[str, torch.Tensor]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    prefixes = sorted(key[: -len("lora_A")] for key in state_dict if key.endswith("lora_A"))
    if not prefixes:
        raise ValueError("checkpoint contains no LoRA tensors")
    scaling = float(alpha) / rank
    consumed: set[str] = set()
    merged: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        a_key = f"{prefix}lora_A"
        b_key = f"{prefix}lora_B"
        weight_key = f"{prefix}base.weight"
        if b_key not in state_dict or weight_key not in state_dict:
            raise ValueError(f"incomplete LoRA tensors under {prefix}")
        base_weight = state_dict[weight_key]
        update = state_dict[b_key].float() @ state_dict[a_key].float()
        merged[f"{prefix}weight"] = (base_weight.float() + scaling * update).to(base_weight.dtype)
        consumed.update({a_key, b_key, weight_key})
        bias_key = f"{prefix}base.bias"
        if bias_key in state_dict:
            merged[f"{prefix}bias"] = state_dict[bias_key]
            consumed.add(bias_key)

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if ".lora_A" in key or ".lora_B" in key or ".base.weight" in key or ".base.bias" in key:
            raise ValueError(f"unconsumed LoRA-wrapped tensor: {key}")
        merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint lacks model_state_dict")
    saved_args = checkpoint.get("args", {})
    rank = args.rank or int(saved_args.get("elf_lora_rank", 0))
    alpha = args.alpha or float(saved_args.get("elf_lora_alpha", 0.0))
    checkpoint["model_state_dict"] = merge_lora_state_dict(
        checkpoint["model_state_dict"], rank=rank, alpha=alpha
    )
    checkpoint["merged_lora"] = {
        "source": str(Path(args.input)),
        "rank": rank,
        "alpha": alpha,
    }
    if isinstance(checkpoint.get("args"), dict):
        checkpoint["args"] = dict(checkpoint["args"])
        checkpoint["args"]["elf_lora_rank"] = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Merged rank={rank} alpha={alpha:g} LoRA checkpoint to {output}")


if __name__ == "__main__":
    main()
