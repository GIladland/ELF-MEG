#!/usr/bin/env python
"""Linearly interpolate matching tensors in two trusted ELF E2E checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="0=baseline, 1=candidate")
    parser.add_argument(
        "--interpolate-model",
        action="store_true",
        help="Interpolate model_state_dict as well as adapter_state_dict.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def interpolate_state_dict(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if baseline.keys() != candidate.keys():
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"state-dict key mismatch: missing={missing[:10]} extra={extra[:10]}")
    result = {}
    for key, baseline_value in baseline.items():
        candidate_value = candidate[key]
        if baseline_value.shape != candidate_value.shape or baseline_value.dtype != candidate_value.dtype:
            raise ValueError(
                f"tensor mismatch at {key}: "
                f"{tuple(baseline_value.shape)}/{baseline_value.dtype} != "
                f"{tuple(candidate_value.shape)}/{candidate_value.dtype}"
            )
        if baseline_value.is_floating_point() or baseline_value.is_complex():
            result[key] = torch.lerp(baseline_value, candidate_value, alpha)
        else:
            if not torch.equal(baseline_value, candidate_value):
                raise ValueError(f"non-floating checkpoint tensor changed at {key}")
            result[key] = baseline_value
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    for required in ("model_state_dict", "adapter_state_dict"):
        if required not in baseline or required not in candidate:
            raise ValueError(f"both checkpoints must contain {required}")

    payload = {
        "step": 0,
        "epoch": 0.0,
        "score": None,
        "model_state_dict": (
            interpolate_state_dict(
                baseline["model_state_dict"], candidate["model_state_dict"], args.alpha
            )
            if args.interpolate_model
            else baseline["model_state_dict"]
        ),
        "adapter_state_dict": interpolate_state_dict(
            baseline["adapter_state_dict"], candidate["adapter_state_dict"], args.alpha
        ),
        "args": baseline.get("args", {}),
        "config": baseline.get("config", {}),
        "interpolation": {
            "baseline": str(Path(args.baseline)),
            "candidate": str(Path(args.candidate)),
            "alpha": args.alpha,
            "interpolate_model": bool(args.interpolate_model),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Wrote alpha={args.alpha:g} checkpoint to {output}")


if __name__ == "__main__":
    main()
