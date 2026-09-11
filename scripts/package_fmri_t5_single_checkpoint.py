#!/usr/bin/env python
"""Bundle one frozen fMRI-to-T5 checkpoint with its deterministic decode rule.

The resulting ``.pt`` contains every trained adapter/LoRA/lexical-head tensor
from the source checkpoint plus the globally selected decoding configuration
and validation provenance.  Frozen public T5 base weights remain referenced by
model name, as in the training checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics_json)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path.read_text())
    metrics_checkpoint = metrics.get("checkpoint")
    if metrics_checkpoint and Path(metrics_checkpoint) != checkpoint_path:
        raise ValueError(
            f"metrics were generated from {metrics_checkpoint}, not {checkpoint_path}"
        )
    decoding = metrics.get("decoding_config")
    if not isinstance(decoding, dict) or not decoding:
        raise ValueError("metrics JSON lacks a non-empty decoding_config")
    secondary_path_value = decoding.get("secondary_content_head_checkpoint")
    if secondary_path_value:
        secondary_path = Path(secondary_path_value)
        secondary_payload = torch.load(
            secondary_path, map_location="cpu", weights_only=False
        )
        required = ("vocabulary", "head_state_dict", "logit_prior", "input_dim")
        missing = [key for key in required if key not in secondary_payload]
        if missing:
            raise ValueError(
                f"secondary lexical checkpoint lacks required fields: {missing}"
            )
        payload["secondary_content_head_bundle"] = {
            key: secondary_payload[key]
            for key in (*required, "hidden_dim")
            if key in secondary_payload
        }
        payload["secondary_content_head_source"] = str(secondary_path)
        system_kind = (
            "single-checkpoint deterministic decoding with a fixed train-only "
            "dual lexical head; no candidate reranking"
        )
    else:
        system_kind = "single-checkpoint deterministic decoding; no candidate reranking"
    payload["packaged_system"] = {
        "name": args.name,
        "system_kind": system_kind,
        "source_checkpoint": str(checkpoint_path),
        "source_metrics_json": str(metrics_path),
        "decoding_config": decoding,
        "validation_metrics": {
            key: metrics.get(key) for key in (
                "content_words_overlap",
                "content_words_overlap_precision",
                "content_words_overlap_recall",
                "words_overlap",
                "word_error_rate",
                "deranged_content_f1_mean",
                "deranged_content_f1_max",
                "conditional_content_margin",
                "beats_all_derangements",
            )
        },
        "scientific_scope": metrics.get("scientific_scope"),
        "test107_status_at_packaging": "sealed and not loaded",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    verification = torch.load(destination, map_location="cpu", weights_only=False)
    if verification.get("packaged_system") != payload["packaged_system"]:
        raise RuntimeError("packaged checkpoint verification failed")
    print(json.dumps({
        "output": str(destination),
        "bytes": destination.stat().st_size,
        **payload["packaged_system"],
    }, indent=2))


if __name__ == "__main__":
    main()
