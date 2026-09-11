#!/usr/bin/env python
"""Bundle a brain-conditioned lexical ELF diffusion system into one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


T5_BASELINE = {
    "content_words_overlap": 0.043743318085423345,
    "words_overlap": 0.08267770173628086,
    "word_error_rate": 0.987218045112782,
    "bertscore_f1": -0.0882781825,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lexical-checkpoint", required=True)
    parser.add_argument("--ordered-checkpoint", default="")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--deranged-metrics", nargs="*", default=[])
    parser.add_argument("--bertscore-json", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--lexical-topk", type=int, required=True)
    parser.add_argument("--lexical-temperature", type=float, required=True)
    parser.add_argument("--lexical-prior-subtraction", type=float, required=True)
    parser.add_argument("--lexical-max-prior", type=float, required=True)
    parser.add_argument("--lexical-bias-strength", type=float, required=True)
    parser.add_argument("--lexical-bias-max-positions", type=int, default=0)
    parser.add_argument(
        "--lexical-bias-mode", choices=("scattered", "sequence"), required=True
    )
    parser.add_argument("--lexical-context-mode", required=True)
    parser.add_argument("--ordered-prior-subtraction", type=float, default=0.0)
    parser.add_argument("--ordered-min-prior-probability", type=float, default=0.0)
    parser.add_argument("--ordered-min-margin", type=float, default=0.0)
    parser.add_argument("--ordered-bias-strength", type=float, default=0.0)
    parser.add_argument("--ordered-max-positions", type=int, default=0)
    parser.add_argument(
        "--require-beats-t5-all",
        action="store_true",
        help="Refuse packaging unless all four val266 metrics and brain controls clear the frozen T5 baseline.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def unique_suffix_tensor(state: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one adapter tensor ending in {suffix!r}, found {len(matches)}")
    return matches[0].detach().cpu()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    lexical_path = Path(args.lexical_checkpoint)
    ordered_path = Path(args.ordered_checkpoint) if args.ordered_checkpoint else None
    metrics_path = Path(args.metrics_json)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lexical = torch.load(lexical_path, map_location="cpu", weights_only=False)
    ordered = (
        torch.load(ordered_path, map_location="cpu", weights_only=False)
        if ordered_path is not None
        else None
    )
    metrics = load_json(metrics_path)
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, dict):
        raise ValueError("Source checkpoint lacks adapter_state_dict")
    required = ("vocabulary", "head_state_dict", "feature", "input_dim", "hidden_dim")
    missing = [key for key in required if key not in lexical]
    if missing:
        raise ValueError(f"Lexical checkpoint lacks required fields: {missing}")
    if lexical["feature"] != "semantic":
        raise ValueError("Packaged lexical head must consume frozen MRI2SEM semantic output")
    ordered_required = (
        "vocabulary", "head_state_dict", "log_prior", "feature", "input_dim",
        "positions", "hidden_dim",
    )
    if ordered is not None:
        ordered_missing = [key for key in ordered_required if key not in ordered]
        if ordered_missing:
            raise ValueError(
                f"Ordered checkpoint lacks required fields: {ordered_missing}"
            )
        if ordered["feature"] != "semantic":
            raise ValueError(
                "Packaged ordered head must consume frozen MRI2SEM semantic output"
            )

    deranged_payloads = [load_json(path) for path in args.deranged_metrics]
    matched_content = float(metrics["generation_quality"]["content_words_overlap"])
    deranged_content = [
        float(item["generation_quality"]["content_words_overlap"])
        for item in deranged_payloads
    ]
    validation = {
        "content_words_overlap": matched_content,
        "words_overlap": float(metrics["generation_quality"]["words_overlap"]),
        "word_error_rate": float(metrics["generation_quality"]["word_error_rate"]),
        "generation_t5_retrieval": metrics.get("generation_t5_retrieval"),
        "deranged_content_values": deranged_content,
        "deranged_content_mean": (
            float(np.mean(deranged_content)) if deranged_content else None
        ),
        "deranged_content_max": max(deranged_content) if deranged_content else None,
        "conditional_content_margin": (
            matched_content - float(np.mean(deranged_content))
            if deranged_content
            else None
        ),
        "beats_all_derangements": (
            matched_content > max(deranged_content) if deranged_content else None
        ),
    }
    if args.bertscore_json:
        bertscore = load_json(args.bertscore_json)
        validation["bertscore"] = bertscore.get("summary", bertscore.get("mean"))
    if args.require_beats_t5_all:
        bertscore_summary = validation.get("bertscore")
        bertscore_f1 = (
            bertscore_summary.get("bertscore_f1")
            if isinstance(bertscore_summary, dict)
            else None
        )
        checks = {
            "content_words_overlap": (
                validation["content_words_overlap"]
                > T5_BASELINE["content_words_overlap"]
            ),
            "words_overlap": (
                validation["words_overlap"] > T5_BASELINE["words_overlap"]
            ),
            "word_error_rate": (
                validation["word_error_rate"] < T5_BASELINE["word_error_rate"]
            ),
            "bertscore_f1": (
                bertscore_f1 is not None
                and float(bertscore_f1) > T5_BASELINE["bertscore_f1"]
            ),
            "beats_all_derangements": validation["beats_all_derangements"] is True,
        }
        if not all(checks.values()):
            raise ValueError(
                "Candidate does not satisfy the all-metric T5 promotion contract: "
                + json.dumps(checks, sort_keys=True)
            )
        validation["beats_t5_all_contract"] = checks
        validation["t5_baseline"] = T5_BASELINE

    # The adapter state already contains the frozen head tensors, reconstructed
    # train-only prior, word prototypes, and token ids.  The bundle supplies
    # the non-tensor vocabulary and enough architecture metadata to rebuild it.
    payload["packaged_lexical_head_bundle"] = {
        key: lexical[key] for key in required
    }
    payload["packaged_lexical_head_bundle"]["logit_prior"] = unique_suffix_tensor(
        adapter_state, "lexical_logit_prior"
    )
    if ordered is not None:
        payload["packaged_ordered_head_bundle"] = {
            key: ordered[key] for key in ordered_required
        }
    payload["packaged_system"] = {
        "name": args.name,
        "system_kind": (
            "single-checkpoint deterministic brain-conditioned ELF diffusion with "
            "one-shot lexical constraint and optional ordered-word repair; "
            "no candidate reranking"
        ),
        "source_checkpoint": str(checkpoint_path),
        "source_lexical_checkpoint": str(lexical_path),
        "source_ordered_checkpoint": (
            str(ordered_path) if ordered_path is not None else None
        ),
        "source_metrics_json": str(metrics_path),
        "deranged_metrics_json": [str(path) for path in args.deranged_metrics],
        "decoding_config": {
            "lexical_topk": args.lexical_topk,
            "lexical_temperature": args.lexical_temperature,
            "lexical_prior_subtraction": args.lexical_prior_subtraction,
            "lexical_max_prior_probability": args.lexical_max_prior,
            "lexical_decode_bias_strength": args.lexical_bias_strength,
            "lexical_decode_bias_max_positions": args.lexical_bias_max_positions,
            "lexical_decode_bias_once": True,
            "lexical_decode_bias_mode": args.lexical_bias_mode,
            "lexical_context_mode": args.lexical_context_mode,
            "ordered_prior_subtraction": args.ordered_prior_subtraction,
            "ordered_min_prior_probability": args.ordered_min_prior_probability,
            "ordered_min_margin": args.ordered_min_margin,
            "ordered_decode_bias_strength": args.ordered_bias_strength,
            "ordered_decode_max_positions": args.ordered_max_positions,
        },
        "validation_metrics": validation,
        "scientific_scope": (
            "MRI2SEM, lexical head, ELF adapter, and decoding selected with train11725/"
            "val266 only; lexical head fit on train11725; "
            + ("ordered head fit on train11725; " if ordered is not None else "")
            + "sealed test107 not loaded"
        ),
        "test107_status_at_packaging": "sealed and not loaded",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    verified = torch.load(output, map_location="cpu", weights_only=False)
    if verified.get("packaged_system") != payload["packaged_system"]:
        raise RuntimeError("Packaged checkpoint metadata verification failed")
    if verified["packaged_lexical_head_bundle"]["vocabulary"] != lexical["vocabulary"]:
        raise RuntimeError("Packaged lexical vocabulary verification failed")
    if ordered is not None and (
        verified["packaged_ordered_head_bundle"]["vocabulary"]
        != ordered["vocabulary"]
    ):
        raise RuntimeError("Packaged ordered vocabulary verification failed")
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                **payload["packaged_system"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
