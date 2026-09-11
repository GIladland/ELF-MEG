#!/usr/bin/env python
"""Invert semantic embeddings from an NPZ with a pretrained Vec2Text corrector."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--text-key", default="sentence")
    parser.add_argument("--embedder", default="text-embedding-ada-002")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--sequence-beam-width", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--word-cap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def cap_words(text: str, limit: int) -> str:
    if limit <= 0:
        return text.strip()
    return " ".join(text.strip().split()[:limit])


def main() -> None:
    args = parse_args()
    if args.num_steps < 0:
        raise ValueError("--num-steps must be nonnegative")
    if args.num_steps == 0 and args.sequence_beam_width != 0:
        raise ValueError("--sequence-beam-width must be zero when --num-steps is zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.input_npz, allow_pickle=True) as payload:
        if args.embedding_key not in payload or args.text_key not in payload:
            raise KeyError(
                f"Expected {args.embedding_key!r} and {args.text_key!r}; "
                f"found {payload.files}"
            )
        embeddings = np.asarray(payload[args.embedding_key], dtype=np.float32)
        targets = [str(value) for value in payload[args.text_key].tolist()]
    if embeddings.ndim != 2 or len(embeddings) != len(targets):
        raise ValueError(
            f"Invalid aligned arrays: embeddings={embeddings.shape}, targets={len(targets)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    # Vec2Text's inference trainer initializes four reporting-only `evaluate`
    # metrics. They are never used by inversion, and loading them can trigger
    # network downloads on a compute node. Disable only that reporting hook.
    import evaluate

    evaluate.load = lambda *args, **kwargs: None
    import vec2text

    device = resolve_device(args.device)
    corrector = vec2text.load_pretrained_corrector(args.embedder)
    corrector.inversion_trainer.model.to(device).eval()
    corrector.model.to(device).eval()

    generated_native: list[str] = []
    recursive_steps = None if args.num_steps == 0 else args.num_steps
    with torch.inference_mode():
        for start in range(0, len(embeddings), args.batch_size):
            stop = min(start + args.batch_size, len(embeddings))
            batch = torch.as_tensor(
                embeddings[start:stop], dtype=torch.float32, device=device
            )
            decoded = vec2text.invert_embeddings(
                embeddings=batch,
                corrector=corrector,
                num_steps=recursive_steps,
                sequence_beam_width=args.sequence_beam_width,
            )
            generated_native.extend(str(value).strip() for value in decoded)
            print(f"decoded {stop}/{len(embeddings)}", flush=True)

    generated = [cap_words(value, args.word_cap) for value in generated_native]
    result = {
        "contract": {
            "method": "pretrained_vec2text",
            "input_npz": str(args.input_npz),
            "embedding_key": args.embedding_key,
            "embedder": args.embedder,
            "num_steps": args.num_steps,
            "sequence_beam_width": args.sequence_beam_width,
            "batch_size": args.batch_size,
            "word_cap": args.word_cap,
            "seed": args.seed,
            "n": len(targets),
        },
        "targets": targets,
        "generated": generated,
        "generated_native": generated_native,
        "target_word_counts": [len(value.split()) for value in targets],
        "generated_word_counts": [len(value.split()) for value in generated],
        "generated_native_word_counts": [len(value.split()) for value in generated_native],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    native_path = args.output_json.with_name(
        f"{args.output_json.stem}.native{args.output_json.suffix}"
    )
    native_result = dict(result)
    native_result["generated"] = generated_native
    native_result["contract"] = {**result["contract"], "word_cap": None}
    native_path.write_text(
        json.dumps(native_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output_json}")
    print(f"saved {native_path}")


if __name__ == "__main__":
    main()
