#!/usr/bin/env python
"""Export simple synthetic sentences as normalized T5 latents for ELF."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from scripts.export_synthetic_sentence_embeddings import build_sentences
from scripts.meg_context_overfit import encode_text_batched, tokenize_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-model-name", default="t5-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=0, help="0 means use the full target T5 sequence as context.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    sentences = build_sentences(args.count, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask = tokenize_sentences(tokenizer, sentences)

    _encoder_config, encoder = get_encoder(args.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    latents = encode_text_batched(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder=encoder,
        latent_mean=0.0,
        latent_std=0.2,
        device=device,
        batch_size=args.batch_size,
    ).numpy()
    masks = attention_mask.numpy().astype(np.float32)
    ids = input_ids.numpy().astype(np.int64)
    context_length = latents.shape[1] if args.context_length <= 0 else min(args.context_length, latents.shape[1])
    context = latents[:, :context_length]
    context_mask = masks[:, :context_length]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "count": len(sentences),
        "encoder_model_name": args.encoder_model_name,
        "target_t5_latents_shape": list(latents.shape),
        "context_shape": list(context.shape),
        "seed": args.seed,
        "sample": sentences[:5],
    }
    np.savez_compressed(
        output,
        target_t5_latents=latents.astype(np.float32),
        t5_attention_mask=masks,
        t5_input_ids=ids,
        context=context.astype(np.float32),
        context_mask=context_mask.astype(np.float32),
        sentence=np.asarray(sentences, dtype=str),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
