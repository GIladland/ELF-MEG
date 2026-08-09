#!/usr/bin/env python
"""Export synthetic sentences with token-level source embeddings and T5 targets."""

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
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from modules.t5_encoder import get_encoder
from scripts.export_synthetic_sentence_embeddings import build_sentences
from scripts.meg_context_overfit import encode_text_batched, tokenize_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--target-encoder-model-name", default="t5-small")
    parser.add_argument("--source-max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize-source-tokens", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def encode_source_tokens(
    *,
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    latents = []
    masks = []
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        encoded = tokenizer(
            sentences[start:end],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.float()
        if normalize:
            hidden = F.normalize(hidden, dim=-1)
        hidden = hidden * attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        latents.append(hidden.cpu().numpy())
        masks.append(attention_mask.cpu().numpy().astype(np.float32))
        print(f"encoded source {end}/{len(sentences)}", flush=True)
    return np.concatenate(latents, axis=0), np.concatenate(masks, axis=0)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    sentences = build_sentences(args.count, args.seed)

    source_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    source_model = AutoModel.from_pretrained(args.embedding_model_name).to(device).eval()
    source_latents, source_mask = encode_source_tokens(
        sentences=sentences,
        tokenizer=source_tokenizer,
        model=source_model,
        device=device,
        max_length=args.source_max_length,
        batch_size=args.batch_size,
        normalize=args.normalize_source_tokens,
    )

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_encoder_model_name)
    if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token is not None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_ids, target_mask = tokenize_sentences(target_tokenizer, sentences)
    _encoder_config, target_encoder = get_encoder(args.target_encoder_model_name, dtype=torch.float32)
    target_encoder = target_encoder.to(device).eval()
    for param in target_encoder.parameters():
        param.requires_grad_(False)
    target_latents = encode_text_batched(
        input_ids=target_ids,
        attention_mask=target_mask,
        encoder=target_encoder,
        latent_mean=0.0,
        latent_std=0.2,
        device=device,
        batch_size=args.batch_size,
    ).numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "count": len(sentences),
        "embedding_model_name": args.embedding_model_name,
        "target_encoder_model_name": args.target_encoder_model_name,
        "input_embeddings_shape": list(source_latents.shape),
        "target_t5_latents_shape": list(target_latents.shape),
        "source_max_length": args.source_max_length,
        "normalize_source_tokens": args.normalize_source_tokens,
        "seed": args.seed,
        "sample": sentences[:5],
    }
    np.savez_compressed(
        output,
        input_embeddings=source_latents.astype(np.float32),
        input_attention_mask=source_mask.astype(np.float32),
        target_t5_latents=target_latents.astype(np.float32),
        t5_attention_mask=target_mask.numpy().astype(np.float32),
        t5_input_ids=target_ids.numpy().astype(np.int64),
        sentence=np.asarray(sentences, dtype=str),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
