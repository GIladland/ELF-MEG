#!/usr/bin/env python
"""Attach cached T5 target latents, masks, and token ids to a text NPZ.

The resulting file can be consumed by ``decode_elf_context_npz.py`` and
``train_npz_t5_bridge_to_elf.py``.  It is intentionally generic: row order is
preserved exactly, so protected-split policies remain properties of the input
NPZ rather than being silently changed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--target-latents-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--semantic-key", default="input_embeddings")
    parser.add_argument("--encoder-model-name", default="t5-small")
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [
        str(value.decode("utf-8") if isinstance(value, bytes) else value)
        for value in array.tolist()
    ]


def main() -> None:
    args = parse_args()
    source = np.load(args.source_npz, allow_pickle=True)
    sentences = strings(source[args.sentence_key])
    latents = np.load(args.target_latents_cache, mmap_mode="r")
    if len(sentences) != len(latents):
        raise ValueError(
            f"Row mismatch: sentences={len(sentences)} target_latents={len(latents)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=int(latents.shape[1]),
        return_tensors="np",
    )
    ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
    masks = np.asarray(tokenized["attention_mask"], dtype=np.float32)
    if ids.shape != tuple(latents.shape[:2]):
        raise ValueError(
            f"Token/cache shape mismatch: ids={ids.shape} latents={latents.shape}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_npz": args.source_npz,
        "target_latents_cache": args.target_latents_cache,
        "output": str(output),
        "rows": len(sentences),
        "target_t5_latents_shape": list(latents.shape),
        "semantic_shape": list(source[args.semantic_key].shape),
        "encoder_model_name": args.encoder_model_name,
        "row_order_preserved": True,
    }
    semantic = np.asarray(source[args.semantic_key], dtype=np.float32)
    np.savez_compressed(
        output,
        input_embeddings=semantic,
        # A single language-agnostic semantic token. This lets the compact
        # T5LatentBridge resample ADA into an ordered T5 context without
        # pretending that adjacent ADA coordinates are meaningful tokens.
        ada_token=semantic[:, None, :],
        ada_token_mask=np.ones((len(semantic), 1), dtype=np.float32),
        target_t5_latents=np.asarray(latents, dtype=np.float32),
        t5_attention_mask=masks,
        t5_input_ids=ids,
        context=np.asarray(latents, dtype=np.float32),
        context_mask=masks,
        sentence=np.asarray(sentences, dtype=str),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
