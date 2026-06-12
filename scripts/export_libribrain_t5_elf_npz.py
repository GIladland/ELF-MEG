#!/usr/bin/env python
"""Export LibriBrain Sherlock sentences as normalized T5 latents for ELF."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from utils.encoder_utils import encode_text
from utils.libribrain_utils import build_libribrain_sentence_dataset_per_book


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--semantic-data-path", default=None)
    parser.add_argument("--pnpl-root", required=True)
    parser.add_argument("--books", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder-model-name", default="t5-small")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-type", default="ADA")
    parser.add_argument("--set-name", default="sentences")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-files", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-examples", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def encode_sentences(
    *,
    sentences: list[str],
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_latents = []
    all_masks = []
    all_ids = []
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        encoded = tokenizer(
            sentences[start:end],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        latents = encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder=encoder,
            latent_mean=0.0,
            latent_std=0.2,
            use_bf16=True,
        )
        all_latents.append(latents.float().cpu().numpy())
        all_masks.append(attention_mask.float().cpu().numpy())
        all_ids.append(input_ids.cpu().numpy())
        logger.info("encoded %d/%d", end, len(sentences))
    return (
        np.concatenate(all_latents, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_ids, axis=0),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = build_libribrain_sentence_dataset_per_book(
        data_path=args.data_path,
        books=args.books,
        pnpl_root=args.pnpl_root,
        set_name=args.set_name,
        semantic_data_path=args.semantic_data_path,
        embedding_type=args.embedding_type,
        num_workers=args.num_workers,
        preload_files=args.preload_files,
    )
    sentences = []
    rows = []
    for sample in getattr(dataset, "samples", []):
        if len(sample) != 8:
            continue
        subject, session, task, run, onset, offset, _semantic_vector, sentence = sample
        sentence = str(sentence).strip()
        if not sentence:
            continue
        rows.append({
            "subject": str(subject),
            "session": str(session),
            "task": str(task),
            "run": str(run),
            "onset": float(onset),
            "offset": float(offset),
            "sentence": sentence,
        })
        sentences.append(sentence)
        if args.num_examples > 0 and len(sentences) >= args.num_examples:
            break
    logger.info("loaded %d non-empty sentences from books=%s", len(sentences), args.books)
    if not sentences:
        raise RuntimeError("No sentences found")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _encoder_config, encoder = get_encoder(args.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    latents, masks, input_ids = encode_sentences(
        sentences=sentences,
        tokenizer=tokenizer,
        encoder=encoder,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target_t5_latents=latents,
        t5_attention_mask=masks,
        t5_input_ids=input_ids,
        sentence=np.asarray(sentences, dtype=object),
        rows=np.asarray(rows, dtype=object),
    )
    summary = {
        "output": str(output),
        "n": len(sentences),
        "books": args.books,
        "max_length": args.max_length,
        "target_t5_latents_shape": list(latents.shape),
    }
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved %s", output)


if __name__ == "__main__":
    main()
