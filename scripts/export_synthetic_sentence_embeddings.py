#!/usr/bin/env python
"""Export synthetic sentence pairs with local HuggingFace sentence embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ADJECTIVES = [
    "red", "blue", "green", "small", "bright", "quiet", "silver", "wooden",
    "old", "fresh", "warm", "cold", "plain", "heavy", "narrow", "round",
]

SUBJECTS = [
    "fox", "train", "musician", "boat", "teacher", "gardener", "baker",
    "doctor", "student", "painter", "runner", "engineer", "librarian",
    "carpenter", "photograph", "lantern", "notebook", "market", "river",
    "mountain",
]

VERBS = [
    "moved past", "rested beside", "waited near", "turned toward",
    "leaned against", "crossed over", "stood behind", "glowed above",
    "slipped under", "circled around", "balanced on", "drifted beyond",
]

OBJECTS = [
    "the sleeping dog", "the empty station", "a narrow window",
    "the kitchen table", "a quiet harbor", "the folded letter",
    "a row of pine trees", "the marble floor", "a dusty cabinet",
    "the river bank", "a broken fence", "the western gallery",
    "a stack of books", "the garden wall", "a brass lantern",
]

PLACES = [
    "museum", "courtyard", "workshop", "library", "harbor", "garden",
    "station", "kitchen", "market", "classroom", "theater", "bridge",
]

TIMES = [
    "before sunrise", "during the rain", "after the meeting",
    "in complete silence", "near the old museum", "beside the notebook",
    "under a pale sky", "while bells rang", "at the edge of town",
    "behind the blue house",
]

TEMPLATES = [
    "The {adj} {subject} {verb} {object} {time}.",
    "A {adj} {subject} {verb} {object} near the {place}.",
    "The {subject} {verb} the {adj} {object_noun} {time}.",
    "In the {place}, the {adj} {subject} {verb} {object}.",
]

OBJECT_NOUNS = [
    "notebook", "window", "drawer", "lamp", "boat", "letter", "violin",
    "camera", "clock", "basket", "bicycle", "map", "cup", "box", "chair",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local/free HuggingFace encoder used for conditioning vectors.",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--input-prefix",
        default="",
        help="Optional text prepended only for the embedding encoder, e.g. 'query: ' for E5.",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "cls"],
        help="Sentence pooling strategy for the encoder hidden states.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sentences(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sentences: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 100
    while len(sentences) < count and attempts < max_attempts:
        attempts += 1
        template = rng.choice(TEMPLATES)
        sentence = template.format(
            adj=rng.choice(ADJECTIVES),
            subject=rng.choice(SUBJECTS),
            verb=rng.choice(VERBS),
            object=rng.choice(OBJECTS),
            object_noun=rng.choice(OBJECT_NOUNS),
            place=rng.choice(PLACES),
            time=rng.choice(TIMES),
        )
        if sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)
    if len(sentences) < count:
        raise RuntimeError(f"Could only generate {len(sentences)} unique sentences out of {count}")
    return sentences


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled


@torch.no_grad()
def encode_sentences(
    *,
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    pooling: str,
    normalize: bool,
) -> np.ndarray:
    vectors = []
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        encoded = tokenizer(
            sentences[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        if pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        if normalize:
            pooled = F.normalize(pooled.float(), dim=-1)
        vectors.append(pooled.float().cpu().numpy())
        print(f"encoded {end}/{len(sentences)}", flush=True)
    return np.concatenate(vectors, axis=0)


def main() -> None:
    args = parse_args()
    sentences = build_sentences(args.count, args.seed)
    embedding_inputs = [f"{args.input_prefix}{sentence}" for sentence in sentences]
    device = resolve_device(args.device)
    print(f"Using device={device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    embeddings = encode_sentences(
        sentences=embedding_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        normalize=not args.no_normalize,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "count": len(sentences),
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "input_prefix": args.input_prefix,
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "seed": args.seed,
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
