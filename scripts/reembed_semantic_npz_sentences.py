#!/usr/bin/env python
"""Re-embed the sentence field of a semantic NPZ with a different HF encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--embedding-model-name", required=True)
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    value = str(data[schema_key].tolist())
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    if args.sentence_key not in data.files:
        raise KeyError(f"{args.input_npz} has no sentence key {args.sentence_key!r}")
    sentences = _strings(data[args.sentence_key])
    n = len(sentences)

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
    embedding_inputs = [f"{args.input_prefix}{sentence}" for sentence in sentences]
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

    output_arrays = {}
    for key in data.files:
        if key == args.input_key:
            continue
        output_arrays[key] = data[key]

    summary = {
        "output": args.output,
        "input_npz": args.input_npz,
        "input_key": args.input_key,
        "sentence_key": args.sentence_key,
        "count": n,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "input_prefix": args.input_prefix,
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "source_schema": _schema(data, args.schema_key),
        "sample_sentences": sentences[:5],
    }
    output_arrays[args.input_key] = embeddings.astype(np.float32)
    output_arrays[args.schema_key] = np.asarray(json.dumps(summary))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **output_arrays)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
