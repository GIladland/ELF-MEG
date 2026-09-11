#!/usr/bin/env python
"""Replace semantic arrays in a packed MEG NPZ with an aligned sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--sidecar-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-key", default="minilm_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--sidecar-sentence-key", default="lexical_element")
    parser.add_argument(
        "--replace-keys",
        nargs="+",
        default=["input_embeddings", "semantic_vectors", "embeddings_ada", "minilm_embeddings"],
    )
    return parser.parse_args()


def text_values(array: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value).strip() for value in array.tolist()]


def main() -> None:
    args = parse_args()
    packed = np.load(args.input_npz, allow_pickle=True)
    sidecar = np.load(args.sidecar_npz, allow_pickle=True)
    if args.sentence_key not in packed.files:
        raise KeyError(f"Missing packed sentence key: {args.sentence_key}")
    if args.sidecar_sentence_key not in sidecar.files:
        raise KeyError(f"Missing sidecar sentence key: {args.sidecar_sentence_key}")
    if args.embedding_key not in sidecar.files:
        raise KeyError(f"Missing sidecar embedding key: {args.embedding_key}")

    packed_text = text_values(packed[args.sentence_key])
    sidecar_text = text_values(sidecar[args.sidecar_sentence_key])
    if packed_text != sidecar_text:
        mismatches = [
            {"index": index, "packed": left, "sidecar": right}
            for index, (left, right) in enumerate(zip(packed_text, sidecar_text))
            if left != right
        ]
        raise ValueError(
            f"Packed/sidecar text mismatch: packed={len(packed_text)} sidecar={len(sidecar_text)} "
            f"examples={mismatches[:5]}"
        )

    embeddings = np.asarray(sidecar[args.embedding_key], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(packed_text):
        raise ValueError(
            f"Expected aligned 2D embeddings, got {embeddings.shape} for {len(packed_text)} sentences"
        )

    replace_keys = set(args.replace_keys)
    output_arrays = {key: packed[key] for key in packed.files if key not in replace_keys and key != "schema_json"}
    for key in args.replace_keys:
        output_arrays[key] = embeddings

    summary = {
        "input_npz": args.input_npz,
        "sidecar_npz": args.sidecar_npz,
        "output": args.output,
        "count": len(packed_text),
        "embedding_key": args.embedding_key,
        "embedding_shape": list(embeddings.shape),
        "replace_keys": args.replace_keys,
        "text_mismatch_count": 0,
    }
    output_arrays["schema_json"] = np.asarray(json.dumps(summary))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **output_arrays)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
