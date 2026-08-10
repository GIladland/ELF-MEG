#!/usr/bin/env python
"""Copy embeddings from one NPZ into another by normalized sentence text."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-npz", required=True, help="NPZ whose row order/metadata should be preserved.")
    parser.add_argument("--embedding-npz", required=True, help="NPZ providing replacement embeddings.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--template-input-key", default="input_embeddings")
    parser.add_argument("--embedding-input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument(
        "--prefer-source-regex",
        default="",
        help="Prefer source_npz values matching this regex when duplicate sentence keys exist.",
    )
    parser.add_argument("--max-examples", type=int, default=5)
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


def sentence_key(sentence: str) -> str:
    return " ".join(WORD_RE.findall(str(sentence).casefold()))


def source_values(data: np.lib.npyio.NpzFile, n: int) -> list[str]:
    if "source_npz" not in data.files:
        return [""] * n
    values = _strings(data["source_npz"])
    if len(values) != n:
        return [""] * n
    return values


def main() -> None:
    args = parse_args()
    template = np.load(args.template_npz, allow_pickle=True)
    embedding = np.load(args.embedding_npz, allow_pickle=True)
    for path, data, input_key in [
        (args.template_npz, template, args.template_input_key),
        (args.embedding_npz, embedding, args.embedding_input_key),
    ]:
        if args.sentence_key not in data.files:
            raise KeyError(f"{path} has no sentence key {args.sentence_key!r}")
        if input_key not in data.files:
            raise KeyError(f"{path} has no embedding key {input_key!r}")

    template_sentences = _strings(template[args.sentence_key])
    embedding_sentences = _strings(embedding[args.sentence_key])
    template_n = len(template_sentences)
    embedding_n = len(embedding_sentences)
    if len(template[args.template_input_key]) != template_n:
        raise ValueError(f"Template embedding count does not match sentence count: {args.template_npz}")
    if len(embedding[args.embedding_input_key]) != embedding_n:
        raise ValueError(f"Source embedding count does not match sentence count: {args.embedding_npz}")

    prefer_re = re.compile(args.prefer_source_regex) if args.prefer_source_regex else None
    embedding_sources = source_values(embedding, embedding_n)
    by_sentence: dict[str, list[int]] = defaultdict(list)
    for idx, sentence in enumerate(embedding_sentences):
        key = sentence_key(sentence)
        if key:
            by_sentence[key].append(idx)

    selected_indices = []
    fallback_duplicate_count = 0
    preferred_count = 0
    missing = []
    for sentence in template_sentences:
        key = sentence_key(sentence)
        candidates = by_sentence.get(key, [])
        if not candidates:
            missing.append(sentence)
            selected_indices.append(-1)
            continue
        chosen = candidates[0]
        if prefer_re is not None:
            preferred = [idx for idx in candidates if prefer_re.search(embedding_sources[idx])]
            if preferred:
                chosen = preferred[0]
                preferred_count += 1
        if len(candidates) > 1 and chosen == candidates[0]:
            fallback_duplicate_count += 1
        selected_indices.append(chosen)
    if missing:
        examples = "\n".join(f"- {sentence}" for sentence in missing[: args.max_examples])
        raise RuntimeError(f"Missing {len(missing)} template sentences in embedding NPZ. Examples:\n{examples}")

    selected = np.asarray(selected_indices, dtype=np.int64)
    replacement_embeddings = np.asarray(embedding[args.embedding_input_key][selected], dtype=np.float32)
    output_arrays = {}
    for key in template.files:
        if key in {args.template_input_key, args.schema_key}:
            continue
        output_arrays[key] = template[key]
    output_arrays[args.template_input_key] = replacement_embeddings

    selected_sources = [embedding_sources[int(idx)] for idx in selected.tolist()]
    summary = {
        "template_npz": args.template_npz,
        "embedding_npz": args.embedding_npz,
        "output": args.output,
        "sentence_key": args.sentence_key,
        "template_input_key": args.template_input_key,
        "embedding_input_key": args.embedding_input_key,
        "template_examples": int(template_n),
        "embedding_examples": int(embedding_n),
        "embedding_shape": list(replacement_embeddings.shape),
        "prefer_source_regex": args.prefer_source_regex,
        "preferred_count": int(preferred_count),
        "fallback_duplicate_count": int(fallback_duplicate_count),
        "selected_source_counts": dict(Counter(selected_sources)),
        "template_schema": _schema(template, args.schema_key),
        "embedding_schema": _schema(embedding, args.schema_key),
        "sample_sentences": template_sentences[: args.max_examples],
    }
    output_arrays[args.schema_key] = np.asarray(json.dumps(summary))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
