#!/usr/bin/env python
"""Re-embed an NPZ sentence field with OpenAI embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np


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
    parser.add_argument("--model", default="text-embedding-ada-002")
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--initial-backoff", type=float, default=2.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def _split_counts(data: np.lib.npyio.NpzFile, total: int) -> tuple[int | None, int | None]:
    if "split" in data.files:
        split = _strings(data["split"])
        train = sum(item == "train" for item in split)
        val = sum(item == "val" for item in split)
        if train + val == total:
            return train, val
    schema = _schema(data, "schema_json")
    if isinstance(schema, dict):
        train = schema.get("train_examples")
        val = schema.get("val_examples")
        if isinstance(train, int) and isinstance(val, int):
            return train, val
    return None, None


def _embedding_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(f"{output.stem}.openai_embeddings.tmp.npy"),
        output.with_name(f"{output.stem}.openai_progress.json"),
    )


def _client_embed(
    texts: list[str],
    *,
    model: str,
    api_key_env: str,
) -> np.ndarray:
    try:
        from openai import OpenAI
    except Exception:
        return _http_embed(texts, model=model, api_key_env=api_key_env)

    client = OpenAI(api_key=os.environ.get(api_key_env))
    try:
        response = client.embeddings.create(model=model, input=texts, encoding_format="float")
    except TypeError:
        response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.asarray([item.embedding for item in ordered], dtype=np.float32)


def _http_embed(
    texts: list[str],
    *,
    model: str,
    api_key_env: str,
) -> np.ndarray:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env}; cannot call OpenAI embeddings API.")
    payload = json.dumps({"model": model, "input": texts, "encoding_format": "float"}).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    ordered = sorted(data["data"], key=lambda item: item["index"])
    return np.asarray([item["embedding"] for item in ordered], dtype=np.float32)


def embed_with_retries(
    texts: list[str],
    *,
    model: str,
    api_key_env: str,
    max_retries: int,
    initial_backoff: float,
) -> np.ndarray:
    delay = initial_backoff
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _client_embed(texts, model=model, api_key_env=api_key_env)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_for = delay * (1.0 + 0.1 * attempt)
            print(f"embedding batch failed ({type(exc).__name__}); retrying in {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
            delay = min(delay * 2.0, 120.0)
    raise RuntimeError("OpenAI embedding request failed after retries") from last_error


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, np.maximum(norms, 1e-12), out=values)
    return values


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"Output exists, reusing: {output}")
        return
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Missing {args.api_key_env}; cannot call OpenAI embeddings API.")

    data = np.load(args.input_npz, allow_pickle=True)
    if args.sentence_key not in data.files:
        raise KeyError(f"{args.input_npz} has no sentence key {args.sentence_key!r}")
    sentences = _strings(data[args.sentence_key])
    embedding_inputs = [f"{args.input_prefix}{sentence}" for sentence in sentences]
    total = len(embedding_inputs)
    train_examples, val_examples = _split_counts(data, total)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_embeddings_path, progress_path = _embedding_paths(output)

    progress: dict[str, Any] = {}
    if tmp_embeddings_path.exists() and progress_path.exists() and not args.overwrite:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        embeddings = np.load(tmp_embeddings_path, mmap_mode="r+")
        done = int(progress.get("rows_done", 0))
        if embeddings.shape[0] != total:
            raise ValueError(
                f"Resume file row count {embeddings.shape[0]} does not match input count {total}: {tmp_embeddings_path}"
            )
        print(f"Resuming {tmp_embeddings_path} from row {done}/{total}", flush=True)
    else:
        first_batch = embedding_inputs[: max(1, min(args.batch_size, total))]
        first_embeddings = embed_with_retries(
            first_batch,
            model=args.model,
            api_key_env=args.api_key_env,
            max_retries=args.max_retries,
            initial_backoff=args.initial_backoff,
        )
        if not args.no_normalize:
            first_embeddings = normalize_rows(first_embeddings)
        embeddings = np.lib.format.open_memmap(
            tmp_embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(total, int(first_embeddings.shape[1])),
        )
        end = len(first_batch)
        embeddings[:end] = first_embeddings
        embeddings.flush()
        done = end
        progress = {
            "input_npz": args.input_npz,
            "output": args.output,
            "model": args.model,
            "rows_total": total,
            "rows_done": done,
            "embedding_dim": int(first_embeddings.shape[1]),
            "normalized": not args.no_normalize,
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"embedded rows 0:{end} / {total}", flush=True)

    batch_size = max(1, args.batch_size)
    for start in range(done, total, batch_size):
        end = min(start + batch_size, total)
        batch = embed_with_retries(
            embedding_inputs[start:end],
            model=args.model,
            api_key_env=args.api_key_env,
            max_retries=args.max_retries,
            initial_backoff=args.initial_backoff,
        )
        if not args.no_normalize:
            batch = normalize_rows(batch)
        embeddings[start:end] = batch
        embeddings.flush()
        progress["rows_done"] = end
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"embedded rows {start}:{end} / {total}", flush=True)

    output_arrays = {key: data[key] for key in data.files if key not in {args.input_key, args.schema_key}}
    output_arrays[args.input_key] = np.asarray(embeddings, dtype=np.float32)
    summary = {
        "output": args.output,
        "input_npz": args.input_npz,
        "input_key": args.input_key,
        "sentence_key": args.sentence_key,
        "count": total,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "embedding_model_name": args.model,
        "embedding_shape": list(embeddings.shape),
        "input_prefix": args.input_prefix,
        "normalized": not args.no_normalize,
        "source_schema": _schema(data, args.schema_key),
        "sample_sentences": sentences[:5],
    }
    output_arrays[args.schema_key] = np.asarray(json.dumps(summary))
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with tmp_output.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    os.replace(tmp_output, output)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
