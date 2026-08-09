#!/usr/bin/env python
"""Append retrieved train-neighbor text features to semantic-vector NPZ rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from modules.t5_encoder import get_encoder
from scripts.meg_context_overfit import encode_text_batched, mean_pool_latents, tokenize_sentences


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--val-num-examples", type=int, default=0)
    parser.add_argument("--neighbor-top-k", type=int, default=5)
    parser.add_argument(
        "--bank-npz",
        action="append",
        default=[],
        help="Optional retrieval bank NPZ. If omitted, retrieve from the source NPZ train split.",
    )
    parser.add_argument("--bank-input-key", default="input_embeddings")
    parser.add_argument("--bank-sentence-key", default="sentence")
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--t5-batch-size", type=int, default=128)
    parser.add_argument("--encoder-model-name", default="t5-small")
    parser.add_argument("--latent-mean", type=float, default=0.0)
    parser.add_argument("--latent-std", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exclude-identical-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-append-query-vector", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    try:
        return json.loads(str(data[schema_key].tolist()))
    except json.JSONDecodeError:
        return str(data[schema_key].tolist())


def normalize_text(sentence: str) -> str:
    return " ".join(token.lower() for token in WORD_RE.findall(str(sentence)))


def word_counts(sentence: str) -> Counter[str]:
    return Counter(token.lower() for token in WORD_RE.findall(str(sentence)))


def word_f1(query: str, neighbor: str) -> float:
    query_counts = word_counts(query)
    neighbor_counts = word_counts(neighbor)
    if not query_counts or not neighbor_counts:
        return 0.0
    overlap = sum(min(count, neighbor_counts.get(token, 0)) for token, count in query_counts.items())
    precision = overlap / max(1, sum(neighbor_counts.values()))
    recall = overlap / max(1, sum(query_counts.values()))
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "ge_0.5": float(np.mean(arr >= 0.5)),
        "ge_0.7": float(np.mean(arr >= 0.7)),
        "ge_0.8": float(np.mean(arr >= 0.8)),
    }


def split_indices(
    data: np.lib.npyio.NpzFile,
    total_n: int,
    split_key: str,
    val_num_examples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if split_key in data.files:
        split = _strings(data[split_key])
        train = np.asarray([idx for idx, value in enumerate(split[:total_n]) if value == "train"], dtype=np.int64)
        val = np.asarray([idx for idx, value in enumerate(split[:total_n]) if value == "val"], dtype=np.int64)
        if len(train) and len(val):
            return train, val
    if val_num_examples <= 0:
        raise ValueError("NPZ has no usable split field; pass --val-num-examples.")
    val_n = min(val_num_examples, max(0, total_n - 1))
    return (
        np.arange(0, total_n - val_n, dtype=np.int64),
        np.arange(total_n - val_n, total_n, dtype=np.int64),
    )


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def load_bank_npzs(
    *,
    paths: list[str],
    input_key: str,
    sentence_key: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    vectors = []
    sentences: list[str] = []
    source_paths: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            bank_vectors = np.asarray(data[input_key], dtype=np.float32)
            bank_sentences = _strings(data[sentence_key])
        if bank_vectors.shape[0] != len(bank_sentences):
            raise ValueError(
                f"Bank vector/sentence row mismatch in {path}: "
                f"{bank_vectors.shape[0]} vs {len(bank_sentences)}"
            )
        vectors.append(bank_vectors)
        sentences.extend(bank_sentences)
        source_paths.extend([path] * len(bank_sentences))
    if not vectors:
        raise ValueError("No retrieval bank vectors loaded.")
    return np.concatenate(vectors, axis=0), sentences, np.asarray(source_paths, dtype=object)


def find_neighbors(
    *,
    query_vectors: np.ndarray,
    query_sentences: list[str],
    bank_vectors: np.ndarray,
    bank_sentences: list[str],
    top_k: int,
    batch_size: int,
    device: torch.device,
    exclude_identical_text: bool,
    self_lookup: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_query = normalize_rows(query_vectors)
    normalized_bank = normalize_rows(bank_vectors)
    bank_tensor = torch.as_tensor(normalized_bank, dtype=torch.float32, device=device).T.contiguous()
    query_indices = np.arange(query_vectors.shape[0], dtype=np.int64)
    text_to_train_locals: dict[str, list[int]] = defaultdict(list)
    if exclude_identical_text:
        for local, sentence in enumerate(bank_sentences):
            text_to_train_locals[normalize_text(sentence)].append(local)

    neighbor_local = np.full((query_vectors.shape[0], top_k), -1, dtype=np.int64)
    neighbor_scores = np.full((query_vectors.shape[0], top_k), -np.inf, dtype=np.float32)
    self_lookup = self_lookup or {}
    for start in range(0, query_vectors.shape[0], batch_size):
        end = min(start + batch_size, query_vectors.shape[0])
        query_tensor = torch.as_tensor(normalized_query[start:end], dtype=torch.float32, device=device)
        sims = query_tensor @ bank_tensor
        for row, global_idx in enumerate(query_indices[start:end].tolist()):
            train_local = self_lookup.get(int(global_idx))
            if train_local is not None:
                sims[row, train_local] = -float("inf")
            if exclude_identical_text:
                key = normalize_text(query_sentences[int(global_idx)])
                for matching_local in text_to_train_locals.get(key, []):
                    sims[row, matching_local] = -float("inf")
        k = min(top_k, sims.shape[1])
        scores, indices = torch.topk(sims, k=k, dim=1)
        neighbor_local[start:end, :k] = indices.detach().cpu().numpy().astype(np.int64)
        neighbor_scores[start:end, :k] = scores.detach().cpu().numpy().astype(np.float32)
        print(f"retrieved {end}/{query_vectors.shape[0]}", flush=True)
    return neighbor_local, neighbor_scores


def encode_t5_pooled(
    *,
    sentences: list[str],
    encoder_model_name: str,
    latent_mean: float,
    latent_std: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask = tokenize_sentences(tokenizer, sentences)
    _encoder_config, encoder = get_encoder(encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    latents = encode_text_batched(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder=encoder,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=device,
        batch_size=batch_size,
    )
    pooled = mean_pool_latents(latents, attention_mask.to(torch.float32))
    return pooled.detach().cpu().numpy().astype(np.float32)


def first_dim_array(data: np.lib.npyio.NpzFile, key: str, total_n: int) -> np.ndarray | None:
    if key not in data.files:
        return None
    value = data[key]
    if getattr(value, "shape", None) and value.shape[0] >= total_n:
        return value[:total_n]
    return None


def main() -> None:
    args = parse_args()
    if args.neighbor_top_k <= 0:
        raise ValueError("--neighbor-top-k must be positive.")

    data = np.load(args.npz, allow_pickle=True)
    vectors = np.asarray(data[args.input_key], dtype=np.float32)
    sentences = _strings(data[args.sentence_key])
    total_n = len(sentences)
    if vectors.shape[0] != total_n:
        raise ValueError(f"Vector/sentence row mismatch: {vectors.shape[0]} vs {total_n}")
    train_indices, val_indices = split_indices(data, total_n, args.split_key, args.val_num_examples)
    device = resolve_device(args.device)
    print(
        f"Loaded total_n={total_n} train_n={len(train_indices)} val_n={len(val_indices)} "
        f"dim={vectors.shape[1]} top_k={args.neighbor_top_k} device={device}",
        flush=True,
    )

    if args.bank_npz:
        bank_vectors, bank_sentences, bank_source_npz = load_bank_npzs(
            paths=args.bank_npz,
            input_key=args.bank_input_key,
            sentence_key=args.bank_sentence_key,
        )
        self_lookup: dict[int, int] = {}
    else:
        bank_vectors = vectors[train_indices]
        bank_sentences = [sentences[int(idx)] for idx in train_indices.tolist()]
        bank_source_npz = np.asarray([args.npz] * len(bank_sentences), dtype=object)
        self_lookup = {int(global_idx): local for local, global_idx in enumerate(train_indices.tolist())}

    neighbor_local, neighbor_scores = find_neighbors(
        query_vectors=vectors,
        query_sentences=sentences,
        bank_vectors=bank_vectors,
        bank_sentences=bank_sentences,
        top_k=args.neighbor_top_k,
        batch_size=args.retrieval_batch_size,
        device=device,
        exclude_identical_text=args.exclude_identical_text,
        self_lookup=self_lookup,
    )
    unique_neighbor_local, inverse = np.unique(neighbor_local.reshape(-1), return_inverse=True)
    unique_neighbor_sentences = [bank_sentences[int(idx)] for idx in unique_neighbor_local.tolist()]
    unique_t5_pooled = encode_t5_pooled(
        sentences=unique_neighbor_sentences,
        encoder_model_name=args.encoder_model_name,
        latent_mean=args.latent_mean,
        latent_std=args.latent_std,
        device=device,
        batch_size=args.t5_batch_size,
    )
    neighbor_features = unique_t5_pooled[inverse].reshape(total_n, -1)

    feature_parts = []
    if not args.no_append_query_vector:
        feature_parts.append(vectors)
    feature_parts.append(neighbor_features)
    if args.include_scores:
        feature_parts.append(neighbor_scores.astype(np.float32))
    augmented_vectors = np.concatenate(feature_parts, axis=1).astype(np.float32)

    neighbor_sentences = np.asarray(
        [[bank_sentences[int(local_idx)] for local_idx in row] for row in neighbor_local],
        dtype=object,
    )
    top1_f1 = [
        word_f1(sentences[row_idx], neighbor_sentences[row_idx, 0])
        for row_idx in range(total_n)
    ]
    best_topk_f1 = [
        max(word_f1(sentences[row_idx], neighbor_sentences[row_idx, col]) for col in range(args.neighbor_top_k))
        for row_idx in range(total_n)
    ]
    val_set = set(int(idx) for idx in val_indices.tolist())
    train_set = set(int(idx) for idx in train_indices.tolist())
    train_top1 = [top1_f1[idx] for idx in train_set]
    val_top1 = [top1_f1[idx] for idx in val_set]
    train_best = [best_topk_f1[idx] for idx in train_set]
    val_best = [best_topk_f1[idx] for idx in val_set]

    rows = first_dim_array(data, args.rows_key, total_n)
    source_npz = first_dim_array(data, "source_npz", total_n)
    split = first_dim_array(data, args.split_key, total_n)
    if split is None:
        split = np.asarray(["train" if idx in train_set else "val" for idx in range(total_n)], dtype=object)

    summary = {
        "output": args.output,
        "source_npz": args.npz,
        "input_key": args.input_key,
        "sentence_key": args.sentence_key,
        "total_examples": int(total_n),
        "train_examples": int(len(train_indices)),
        "val_examples": int(len(val_indices)),
        "source_embedding_shape": list(vectors.shape),
        "bank_npz": args.bank_npz if args.bank_npz else [args.npz],
        "bank_examples": int(len(bank_sentences)),
        "bank_retrieval_mode": "external_bank" if args.bank_npz else "source_train_split",
        "unique_retrieved_bank_examples": int(len(unique_neighbor_local)),
        "augmented_embedding_shape": list(augmented_vectors.shape),
        "neighbor_top_k": int(args.neighbor_top_k),
        "neighbor_feature": f"{args.encoder_model_name}_mean_pooled_latent",
        "encoder_model_name": args.encoder_model_name,
        "latent_mean": float(args.latent_mean),
        "latent_std": float(args.latent_std),
        "exclude_identical_text": bool(args.exclude_identical_text),
        "include_scores": bool(args.include_scores),
        "append_query_vector": not args.no_append_query_vector,
        "retrieval_word_f1": {
            "train_top1": summarize(train_top1),
            "train_best_topk": summarize(train_best),
            "val_top1": summarize(val_top1),
            "val_best_topk": summarize(val_best),
        },
        "sample_val_neighbors": [
            {
                "query_index": int(idx),
                "query": sentences[int(idx)],
                "neighbors": [
                    {
                        "bank_index": int(neighbor_local[int(idx), col]),
                        "bank_npz": str(bank_source_npz[int(neighbor_local[int(idx), col])]),
                        "score": float(neighbor_scores[int(idx), col]),
                        "word_f1": word_f1(sentences[int(idx)], neighbor_sentences[int(idx), col]),
                        "sentence": str(neighbor_sentences[int(idx), col]),
                    }
                    for col in range(args.neighbor_top_k)
                ],
            }
            for idx in val_indices[: min(10, len(val_indices))]
        ],
        "source_schema": _schema(data, args.schema_key),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, np.ndarray] = {
        "input_embeddings": augmented_vectors,
        "sentence": np.asarray(sentences, dtype=object),
        "split": split,
        "retrieval_neighbor_indices": neighbor_local.astype(np.int64),
        "retrieval_neighbor_scores": neighbor_scores.astype(np.float32),
        "retrieval_neighbor_sentences": neighbor_sentences,
        "retrieval_neighbor_source_npz": bank_source_npz[neighbor_local],
        "schema_json": np.asarray(json.dumps(summary)),
    }
    if rows is not None:
        save_kwargs["rows"] = rows
    if source_npz is not None:
        save_kwargs["source_npz"] = source_npz
    np.savez_compressed(output, **save_kwargs)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
