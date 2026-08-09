#!/usr/bin/env python
"""Analyze lexical/semantic nearest-neighbor coverage in semantic-vector NPZs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", action="append", required=True)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--output-json", default="")
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--val-num-examples", type=int, default=0)
    parser.add_argument("--train-query-sample-size", type=int, default=4096)
    parser.add_argument("--val-query-sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exclude-identical-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--examples-per-set", type=int, default=5)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def normalize_text(sentence: str) -> str:
    return re.sub(r"\s+", " ", str(sentence).strip()).casefold()


def word_set(sentence: str) -> set[str]:
    return set(word.lower() for word in WORD_RE.findall(str(sentence)))


def word_metrics(target: str, neighbor: str) -> dict[str, float]:
    target_words = word_set(target)
    neighbor_words = word_set(neighbor)
    if not target_words and not neighbor_words:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0, "jaccard": 1.0}
    overlap = len(target_words & neighbor_words)
    recall = overlap / max(1, len(target_words))
    precision = overlap / max(1, len(neighbor_words))
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    union = len(target_words | neighbor_words)
    jaccard = overlap / max(1, union)
    return {"recall": recall, "precision": precision, "f1": f1, "jaccard": jaccard}


def summarize(values: Sequence[float]) -> dict[str, float]:
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


def split_indices(data: np.lib.npyio.NpzFile, total_n: int, split_key: str, val_num_examples: int) -> tuple[np.ndarray, np.ndarray]:
    if split_key in data.files:
        split = _strings(data[split_key])
        train = np.asarray([idx for idx, value in enumerate(split) if value == "train"], dtype=np.int64)
        val = np.asarray([idx for idx, value in enumerate(split) if value == "val"], dtype=np.int64)
        if len(train) and len(val):
            return train, val
    if val_num_examples <= 0:
        raise ValueError("NPZ has no usable split field; pass --val-num-examples.")
    val_n = min(val_num_examples, max(0, total_n - 1))
    return (
        np.arange(0, total_n - val_n, dtype=np.int64),
        np.arange(total_n - val_n, total_n, dtype=np.int64),
    )


def group_for_row(row: object) -> str:
    if isinstance(row, dict):
        corpus = str(row.get("corpus", "") or row.get("dataset", "") or row.get("coverage_source_name", ""))
        if corpus.startswith("Sherlock"):
            return "Sherlock"
        return corpus or "unknown"
    return "unknown"


def nearest_indices(
    *,
    query_vectors: np.ndarray,
    target_vectors: np.ndarray,
    query_global_indices: np.ndarray,
    target_global_indices: np.ndarray,
    query_text_keys: list[str],
    target_text_keys: list[str],
    batch_size: int,
    device: torch.device,
    exclude_self: bool,
    exclude_identical_text: bool,
) -> tuple[np.ndarray, np.ndarray]:
    target_tensor = torch.as_tensor(target_vectors, dtype=torch.float32, device=device).T.contiguous()
    best_scores = np.full((len(query_vectors),), -np.inf, dtype=np.float32)
    best_local = np.full((len(query_vectors),), -1, dtype=np.int64)
    target_index_lookup = {int(index): local for local, index in enumerate(target_global_indices.tolist())}
    text_to_target_locals: dict[str, list[int]] = defaultdict(list)
    if exclude_identical_text:
        for local, key in enumerate(target_text_keys):
            text_to_target_locals[key].append(local)

    for start in range(0, len(query_vectors), batch_size):
        end = min(start + batch_size, len(query_vectors))
        query_tensor = torch.as_tensor(query_vectors[start:end], dtype=torch.float32, device=device)
        sims = query_tensor @ target_tensor
        for local_query, global_query_idx in enumerate(query_global_indices[start:end].tolist()):
            if exclude_self and global_query_idx in target_index_lookup:
                sims[local_query, target_index_lookup[int(global_query_idx)]] = -float("inf")
            if exclude_identical_text:
                for target_local in text_to_target_locals.get(query_text_keys[start + local_query], []):
                    sims[local_query, target_local] = -float("inf")
        scores, indices = sims.max(dim=1)
        best_scores[start:end] = scores.detach().cpu().numpy()
        best_local[start:end] = indices.detach().cpu().numpy().astype(np.int64)
        print(f"nearest {end}/{len(query_vectors)}", flush=True)
    return best_local, best_scores


def evaluate_direction(
    *,
    label: str,
    sentences: list[str],
    rows: np.ndarray | None,
    embeddings: np.ndarray,
    query_indices: np.ndarray,
    target_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    exclude_self: bool,
    exclude_identical_text: bool,
) -> dict:
    target_vectors = embeddings[target_indices]
    query_vectors = embeddings[query_indices]
    target_norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    target_vectors = target_vectors / np.clip(target_norms, 1e-12, None)
    query_vectors = query_vectors / np.clip(query_norms, 1e-12, None)
    text_keys = [normalize_text(sentence) for sentence in sentences]
    best_local, best_scores = nearest_indices(
        query_vectors=query_vectors,
        target_vectors=target_vectors,
        query_global_indices=query_indices,
        target_global_indices=target_indices,
        query_text_keys=[text_keys[int(idx)] for idx in query_indices],
        target_text_keys=[text_keys[int(idx)] for idx in target_indices],
        batch_size=batch_size,
        device=device,
        exclude_self=exclude_self,
        exclude_identical_text=exclude_identical_text,
    )

    records = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for local_query, local_target in enumerate(best_local.tolist()):
        query_idx = int(query_indices[local_query])
        target_idx = int(target_indices[local_target]) if local_target >= 0 else -1
        metrics = word_metrics(sentences[query_idx], sentences[target_idx]) if target_idx >= 0 else {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
        }
        row = rows[query_idx] if rows is not None and query_idx < len(rows) else {}
        group = group_for_row(row)
        record = {
            "query_index": query_idx,
            "neighbor_index": target_idx,
            "cosine": float(best_scores[local_query]),
            **metrics,
            "query": sentences[query_idx],
            "neighbor": sentences[target_idx] if target_idx >= 0 else "",
            "group": group,
        }
        records.append(record)
        grouped[group].append(record)

    result = {
        "label": label,
        "query_count": int(len(query_indices)),
        "target_count": int(len(target_indices)),
        "cosine": summarize([record["cosine"] for record in records]),
        "word_recall": summarize([record["recall"] for record in records]),
        "word_f1": summarize([record["f1"] for record in records]),
        "word_precision": summarize([record["precision"] for record in records]),
        "word_jaccard": summarize([record["jaccard"] for record in records]),
        "groups": {},
        "high_examples": sorted(records, key=lambda item: item["recall"], reverse=True)[:5],
        "low_examples": sorted(records, key=lambda item: item["recall"])[:5],
    }
    for group, group_records in sorted(grouped.items()):
        result["groups"][group] = {
            "query_count": int(len(group_records)),
            "word_recall": summarize([record["recall"] for record in group_records]),
            "word_f1": summarize([record["f1"] for record in group_records]),
            "cosine": summarize([record["cosine"] for record in group_records]),
        }
    return result


def analyze_one(path: str, name: str, args: argparse.Namespace, device: torch.device) -> dict:
    rng = np.random.default_rng(args.seed)
    data = np.load(path, allow_pickle=True)
    embeddings = np.asarray(data[args.input_key], dtype=np.float32)
    sentences = _strings(data[args.sentence_key])
    rows = data[args.rows_key] if args.rows_key in data.files else None
    train_indices, val_indices = split_indices(data, len(sentences), args.split_key, args.val_num_examples)

    train_query_indices = train_indices
    if args.train_query_sample_size > 0 and len(train_query_indices) > args.train_query_sample_size:
        train_query_indices = np.sort(
            rng.choice(train_query_indices, size=args.train_query_sample_size, replace=False)
        ).astype(np.int64)
    val_query_indices = val_indices
    if args.val_query_sample_size > 0 and len(val_query_indices) > args.val_query_sample_size:
        val_query_indices = np.sort(
            rng.choice(val_query_indices, size=args.val_query_sample_size, replace=False)
        ).astype(np.int64)

    print(f"=== {name}: train_n={len(train_indices)} val_n={len(val_indices)}", flush=True)
    train_to_train = evaluate_direction(
        label="train_to_train_leave_one",
        sentences=sentences,
        rows=rows,
        embeddings=embeddings,
        query_indices=train_query_indices,
        target_indices=train_indices,
        batch_size=args.batch_size,
        device=device,
        exclude_self=True,
        exclude_identical_text=args.exclude_identical_text,
    )
    val_to_train = evaluate_direction(
        label="val_to_train",
        sentences=sentences,
        rows=rows,
        embeddings=embeddings,
        query_indices=val_query_indices,
        target_indices=train_indices,
        batch_size=args.batch_size,
        device=device,
        exclude_self=False,
        exclude_identical_text=args.exclude_identical_text,
    )
    return {
        "name": name,
        "npz": path,
        "total_n": int(len(sentences)),
        "train_n": int(len(train_indices)),
        "val_n": int(len(val_indices)),
        "train_query_n": int(len(train_query_indices)),
        "val_query_n": int(len(val_query_indices)),
        "train_to_train": train_to_train,
        "val_to_train": val_to_train,
    }


def print_brief(result: dict) -> None:
    def line(section: str) -> str:
        item = result[section]
        recall = item["word_recall"]
        f1 = item["word_f1"]
        cosine = item["cosine"]
        return (
            f"{section}: recall_mean={recall.get('mean', float('nan')):.3f} "
            f"recall_ge0.7={recall.get('ge_0.7', float('nan')):.3f} "
            f"f1_mean={f1.get('mean', float('nan')):.3f} "
            f"cos_mean={cosine.get('mean', float('nan')):.3f}"
        )
    print(f"RESULT {result['name']} train_n={result['train_n']} val_n={result['val_n']}")
    print("  " + line("train_to_train"))
    print("  " + line("val_to_train"))


def main() -> None:
    args = parse_args()
    names = list(args.name)
    while len(names) < len(args.npz):
        names.append(Path(args.npz[len(names)]).stem)
    if len(names) > len(args.npz):
        raise ValueError("More --name values than --npz values.")
    device = resolve_device(args.device)
    results = [analyze_one(path, name, args, device) for path, name in zip(args.npz, names)]
    for result in results:
        print_brief(result)
    output = {"results": results}
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {output_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
