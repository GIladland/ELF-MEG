#!/usr/bin/env python
"""Select candidate semantic-vector rows nearest to a query semantic corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--query-sample-size", type=int, default=8192)
    parser.add_argument("--candidate-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-mode", choices=["global", "query_balanced"], default="global")
    parser.add_argument("--per-query-candidates", type=int, default=64)
    parser.add_argument("--score-mode", choices=["max", "mean", "hybrid"], default="max")
    parser.add_argument("--hybrid-mean-weight", type=float, default=0.1)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str):
    if schema_key not in data.files:
        return None
    try:
        return json.loads(str(data[schema_key].tolist()))
    except json.JSONDecodeError:
        return str(data[schema_key].tolist())


def normalize_sentence_key(sentence: str) -> str:
    return " ".join(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(sentence).lower()))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_blocked_sentences(paths: list[str], sentence_key: str) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(normalize_sentence_key(sentence) for sentence in _strings(data[sentence_key]))
    return blocked


def maybe_normalize(array: np.ndarray, *, normalize: bool) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if not normalize:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def score_candidates(
    candidates: np.ndarray,
    queries: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    score_mode: str,
    hybrid_mean_weight: float,
) -> np.ndarray:
    scores = np.empty((candidates.shape[0],), dtype=np.float32)
    query_tensor = torch.as_tensor(queries, dtype=torch.float32, device=device)
    query_t = query_tensor.T.contiguous()
    query_centroid = F.normalize(query_tensor.mean(dim=0, keepdim=True), dim=-1)
    for start in range(0, candidates.shape[0], batch_size):
        end = min(start + batch_size, candidates.shape[0])
        candidate_tensor = torch.as_tensor(candidates[start:end], dtype=torch.float32, device=device)
        if score_mode == "mean":
            batch_scores = candidate_tensor @ query_centroid.T
            batch_scores = batch_scores.squeeze(-1)
        else:
            sims = candidate_tensor @ query_t
            batch_scores = sims.max(dim=1).values
            if score_mode == "hybrid":
                mean_scores = (candidate_tensor @ query_centroid.T).squeeze(-1)
                batch_scores = batch_scores + hybrid_mean_weight * mean_scores
        scores[start:end] = batch_scores.detach().cpu().numpy()
        print(f"scored {end}/{candidates.shape[0]}", flush=True)
    return scores


def score_candidates_query_balanced(
    candidates: np.ndarray,
    queries: np.ndarray,
    *,
    blocked_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    per_query_candidates: int,
) -> np.ndarray:
    if per_query_candidates <= 0:
        raise ValueError("--per-query-candidates must be positive for query_balanced mode.")
    query_tensor = torch.as_tensor(queries, dtype=torch.float32, device=device)
    n_queries = int(query_tensor.shape[0])
    k = min(per_query_candidates, max(1, candidates.shape[0]))
    top_scores = torch.full((n_queries, k), -float("inf"), dtype=torch.float32, device=device)
    top_indices = torch.full((n_queries, k), -1, dtype=torch.long, device=device)
    for start in range(0, candidates.shape[0], batch_size):
        end = min(start + batch_size, candidates.shape[0])
        candidate_tensor = torch.as_tensor(candidates[start:end], dtype=torch.float32, device=device)
        sims = query_tensor @ candidate_tensor.T
        local_blocked = blocked_mask[start:end]
        if local_blocked.any():
            blocked_tensor = torch.as_tensor(local_blocked, dtype=torch.bool, device=device)
            sims[:, blocked_tensor] = -float("inf")
        batch_k = min(k, end - start)
        batch_scores, batch_local_indices = torch.topk(sims, k=batch_k, dim=1)
        batch_indices = batch_local_indices.to(torch.long) + start
        merged_scores = torch.cat([top_scores, batch_scores], dim=1)
        merged_indices = torch.cat([top_indices, batch_indices], dim=1)
        top_scores, merged_positions = torch.topk(merged_scores, k=k, dim=1)
        top_indices = torch.gather(merged_indices, 1, merged_positions)
        print(f"query_balanced_scored {end}/{candidates.shape[0]}", flush=True)

    flat_indices = top_indices.detach().cpu().numpy().reshape(-1)
    flat_scores = top_scores.detach().cpu().numpy().reshape(-1)
    valid = (flat_indices >= 0) & np.isfinite(flat_scores)
    scores = np.full((candidates.shape[0],), -np.inf, dtype=np.float32)
    if valid.any():
        np.maximum.at(scores, flat_indices[valid], flat_scores[valid].astype(np.float32))
    return scores


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    candidate_data = np.load(args.candidate_npz, allow_pickle=True)
    query_data = np.load(args.query_npz, allow_pickle=True)
    candidate_sentences = np.asarray(_strings(candidate_data[args.sentence_key]), dtype=object)
    query_sentences = np.asarray(_strings(query_data[args.sentence_key]), dtype=object)
    candidates = maybe_normalize(
        candidate_data[args.input_key],
        normalize=not args.no_normalize,
    )
    queries = maybe_normalize(
        query_data[args.input_key],
        normalize=not args.no_normalize,
    )
    if args.query_sample_size > 0 and len(queries) > args.query_sample_size:
        query_indices = np.sort(rng.choice(len(queries), size=args.query_sample_size, replace=False))
        queries_for_score = queries[query_indices]
    else:
        query_indices = np.arange(len(queries), dtype=np.int64)
        queries_for_score = queries

    blocked = load_blocked_sentences(args.blocked_npz, args.sentence_key)
    blocked_mask = np.asarray(
        [normalize_sentence_key(sentence) in blocked for sentence in candidate_sentences.tolist()],
        dtype=bool,
    )
    if args.selection_mode == "query_balanced":
        scores = score_candidates_query_balanced(
            candidates,
            queries_for_score,
            blocked_mask=blocked_mask,
            device=device,
            batch_size=args.candidate_batch_size,
            per_query_candidates=args.per_query_candidates,
        )
    else:
        scores = score_candidates(
            candidates,
            queries_for_score,
            device=device,
            batch_size=args.candidate_batch_size,
            score_mode=args.score_mode,
            hybrid_mean_weight=args.hybrid_mean_weight,
        )
        scores[blocked_mask] = -np.inf
    selectable = int(np.isfinite(scores).sum())
    if selectable <= 0:
        raise RuntimeError("No candidate rows remained after blocking.")
    count = min(args.count, selectable) if args.count > 0 else selectable
    top_unsorted = np.argpartition(-scores, count - 1)[:count]
    selected = top_unsorted[np.argsort(-scores[top_unsorted], kind="stable")]

    rows = (
        candidate_data["rows"][selected]
        if "rows" in candidate_data.files
        else np.asarray([{} for _ in range(len(selected))], dtype=object)
    )
    source_npz = np.asarray([args.candidate_npz] * len(selected), dtype=object)
    split = np.asarray(["train"] * len(selected), dtype=object)
    selected_scores = scores[selected].astype(np.float32)
    selected_sentences = candidate_sentences[selected]
    selected_embeddings = candidates[selected]

    summary = {
        "output": args.output,
        "candidate_npz": args.candidate_npz,
        "query_npz": args.query_npz,
        "blocked_npz": args.blocked_npz,
        "candidate_count": int(len(candidate_sentences)),
        "query_count": int(len(query_sentences)),
        "query_sample_size": int(len(queries_for_score)),
        "query_sample_indices_sha1": __import__("hashlib").sha1(query_indices.tobytes()).hexdigest(),
        "blocked_candidates": int(blocked_mask.sum()),
        "selectable_candidates": selectable,
        "selected_count": int(len(selected)),
        "selection_mode": args.selection_mode,
        "per_query_candidates": args.per_query_candidates,
        "score_mode": args.score_mode,
        "hybrid_mean_weight": args.hybrid_mean_weight,
        "score_min": float(np.min(selected_scores)),
        "score_mean": float(np.mean(selected_scores)),
        "score_max": float(np.max(selected_scores)),
        "candidate_schema": _schema(candidate_data, args.schema_key),
        "query_schema": _schema(query_data, args.schema_key),
        "sample": [
            {"score": float(selected_scores[i]), "sentence": str(selected_sentences[i])}
            for i in range(min(10, len(selected_sentences)))
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        input_embeddings=selected_embeddings.astype(np.float32),
        sentence=selected_sentences,
        rows=rows,
        source_npz=source_npz,
        split=split,
        selection_score=selected_scores,
        source_index=selected.astype(np.int64),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
