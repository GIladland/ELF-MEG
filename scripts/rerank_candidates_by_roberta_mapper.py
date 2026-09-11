#!/usr/bin/env python
"""Target-free candidate reranking via a train-only MiniLM-to-RoBERTa map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_generated_candidates_by_semantic import retrieval_metrics, text_metrics, words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument(
        "--train-input-npz",
        type=Path,
        default=None,
        help="Optional noisy/OOF semantic inputs aligned to the train text rows.",
    )
    parser.add_argument("--train-input-start", type=int, default=0)
    parser.add_argument("--val-semantic-npz", type=Path, required=True)
    parser.add_argument("--val-input-start", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=0)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=11725)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--story-key", default="story")
    parser.add_argument(
        "--input-delay-blocks",
        type=int,
        default=1,
        help="Mean-pool this many equal semantic delay blocks before ridge fitting.",
    )
    parser.add_argument("--model-name", default="roberta-large")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument(
        "--length-penalty",
        type=float,
        action="append",
        default=[],
        help=(
            "Optional global penalty multiplied by the absolute deviation from "
            "--target-word-count. May be repeated; every setting is written as a "
            "separate metrics JSON. Selection still never reads validation text."
        ),
    )
    parser.add_argument("--target-word-count", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def pool_delay_blocks(values: np.ndarray, blocks: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"semantic inputs must have two dimensions, got {values.shape}")
    if blocks <= 0 or values.shape[1] % blocks:
        raise ValueError(
            f"cannot split semantic dimension {values.shape[1]} into {blocks} delay blocks"
        )
    if blocks == 1:
        return values
    return values.reshape(values.shape[0], blocks, values.shape[1] // blocks).mean(axis=1)


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    *,
    tokenizer,
    model,
    layer: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[layer]
        mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        rows.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = np.linalg.solve(gram, centered_x.T @ centered_y)
    return weights.astype(np.float32), x_mean.astype(np.float32), y_mean.astype(np.float32)


def apply_ridge(
    x: np.ndarray, weights: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray
) -> np.ndarray:
    return normalize_rows((np.asarray(x, dtype=np.float32) - x_mean) @ weights + y_mean)


def grouped_train_validation_split(stories: Sequence[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(str(value) for value in stories)), dtype=object)
    if unique.size < 2:
        split = max(1, int(0.8 * len(stories)))
        return np.arange(split), np.arange(split, len(stories))
    generator = np.random.default_rng(seed)
    generator.shuffle(unique)
    val_count = max(1, int(round(0.2 * unique.size)))
    held_out = set(unique[:val_count].tolist())
    val = np.asarray([index for index, story in enumerate(stories) if str(story) in held_out])
    train = np.asarray([index for index, story in enumerate(stories) if str(story) not in held_out])
    return train, val


def select_with_length_penalty(
    scores: np.ndarray,
    candidates: Sequence[Sequence[str]],
    *,
    target_word_count: int,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select candidates using mapped semantics and a reference-free length prior."""
    if penalty < 0:
        raise ValueError("length penalty must be non-negative")
    candidate_lengths = np.asarray(
        [[len(words(text)) for text in run] for run in candidates], dtype=np.float32
    ).T
    if candidate_lengths.shape != scores.shape:
        raise ValueError(
            f"candidate length/score mismatch: {candidate_lengths.shape} != {scores.shape}"
        )
    adjusted_scores = scores - penalty * np.abs(candidate_lengths - target_word_count)
    return np.argmax(adjusted_scores, axis=1), adjusted_scores


def main() -> None:
    args = parse_args()
    with np.load(args.train_npz, allow_pickle=True) as data:
        default_train_x = data[args.embedding_key][: args.train_rows]
        train_text = [str(value) for value in data[args.sentence_key][: args.train_rows].tolist()]
        stories = (
            [str(value) for value in data[args.story_key][: args.train_rows].tolist()]
            if args.story_key in data
            else [str(index) for index in range(args.train_rows)]
        )
    if args.train_input_npz is None:
        train_x = default_train_x
        train_input_path = args.train_npz
    else:
        train_input_path = args.train_input_npz
        start = args.train_input_start
        stop = start + args.train_rows
        with np.load(args.train_input_npz, allow_pickle=True) as data:
            train_x = data[args.embedding_key][start:stop]
            if train_x.shape[0] != args.train_rows:
                raise ValueError(
                    f"train input slice has {train_x.shape[0]} rows, expected {args.train_rows}"
                )
            if args.sentence_key in data:
                input_text = [str(value) for value in data[args.sentence_key][start:stop].tolist()]
                if input_text != train_text:
                    raise ValueError("train input sentences do not align to train text rows")
    train_x = normalize_rows(pool_delay_blocks(train_x, args.input_delay_blocks))

    with np.load(args.val_semantic_npz, allow_pickle=True) as data:
        val_start = args.val_input_start
        val_stop = None if args.val_rows <= 0 else val_start + args.val_rows
        val_x = data[args.embedding_key][val_start:val_stop]
        val_targets = [
            str(value) for value in data[args.sentence_key][val_start:val_stop].tolist()
        ]
    val_x = normalize_rows(pool_delay_blocks(val_x, args.input_delay_blocks))

    candidates = []
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if [str(value) for value in payload["targets"]] != val_targets:
            raise ValueError(f"target row mismatch: {path}")
        candidates.append([str(value) for value in payload["generated"]])

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model_name, local_files_only=args.local_files_only).to(device).eval()
    if args.cache is not None and args.cache.exists():
        train_roberta = np.load(args.cache)
        if train_roberta.shape[0] != args.train_rows:
            raise ValueError(f"stale RoBERTa cache shape: {train_roberta.shape}")
    else:
        train_roberta = encode_texts(
            train_text,
            tokenizer=tokenizer,
            model=model,
            layer=args.layer,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.cache, train_roberta)

    fit_indices, held_out_indices = grouped_train_validation_split(stories, args.seed)
    ridge_grid = (0.01, 0.1, 1.0, 10.0, 100.0)
    ridge_scores = {}
    for ridge in ridge_grid:
        fit = fit_ridge(train_x[fit_indices], train_roberta[fit_indices], ridge)
        predicted = apply_ridge(train_x[held_out_indices], *fit)
        ridge_scores[ridge] = float(
            np.mean(np.sum(predicted * normalize_rows(train_roberta[held_out_indices]), axis=1))
        )
    selected_ridge = max(ridge_grid, key=lambda value: ridge_scores[value])
    mapper = fit_ridge(train_x, train_roberta, selected_ridge)
    predicted_val_roberta = apply_ridge(val_x, *mapper)

    flat_candidates = [text for run in candidates for text in run]
    flat_embeddings = encode_texts(
        flat_candidates,
        tokenizer=tokenizer,
        model=model,
        layer=args.layer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    candidate_embeddings = flat_embeddings.reshape(len(candidates), len(val_targets), -1).transpose(1, 0, 2)
    scores = np.einsum("nkd,nd->nk", candidate_embeddings, predicted_val_roberta)
    # Target embeddings are computed only after selection and only for reporting.
    target_embeddings = encode_texts(
        val_targets,
        tokenizer=tokenizer,
        model=model,
        layer=args.layer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    penalties = [0.0]
    penalties.extend(value for value in args.length_penalty if value != 0.0)
    seen_penalties = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for penalty in penalties:
        if penalty in seen_penalties:
            continue
        seen_penalties.add(penalty)
        selected_indices, adjusted_scores = select_with_length_penalty(
            scores,
            candidates,
            target_word_count=args.target_word_count,
            penalty=penalty,
        )
        selected_text = [candidates[index][row] for row, index in enumerate(selected_indices)]
        selected_embeddings = candidate_embeddings[np.arange(len(val_targets)), selected_indices]
        quality = text_metrics(selected_text, val_targets)
        retrieval = retrieval_metrics(selected_embeddings, target_embeddings)
        method = "cosine(mapped train-only MiniLM brain vector, RoBERTa-layer candidate mean)"
        if penalty:
            method += (
                f" - {penalty:g} * abs(generated_word_count - "
                f"{args.target_word_count})"
            )
        result = {
            "step": 0,
            "epoch": 0.0,
            "split": "validation",
            "eval_num_examples": len(val_targets),
            "num_eval_examples": len(val_targets),
            "generated": selected_text,
            "targets": val_targets,
            "generation_quality": quality,
            "word_overlap": {"summary": quality},
            "selection": {
                "method": method,
                "uses_validation_reference_for_selection": False,
                "uses_fixed_target_word_count": bool(penalty),
                "target_word_count": args.target_word_count,
                "length_penalty": penalty,
                "train_rows": args.train_rows,
                "train_input_npz": str(train_input_path),
                "train_input_start": args.train_input_start,
                "val_input_npz": str(args.val_semantic_npz),
                "val_input_start": args.val_input_start,
                "input_delay_blocks": args.input_delay_blocks,
                "model_name": args.model_name,
                "layer": args.layer,
                "ridge_grid": list(ridge_grid),
                "ridge_heldout_cosine": {str(key): value for key, value in ridge_scores.items()},
                "selected_ridge": selected_ridge,
                "candidate_count": len(candidates),
                "candidate_metrics": [str(path) for path in args.metrics],
                "selection_counts": {
                    str(index): int(np.sum(selected_indices == index))
                    for index in range(len(candidates))
                },
            },
            "generation_roberta_retrieval": retrieval,
            "selected_candidate_index": selected_indices.tolist(),
            "selected_score": adjusted_scores[
                np.arange(len(val_targets)), selected_indices
            ].tolist(),
        }
        if penalty == 0.0:
            output_path = args.output
        else:
            tag = f"{penalty:g}".replace(".", "p")
            output_path = args.output.with_name(f"{args.output.stem}.lengthp{tag}{args.output.suffix}")
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summaries[str(penalty)] = {
            "output": str(output_path),
            "selection": result["selection"],
            "quality": quality,
            "retrieval": retrieval,
        }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
