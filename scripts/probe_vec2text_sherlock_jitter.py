#!/usr/bin/env python
"""Probe Vec2Text vector jitter as a target-neighborhood generator.

This is intentionally a diagnostic script. It starts from a small held-out
set, jitters non-MiniLM embeddings, decodes them with Vec2Text, then scores the
decoded candidates back in MiniLM and lexical space.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-npz", required=True)
    parser.add_argument("--geometry-npz", required=True)
    parser.add_argument("--baseline-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-heldout", type=int, default=5)
    parser.add_argument("--heldout-start", type=int, default=0)
    parser.add_argument("--samples-per-scale", type=int, default=20)
    parser.add_argument("--jitter-angles", default="0,0.02,0.04,0.08,0.14")
    parser.add_argument("--neighbor-k", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--gtr-model-name", default="sentence-transformers/gtr-t5-base")
    parser.add_argument("--minilm-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--gtr-batch-size", type=int, default=64)
    parser.add_argument("--minilm-batch-size", type=int, default=128)
    parser.add_argument("--gtr-max-length", type=int, default=128)
    parser.add_argument("--minilm-max-length", type=int, default=64)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vec2text-pretrained-embedder", default="gtr-base")
    parser.add_argument("--vec2text-steps", type=int, default=8)
    parser.add_argument("--sequence-beam-width", type=int, default=0)
    parser.add_argument("--vec2text-batch-size", type=int, default=16)
    parser.add_argument("--max-generated-words", type=int, default=40)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def words(sentence: str) -> list[str]:
    return WORD_RE.findall(str(sentence))


def canonical(sentence: str) -> str:
    return " ".join(word.lower() for word in words(sentence))


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


def word_metrics(target: str, candidate: str) -> dict[str, float]:
    target_words = set(word.lower() for word in words(target))
    candidate_words = set(word.lower() for word in words(candidate))
    overlap = len(target_words & candidate_words)
    recall = overlap / max(1, len(target_words))
    precision = overlap / max(1, len(candidate_words))
    f1 = 0.0 if recall + precision <= 0 else 2.0 * recall * precision / (recall + precision)
    return {
        "word_recall": float(recall),
        "word_precision": float(precision),
        "word_f1": float(f1),
        "word_jaccard": float(overlap / max(1, len(target_words | candidate_words))),
        "word_overlap_count": int(overlap),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ["mean", "median", "min", "max", "p10", "p90"]}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one jitter angle")
    return parsed


def clean_generated(text: str, max_words: int) -> str:
    text = " ".join(str(text).replace("\n", " ").split()).strip()
    if not text:
        return ""
    tokens = words(text)
    if max_words > 0 and len(tokens) > max_words:
        tokens = tokens[:max_words]
    if tokens:
        text = " ".join(tokens)
    if text and text[-1] not in ".!?":
        text = f"{text}."
    return text


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def encode_gtr_raw(
    *,
    sentences: list[str],
    model_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
    local_files_only: bool,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if "t5" in model_name.lower():
        encoder = T5EncoderModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    else:
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        encoder = getattr(model, "encoder", model).to(device)
    encoder.eval()
    outputs = []
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        encoded = tokenizer(
            sentences[start:end],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            pooled = mean_pool(result.last_hidden_state, encoded["attention_mask"])
        outputs.append(pooled.float().cpu().numpy())
        print(f"gtr_encoded {end}/{len(sentences)}", flush=True)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def choose_heldout_indices(total: int, start: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("--num-heldout must be positive")
    end = min(total, start + count)
    indices = np.arange(start, end, dtype=np.int64)
    if len(indices) < count:
        raise ValueError(f"Requested {count} held-out rows from {start}, only {len(indices)} available")
    return indices


def nearest_geometry(
    heldout_unit: np.ndarray,
    geometry_unit: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    sims = heldout_unit @ geometry_unit.T
    top_k = min(k, geometry_unit.shape[0])
    top_indices = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
    sorted_indices = []
    sorted_scores = []
    for row_idx, candidates in enumerate(top_indices):
        order = np.argsort(-sims[row_idx, candidates])
        sorted_indices.append(candidates[order])
        sorted_scores.append(sims[row_idx, candidates][order])
    return np.stack(sorted_indices), np.stack(sorted_scores)


def random_tangent_direction(
    *,
    source_unit: np.ndarray,
    neighbor_units: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    local = neighbor_units - (neighbor_units @ source_unit)[:, None] * source_unit[None, :]
    local_norms = np.linalg.norm(local, axis=1)
    local = local[local_norms > 1e-7] / local_norms[local_norms > 1e-7, None]
    if local.shape[0] == 0:
        direction = rng.normal(size=source_unit.shape).astype(np.float32)
        direction = direction - float(direction @ source_unit) * source_unit
    else:
        weights = rng.normal(size=(local.shape[0],)).astype(np.float32)
        direction = weights @ local
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-7:
        direction = rng.normal(size=source_unit.shape).astype(np.float32)
        direction = direction - float(direction @ source_unit) * source_unit
        norm = float(np.linalg.norm(direction))
    return (direction / max(norm, 1e-7)).astype(np.float32)


def slerp_toward_neighbor(source_unit: np.ndarray, neighbor_unit: np.ndarray, angle: float) -> np.ndarray:
    dot = float(np.clip(source_unit @ neighbor_unit, -1.0, 1.0))
    theta = math.acos(dot)
    if theta <= 1e-6:
        return source_unit.copy()
    alpha = min(1.0, max(0.0, angle / theta))
    sin_theta = math.sin(theta)
    mixed = (
        math.sin((1.0 - alpha) * theta) / sin_theta * source_unit
        + math.sin(alpha * theta) / sin_theta * neighbor_unit
    )
    return (mixed / np.clip(np.linalg.norm(mixed), 1e-12, None)).astype(np.float32)


def build_jittered_embeddings(
    *,
    heldout_gtr: np.ndarray,
    geometry_gtr: np.ndarray,
    neighbor_indices: np.ndarray,
    angles: list[float],
    samples_per_scale: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    heldout_unit = normalize_rows(heldout_gtr)
    geometry_unit = normalize_rows(geometry_gtr)
    heldout_norms = np.linalg.norm(heldout_gtr, axis=1)
    generated_units: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for target_local_idx, source_unit in enumerate(heldout_unit):
        neighbors = geometry_unit[neighbor_indices[target_local_idx]]
        source_norm = float(heldout_norms[target_local_idx])
        for angle in angles:
            if angle == 0:
                generated_units.append(source_unit)
                rows.append(
                    {
                        "target_local_index": int(target_local_idx),
                        "method": "exact",
                        "jitter_angle": 0.0,
                        "geometry_neighbor_index": int(neighbor_indices[target_local_idx, 0]),
                    }
                )
                continue
            for sample_idx in range(samples_per_scale):
                use_tangent = sample_idx % 2 == 0
                if use_tangent:
                    direction = random_tangent_direction(
                        source_unit=source_unit,
                        neighbor_units=neighbors,
                        rng=rng,
                    )
                    unit = math.cos(angle) * source_unit + math.sin(angle) * direction
                    method = "local_tangent"
                    neighbor_index = int(neighbor_indices[target_local_idx, sample_idx % len(neighbors)])
                else:
                    slot = int(rng.integers(0, len(neighbors)))
                    unit = slerp_toward_neighbor(source_unit, neighbors[slot], angle)
                    method = "train_slerp"
                    neighbor_index = int(neighbor_indices[target_local_idx, slot])
                unit = unit / np.clip(np.linalg.norm(unit), 1e-12, None)
                generated_units.append(unit.astype(np.float32))
                rows.append(
                    {
                        "target_local_index": int(target_local_idx),
                        "method": method,
                        "jitter_angle": float(angle),
                        "geometry_neighbor_index": neighbor_index,
                        "source_norm": source_norm,
                    }
                )
    embeddings = np.stack(generated_units).astype(np.float32)
    norms = np.asarray([heldout_norms[int(row["target_local_index"])] for row in rows], dtype=np.float32)
    embeddings = embeddings * norms[:, None]
    return embeddings, rows


def load_vec2text_corrector(embedder: str):
    import vec2text

    corrector = vec2text.load_pretrained_corrector(embedder)
    return vec2text, corrector


@torch.inference_mode()
def invert_with_vec2text(
    *,
    embeddings: np.ndarray,
    embedder: str,
    steps: int,
    sequence_beam_width: int,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    vec2text, corrector = load_vec2text_corrector(embedder)
    corrector.inversion_trainer.model.to(device)
    corrector.model.to(device)
    outputs: list[str] = []
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch = torch.as_tensor(embeddings[start:end], dtype=torch.float32, device=device)
        decoded = vec2text.invert_embeddings(
            embeddings=batch,
            corrector=corrector,
            num_steps=steps,
            sequence_beam_width=sequence_beam_width,
        )
        outputs.extend(decoded)
        print(f"vec2text_inverted {end}/{len(embeddings)}", flush=True)
    return outputs


def best_train_baselines(
    *,
    heldout_sentences: list[str],
    heldout_embeddings: np.ndarray,
    baseline_sentences: list[str],
    baseline_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    query = normalize_rows(heldout_embeddings)
    bank = normalize_rows(baseline_embeddings)
    sims = query @ bank.T
    indices = np.argmax(sims, axis=1)
    rows = []
    for idx, candidate_idx in enumerate(indices.tolist()):
        metrics = word_metrics(heldout_sentences[idx], baseline_sentences[candidate_idx])
        rows.append(
            {
                "target_local_index": int(idx),
                "candidate_index": int(candidate_idx),
                "cosine": float(sims[idx, candidate_idx]),
                "sentence": baseline_sentences[candidate_idx],
                **metrics,
            }
        )
    return rows


def score_candidates(
    *,
    candidates: list[str],
    jitter_rows: list[dict[str, Any]],
    heldout_sentences: list[str],
    heldout_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    geometry_sentences: list[str],
) -> list[dict[str, Any]]:
    heldout_norm = normalize_rows(heldout_embeddings)
    candidate_norm = normalize_rows(candidate_embeddings)
    scored = []
    seen_by_target: set[tuple[int, str]] = set()
    for idx, (candidate, row) in enumerate(zip(candidates, jitter_rows)):
        target_idx = int(row["target_local_index"])
        candidate = clean_generated(candidate, max_words=0)
        if not candidate:
            continue
        key = (target_idx, canonical(candidate))
        duplicate_for_target = key in seen_by_target
        seen_by_target.add(key)
        cosine = float(heldout_norm[target_idx] @ candidate_norm[idx])
        geometry_idx = int(row["geometry_neighbor_index"])
        scored.append(
            {
                "candidate_index": int(idx),
                "target_local_index": target_idx,
                "target_sentence": heldout_sentences[target_idx],
                "sentence": candidate,
                "method": str(row["method"]),
                "jitter_angle": float(row["jitter_angle"]),
                "minilm_cosine_to_target": cosine,
                "duplicate_for_target": bool(duplicate_for_target),
                "geometry_neighbor_index": geometry_idx,
                "geometry_neighbor_sentence": geometry_sentences[geometry_idx],
                **word_metrics(heldout_sentences[target_idx], candidate),
            }
        )
    return scored


def summarize_by_group(scored: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        best_by_target = []
        by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["duplicate_for_target"]:
                continue
            by_target[int(row["target_local_index"])].append(row)
        for target_rows in by_target.values():
            best_by_target.append(
                max(
                    target_rows,
                    key=lambda item: (
                        float(item["word_f1"]),
                        float(item["word_recall"]),
                        float(item["minilm_cosine_to_target"]),
                    ),
                )
            )
        summary = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        summary.update(
            {
                "candidate_count": int(len(rows)),
                "target_count": int(len(by_target)),
                "mean_best_word_f1": float(np.mean([row["word_f1"] for row in best_by_target])) if best_by_target else 0.0,
                "mean_best_word_recall": float(np.mean([row["word_recall"] for row in best_by_target])) if best_by_target else 0.0,
                "mean_best_minilm_cosine": float(np.mean([row["minilm_cosine_to_target"] for row in best_by_target])) if best_by_target else 0.0,
                "best_examples": sorted(
                    best_by_target,
                    key=lambda item: (
                        float(item["word_f1"]),
                        float(item["word_recall"]),
                        float(item["minilm_cosine_to_target"]),
                    ),
                    reverse=True,
                )[:5],
            }
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    angles = parse_float_list(args.jitter_angles)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with np.load(args.heldout_npz, allow_pickle=True) as data:
        heldout_sentences_all = strings(data[args.sentence_key])
        heldout_embeddings_all = np.asarray(data[args.input_key], dtype=np.float32)
    heldout_indices = choose_heldout_indices(len(heldout_sentences_all), args.heldout_start, args.num_heldout)
    heldout_sentences = [heldout_sentences_all[int(idx)] for idx in heldout_indices]
    heldout_minilm = heldout_embeddings_all[heldout_indices]

    with np.load(args.geometry_npz, allow_pickle=True) as data:
        geometry_sentences = strings(data[args.sentence_key])
    with np.load(args.baseline_npz, allow_pickle=True) as data:
        baseline_sentences = strings(data[args.sentence_key])
        baseline_minilm = np.asarray(data[args.input_key], dtype=np.float32)

    print("selected held-out sentences:", flush=True)
    for idx, sentence in zip(heldout_indices.tolist(), heldout_sentences):
        print(f"[{idx}] {sentence}", flush=True)

    geometry_gtr = encode_gtr_raw(
        sentences=geometry_sentences,
        model_name=args.gtr_model_name,
        device=device,
        max_length=args.gtr_max_length,
        batch_size=args.gtr_batch_size,
        local_files_only=args.local_files_only,
    )
    heldout_gtr = encode_gtr_raw(
        sentences=heldout_sentences,
        model_name=args.gtr_model_name,
        device=device,
        max_length=args.gtr_max_length,
        batch_size=args.gtr_batch_size,
        local_files_only=args.local_files_only,
    )
    neighbor_indices, neighbor_scores = nearest_geometry(
        normalize_rows(heldout_gtr),
        normalize_rows(geometry_gtr),
        args.neighbor_k,
    )
    jittered_gtr, jitter_rows = build_jittered_embeddings(
        heldout_gtr=heldout_gtr,
        geometry_gtr=geometry_gtr,
        neighbor_indices=neighbor_indices,
        angles=angles,
        samples_per_scale=args.samples_per_scale,
        seed=args.seed,
    )
    decoded = invert_with_vec2text(
        embeddings=jittered_gtr,
        embedder=args.vec2text_pretrained_embedder,
        steps=args.vec2text_steps,
        sequence_beam_width=args.sequence_beam_width,
        batch_size=args.vec2text_batch_size,
        device=device,
    )
    cleaned = [clean_generated(text, args.max_generated_words) for text in decoded]

    tokenizer = AutoTokenizer.from_pretrained(args.minilm_model_name, local_files_only=args.local_files_only)
    minilm_model = AutoModel.from_pretrained(args.minilm_model_name, local_files_only=args.local_files_only).to(device)
    minilm_model.eval()
    candidate_minilm = encode_sentences(
        sentences=cleaned,
        tokenizer=tokenizer,
        model=minilm_model,
        device=device,
        max_length=args.minilm_max_length,
        batch_size=args.minilm_batch_size,
        pooling="mean",
        normalize=True,
    )
    scored = score_candidates(
        candidates=cleaned,
        jitter_rows=jitter_rows,
        heldout_sentences=heldout_sentences,
        heldout_embeddings=heldout_minilm,
        candidate_embeddings=candidate_minilm,
        geometry_sentences=geometry_sentences,
    )
    train_baseline = best_train_baselines(
        heldout_sentences=heldout_sentences,
        heldout_embeddings=heldout_minilm,
        baseline_sentences=baseline_sentences,
        baseline_embeddings=baseline_minilm,
    )

    best_by_target = []
    for target_idx in range(len(heldout_sentences)):
        target_rows = [
            row
            for row in scored
            if int(row["target_local_index"]) == target_idx and not row["duplicate_for_target"]
        ]
        if target_rows:
            best_by_target.append(
                max(
                    target_rows,
                    key=lambda item: (
                        float(item["word_f1"]),
                        float(item["word_recall"]),
                        float(item["minilm_cosine_to_target"]),
                    ),
                )
            )

    by_angle_method = summarize_by_group(scored, ("jitter_angle", "method"))
    by_angle = summarize_by_group(scored, ("jitter_angle",))
    chosen = max(
        by_angle,
        key=lambda item: (
            float(item["mean_best_word_f1"]),
            float(item["mean_best_word_recall"]),
            float(item["mean_best_minilm_cosine"]),
        ),
    )

    summary = {
        "config": vars(args),
        "warning": "This probe starts from held-out embeddings; generated candidates are target-derived diagnostics, not clean training data.",
        "heldout_indices": heldout_indices.tolist(),
        "heldout_sentences": heldout_sentences,
        "geometry_npz": args.geometry_npz,
        "baseline_npz": args.baseline_npz,
        "heldout_npz": args.heldout_npz,
        "nearest_geometry_examples": [
            {
                "target_local_index": int(target_idx),
                "target_sentence": heldout_sentences[target_idx],
                "neighbors": [
                    {
                        "geometry_index": int(index),
                        "gtr_cosine": float(score),
                        "sentence": geometry_sentences[int(index)],
                    }
                    for index, score in zip(neighbor_indices[target_idx, :5], neighbor_scores[target_idx, :5])
                ],
            }
            for target_idx in range(len(heldout_sentences))
        ],
        "train_baseline_top1": train_baseline,
        "train_baseline_summary": {
            "word_f1": summarize([row["word_f1"] for row in train_baseline]),
            "word_recall": summarize([row["word_recall"] for row in train_baseline]),
            "minilm_cosine": summarize([row["cosine"] for row in train_baseline]),
        },
        "vec2text_best_by_target": best_by_target,
        "vec2text_summary": {
            "word_f1": summarize([row["word_f1"] for row in best_by_target]),
            "word_recall": summarize([row["word_recall"] for row in best_by_target]),
            "minilm_cosine": summarize([row["minilm_cosine_to_target"] for row in best_by_target]),
        },
        "by_jitter_angle": by_angle,
        "by_jitter_angle_and_method": by_angle_method,
        "chosen_jitter": chosen,
    }

    json_path = output_dir / "summary.json"
    jsonl_path = output_dir / "candidates.jsonl"
    npz_path = output_dir / "candidates_minilm_targetderived.npz"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")
    np.savez_compressed(
        npz_path,
        input_embeddings=candidate_minilm.astype(np.float32),
        sentence=np.asarray(cleaned, dtype=object),
        rows=np.asarray(jitter_rows, dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
