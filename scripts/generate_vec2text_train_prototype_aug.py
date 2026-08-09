#!/usr/bin/env python
"""Generate non-leaky Sherlock-style Vec2Text augmentations from train prototypes.

The target validation split is never used for generation or candidate
selection. We optionally carve an inner-dev split out of the training NPZ so
downstream selection can tune coverage without looking at Sherlock12.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import DEFAULT_META_RE, keep_sentence, words


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-npz", required=True)
    parser.add_argument("--base-train-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--inner-dev-output", required=True)
    parser.add_argument("--inner-dev-fraction", type=float, default=0.20)
    parser.add_argument("--inner-dev-max", type=int, default=768)
    parser.add_argument("--inner-dev-min", type=int, default=256)
    parser.add_argument("--prototype-count", type=int, default=3000)
    parser.add_argument("--target-count", type=int, default=50000)
    parser.add_argument("--raw-multiplier", type=float, default=1.6)
    parser.add_argument("--jitter-angles", default="0.04,0.08,0.10")
    parser.add_argument("--neighbor-k", type=int, default=48)
    parser.add_argument("--density-k", type=int, default=32)
    parser.add_argument("--density-batch-size", type=int, default=512)
    parser.add_argument("--gtr-model-name", default="sentence-transformers/gtr-t5-base")
    parser.add_argument("--minilm-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--vec2text-pretrained-embedder", default="gtr-base")
    parser.add_argument("--vec2text-steps", type=int, default=6)
    parser.add_argument("--sequence-beam-width", type=int, default=0)
    parser.add_argument("--vec2text-batch-size", type=int, default=16)
    parser.add_argument("--gtr-batch-size", type=int, default=96)
    parser.add_argument("--minilm-batch-size", type=int, default=256)
    parser.add_argument("--gtr-max-length", type=int, default=128)
    parser.add_argument("--minilm-max-length", type=int, default=64)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument(
        "--filter-mode",
        choices=["short_content", "simple_sv", "simple_sv_no_coord"],
        default="simple_sv_no_coord",
    )
    parser.add_argument("--exclude-regex", default=DEFAULT_META_RE)
    parser.add_argument("--min-source-word-recall", type=float, default=0.20)
    parser.add_argument("--min-source-minilm-cosine", type=float, default=0.60)
    parser.add_argument("--max-source-minilm-cosine", type=float, default=0.995)
    parser.add_argument("--max-generated-words", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def canonical(sentence: str) -> str:
    return " ".join(word.lower() for word in words(str(sentence)))


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one jitter angle")
    if any(angle <= 0 for angle in parsed):
        raise ValueError("Use only nonzero jitter angles for the non-leaky train-prototype generator")
    return parsed


def word_set(sentence: str) -> set[str]:
    return set(word.lower() for word in WORD_RE.findall(str(sentence)))


def word_recall(source: str, candidate: str) -> float:
    source_words = word_set(source)
    candidate_words = word_set(candidate)
    if not source_words:
        return 0.0
    return float(len(source_words & candidate_words) / len(source_words))


def clean_generated(text: str, max_words: int) -> str:
    text = " ".join(str(text).replace("\n", " ").split()).strip()
    tokens = words(text)
    if max_words > 0 and len(tokens) > max_words:
        tokens = tokens[:max_words]
    if not tokens:
        return ""
    text = " ".join(tokens)
    if text and text[-1] not in ".!?":
        text = f"{text}."
    return text


def load_npz(path: str, input_key: str = "input_embeddings", sentence_key: str = "sentence") -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        sentences = strings(data[sentence_key])
        embeddings = np.asarray(data[input_key], dtype=np.float32)
        rows = data["rows"] if "rows" in data.files else np.asarray([{} for _ in sentences], dtype=object)
    return sentences, embeddings, rows


def save_inner_dev(
    *,
    output: str,
    base_train_npz: str,
    sentences: list[str],
    embeddings: np.ndarray,
    rows: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output_path),
        "base_train_npz": base_train_npz,
        "inner_dev_count": int(len(indices)),
        "seed": int(seed),
        "warning": "This is an inner-dev split carved only from training text; not Sherlock12.",
        "sample": [sentences[int(idx)] for idx in indices[:10].tolist()],
    }
    np.savez_compressed(
        output_path,
        input_embeddings=embeddings[indices].astype(np.float32),
        sentence=np.asarray([sentences[int(idx)] for idx in indices.tolist()], dtype=object),
        rows=rows[indices],
        split=np.asarray(["val"] * len(indices), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def choose_inner_dev(total: int, *, fraction: float, minimum: int, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(round(total * fraction))
    count = min(maximum, max(minimum, count))
    count = min(total // 2, count)
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64)


@torch.inference_mode()
def density_scores(embeddings: np.ndarray, *, k: int, batch_size: int, device: torch.device) -> np.ndarray:
    unit = normalize_rows(embeddings)
    target = torch.as_tensor(unit, dtype=torch.float32, device=device).T.contiguous()
    scores = np.zeros((len(unit),), dtype=np.float32)
    top_k = min(k + 1, len(unit))
    for start in range(0, len(unit), batch_size):
        end = min(start + batch_size, len(unit))
        query = torch.as_tensor(unit[start:end], dtype=torch.float32, device=device)
        sims = query @ target
        row = torch.arange(end - start, device=device)
        sims[row, torch.arange(start, end, device=device)] = -float("inf")
        values = torch.topk(sims, k=min(k, len(unit) - 1), dim=1).values
        scores[start:end] = values.mean(dim=1).detach().cpu().numpy()
        print(f"density_scored {end}/{len(unit)}", flush=True)
    return scores


def eligible_prototypes(
    *,
    sentences: list[str],
    embeddings: np.ndarray,
    blocked_keys: set[str],
    args: argparse.Namespace,
) -> np.ndarray:
    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE) if args.exclude_regex else None
    keep = []
    seen = set()
    for idx, sentence in enumerate(sentences):
        key = canonical(sentence)
        if not key or key in blocked_keys or key in seen:
            continue
        seen.add(key)
        ok, _reason = keep_sentence(
            sentence,
            min_words=args.min_words,
            max_words=args.max_words,
            min_alpha_fraction=0.75,
            min_content_words=2,
            require_verb=True,
            filter_mode="short_content",
            exclude_re=exclude_re,
        )
        if ok:
            keep.append(idx)
    keep_arr = np.asarray(keep, dtype=np.int64)
    if len(keep_arr) == 0:
        raise RuntimeError("No eligible training prototypes survived filters.")
    return keep_arr


def select_dense_prototypes(
    *,
    sentences: list[str],
    embeddings: np.ndarray,
    eligible: np.ndarray,
    count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    eligible_embeddings = embeddings[eligible]
    density = density_scores(
        eligible_embeddings,
        k=args.density_k,
        batch_size=args.density_batch_size,
        device=device,
    )
    lengths = np.asarray([len(words(sentences[int(idx)])) for idx in eligible.tolist()], dtype=np.float32)
    length_score = 1.0 - np.abs(lengths - 10.0) / 10.0
    length_score = np.clip(length_score, 0.0, 1.0)
    score = density + 0.03 * length_score
    order = np.argsort(-score)
    chosen = eligible[order[: min(count, len(order))]]
    print(f"selected_prototypes {len(chosen)}/{len(eligible)}", flush=True)
    for idx in chosen[:10].tolist():
        print(f"prototype[{idx}] {sentences[int(idx)]}", flush=True)
    return chosen.astype(np.int64)


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


def nearest_neighbor_indices(
    *,
    prototype_units: np.ndarray,
    geometry_units: np.ndarray,
    prototype_positions: np.ndarray,
    k: int,
    device: torch.device,
) -> np.ndarray:
    target = torch.as_tensor(geometry_units, dtype=torch.float32, device=device).T.contiguous()
    out = []
    batch_size = 512
    top_k = min(k + 1, len(geometry_units))
    for start in range(0, len(prototype_units), batch_size):
        end = min(start + batch_size, len(prototype_units))
        query = torch.as_tensor(prototype_units[start:end], dtype=torch.float32, device=device)
        sims = query @ target
        for row_idx, position in enumerate(prototype_positions[start:end].tolist()):
            sims[row_idx, int(position)] = -float("inf")
        indices = torch.topk(sims, k=min(k, len(geometry_units) - 1), dim=1).indices
        out.append(indices.detach().cpu().numpy())
        print(f"gtr_neighbors {end}/{len(prototype_units)}", flush=True)
    return np.concatenate(out, axis=0).astype(np.int64)


def random_tangent_direction(
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
    selected_positions: np.ndarray,
    geometry_gtr: np.ndarray,
    neighbor_indices: np.ndarray,
    angles: list[float],
    raw_count: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    units = normalize_rows(geometry_gtr)
    norms = np.linalg.norm(geometry_gtr, axis=1)
    embeddings = []
    rows = []
    cursor = 0
    while len(rows) < raw_count:
        proto_slot = cursor % len(selected_positions)
        position = int(selected_positions[proto_slot])
        source_unit = units[position]
        neighbors = units[neighbor_indices[proto_slot]]
        angle = float(angles[(cursor // len(selected_positions)) % len(angles)])
        if cursor % 2 == 0:
            direction = random_tangent_direction(source_unit, neighbors, rng)
            unit = math.cos(angle) * source_unit + math.sin(angle) * direction
            method = "local_tangent"
            neighbor_position = int(neighbor_indices[proto_slot, cursor % neighbor_indices.shape[1]])
        else:
            neighbor_slot = int(rng.integers(0, neighbor_indices.shape[1]))
            neighbor_position = int(neighbor_indices[proto_slot, neighbor_slot])
            unit = slerp_toward_neighbor(source_unit, units[neighbor_position], angle)
            method = "train_slerp"
        unit = unit / np.clip(np.linalg.norm(unit), 1e-12, None)
        embeddings.append((unit * norms[position]).astype(np.float32))
        rows.append(
            {
                "prototype_slot": int(proto_slot),
                "prototype_position": position,
                "method": method,
                "jitter_angle": angle,
                "gtr_neighbor_position": neighbor_position,
            }
        )
        cursor += 1
    return np.stack(embeddings).astype(np.float32), rows


def load_vec2text_corrector(embedder: str):
    import vec2text

    return vec2text, vec2text.load_pretrained_corrector(embedder)


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


def blocked_sentence_keys(paths: list[str], base_sentences: list[str], inner_dev_sentences: list[str]) -> set[str]:
    blocked = {canonical(sentence) for sentence in base_sentences}
    blocked.update(canonical(sentence) for sentence in inner_dev_sentences)
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in strings(data["sentence"]))
    blocked.discard("")
    return blocked


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    angles = parse_float_list(args.jitter_angles)
    device = resolve_device(args.device)

    base_sentences, base_embeddings, base_rows = load_npz(args.base_train_npz)
    inner_dev_indices = choose_inner_dev(
        len(base_sentences),
        fraction=args.inner_dev_fraction,
        minimum=args.inner_dev_min,
        maximum=args.inner_dev_max,
        seed=args.seed,
    )
    inner_dev_sentences = [base_sentences[int(idx)] for idx in inner_dev_indices.tolist()]
    save_inner_dev(
        output=args.inner_dev_output,
        base_train_npz=args.base_train_npz,
        sentences=base_sentences,
        embeddings=base_embeddings,
        rows=base_rows,
        indices=inner_dev_indices,
        seed=args.seed,
    )

    prototype_sentences, prototype_minilm, _prototype_rows = load_npz(args.prototype_npz)
    block_for_prototypes = {canonical(sentence) for sentence in inner_dev_sentences}
    for path in args.blocked_npz:
        with np.load(path, allow_pickle=True) as data:
            block_for_prototypes.update(canonical(sentence) for sentence in strings(data["sentence"]))
    eligible = eligible_prototypes(
        sentences=prototype_sentences,
        embeddings=prototype_minilm,
        blocked_keys=block_for_prototypes,
        args=args,
    )
    selected_global = select_dense_prototypes(
        sentences=prototype_sentences,
        embeddings=prototype_minilm,
        eligible=eligible,
        count=args.prototype_count,
        args=args,
        device=device,
    )

    eligible_sentences = [prototype_sentences[int(idx)] for idx in eligible.tolist()]
    eligible_minilm = prototype_minilm[eligible]
    selected_positions = np.asarray(
        [int(np.where(eligible == idx)[0][0]) for idx in selected_global.tolist()],
        dtype=np.int64,
    )
    gtr_geometry = encode_gtr_raw(
        sentences=eligible_sentences,
        model_name=args.gtr_model_name,
        device=device,
        max_length=args.gtr_max_length,
        batch_size=args.gtr_batch_size,
        local_files_only=args.local_files_only,
    )
    neighbors = nearest_neighbor_indices(
        prototype_units=normalize_rows(gtr_geometry[selected_positions]),
        geometry_units=normalize_rows(gtr_geometry),
        prototype_positions=selected_positions,
        k=args.neighbor_k,
        device=device,
    )
    raw_count = max(args.target_count, int(math.ceil(args.target_count * args.raw_multiplier)))
    jittered, jitter_rows = build_jittered_embeddings(
        selected_positions=selected_positions,
        geometry_gtr=gtr_geometry,
        neighbor_indices=neighbors,
        angles=angles,
        raw_count=raw_count,
        seed=args.seed,
    )
    decoded = invert_with_vec2text(
        embeddings=jittered,
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

    blocked = blocked_sentence_keys(args.blocked_npz, base_sentences, inner_dev_sentences)
    accepted_indices = []
    accepted_rows = []
    seen = set(blocked)
    reject_counts: dict[str, int] = {}
    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE) if args.exclude_regex else None
    eligible_minilm_unit = normalize_rows(eligible_minilm)
    candidate_minilm_unit = normalize_rows(candidate_minilm)
    for idx, sentence in enumerate(cleaned):
        row = jitter_rows[idx]
        key = canonical(sentence)
        reason = ""
        if not key:
            reason = "empty"
        elif key in seen:
            reason = "duplicate_or_blocked"
        if not reason:
            ok, keep_reason = keep_sentence(
                sentence,
                min_words=args.min_words,
                max_words=args.max_words,
                min_alpha_fraction=0.75,
                min_content_words=2,
                require_verb=True,
                filter_mode=args.filter_mode,
                exclude_re=exclude_re,
            )
            if not ok:
                reason = keep_reason
        source_position = int(row["prototype_position"])
        source_sentence = eligible_sentences[source_position]
        source_recall = word_recall(source_sentence, sentence)
        source_cosine = float(candidate_minilm_unit[idx] @ eligible_minilm_unit[source_position])
        if not reason and source_recall < args.min_source_word_recall:
            reason = "low_source_word_recall"
        if not reason and source_cosine < args.min_source_minilm_cosine:
            reason = "low_source_minilm_cosine"
        if not reason and source_cosine > args.max_source_minilm_cosine:
            reason = "too_close_to_source"
        if reason:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        seen.add(key)
        row = dict(row)
        row.update(
            {
                "source_sentence": source_sentence,
                "gtr_neighbor_sentence": eligible_sentences[int(row["gtr_neighbor_position"])],
                "source_word_recall": float(source_recall),
                "source_minilm_cosine": float(source_cosine),
                "augmentation_source_index": int(selected_global[int(row["prototype_slot"])]),
                "corpus": "vec2text_train_prototype_gtr",
            }
        )
        accepted_indices.append(idx)
        accepted_rows.append(row)
        if len(accepted_indices) >= args.target_count:
            break

    if not accepted_indices:
        raise RuntimeError(f"No generated candidates accepted. Reject counts: {reject_counts}")
    keep = np.asarray(accepted_indices, dtype=np.int64)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output_path),
        "prototype_npz": args.prototype_npz,
        "base_train_npz": args.base_train_npz,
        "blocked_npz": args.blocked_npz,
        "inner_dev_output": args.inner_dev_output,
        "inner_dev_count": int(len(inner_dev_indices)),
        "eligible_prototype_count": int(len(eligible)),
        "selected_prototype_count": int(len(selected_global)),
        "raw_count": int(raw_count),
        "accepted_count": int(len(keep)),
        "target_count": int(args.target_count),
        "reject_counts": reject_counts,
        "jitter_angles": angles,
        "vec2text_steps": int(args.vec2text_steps),
        "sequence_beam_width": int(args.sequence_beam_width),
        "filter_mode": args.filter_mode,
        "min_source_word_recall": float(args.min_source_word_recall),
        "min_source_minilm_cosine": float(args.min_source_minilm_cosine),
        "max_source_minilm_cosine": float(args.max_source_minilm_cosine),
        "seed": int(args.seed),
        "warning": "Generated only from training text/prototypes. Sherlock12 may be blocked for exact text only but is not used for selection.",
        "sample": [
            {
                "sentence": cleaned[int(idx)],
                **accepted_rows[pos],
            }
            for pos, idx in enumerate(keep[:20].tolist())
        ],
    }
    np.savez_compressed(
        output_path,
        input_embeddings=candidate_minilm[keep].astype(np.float32),
        sentence=np.asarray([cleaned[int(idx)] for idx in keep.tolist()], dtype=object),
        rows=np.asarray(accepted_rows, dtype=object),
        split=np.asarray(["train"] * len(keep), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
