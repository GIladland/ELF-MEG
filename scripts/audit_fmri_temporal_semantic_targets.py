#!/usr/bin/env python
"""Compare one leakage-safe fMRI model with alternative val semantic targets.

Only ``val_x`` is read from the brain archive.  Target archives must contain
the concatenated train11725/val266 rows; test107 is neither required nor read.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modules.fmri2sem_bridge import load_mri2sem_model


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Named target archive formatted NAME=PATH; repeat for each target.",
    )
    parser.add_argument("--brain-val-key", default="val_x")
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--mri-output-dim", type=int, default=384)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mri-dropout", type=float, default=0.1)
    parser.add_argument("--semantic-mapper-hidden-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def named_paths(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate target name {name!r}")
        result[name] = path
    return result


def strings(values: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in values.tolist()]


def retrieval_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = F.normalize(predicted.float(), p=2, dim=-1)
    target = F.normalize(target.float(), p=2, dim=-1)
    similarities = predicted @ target.T
    matched = similarities.diag()
    ranks = 1 + (similarities > matched[:, None]).sum(dim=1)
    return {
        "matched_cosine_mean": float(matched.mean()),
        "matched_cosine_median": float(matched.median()),
        "top1": float((ranks == 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }


def load_semantic_mapper(
    checkpoint_path: str,
    *,
    input_dim: int,
    hidden_dim: int,
) -> nn.Module | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("adapter_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        return None
    mapper = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, input_dim),
    )
    for prefix in ("semantic_projector.semantic_mapper.", "semantic_mapper."):
        mapper_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if mapper_state:
            mapper.load_state_dict(mapper_state, strict=True)
            return mapper
    return None


def token_f1(left: str, right: str) -> float:
    left_tokens = set(WORD_RE.findall(left.lower()))
    right_tokens = set(WORD_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    brain = np.load(args.brain_npz, allow_pickle=True, mmap_mode="r")
    val_x = brain[args.brain_val_key]
    if len(val_x) != args.val_count:
        raise ValueError(f"Expected {args.val_count} validation rows, got {len(val_x)}")

    model = load_mri2sem_model(
        args.checkpoint,
        input_dim=int(val_x.shape[1]),
        output_dim=args.mri_output_dim,
        hidden_dim=args.mri_hidden_dim,
        res_blocks=args.mri_res_blocks,
        dropout=args.mri_dropout,
    ).to(device).eval()
    predicted_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(val_x), args.batch_size):
            batch = torch.as_tensor(val_x[start : start + args.batch_size], dtype=torch.float32, device=device)
            predicted_chunks.append(model(batch).float().cpu())
    predicted = torch.cat(predicted_chunks)
    mapper = load_semantic_mapper(
        args.checkpoint,
        input_dim=args.mri_output_dim,
        hidden_dim=args.semantic_mapper_hidden_dim,
    )
    mapped_predicted = predicted
    if mapper is not None:
        mapper = mapper.to(device).eval()
        mapped_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(predicted), args.batch_size):
                batch = predicted[start : start + args.batch_size].to(device)
                mapped_chunks.append((batch + mapper(batch)).float().cpu())
        mapped_predicted = torch.cat(mapped_chunks)

    targets: dict[str, dict[str, object]] = {}
    for name, path in named_paths(args.target).items():
        archive = np.load(path, allow_pickle=True)
        embeddings = np.asarray(archive[args.embedding_key], dtype=np.float32)
        sentences = strings(archive[args.sentence_key])
        expected = args.train_count + args.val_count
        if len(embeddings) != expected or len(sentences) != expected:
            raise ValueError(f"{name} expected {expected} rows, got {len(embeddings)}/{len(sentences)}")
        targets[name] = {
            "path": path,
            "embeddings": torch.from_numpy(embeddings[args.train_count :]),
            "sentences": sentences[args.train_count :],
        }

    result_targets = {
        name: {
            "path": payload["path"],
            "sentence_word_count_mean": float(
                np.mean([len(WORD_RE.findall(sentence)) for sentence in payload["sentences"]])
            ),
            "raw_mri_head_retrieval": retrieval_metrics(predicted, payload["embeddings"]),
            "post_semantic_mapper_retrieval": retrieval_metrics(
                mapped_predicted, payload["embeddings"]
            ),
        }
        for name, payload in targets.items()
    }
    pairwise: dict[str, dict[str, float]] = {}
    names = list(targets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_embeddings = F.normalize(targets[left]["embeddings"].float(), p=2, dim=-1)
            right_embeddings = F.normalize(targets[right]["embeddings"].float(), p=2, dim=-1)
            pairwise[f"{left}__{right}"] = {
                "matched_embedding_cosine_mean": float((left_embeddings * right_embeddings).sum(dim=-1).mean()),
                "sentence_set_word_f1_mean": float(
                    np.mean(
                        [
                            token_f1(left_sentence, right_sentence)
                            for left_sentence, right_sentence in zip(
                                targets[left]["sentences"], targets[right]["sentences"]
                            )
                        ]
                    )
                ),
            }

    output = {
        "status": "complete",
        "scientific_scope": "val266 only; test107 brain and text are not loaded",
        "args": vars(args),
        "brain_val_shape": list(val_x.shape),
        "predicted_shape": list(predicted.shape),
        "semantic_mapper_loaded": mapper is not None,
        "targets": result_targets,
        "pairwise_targets": pairwise,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
