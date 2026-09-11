#!/usr/bin/env python3
"""Select one generated candidate per row using only the input semantic vector.

The selection score is cosine similarity between an exact MiniLM embedding of a
generated candidate and the corresponding normalized semantic input. Reference
texts are used only after selection to report validation/test metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CONTENT_STOPWORDS = set(
    """
    a about after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into
    is it its itself just me more most my no nor not of off on once only or other our ours
    ourselves out over own quite really same she should so some such sure than that the
    their theirs them themselves then there these they this those through to too under
    until up us very was we well were what when where which while who whom whose why will
    with would yes you your yours yourself yourselves
    """.split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def overlap_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def text_metrics(generated: Sequence[str], targets: Sequence[str]) -> dict[str, float | int]:
    total_errors = 0
    total_reference_words = 0
    overlap = []
    content_overlap = []
    for generated_text, target_text in zip(generated, targets):
        generated_words = words(generated_text)
        target_words = words(target_text)
        total_errors += edit_distance(target_words, generated_words)
        total_reference_words += len(target_words)
        overlap.append(overlap_f1(generated_words, target_words))
        content_overlap.append(
            overlap_f1(
                [token for token in generated_words if token not in CONTENT_STOPWORDS],
                [token for token in target_words if token not in CONTENT_STOPWORDS],
            )
        )
    return {
        "word_error_rate": total_errors / max(1, total_reference_words),
        "word_error_reference_words": total_reference_words,
        "words_overlap": float(np.mean(overlap)),
        "content_words_overlap": float(np.mean(content_overlap)),
    }


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    output = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        encoded = tokenizer(
            list(texts[start:end]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        output.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(output, axis=0)


def retrieval_metrics(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    similarity = normalize_rows(query) @ normalize_rows(target).T
    diagonal = np.diag(similarity)
    ranks = 1 + np.sum(similarity > diagonal[:, None], axis=1)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.source_npz, allow_pickle=True) as source:
        semantic = normalize_rows(source[args.embedding_key])
        source_targets = [str(value) for value in source["sentence"].tolist()]

    runs = []
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = [str(value) for value in payload["generated"]]
        targets = [str(value) for value in payload["targets"]]
        if targets != source_targets:
            raise ValueError(f"Target rows do not align with {args.source_npz}: {path}")
        if len(generated) != len(semantic):
            raise ValueError(f"Row count mismatch for {path}: {len(generated)} != {len(semantic)}")
        runs.append({"path": str(path), "generated": generated})

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model_name, local_files_only=args.local_files_only).to(device).eval()

    target_embeddings = encode_texts(
        source_targets,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    candidate_embeddings = []
    baselines = []
    for run in runs:
        embeddings = encode_texts(
            run["generated"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        candidate_embeddings.append(embeddings)
        baselines.append(
            {
                "path": run["path"],
                "selection_cosine_mean": float(np.mean(np.sum(embeddings * semantic, axis=1))),
                "text": text_metrics(run["generated"], source_targets),
                "minilm_retrieval": retrieval_metrics(embeddings, target_embeddings),
            }
        )

    candidate_tensor = np.stack(candidate_embeddings, axis=1)
    scores = np.einsum("nkd,nd->nk", candidate_tensor, semantic)
    selected_indices = np.argmax(scores, axis=1)
    selected_embeddings = candidate_tensor[np.arange(len(semantic)), selected_indices]
    selected_text = [runs[index]["generated"][row] for row, index in enumerate(selected_indices)]
    result = {
        "step": 0,
        "epoch": 0.0,
        "split": "validation",
        "eval_num_examples": len(source_targets),
        "num_eval_examples": len(source_targets),
        "generated": selected_text,
        "targets": source_targets,
        "selection": {
            "method": "maximum cosine(exact-MiniLM(candidate), input_semantic)",
            "uses_reference_text_for_selection": False,
            "source_npz": str(args.source_npz),
            "model_name": args.model_name,
            "candidate_count": len(runs),
            "selection_counts": {
                str(index): int(np.sum(selected_indices == index)) for index in range(len(runs))
            },
            "selection_cosine_mean": float(np.mean(scores[np.arange(len(semantic)), selected_indices])),
        },
        "metrics": {
            "text": text_metrics(selected_text, source_targets),
            "minilm_retrieval": retrieval_metrics(selected_embeddings, target_embeddings),
        },
        "generation_quality": text_metrics(selected_text, source_targets),
        "word_overlap": {
            "summary": text_metrics(selected_text, source_targets),
        },
        "generation_minilm_retrieval": retrieval_metrics(selected_embeddings, target_embeddings),
        "baselines": baselines,
        "targets": source_targets,
        "selected_generated": selected_text,
        "selected_candidate_index": selected_indices.tolist(),
        "selected_cosine": scores[np.arange(len(semantic)), selected_indices].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selection", "metrics", "baselines")}, indent=2))


if __name__ == "__main__":
    main()
