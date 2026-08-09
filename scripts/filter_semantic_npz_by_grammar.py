#!/usr/bin/env python
"""Filter a semantic NPZ with a grammatical acceptability classifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="textattack/roberta-base-CoLA")
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--source-index-key", default="augmentation_source_index")
    parser.add_argument("--min-grammar-score", type=float, default=0.70)
    parser.add_argument("--min-source-content-recall", type=float, default=0.55)
    parser.add_argument("--max-per-source", type=int, default=32)
    parser.add_argument("--keep-best-per-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    try:
        return json.loads(str(data[schema_key].tolist()))
    except json.JSONDecodeError:
        return str(data[schema_key].tolist())


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def acceptability_index(model: AutoModelForSequenceClassification) -> int:
    id2label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    for idx, label in id2label.items():
        if "acceptable" in label and "unacceptable" not in label:
            return idx
    for idx, label in id2label.items():
        if label in {"label_1", "1"}:
            return idx
    return 1 if model.config.num_labels > 1 else 0


def sentence_for_scoring(sentence: str) -> str:
    sentence = str(sentence).strip()
    if sentence and sentence[-1] not in ".!?":
        sentence = f"{sentence}."
    return sentence


def score_sentences(
    *,
    sentences: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    acceptable_idx = acceptability_index(model)
    scores = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            end = min(start + batch_size, len(sentences))
            batch = tokenizer(
                [sentence_for_scoring(sentence) for sentence in sentences[start:end]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, acceptable_idx]
            scores.extend(probs.detach().cpu().numpy().tolist())
            print(f"scored {end}/{len(sentences)}", flush=True)
    return np.asarray(scores, dtype=np.float32), {
        "model_name": model_name,
        "id2label": model.config.id2label,
        "acceptable_idx": int(acceptable_idx),
    }


def source_index(row: object, key: str) -> int:
    if isinstance(row, dict) and key in row:
        try:
            return int(row[key])
        except (TypeError, ValueError):
            return -1
    return -1


def source_recall(row: object) -> float:
    if isinstance(row, dict):
        for key in ("source_content_recall", "source_word_recall", "source_content_f1"):
            try:
                value = row.get(key)
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def select_indices(
    *,
    rows: np.ndarray,
    scores: np.ndarray,
    source_index_key: str,
    min_grammar_score: float,
    min_source_content_recall: float,
    max_per_source: int,
    keep_best_per_source: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows.tolist()):
        grouped[source_index(row, source_index_key)].append(idx)

    keep: list[int] = []
    per_source_counts: dict[int, int] = {}
    fallback_kept = 0
    for source_idx, indices in sorted(grouped.items()):
        ranked = sorted(
            indices,
            key=lambda idx: (float(scores[idx]), source_recall(rows[idx])),
            reverse=True,
        )
        selected = [
            idx
            for idx in ranked
            if float(scores[idx]) >= min_grammar_score
            and source_recall(rows[idx]) >= min_source_content_recall
        ][:max_per_source]
        if not selected and keep_best_per_source and ranked:
            best = ranked[0]
            if source_recall(rows[best]) >= min_source_content_recall:
                selected = [best]
                fallback_kept += 1
        keep.extend(selected)
        per_source_counts[int(source_idx)] = len(selected)

    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return keep_arr, {
        "source_count": len(grouped),
        "sources_with_kept": int(sum(1 for value in per_source_counts.values() if value > 0)),
        "fallback_kept": int(fallback_kept),
        "per_source_count_summary": dict(Counter(per_source_counts.values())),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data = np.load(args.input, allow_pickle=True)
    sentences = _strings(data[args.sentence_key])
    rows = data[args.rows_key] if args.rows_key in data.files else np.asarray([{} for _ in sentences], dtype=object)
    scores, model_info = score_sentences(
        sentences=sentences,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        local_files_only=args.local_files_only,
    )
    keep, selection = select_indices(
        rows=rows,
        scores=scores,
        source_index_key=args.source_index_key,
        min_grammar_score=args.min_grammar_score,
        min_source_content_recall=args.min_source_content_recall,
        max_per_source=args.max_per_source,
        keep_best_per_source=args.keep_best_per_source,
    )
    if len(keep) == 0:
        raise RuntimeError("No rows survived grammar filtering.")

    scored_rows = []
    for idx, row in enumerate(rows.tolist()):
        row_dict = dict(row) if isinstance(row, dict) else {"row": row}
        row_dict["grammar_score"] = float(scores[idx])
        scored_rows.append(row_dict)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "input": args.input,
        "input_count": len(sentences),
        "kept_count": int(len(keep)),
        "min_grammar_score": float(args.min_grammar_score),
        "min_source_content_recall": float(args.min_source_content_recall),
        "max_per_source": int(args.max_per_source),
        "grammar_score_mean": float(scores.mean()),
        "grammar_score_kept_mean": float(scores[keep].mean()),
        "grammar_score_kept_min": float(scores[keep].min()),
        "model": model_info,
        "selection": selection,
        "source_schema": _schema(data, args.schema_key),
        "sample": [scored_rows[int(idx)] for idx in keep[:30].tolist()],
    }
    first_dim_keys = {
        key
        for key in data.files
        if getattr(data[key], "shape", None) and data[key].shape[0] == len(sentences)
    }
    output_kwargs = {}
    for key in first_dim_keys:
        output_kwargs[key] = data[key][keep]
    output_kwargs[args.rows_key] = np.asarray([scored_rows[int(idx)] for idx in keep.tolist()], dtype=object)
    output_kwargs[args.sentence_key] = np.asarray([sentences[int(idx)] for idx in keep.tolist()], dtype=object)
    output_kwargs[args.schema_key] = np.asarray(json.dumps(summary))
    np.savez_compressed(output, **output_kwargs)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
