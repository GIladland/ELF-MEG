#!/usr/bin/env python
"""Export MRI2SEM segment predictions into simple NPZs for ELF bridge tests."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--segment-json", required=True)
    parser.add_argument("--tr-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="mri2sem_tang_gpt_lanczos_seg10_test")
    parser.add_argument("--vector-key", choices=["pred", "target"], default="pred")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text.lower()))


def save_npz(
    *,
    output_path: Path,
    embeddings: np.ndarray,
    sentences: list[str],
    stories: np.ndarray,
    start_tr: np.ndarray,
    stop_tr: np.ndarray,
    source_vectors: np.ndarray,
    schema: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        input_embeddings=embeddings.astype(np.float32, copy=False),
        sentence=np.asarray(sentences, dtype=object),
        story=stories,
        start_tr=start_tr,
        stop_tr=stop_tr,
        source_vectors=source_vectors.astype(np.float32, copy=False),
        schema_json=np.asarray(json.dumps(schema, indent=2), dtype=object),
    )


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions_npz)
    out_dir = Path(args.output_dir)
    pred_npz = np.load(predictions_path, allow_pickle=True)
    segment_meta = json.loads(Path(args.segment_json).read_text(encoding="utf-8"))
    tr_meta = json.loads(Path(args.tr_json).read_text(encoding="utf-8"))

    vectors = np.asarray(pred_npz[args.vector_key], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] % args.embedding_dim != 0:
        raise ValueError(f"Expected 2D vectors with dim multiple of {args.embedding_dim}; got {vectors.shape}")
    delays = vectors.shape[1] // args.embedding_dim
    sentences = strings(pred_npz["text"])
    word_counts = [word_count(text) for text in sentences]
    stories = pred_npz["story"]
    start_tr = pred_npz["start_tr"]
    stop_tr = pred_npz["stop_tr"]

    base_schema = {
        "source_predictions_npz": str(predictions_path),
        "segment_json": str(Path(args.segment_json)),
        "tr_json": str(Path(args.tr_json)),
        "vector_key": args.vector_key,
        "source_vector_shape": list(vectors.shape),
        "sentence_key": "sentence",
        "input_key": "input_embeddings",
        "story_key": "story",
        "start_tr_key": "start_tr",
        "stop_tr_key": "stop_tr",
        "text_warning": (
            "MRI2SEM segment text is a 10-TR aggregation of per-TR text windows; "
            "it is not the same 8-18 word chunking used by the Tang-GPT ELF overfit dataset."
        ),
        "word_count_min": int(min(word_counts)) if word_counts else 0,
        "word_count_median": float(np.median(word_counts)) if word_counts else 0.0,
        "word_count_mean": float(np.mean(word_counts)) if word_counts else 0.0,
        "word_count_max": int(max(word_counts)) if word_counts else 0,
        "unique_stories": sorted(set(strings(stories))),
        "segment_meta": {
            key: segment_meta.get(key)
            for key in [
                "target_label",
                "target_kind",
                "target_representation",
                "segment_trs",
                "segment_stride_trs",
                "train_segment_stride_trs",
                "eval_segment_stride_trs",
                "input_pooling",
                "target_pooling",
                "response_lags",
                "stim_delays",
                "n_voxels",
                "shapes",
            ]
        },
        "tr_meta": {
            key: tr_meta.get(key)
            for key in [
                "subject",
                "target_kind",
                "tang_gpt_layer",
                "tang_gpt_context_words",
                "tang_trim_start",
                "tang_trim_end",
                "lanczos_window",
                "tr",
                "text_window_sec",
                "train_stories",
                "val_stories",
                "test_stories",
            ]
        },
    }

    full = l2_normalize(vectors) if args.normalize else vectors
    full_schema = {
        **base_schema,
        "representation": f"{args.vector_key}_native_{vectors.shape[1]}",
        "embedding_shape": list(full.shape),
        "normalized": bool(args.normalize),
    }
    full_path = out_dir / f"{args.prefix}_{args.vector_key}3072.npz"
    save_npz(
        output_path=full_path,
        embeddings=full,
        sentences=sentences,
        stories=stories,
        start_tr=start_tr,
        stop_tr=stop_tr,
        source_vectors=vectors,
        schema=full_schema,
    )

    avg = vectors.reshape(vectors.shape[0], delays, args.embedding_dim).mean(axis=1)
    avg = l2_normalize(avg)
    avg_schema = {
        **base_schema,
        "representation": f"{args.vector_key}_delay_mean_{args.embedding_dim}",
        "embedding_shape": list(avg.shape),
        "delay_count": int(delays),
        "delay_order": segment_meta.get("stim_delays"),
        "delay_reduction": "reshape(n, delay_count, embedding_dim).mean(axis=1), then L2 normalize",
        "normalized": True,
    }
    avg_path = out_dir / f"{args.prefix}_{args.vector_key}avg768.npz"
    save_npz(
        output_path=avg_path,
        embeddings=avg,
        sentences=sentences,
        stories=stories,
        start_tr=start_tr,
        stop_tr=stop_tr,
        source_vectors=vectors,
        schema=avg_schema,
    )

    summary = {
        "created": [str(full_path), str(avg_path)],
        "n": int(vectors.shape[0]),
        "native_dim": int(vectors.shape[1]),
        "avg_dim": int(avg.shape[1]),
        "word_counts": {
            "min": base_schema["word_count_min"],
            "median": base_schema["word_count_median"],
            "mean": base_schema["word_count_mean"],
            "max": base_schema["word_count_max"],
        },
        "sample": [
            {
                "story": str(stories[i]),
                "start_tr": int(start_tr[i]),
                "stop_tr": int(stop_tr[i]),
                "text": sentences[i],
            }
            for i in range(min(5, len(sentences)))
        ],
    }
    summary_path = out_dir / f"{args.prefix}_{args.vector_key}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
