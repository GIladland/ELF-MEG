#!/usr/bin/env python
"""Export MRI2SEM oracle target splits as native semantic-to-ELF NPZs."""

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


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="uts01_tang_gpt_lanczos_seg10_oracle3072")
    parser.add_argument(
        "--layout",
        choices=["separate", "train-val", "both"],
        default="both",
        help="Write per-split files, the train+val file consumed by ELF, or both.",
    )
    parser.add_argument(
        "--predicted-test-npz",
        default="",
        help="Optional ELF-format MRI2SEM prediction NPZ to verify row-for-row test alignment.",
    )
    parser.add_argument("--expected-dim", type=int, default=3072)
    parser.add_argument("--delay-dim", type=int, default=768)
    parser.add_argument("--expected-train", type=int, default=11725)
    parser.add_argument("--expected-val", type=int, default=266)
    parser.add_argument("--expected-test", type=int, default=107)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text.lower()))


def load_split(data: np.lib.npyio.NpzFile, split: str, expected_n: int, expected_dim: int) -> dict[str, np.ndarray]:
    source_keys = {
        "input_embeddings": f"{split}_y",
        "sentence": f"{split}_text",
        "story": f"{split}_story",
        "start_tr": f"{split}_start_tr",
        "stop_tr": f"{split}_stop_tr",
    }
    missing = [source_key for source_key in source_keys.values() if source_key not in data.files]
    if missing:
        raise KeyError(f"Missing {split} keys in source NPZ: {missing}")

    arrays = {output_key: np.asarray(data[source_key]) for output_key, source_key in source_keys.items()}
    embeddings = np.asarray(arrays["input_embeddings"], dtype=np.float32)
    if embeddings.shape != (expected_n, expected_dim):
        raise ValueError(
            f"Expected {split}_y shape {(expected_n, expected_dim)}, got {embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{split}_y contains non-finite values.")
    for key in ("sentence", "story", "start_tr", "stop_tr"):
        if arrays[key].shape != (expected_n,):
            raise ValueError(f"Expected {split} {key} shape {(expected_n,)}, got {arrays[key].shape}.")

    arrays["input_embeddings"] = embeddings
    arrays["sentence"] = np.asarray(strings(arrays["sentence"]), dtype=object)
    arrays["story"] = np.asarray(strings(arrays["story"]), dtype=object)
    arrays["start_tr"] = np.asarray(arrays["start_tr"], dtype=np.int64)
    arrays["stop_tr"] = np.asarray(arrays["stop_tr"], dtype=np.int64)
    arrays["split"] = np.full((expected_n,), split, dtype=object)
    return arrays


def split_summary(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    embeddings = arrays["input_embeddings"]
    word_counts = np.asarray([word_count(text) for text in strings(arrays["sentence"])])
    norms = np.linalg.norm(embeddings, axis=1)
    # MRI2SEM stores inclusive start/stop TR indices, so 10 TRs span stop-start=9.
    segment_lengths = arrays["stop_tr"] - arrays["start_tr"] + 1
    return {
        "n": int(embeddings.shape[0]),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "finite": bool(np.isfinite(embeddings).all()),
        "normalized": False,
        "vector_norm": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
        "word_count": {
            "min": int(word_counts.min()),
            "median": float(np.median(word_counts)),
            "mean": float(word_counts.mean()),
            "max": int(word_counts.max()),
        },
        "segment_trs": sorted(int(value) for value in np.unique(segment_lengths)),
        "unique_stories": sorted(set(strings(arrays["story"]))),
    }


def schema_for(
    *,
    dataset_path: Path,
    metadata_path: Path,
    metadata: dict,
    split_names: list[str],
    arrays: dict[str, np.ndarray],
    delay_dim: int,
) -> dict[str, object]:
    embedding_dim = int(arrays["input_embeddings"].shape[1])
    delays = metadata.get("stim_delays") or metadata.get("args", {}).get("stim_delays")
    return {
        "source_dataset_npz": str(dataset_path),
        "source_metadata_json": str(metadata_path),
        "source_target_keys": [f"{split}_y" for split in split_names],
        "representation": "oracle_native_4delay_tang_openai_gpt_segment_mean",
        "input_key": "input_embeddings",
        "sentence_key": "sentence",
        "metadata_keys": ["story", "start_tr", "stop_tr", "split"],
        "embedding_family": "Tang/OpenAI-GPT",
        "tang_gpt_layer": metadata.get("args", {}).get("tang_gpt_layer", 9),
        "tang_gpt_context_words": metadata.get("args", {}).get("tang_gpt_context_words", 5),
        "embedding_dim": embedding_dim,
        "delay_embedding_dim": delay_dim,
        "delay_count": embedding_dim // delay_dim,
        "delay_order": delays,
        "concatenation": "np.hstack([delay1, delay2, delay3, delay4])",
        "segment_trs": metadata.get("segment_trs"),
        "tr_seconds": metadata.get("args", {}).get("tr", 2.0),
        "target_pooling": metadata.get("target_pooling"),
        "train_segment_stride_trs": metadata.get("train_segment_stride_trs"),
        "eval_segment_stride_trs": metadata.get("eval_segment_stride_trs"),
        "normalized": False,
        "split_order": split_names,
        "shape": list(arrays["input_embeddings"].shape),
    }


def save_elf_npz(path: Path, arrays: dict[str, np.ndarray], schema: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        input_embeddings=arrays["input_embeddings"].astype(np.float32, copy=False),
        sentence=arrays["sentence"],
        story=arrays["story"],
        start_tr=arrays["start_tr"],
        stop_tr=arrays["stop_tr"],
        split=arrays["split"],
        schema_json=np.asarray(json.dumps(schema, indent=2), dtype=object),
    )


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ("input_embeddings", "sentence", "story", "start_tr", "stop_tr", "split")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def validate_native_contract(
    metadata: dict,
    split_arrays: dict[str, dict[str, np.ndarray]],
    *,
    expected_dim: int,
    delay_dim: int,
) -> None:
    args = metadata.get("args") or {}
    checks = {
        "stim_delays": (metadata.get("stim_delays"), [1, 2, 3, 4]),
        "segment_trs": (metadata.get("segment_trs"), 10),
        "target_pooling": (metadata.get("target_pooling"), "mean"),
        "tr": (args.get("tr"), 2.0),
        "tang_gpt_layer": (args.get("tang_gpt_layer"), 9),
        "tang_gpt_context_words": (args.get("tang_gpt_context_words"), 5),
        "delay_count": (expected_dim // delay_dim, 4),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    for split, arrays in split_arrays.items():
        # start_tr and stop_tr are inclusive in the MRI2SEM segment dataset.
        lengths = np.unique(arrays["stop_tr"] - arrays["start_tr"] + 1).tolist()
        if lengths != [10]:
            mismatches[f"{split}_segment_lengths"] = {"actual": lengths, "expected": [10]}
    if mismatches:
        raise ValueError(f"Source does not satisfy the native 3072 Tang-GPT contract: {mismatches}")


def verify_predicted_alignment(predicted_path: Path, oracle_test: dict[str, np.ndarray], expected_dim: int) -> dict:
    with np.load(predicted_path, allow_pickle=True) as predicted:
        required = ("input_embeddings", "sentence", "story", "start_tr", "stop_tr")
        missing = [key for key in required if key not in predicted.files]
        if missing:
            raise KeyError(f"Predicted test NPZ is missing keys: {missing}")
        predicted_vectors = np.asarray(predicted["input_embeddings"])
        expected_shape = (oracle_test["input_embeddings"].shape[0], expected_dim)
        if predicted_vectors.shape != expected_shape:
            raise ValueError(f"Predicted test shape {predicted_vectors.shape} != {expected_shape}.")
        if not np.isfinite(predicted_vectors).all():
            raise ValueError("Predicted test vectors contain non-finite values.")

        checks = {
            "sentence": strings(predicted["sentence"]) == strings(oracle_test["sentence"]),
            "story": strings(predicted["story"]) == strings(oracle_test["story"]),
            "start_tr": np.array_equal(predicted["start_tr"], oracle_test["start_tr"]),
            "stop_tr": np.array_equal(predicted["stop_tr"], oracle_test["stop_tr"]),
        }
        if not all(checks.values()):
            raise ValueError(f"Predicted and oracle test rows are not aligned: {checks}")
        return {
            "path": str(predicted_path),
            "shape": list(predicted_vectors.shape),
            "finite": True,
            "row_alignment": checks,
        }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_npz)
    metadata_path = Path(args.metadata_json)
    output_dir = Path(args.output_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if args.expected_dim % args.delay_dim != 0:
        raise ValueError("--expected-dim must be divisible by --delay-dim.")
    expected_counts = {
        "train": args.expected_train,
        "val": args.expected_val,
        "test": args.expected_test,
    }
    metadata_shapes = metadata.get("shapes") or {}
    for split, expected_n in expected_counts.items():
        declared = metadata_shapes.get(f"{split}_y")
        if declared is not None and list(declared) != [expected_n, args.expected_dim]:
            raise ValueError(
                f"Metadata declares {split}_y={declared}, expected {[expected_n, args.expected_dim]}."
            )

    with np.load(dataset_path, allow_pickle=True) as source:
        split_arrays = {
            split: load_split(source, split, expected_counts[split], args.expected_dim)
            for split in SPLITS
        }
    validate_native_contract(
        metadata,
        split_arrays,
        expected_dim=args.expected_dim,
        delay_dim=args.delay_dim,
    )

    created: list[str] = []
    if args.layout in {"separate", "both"}:
        for split in SPLITS:
            arrays = split_arrays[split]
            path = output_dir / f"{args.prefix}_{split}.npz"
            schema = schema_for(
                dataset_path=dataset_path,
                metadata_path=metadata_path,
                metadata=metadata,
                split_names=[split],
                arrays=arrays,
                delay_dim=args.delay_dim,
            )
            save_elf_npz(path, arrays, schema)
            created.append(str(path))

    if args.layout in {"train-val", "both"}:
        train_val = concatenate([split_arrays["train"], split_arrays["val"]])
        path = output_dir / f"{args.prefix}_train_val.npz"
        schema = schema_for(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            metadata=metadata,
            split_names=["train", "val"],
            arrays=train_val,
            delay_dim=args.delay_dim,
        )
        schema["elf_training_contract"] = {
            "train_rows": args.expected_train,
            "validation_rows": args.expected_val,
            "validation_is_tail": True,
            "trainer_argument": f"--val-num-examples {args.expected_val}",
            "test_excluded": True,
        }
        save_elf_npz(path, train_val, schema)
        created.append(str(path))

    predicted_alignment = None
    if args.predicted_test_npz:
        predicted_alignment = verify_predicted_alignment(
            Path(args.predicted_test_npz), split_arrays["test"], args.expected_dim
        )

    summary = {
        "created": created,
        "dataset_npz": str(dataset_path),
        "metadata_json": str(metadata_path),
        "embedding_contract": {
            "shape": args.expected_dim,
            "meaning": f"{args.expected_dim // args.delay_dim} x {args.delay_dim}",
            "order": "np.hstack([delay1, delay2, delay3, delay4])",
            "delays": metadata.get("stim_delays"),
            "normalized": False,
        },
        "splits": {split: split_summary(arrays) for split, arrays in split_arrays.items()},
        "predicted_test_alignment": predicted_alignment,
    }
    summary_path = output_dir / f"{args.prefix}_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
