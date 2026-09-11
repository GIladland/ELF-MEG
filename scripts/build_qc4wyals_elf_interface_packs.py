#!/usr/bin/env python3
"""Build leakage-auditable qc4wyals ADA packs for ELF interface training.

The final MEG test vectors are written only to standalone evaluation files.
Training packs contain exact semantic/text rows plus train MEG predictions;
validation MEG predictions are always the final 110 rows and are held out via
``--val-num-examples 110`` in the ELF trainer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


SESSION_TO_STORY = {
    "6": "thatthingonmyarm",
    "22": "leavingbaghdad",
    "24": "howtodraw",
    "19": "birthofanation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--knowntext-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--residual-scales",
        default="0.5,1.0,1.5",
        help="Train-residual multipliers used for exact-ADA robustness copies.",
    )
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            for value in np.asarray(values).tolist()
        ],
        dtype=object,
    )


def load_prediction(path: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        required = {
            "input_embeddings",
            "target_embeddings",
            "sentence",
            "session",
            "run",
            "start_samples",
        }
        missing = required.difference(source.files)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        predicted = normalize(source["input_embeddings"])
        exact = normalize(source["target_embeddings"])
        sentence = strings(source["sentence"])
        session = strings(source["session"])
        run = strings(source["run"])
        start = np.asarray(source["start_samples"], dtype=np.int64)
    if predicted.shape != exact.shape or predicted.ndim != 2 or predicted.shape[1] != 1536:
        raise ValueError(f"Unexpected predicted/exact shapes in {path}: {predicted.shape}, {exact.shape}")
    n = predicted.shape[0]
    if any(len(value) != n for value in (sentence, session, run, start)):
        raise ValueError(f"Metadata row mismatch in {path}")
    story = np.asarray([SESSION_TO_STORY.get(value, f"session_{value}") for value in session], dtype=object)
    return {
        "predicted": predicted,
        "exact": exact,
        "sentence": sentence,
        "session": session,
        "run": run,
        "start_samples": start,
        "story": story,
        "source_split": np.full(n, split, dtype=object),
    }


def row_metadata(part: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(part["sentence"])
    return {
        "sentence": part["sentence"],
        "session": part["session"],
        "run": part["run"],
        "start_samples": part["start_samples"],
        "story": part["story"],
        "start_tr": part["start_samples"],
        "stop_tr": part["start_samples"] + 1,
        "split": part["source_split"],
        "source_split": part["source_split"],
    }


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set.intersection(*(set(part) for part in parts))
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def write_pack(
    path: Path,
    embeddings: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    condition_source: np.ndarray,
    schema: dict[str, object],
    exact_targets: np.ndarray | None = None,
) -> None:
    embeddings = normalize(embeddings)
    n = embeddings.shape[0]
    if len(condition_source) != n or any(len(value) != n for value in metadata.values()):
        raise ValueError(f"Row mismatch while writing {path}")
    payload: dict[str, np.ndarray] = {
        "input_embeddings": embeddings,
        **metadata,
        "condition_source": np.asarray(condition_source, dtype=object),
        "schema_json": np.asarray(json.dumps(schema, sort_keys=True)),
    }
    if exact_targets is not None:
        if np.asarray(exact_targets).shape != embeddings.shape:
            raise ValueError(f"Exact-target shape mismatch while writing {path}")
        payload["source_vectors"] = normalize(exact_targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A compressed NPZ can take long enough to write that another scheduled
    # job may try to open it mid-write.  Publish only a completely written
    # archive by replacing the destination atomically on the same filesystem.
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary_path, **payload)
        with np.load(temporary_path, allow_pickle=True) as audit:
            if set(audit.files) != set(payload):
                raise RuntimeError(
                    f"Temporary NPZ key mismatch for {path}: "
                    f"expected={sorted(payload)} actual={sorted(audit.files)}"
                )
            if audit["input_embeddings"].shape != embeddings.shape:
                raise RuntimeError(
                    f"Temporary NPZ shape mismatch for {path}: "
                    f"expected={embeddings.shape} actual={audit['input_embeddings'].shape}"
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"wrote={path} rows={n} dim={embeddings.shape[1]}", flush=True)


def exact_knowntext_rows(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as source:
        exact = normalize(source["input_embeddings"])
        sentence = strings(source["sentence"])
        split = strings(source["source_split"])
    n = len(sentence)
    blank = np.full(n, "", dtype=object)
    zeros = np.zeros(n, dtype=np.int64)
    metadata = {
        "sentence": sentence,
        "session": blank,
        "run": blank,
        "start_samples": zeros,
        "story": np.asarray([f"exact_{value}" for value in split], dtype=object),
        "start_tr": zeros,
        "stop_tr": zeros,
        "split": split,
        "source_split": split,
    }
    return exact, metadata


def main() -> None:
    args = parse_args()
    parts = {
        split: load_prediction(
            args.prediction_dir / f"qc4wyals_{split}_predicted_ada002.npz",
            split,
        )
        for split in ("train", "val", "test")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Standalone split files. The test file is never concatenated into a train pack.
    for split, part in parts.items():
        meta = row_metadata(part)
        condition = np.full(len(part["sentence"]), f"qc4wyals_{split}_prediction", dtype=object)
        base_schema = {
            "schema": "qc4wyals_ada_elf_split_v1",
            "split": split,
            "test_brain_vectors_trainable": False,
        }
        write_pack(
            args.output_dir / f"qc4wyals_{split}_predicted_ada002.npz",
            part["predicted"],
            meta,
            condition_source=condition,
            schema={**base_schema, "role": "predicted_condition"},
            exact_targets=part["exact"],
        )
        write_pack(
            args.output_dir / f"qc4wyals_{split}_exact_ada002.npz",
            part["exact"],
            meta,
            condition_source=np.full(len(condition), f"exact_{split}", dtype=object),
            schema={**base_schema, "role": "exact_alignment_target"},
        )

    train_val_meta = concatenate([row_metadata(parts["train"]), row_metadata(parts["val"])])
    train_val_pred = np.concatenate([parts["train"]["predicted"], parts["val"]["predicted"]])
    train_val_exact = np.concatenate([parts["train"]["exact"], parts["val"]["exact"]])
    train_val_condition = np.concatenate(
        [
            np.full(len(parts["train"]["sentence"]), "qc4wyals_train_prediction", dtype=object),
            np.full(len(parts["val"]["sentence"]), "qc4wyals_val_prediction", dtype=object),
        ]
    )
    train_val_schema = {
        "schema": "qc4wyals_ada_train_valtail_v1",
        "train_rows": int(len(parts["train"]["sentence"])),
        "validation_tail_rows": int(len(parts["val"]["sentence"])),
        "final_test_rows_included": 0,
    }
    write_pack(
        args.output_dir / "qc4wyals_predicted_train_valtail_ada002.npz",
        train_val_pred,
        train_val_meta,
        condition_source=train_val_condition,
        schema={**train_val_schema, "role": "ELF_interface_input"},
        exact_targets=train_val_exact,
    )
    write_pack(
        args.output_dir / "qc4wyals_exact_alignment_train_valtail_ada002.npz",
        train_val_exact,
        train_val_meta,
        condition_source=np.asarray([f"exact_{value}" for value in train_val_meta["split"]], dtype=object),
        schema={**train_val_schema, "role": "semantic_alignment_target"},
    )

    exact_all, exact_all_meta = exact_knowntext_rows(args.knowntext_npz)
    pred_train_meta = row_metadata(parts["train"])
    pred_val_meta = row_metadata(parts["val"])
    mixed_meta = concatenate([exact_all_meta, pred_train_meta, pred_val_meta])
    mixed_embeddings = np.concatenate([exact_all, parts["train"]["predicted"], parts["val"]["predicted"]])
    mixed_targets = np.concatenate([exact_all, parts["train"]["exact"], parts["val"]["exact"]])
    mixed_condition = np.concatenate(
        [
            np.full(len(exact_all), "exact_knowntext", dtype=object),
            np.full(len(parts["train"]["sentence"]), "qc4wyals_train_prediction", dtype=object),
            np.full(len(parts["val"]["sentence"]), "qc4wyals_val_prediction", dtype=object),
        ]
    )
    write_pack(
        args.output_dir / "qc4wyals_mixed_exactall_predtrain_valtail_ada002.npz",
        mixed_embeddings,
        mixed_meta,
        condition_source=mixed_condition,
        schema={
            "schema": "qc4wyals_mixed_exactall_predtrain_valtail_v1",
            "exact_knowntext_rows": int(len(exact_all)),
            "predicted_train_rows": int(len(parts["train"]["sentence"])),
            "validation_tail_rows": int(len(parts["val"]["sentence"])),
            "predicted_test_rows": 0,
        },
        exact_targets=mixed_targets,
    )
    write_pack(
        args.output_dir / "qc4wyals_mixed_exact_alignment_valtail_ada002.npz",
        mixed_targets,
        mixed_meta,
        condition_source=np.full(len(mixed_targets), "exact_alignment_target", dtype=object),
        schema={"schema": "qc4wyals_mixed_exact_alignment_valtail_v1"},
    )

    # Empirical residual augmentation: reuse only train MEG errors, randomly
    # reassigned across exact train conditions to avoid learning row identity.
    scales = [float(value) for value in args.residual_scales.split(",") if value]
    rng = np.random.default_rng(args.seed)
    train_residual = parts["train"]["predicted"] - parts["train"]["exact"]
    robust_vectors: list[np.ndarray] = []
    robust_targets: list[np.ndarray] = []
    robust_meta_parts: list[dict[str, np.ndarray]] = []
    robust_sources: list[np.ndarray] = []
    for scale in scales:
        permutation = rng.permutation(len(train_residual))
        perturbed = normalize(parts["train"]["exact"] + np.float32(scale) * train_residual[permutation])
        robust_vectors.append(perturbed)
        robust_targets.append(parts["train"]["exact"])
        robust_meta_parts.append(pred_train_meta)
        robust_sources.append(np.full(len(perturbed), f"train_residual_aug_scale_{scale:g}", dtype=object))

    robust_meta = concatenate([exact_all_meta, pred_train_meta, *robust_meta_parts, pred_val_meta])
    robust_embeddings = np.concatenate(
        [exact_all, parts["train"]["predicted"], *robust_vectors, parts["val"]["predicted"]]
    )
    robust_exact = np.concatenate(
        [exact_all, parts["train"]["exact"], *robust_targets, parts["val"]["exact"]]
    )
    robust_condition = np.concatenate(
        [
            np.full(len(exact_all), "exact_knowntext", dtype=object),
            np.full(len(parts["train"]["sentence"]), "qc4wyals_train_prediction", dtype=object),
            *robust_sources,
            np.full(len(parts["val"]["sentence"]), "qc4wyals_val_prediction", dtype=object),
        ]
    )
    write_pack(
        args.output_dir / "qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz",
        robust_embeddings,
        robust_meta,
        condition_source=robust_condition,
        schema={
            "schema": "qc4wyals_empirical_train_residual_augmentation_v1",
            "residual_scales": scales,
            "residual_source": "qc4wyals train predictions only",
            "residual_row_assignment": "seeded random permutation",
            "validation_tail_rows": int(len(parts["val"]["sentence"])),
            "predicted_test_rows": 0,
        },
        exact_targets=robust_exact,
    )
    write_pack(
        args.output_dir / "qc4wyals_robust_exact_alignment_valtail_ada002.npz",
        robust_exact,
        robust_meta,
        condition_source=np.full(len(robust_exact), "exact_alignment_target", dtype=object),
        schema={"schema": "qc4wyals_robust_exact_alignment_valtail_v1"},
    )

    summary = {
        "schema": "qc4wyals_elf_interface_pack_summary_v1",
        "rows": {split: int(len(part["sentence"])) for split, part in parts.items()},
        "final_test_brain_vectors_used_for_training": False,
        "knowntext_exact_test_sentences_used_by_ELF_oracle": True,
        "residual_scales": scales,
    }
    (args.output_dir / "interface_pack_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
