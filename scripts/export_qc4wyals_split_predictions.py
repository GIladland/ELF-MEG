#!/usr/bin/env python3
"""Export qc4wyals MEG->ADA predictions for train/val/test packs.

The script imports the canonical MEG2SEM exporter so checkpoint reconstruction,
residual handling, and normalization remain identical to the selected test
export. Outputs are semantic-only and suitable for ELF interface calibration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meg2sem-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-npz", type=Path)
    parser.add_argument("--val-npz", type=Path)
    parser.add_argument("--test-npz", type=Path)
    parser.add_argument(
        "--single-npz",
        type=Path,
        help="Export one named split instead of the canonical train/val/test triplet.",
    )
    parser.add_argument("--single-name", default="single")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_canonical_module(repo: Path):
    path = repo / "utils" / "select_and_export_tang_themoth_apples.py"
    spec = importlib.util.spec_from_file_location("tang_export", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import canonical exporter from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # module body executes, so dynamic imports must be registered first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    canonical = load_canonical_module(args.meg2sem_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.single_npz is not None:
        sources = {args.single_name: args.single_npz}
    else:
        missing = [
            name
            for name, path in (("train", args.train_npz), ("val", args.val_npz), ("test", args.test_npz))
            if path is None
        ]
        if missing:
            raise ValueError(
                "Supply --single-npz or all three canonical splits; missing " + ", ".join(missing)
            )
        sources = {"train": args.train_npz, "val": args.val_npz, "test": args.test_npz}
    summary: dict[str, object] = {
        "schema": "qc4wyals_split_predictions_v1",
        "checkpoint": str(args.checkpoint),
        "splits": {},
    }
    for split, path in sources.items():
        print(f"loading split={split} path={path}", flush=True)
        pack = canonical.load_pack(path)
        prediction = canonical.predict_checkpoint(
            args.checkpoint,
            pack,
            batch_size=args.batch_size,
            device=device,
        )
        target = np.asarray(pack["target"], dtype=np.float32)
        prediction_n = canonical.normalize_rows(prediction)
        target_n = canonical.normalize_rows(target)
        metrics = canonical.retrieval_metrics(prediction, target)
        output = args.output_dir / f"qc4wyals_{split}_predicted_ada002.npz"
        extra_metadata: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=True) as source:
            for key in ("source_original_index", "oof_outer_fold", "oof_role"):
                if key in source.files:
                    extra_metadata[key] = np.asarray(source[key])
        np.savez_compressed(
            output,
            input_embeddings=prediction_n,
            predicted_embeddings_raw=prediction.astype(np.float32),
            target_embeddings=target_n,
            target_embeddings_raw=target,
            sentence=np.asarray(pack["sentence"], dtype=object),
            session=np.asarray(pack["session"], dtype=object),
            run=np.asarray(pack["run"], dtype=object),
            start_samples=np.asarray(pack["start_samples"], dtype=np.int64),
            source_split=np.full(prediction.shape[0], split, dtype=object),
            retrieval_rank=np.asarray(metrics["ranks"], dtype=np.int64),
            selected_checkpoint=np.asarray(str(args.checkpoint), dtype=object),
            **extra_metadata,
            metrics_json=np.asarray(
                json.dumps(
                    {k: v for k, v in metrics.items() if k not in {"ranks", "similarity"}},
                    sort_keys=True,
                ),
                dtype=object,
            ),
        )
        metric_summary = {
            k: v for k, v in metrics.items() if k not in {"ranks", "similarity"}
        }
        summary["splits"][split] = {"output": str(output), **metric_summary}
        print(f"wrote={output} metrics={json.dumps(metric_summary, sort_keys=True)}", flush=True)
        del pack, prediction, target, prediction_n, target_n
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_name = (
        f"qc4wyals_{args.single_name}_prediction_summary.json"
        if args.single_npz is not None
        else "qc4wyals_split_predictions_summary.json"
    )
    (args.output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
