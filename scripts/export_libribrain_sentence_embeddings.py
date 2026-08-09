#!/usr/bin/env python
"""Export LibriBrain Sherlock sentence rows with local sentence embeddings."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import (
    encode_sentences,
    resolve_device,
)

import torch
from transformers import AutoModel, AutoTokenizer


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


SESSION_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_semantic_vectors\.npz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--books", nargs="*", type=int, default=[])
    parser.add_argument("--sherlock1-sessions", nargs="*", type=int, default=[])
    parser.add_argument("--set-name", default="sentences")
    parser.add_argument("--source-name", default="sentence")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def build_sherlock_run_keys(book_idx: int) -> list[tuple[str, str, str, str]]:
    if book_idx == 9:
        return [("0", str(i), "Sherlock9", "1") for i in range(0, 13)]
    if book_idx == 8:
        return [("0", str(i), "Sherlock8", "1") for i in range(1, 11)]
    if book_idx == 7:
        return [("0", str(i), "Sherlock7", "1") for i in range(1, 15)]
    if book_idx == 6:
        return [("0", str(i), "Sherlock6", "1") for i in range(1, 14) if i not in (3, 10)]
    if book_idx == 5:
        return [("0", str(i), "Sherlock5", "1") for i in range(1, 16)]
    if book_idx == 4:
        return [("0", str(i), "Sherlock4", "1") for i in range(1, 13) if i not in (2, 9)]
    if book_idx == 3:
        return [("0", str(i), "Sherlock3", "1") for i in range(1, 13) if i != 2]
    if book_idx == 2:
        return [("0", str(i), "Sherlock2", "1") for i in range(1, 13) if i != 2]
    if book_idx == 1:
        return [("0", str(i), "Sherlock1", "1") for i in range(1, 11)]
    raise ValueError(f"Unsupported Sherlock book index: {book_idx}")


def build_sherlock1_session_run_keys(sessions: list[int]) -> list[tuple[str, str, str, str]]:
    run_keys = []
    for session in sessions:
        session_id = str(session)
        run = "2" if session_id in {"11", "12"} else "1"
        run_keys.append(("0", session_id, "Sherlock1", run))
    return run_keys


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def load_sentences_for_run_key(
    *,
    data_root: Path,
    run_key: tuple[str, str, str, str],
    set_name: str,
    source_name: str,
) -> tuple[list[str], list[dict]]:
    subject, session, task, run = run_key
    path = (
        data_root
        / task
        / "derivatives"
        / "events"
        / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_semantic_vectors.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        if "lexical_element" not in z.files:
            raise KeyError(f"{path} has no lexical_element")
        lexical = _strings(z["lexical_element"])
        sets = _strings(z["set"]) if "set" in z.files else [""] * len(lexical)
        sources = _strings(z["source"]) if "source" in z.files else [""] * len(lexical)
        sentences = []
        rows = []
        for idx, text in enumerate(lexical):
            sentence = text.strip()
            if not sentence:
                continue
            if set_name and sets[idx] != set_name:
                continue
            if source_name and sources[idx] != source_name:
                continue
            sentences.append(sentence)
            rows.append(
                {
                    "path": str(path),
                    "book": task,
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "run": run,
                    "index": idx,
                    "sentence": sentence,
                }
            )
    return sentences, rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    run_keys: list[tuple[str, str, str, str]] = []
    for book in args.books:
        run_keys.extend(build_sherlock_run_keys(book))
    run_keys.extend(build_sherlock1_session_run_keys(args.sherlock1_sessions))
    if not run_keys:
        raise ValueError("Provide --books and/or --sherlock1-sessions.")

    sentences: list[str] = []
    rows: list[dict] = []
    seen: set[str] = set()
    dropped_duplicates = 0
    for run_key in run_keys:
        run_sentences, run_rows = load_sentences_for_run_key(
            data_root=data_root,
            run_key=run_key,
            set_name=args.set_name,
            source_name=args.source_name,
        )
        for sentence, row in zip(run_sentences, run_rows):
            key = sentence.strip()
            if args.dedupe and key in seen:
                dropped_duplicates += 1
                continue
            seen.add(key)
            sentences.append(sentence)
            rows.append(row)
        print(f"loaded {len(run_sentences)} sentences from {run_key}", flush=True)

    if not sentences:
        raise RuntimeError("No sentences collected.")

    embedding_inputs = [f"{args.input_prefix}{sentence}" for sentence in sentences]
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = encode_sentences(
            sentences=embedding_inputs,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pooling=args.pooling,
            normalize=not args.no_normalize,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(output),
        "data_root": str(data_root),
        "books": args.books,
        "sherlock1_sessions": args.sherlock1_sessions,
        "run_keys": run_keys,
        "set_name": args.set_name,
        "source_name": args.source_name,
        "count": len(sentences),
        "dropped_duplicates": dropped_duplicates,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "input_prefix": args.input_prefix,
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "sample": sentences[:5],
    }
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(sentences, dtype=object),
        rows=np.asarray(rows, dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
