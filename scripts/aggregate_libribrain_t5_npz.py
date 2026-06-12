#!/usr/bin/env python
"""Aggregate existing per-session LibriBrain T5 ELF target arrays into one NPZ."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


SESSION_RE = re.compile(r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_semantic_vectors\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--books", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-sessions", nargs="*", default=[])
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--compressed", action="store_true")
    return parser.parse_args()


def sentence_array(z: np.lib.npyio.NpzFile) -> np.ndarray:
    if "sentences" in z.files:
        arr = z["sentences"]
    elif "lexical_element" in z.files:
        arr = z["lexical_element"]
    else:
        raise KeyError("No sentence field found; expected sentences or lexical_element")
    return np.asarray([
        str(x.decode("utf-8") if isinstance(x, bytes) else x).strip()
        for x in arr.tolist()
    ], dtype=object)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    exclude = set(args.exclude_sessions)
    latents = []
    masks = []
    ids = []
    sentences = []
    rows = []
    paths = []
    for book in args.books:
        events_dir = data_root / f"Sherlock{book}" / "derivatives" / "events"
        for path in sorted(events_dir.glob("*_semantic_vectors.npz")):
            name = path.name
            if "backup" in name or "with_gtr" in name:
                continue
            match = SESSION_RE.search(name)
            if not match:
                continue
            session_key = f"{match.group('task')}_ses-{match.group('session')}_run-{match.group('run')}"
            if session_key in exclude:
                continue
            with np.load(path, allow_pickle=True) as z:
                if not {"t5_latents", "t5_attention_mask", "t5_input_ids"}.issubset(z.files):
                    continue
                sent = sentence_array(z)
                valid = np.asarray([bool(str(s).strip()) for s in sent], dtype=bool)
                if not valid.any():
                    continue
                t5_latents = z["t5_latents"][valid].astype(np.float32)
                t5_mask = z["t5_attention_mask"][valid].astype(np.float32)
                t5_ids = z["t5_input_ids"][valid].astype(np.int64)
                sent = sent[valid]
                latents.append(t5_latents)
                masks.append(t5_mask)
                ids.append(t5_ids)
                sentences.extend(sent.tolist())
                kept_indices = np.nonzero(valid)[0]
                for local_idx, sentence in zip(kept_indices.tolist(), sent.tolist()):
                    rows.append({
                        "path": str(path),
                        "book": book,
                        "subject": match.group("subject"),
                        "session": match.group("session"),
                        "task": match.group("task"),
                        "run": match.group("run"),
                        "index": int(local_idx),
                        "sentence": str(sentence),
                    })
                paths.append(str(path))
                if args.num_examples > 0 and len(sentences) >= args.num_examples:
                    break
        if args.num_examples > 0 and len(sentences) >= args.num_examples:
            break
    if not latents:
        raise RuntimeError("No T5 latent arrays found")
    target_t5_latents = np.concatenate(latents, axis=0)
    t5_attention_mask = np.concatenate(masks, axis=0)
    t5_input_ids = np.concatenate(ids, axis=0)
    if args.num_examples > 0:
        n = args.num_examples
        target_t5_latents = target_t5_latents[:n]
        t5_attention_mask = t5_attention_mask[:n]
        t5_input_ids = t5_input_ids[:n]
        sentences = sentences[:n]
        rows = rows[:n]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(
        output,
        target_t5_latents=target_t5_latents,
        t5_attention_mask=t5_attention_mask,
        t5_input_ids=t5_input_ids,
        sentence=np.asarray(sentences, dtype=object),
        rows=np.asarray(rows, dtype=object),
    )
    summary = {
        "output": str(output),
        "n": len(sentences),
        "books": args.books,
        "shape": list(target_t5_latents.shape),
        "num_source_files": len(paths),
        "source_files": paths,
    }
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
