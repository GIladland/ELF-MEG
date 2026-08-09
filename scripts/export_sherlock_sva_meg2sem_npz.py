#!/usr/bin/env python
"""Pack Sherlock SVA MiniLM rows with their sentence-aligned LibriBrain MEG."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EVENT_NAME_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_events\.tsv$"
)
SEMANTIC_NAME_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_semantic_vectors(?:_[^.]+)?\.npz$"
)


@dataclass(frozen=True)
class EventSentence:
    sentence_id: str
    sentence: str
    start_time: float
    end_time: float
    word_count: int
    time_column: str


@dataclass(frozen=True)
class ResolvedRow:
    source_index: int
    row: dict[str, Any]
    sentence: str
    embedding: np.ndarray
    event_path: Path
    h5_path: Path
    subject: str
    session: str
    task: str
    run: str
    event_sentence: EventSentence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sva-npz",
        required=True,
        help="Exact Sherlock SVA MiniLM NPZ used for ELF text experiments.",
    )
    parser.add_argument("--output", required=True, help="Packed MEG+MiniLM output NPZ.")
    parser.add_argument(
        "--sidecar-root",
        default=None,
        help=(
            "Optional root for per-run semantic-vector sidecars. The sidecars store "
            "MiniLM vectors under embeddings_ada so MEG2SEM can read them with "
            "--embedding_type ADA --embedding_dim 384 if pointed to this root."
        ),
    )
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--rows-key", default="rows")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--set-name", default="sentences")
    parser.add_argument("--source-name", default="sentence")
    parser.add_argument("--preprocessing-str", default="bads+headpos+sss+notch+bp+ds")
    parser.add_argument("--segment-ms", type=int, default=3000)
    parser.add_argument("--time-column", default="timemeg")
    parser.add_argument(
        "--fallback-time-column",
        action="append",
        default=["timeds", "timesentence"],
        help="Fallback event TSV time column; can be repeated.",
    )
    parser.add_argument(
        "--fallback-data-root",
        action="append",
        default=[],
        help=(
            "Alternative LibriBrain root to search when row paths are stale. "
            "Expected layout: ROOT/TASK/derivatives/events and serialised."
        ),
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite stale row paths by replacing prefix OLD with NEW; can be repeated.",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply MEG2SEM-style global channel standardization over represented H5 runs.",
    )
    parser.add_argument("--clip-boundary", type=float, default=10.0)
    parser.add_argument(
        "--meg-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for packed MEG. float16 is usually enough and halves disk use.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Debug limit. 0 exports all rows.",
    )
    parser.add_argument("--no-packed", action="store_true", help="Only write sidecars.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def row_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    return {"row": _to_python(value)}


def strings(array: np.ndarray) -> list[str]:
    out = []
    for value in array.tolist():
        value = _to_python(value)
        out.append(str(value).strip())
    return out


def normalized_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def normalized_id(value: Any) -> str:
    if value is None:
        return ""
    raw = str(_to_python(value)).strip()
    if raw == "":
        return ""
    try:
        parsed = float(raw)
    except ValueError:
        return raw
    if math.isfinite(parsed) and parsed.is_integer():
        return str(int(parsed))
    return raw


def fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return None if math.isnan(parsed) else parsed
    except (TypeError, ValueError):
        return None


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    maps = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OLD=NEW for --path-map, got {value!r}")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError(f"Empty OLD prefix in --path-map {value!r}")
        maps.append((old.rstrip("/"), new.rstrip("/")))
    return maps


def apply_path_maps(path: str, maps: list[tuple[str, str]]) -> str:
    for old, new in maps:
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def metadata_from_filename(path: Path) -> dict[str, str]:
    for pattern in (EVENT_NAME_RE, SEMANTIC_NAME_RE):
        match = pattern.match(path.name)
        if match:
            return {key: str(value) for key, value in match.groupdict().items()}
    return {}


def metadata_for_row(row: dict[str, Any], path: Path | None = None) -> dict[str, str]:
    meta = {}
    if path is not None:
        meta.update(metadata_from_filename(path))
    for out_key, keys in {
        "subject": ("subject", "sub"),
        "session": ("session", "ses"),
        "task": ("task", "book", "corpus"),
        "run": ("run",),
    }.items():
        if meta.get(out_key):
            continue
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                meta[out_key] = str(_to_python(value))
                break
    return meta


def event_path_from_row(
    row: dict[str, Any],
    *,
    path_maps: list[tuple[str, str]],
    fallback_roots: list[Path],
) -> Path:
    candidates: list[Path] = []
    for key in ("path", "events_path", "event_path", "source_file"):
        value = row.get(key)
        if value in (None, ""):
            continue
        raw = apply_path_maps(str(value), path_maps)
        path = Path(raw)
        if path.name.endswith("_semantic_vectors.npz"):
            stem = re.sub(r"_semantic_vectors(?:_[^.]+)?\.npz$", "_events.tsv", path.name)
            path = path.with_name(stem)
        candidates.append(path)

    path_meta = metadata_for_row(row, candidates[0] if candidates else None)
    subject = path_meta.get("subject", "")
    session = path_meta.get("session", "")
    task = path_meta.get("task", "")
    run = path_meta.get("run", "")
    if subject and session and task and run:
        for root in fallback_roots:
            candidates.append(
                root
                / task
                / "derivatives"
                / "events"
                / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        raise FileNotFoundError(
            "Could not resolve event TSV. Tried: " + ", ".join(str(p) for p in candidates)
        )
    raise KeyError(f"Row has no usable path metadata: {row}")


def h5_path_for_event_path(
    event_path: Path,
    *,
    preprocessing_str: str,
    fallback_roots: list[Path],
) -> Path:
    meta = metadata_from_filename(event_path)
    if not meta:
        raise ValueError(f"Could not parse event filename: {event_path}")
    subject = meta["subject"]
    session = meta["session"]
    task = meta["task"]
    run = meta["run"]
    fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5"
    candidates = [event_path.parents[1] / "serialised" / fname]
    for root in fallback_roots:
        candidates.append(root / task / "derivatives" / "serialised" / fname)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve H5 for event path "
        f"{event_path}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def choose_time(row: dict[str, str], columns: list[str]) -> tuple[float | None, str]:
    for column in columns:
        value = fnum(row.get(column))
        if value is not None:
            return value, column
    return None, ""


def load_event_sentences(
    event_path: Path,
    *,
    time_columns: list[str],
) -> dict[str, EventSentence]:
    groups: dict[str, list[tuple[float, float | None, float, str, str]]] = defaultdict(list)
    with event_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("kind") or "").strip().lower() != "word":
                continue
            sentence_id = normalized_id(row.get("sentenceidx"))
            if not sentence_id:
                continue
            word = str(row.get("segment") or "").strip()
            if not word:
                continue
            word_idx = fnum(row.get("wordidx"))
            time_sentence = fnum(row.get("timesentence"))
            chosen_time, chosen_column = choose_time(row, time_columns)
            duration = fnum(row.get("duration")) or 0.0
            order = word_idx if word_idx is not None else (time_sentence if time_sentence is not None else 0.0)
            groups[sentence_id].append((order, chosen_time, duration, word, chosen_column))

    out: dict[str, EventSentence] = {}
    for sentence_id, items in groups.items():
        ordered = sorted(items, key=lambda item: (item[0], item[1] if item[1] is not None else 0.0))
        sentence = " ".join(item[3] for item in ordered).strip()
        starts = [item[1] for item in ordered if item[1] is not None]
        if not starts:
            continue
        ends = [
            (item[1] + max(0.0, item[2]))
            for item in ordered
            if item[1] is not None
        ]
        start_time = float(min(starts))
        end_time = float(max(ends)) if ends else start_time
        if end_time <= start_time:
            end_time = start_time
        used_columns = [item[4] for item in ordered if item[4]]
        time_column = Counter(used_columns).most_common(1)[0][0] if used_columns else ""
        out[sentence_id] = EventSentence(
            sentence_id=sentence_id,
            sentence=sentence,
            start_time=start_time,
            end_time=end_time,
            word_count=len(WORD_RE.findall(sentence)),
            time_column=time_column,
        )
    return out


def find_event_sentence(
    *,
    row: dict[str, Any],
    sentence: str,
    event_sentences: dict[str, EventSentence],
    used_sentence_ids: set[str],
) -> EventSentence:
    sentence_id = normalized_id(row.get("sentence_id", row.get("sentenceidx")))
    if sentence_id and sentence_id in event_sentences:
        return event_sentences[sentence_id]

    target = normalized_sentence(sentence)
    matches = [
        event_sentence
        for event_sentence in event_sentences.values()
        if normalized_sentence(event_sentence.sentence) == target
    ]
    unused = [match for match in matches if match.sentence_id not in used_sentence_ids]
    if unused:
        return unused[0]
    if matches:
        return matches[0]

    raise KeyError(
        f"Could not match row sentence {sentence!r} to event file; "
        f"row sentence_id={sentence_id!r}"
    )


def load_sva_rows(
    args: argparse.Namespace,
) -> tuple[list[ResolvedRow], dict[str, Any]]:
    source = Path(args.sva_npz)
    with np.load(source, allow_pickle=True) as data:
        if args.input_key not in data.files:
            raise KeyError(f"{source} has no key {args.input_key!r}. Keys: {data.files}")
        if args.sentence_key not in data.files:
            raise KeyError(f"{source} has no key {args.sentence_key!r}. Keys: {data.files}")
        embeddings = np.asarray(data[args.input_key], dtype=np.float32)
        sentences = strings(data[args.sentence_key])
        rows_raw = data[args.rows_key] if args.rows_key in data.files else np.asarray([{}] * len(sentences), dtype=object)
        schema = None
        if args.schema_key in data.files:
            raw_schema = str(data[args.schema_key].tolist())
            try:
                schema = json.loads(raw_schema)
            except json.JSONDecodeError:
                schema = raw_schema

    if len(sentences) != len(embeddings) or len(rows_raw) != len(sentences):
        raise ValueError(
            f"Shape mismatch: sentences={len(sentences)} embeddings={len(embeddings)} rows={len(rows_raw)}"
        )

    n_rows = len(sentences) if args.max_rows <= 0 else min(args.max_rows, len(sentences))
    path_maps = parse_path_maps(args.path_map)
    fallback_roots = [Path(path) for path in args.fallback_data_root]
    time_columns = [args.time_column] + [col for col in args.fallback_time_column if col != args.time_column]

    event_cache: dict[Path, dict[str, EventSentence]] = {}
    used_by_path: dict[Path, set[str]] = defaultdict(set)
    resolved: list[ResolvedRow] = []
    text_mismatch = 0
    for idx in range(n_rows):
        row = row_to_dict(rows_raw[idx])
        sentence = sentences[idx]
        event_path = event_path_from_row(row, path_maps=path_maps, fallback_roots=fallback_roots)
        if event_path not in event_cache:
            event_cache[event_path] = load_event_sentences(event_path, time_columns=time_columns)
        event_sentence = find_event_sentence(
            row=row,
            sentence=sentence,
            event_sentences=event_cache[event_path],
            used_sentence_ids=used_by_path[event_path],
        )
        used_by_path[event_path].add(event_sentence.sentence_id)
        if normalized_sentence(sentence) != normalized_sentence(event_sentence.sentence):
            text_mismatch += 1

        h5_path = h5_path_for_event_path(
            event_path,
            preprocessing_str=args.preprocessing_str,
            fallback_roots=fallback_roots,
        )
        meta = metadata_for_row(row, event_path)
        resolved.append(
            ResolvedRow(
                source_index=idx,
                row=row,
                sentence=sentence,
                embedding=embeddings[idx],
                event_path=event_path,
                h5_path=h5_path,
                subject=meta["subject"],
                session=meta["session"],
                task=meta["task"],
                run=meta["run"],
                event_sentence=event_sentence,
            )
        )
        if (idx + 1) % 1000 == 0:
            print(f"resolved {idx + 1}/{n_rows}", flush=True)

    return resolved, {
        "source_npz": str(source),
        "source_schema": schema,
        "source_examples": int(len(sentences)),
        "export_examples": int(n_rows),
        "event_files": int(len(event_cache)),
        "text_mismatch_count": int(text_mismatch),
        "time_columns": time_columns,
    }


def h5_stats(h5_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int, float]:
    means = []
    stds = []
    n_samples = []
    n_channels = None
    sfreq = None
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as handle:
            dataset = handle["data"]
            if n_channels is None:
                n_channels = int(dataset.shape[0])
            elif int(dataset.shape[0]) != n_channels:
                raise ValueError(f"Channel mismatch in {h5_path}: {dataset.shape[0]} != {n_channels}")
            file_sfreq = float(handle.attrs["sample_frequency"])
            if sfreq is None:
                sfreq = file_sfreq
            elif abs(file_sfreq - sfreq) > 1e-6:
                raise ValueError(f"Sampling-rate mismatch in {h5_path}: {file_sfreq} != {sfreq}")

            if "channel_means" in dataset.attrs and "channel_stds" in dataset.attrs:
                channel_means = np.asarray(dataset.attrs["channel_means"], dtype=np.float64)
                channel_stds = np.asarray(dataset.attrs["channel_stds"], dtype=np.float64)
            else:
                data = dataset[:, :]
                channel_means = np.mean(data, axis=1, dtype=np.float64)
                channel_stds = np.std(data, axis=1, dtype=np.float64)
            means.append(channel_means)
            stds.append(channel_stds)
            n_samples.append(int(dataset.shape[1]))

    if n_channels is None or sfreq is None:
        raise ValueError("No H5 paths to compute stats from.")

    means_arr = np.asarray(means, dtype=np.float64)
    stds_arr = np.asarray(stds, dtype=np.float64)
    n_arr = np.asarray(n_samples, dtype=np.float64)
    vars_arr = stds_arr**2
    mean_total = np.average(means_arr, axis=0, weights=n_arr)
    within = np.sum(vars_arr * n_arr[:, None], axis=0)
    between = np.sum(((means_arr - mean_total[None, :]) ** 2) * n_arr[:, None], axis=0)
    std_total = np.sqrt((within + between) / np.sum(n_arr))
    std_total = np.where(std_total <= 0, 1.0, std_total)
    return mean_total.astype(np.float32), std_total.astype(np.float32), int(n_channels), float(sfreq)


def h5_metadata(h5_paths: list[Path]) -> tuple[int, float]:
    n_channels = None
    sfreq = None
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as handle:
            dataset = handle["data"]
            if n_channels is None:
                n_channels = int(dataset.shape[0])
            elif int(dataset.shape[0]) != n_channels:
                raise ValueError(f"Channel mismatch in {h5_path}: {dataset.shape[0]} != {n_channels}")
            file_sfreq = float(handle.attrs["sample_frequency"])
            if sfreq is None:
                sfreq = file_sfreq
            elif abs(file_sfreq - sfreq) > 1e-6:
                raise ValueError(f"Sampling-rate mismatch in {h5_path}: {file_sfreq} != {sfreq}")
    if n_channels is None or sfreq is None:
        raise ValueError("No H5 paths to read metadata from.")
    return int(n_channels), float(sfreq)


def fill_meg_array(
    resolved: list[ResolvedRow],
    *,
    segment_ms: int,
    meg_dtype: np.dtype,
    standardize: bool,
    clip_boundary: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    unique_h5 = sorted({row.h5_path for row in resolved})
    if standardize:
        channel_means, channel_stds, n_channels, sfreq = h5_stats(unique_h5)
    else:
        n_channels, sfreq = h5_metadata(unique_h5)
        channel_means = np.zeros((n_channels,), dtype=np.float32)
        channel_stds = np.ones((n_channels,), dtype=np.float32)
    segment_samples = int(round(sfreq * (segment_ms / 1000.0)))
    if segment_samples <= 0:
        raise ValueError(f"--segment-ms must imply at least one sample, got {segment_ms}")

    n = len(resolved)
    meg = np.zeros((n, n_channels, segment_samples), dtype=meg_dtype)
    meg_time_mask = np.zeros((n, segment_samples), dtype=bool)
    meg_lengths = np.zeros((n,), dtype=np.int64)
    start_times = np.zeros((n,), dtype=np.float64)
    end_times = np.zeros((n,), dtype=np.float64)
    start_samples = np.zeros((n,), dtype=np.int64)
    end_samples = np.zeros((n,), dtype=np.int64)

    open_h5: dict[Path, h5py.File] = {}
    try:
        for idx, item in enumerate(resolved):
            handle = open_h5.get(item.h5_path)
            if handle is None:
                handle = h5py.File(item.h5_path, "r")
                open_h5[item.h5_path] = handle
            dataset = handle["data"]
            start = int(round(item.event_sentence.start_time * sfreq))
            requested_end = start + segment_samples
            clamped_start = max(0, start)
            clamped_end = min(int(dataset.shape[1]), requested_end)
            available = max(0, clamped_end - clamped_start)

            if available > 0:
                data = np.asarray(dataset[:, clamped_start:clamped_end], dtype=np.float32)
                if standardize:
                    data = (data - channel_means[:, None]) / channel_stds[:, None]
                if clip_boundary is not None:
                    data = np.clip(data, -float(clip_boundary), float(clip_boundary))
                dest_start = 0 if start >= 0 else -start
                dest_end = min(segment_samples, dest_start + available)
                write_len = max(0, dest_end - dest_start)
                if write_len > 0:
                    meg[idx, :, dest_start:dest_end] = data[:, :write_len].astype(meg_dtype, copy=False)
                    meg_time_mask[idx, dest_start:dest_end] = True
                    meg_lengths[idx] = write_len

            start_times[idx] = item.event_sentence.start_time
            end_times[idx] = item.event_sentence.end_time
            start_samples[idx] = start
            end_samples[idx] = requested_end
            if (idx + 1) % 500 == 0:
                print(f"packed MEG {idx + 1}/{n}", flush=True)
    finally:
        for handle in open_h5.values():
            handle.close()

    return meg, meg_time_mask, meg_lengths, start_times, end_times, start_samples, end_samples, sfreq


def write_sidecars(
    resolved: list[ResolvedRow],
    *,
    sidecar_root: Path,
    set_name: str,
    source_name: str,
    force: bool,
) -> dict[str, Any]:
    by_run: dict[tuple[str, str, str, str], list[ResolvedRow]] = defaultdict(list)
    for item in resolved:
        by_run[(item.subject, item.session, item.task, item.run)].append(item)

    written = []
    for (subject, session, task, run), items in sorted(by_run.items(), key=lambda kv: kv[0]):
        out_dir = sidecar_root / task / "derivatives" / "events"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_semantic_vectors.npz"
        if out_path.exists() and not force:
            raise FileExistsError(f"{out_path} exists; pass --force to overwrite.")

        embeddings = np.stack([item.embedding for item in items]).astype(np.float32)
        start_times = np.asarray([item.event_sentence.start_time for item in items], dtype=np.float64)
        end_times = np.asarray([item.event_sentence.end_time for item in items], dtype=np.float64)
        lexical = np.asarray([item.sentence for item in items], dtype=object)
        set_arr = np.asarray([set_name] * len(items), dtype=object)
        source_arr = np.asarray([source_name] * len(items), dtype=object)
        source_index = np.asarray([item.source_index for item in items], dtype=np.int64)
        sentence_id = np.asarray([item.event_sentence.sentence_id for item in items], dtype=object)
        meta = np.asarray(
            [
                {
                    "embedding_family": "MiniLM pooled",
                    "embedding_key": "embeddings_ada",
                    "note": "MiniLM vectors are stored under embeddings_ada for MEG2SEM loader compatibility.",
                    "N": int(len(items)),
                    "D": int(embeddings.shape[1]),
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "run": run,
                }
            ],
            dtype=object,
        )
        np.savez_compressed(
            out_path,
            embeddings_ada=embeddings,
            minilm_embeddings=embeddings,
            start_times=start_times,
            end_times=end_times,
            lexical_element=lexical,
            set=set_arr,
            source=source_arr,
            source_index=source_index,
            sentence_id=sentence_id,
            meta=meta,
        )
        written.append({"path": str(out_path), "count": int(len(items))})
    return {"sidecar_root": str(sidecar_root), "files": written, "file_count": len(written)}


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.force and not args.no_packed:
        raise FileExistsError(f"{output} exists; pass --force to overwrite.")
    if args.no_packed and not args.sidecar_root:
        raise ValueError("--no-packed requires --sidecar-root.")
    output.parent.mkdir(parents=True, exist_ok=True)

    resolved, load_summary = load_sva_rows(args)
    if not resolved:
        raise RuntimeError("No rows resolved.")

    embeddings = np.stack([item.embedding for item in resolved]).astype(np.float32)
    sentences = np.asarray([item.sentence for item in resolved], dtype=object)
    rows_out = []
    for item in resolved:
        row = dict(item.row)
        row.update(
            {
                "source_index": int(item.source_index),
                "event_path": str(item.event_path),
                "h5_path": str(item.h5_path),
                "event_sentence_id": item.event_sentence.sentence_id,
                "event_sentence": item.event_sentence.sentence,
                "start_time": float(item.event_sentence.start_time),
                "end_time": float(item.event_sentence.end_time),
                "time_column": item.event_sentence.time_column,
                "subject": item.subject,
                "session": item.session,
                "task": item.task,
                "run": item.run,
            }
        )
        rows_out.append(row)

    sidecar_summary = None
    if args.sidecar_root:
        sidecar_summary = write_sidecars(
            resolved,
            sidecar_root=Path(args.sidecar_root),
            set_name=args.set_name,
            source_name=args.source_name,
            force=args.force,
        )

    packed_summary = None
    if not args.no_packed:
        meg_dtype = np.dtype(args.meg_dtype)
        clip_boundary = None if args.clip_boundary < 0 else float(args.clip_boundary)
        (
            meg,
            meg_time_mask,
            meg_lengths,
            start_times,
            end_times,
            start_samples,
            end_samples,
            sfreq,
        ) = fill_meg_array(
            resolved,
            segment_ms=args.segment_ms,
            meg_dtype=meg_dtype,
            standardize=bool(args.standardize),
            clip_boundary=clip_boundary,
        )

        split = np.asarray(["train"] * len(resolved), dtype=object)
        subject = np.asarray([item.subject for item in resolved], dtype=object)
        session = np.asarray([item.session for item in resolved], dtype=object)
        task = np.asarray([item.task for item in resolved], dtype=object)
        run = np.asarray([item.run for item in resolved], dtype=object)
        source_index = np.asarray([item.source_index for item in resolved], dtype=np.int64)
        event_path = np.asarray([str(item.event_path) for item in resolved], dtype=object)
        h5_path = np.asarray([str(item.h5_path) for item in resolved], dtype=object)

        summary = {
            **load_summary,
            "output": str(output),
            "sidecar_summary": sidecar_summary,
            "meg_shape": list(meg.shape),
            "meg_dtype": str(meg.dtype),
            "embedding_shape": list(embeddings.shape),
            "sample_frequency": float(sfreq),
            "segment_ms": int(args.segment_ms),
            "standardized": bool(args.standardize),
            "clip_boundary": clip_boundary,
            "keys": [
                "meg",
                "meg_time_mask",
                "meg_lengths",
                "input_embeddings",
                "semantic_vectors",
                "embeddings_ada",
                "minilm_embeddings",
                "sentence",
                "rows",
            ],
            "run_counts": dict(Counter(f"{item.task}:ses-{item.session}:run-{item.run}" for item in resolved)),
            "task_counts": dict(Counter(item.task for item in resolved)),
            "sample_sentences": [str(sentence) for sentence in sentences[:10].tolist()],
        }
        schema_json = np.asarray(json.dumps(summary))
        np.savez_compressed(
            output,
            meg=meg,
            meg_time_mask=meg_time_mask,
            meg_lengths=meg_lengths,
            input_embeddings=embeddings,
            semantic_vectors=embeddings,
            embeddings_ada=embeddings,
            minilm_embeddings=embeddings,
            sentence=sentences,
            text=sentences,
            rows=np.asarray(rows_out, dtype=object),
            split=split,
            subject=subject,
            session=session,
            task=task,
            run=run,
            source_index=source_index,
            event_path=event_path,
            h5_path=h5_path,
            start_times=start_times,
            end_times=end_times,
            start_samples=start_samples,
            end_samples=end_samples,
            sample_frequency=np.asarray(float(sfreq), dtype=np.float64),
            schema_json=schema_json,
        )
        with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        packed_summary = summary

    final_summary = packed_summary or {**load_summary, "sidecar_summary": sidecar_summary}
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
