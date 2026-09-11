#!/usr/bin/env python
"""Export Tang-aligned TheMoth MiniLM sidecars for apples-to-apples MEG2SEM.

The Tang fMRI segment artifacts already contain the exact ten-word text target,
its normalized MiniLM384 embedding, and its stimulus-clock center.  LibriBrain2
contains MEG for the same thirty TheMoth stories.  This script maps every
shared Tang row onto the matching TheMoth session and writes semantic sidecars
whose MEG window is centered on the same stimulus time.

No brain data are read here.  The output is consumed by MEG2SEM's existing
LibriBrain packed-data exporter, so the brain preprocessing remains identical
to the established MEG pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EVENT_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_events\.tsv$"
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
EXPECTED_VAL_STORIES = frozenset({"thatthingonmyarm", "leavingbaghdad", "howtodraw"})
EXPECTED_TEST_STORIES = frozenset({"birthofanation"})
REQUIRED_ORACLE_KEYS = (
    "story",
    "start_tr",
    "stop_tr",
    "split",
    "sentence",
    "target_center_sec",
    "target_window_mid_sec",
    "input_embeddings",
)


@dataclass(frozen=True)
class EventWord:
    text: str
    token: str
    time_meg: float
    time_podcast: float
    duration: float


@dataclass(frozen=True)
class StoryEvents:
    story: str
    event_path: Path
    subject: str
    session: str
    task: str
    run: str
    words: tuple[EventWord, ...]
    meg_minus_podcast_sec: float
    meg_minus_podcast_max_deviation_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-train-val", type=Path, required=True)
    parser.add_argument("--oracle-test", type=Path, required=True)
    parser.add_argument("--themoth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--segment-ms", type=int, default=20_000)
    parser.add_argument(
        "--center-mode",
        choices=("target_center", "text_mid"),
        default="target_center",
        help=(
            "Center MEG on the fMRI target-center clock (strict 20-second contract) "
            "or on the midpoint of the exact target text (MEG-native timing)."
        ),
    )
    parser.add_argument("--expected-embedding-dim", type=int, default=384)
    parser.add_argument("--expected-word-count", type=int, default=10)
    parser.add_argument("--expected-shared-stories", type=int, default=30)
    parser.add_argument("--min-exact-text-match-fraction", type=float, default=0.99)
    parser.add_argument("--max-clock-error-p95-sec", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def string_array(values: list[Any] | np.ndarray) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def normalized_token(value: str) -> str:
    return "".join(TOKEN_RE.findall(str(value).casefold()))


def normalized_words(value: str) -> list[str]:
    return [token for token in (normalized_token(part) for part in str(value).split()) if token]


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def story_from_wavile(value: str) -> str:
    parts = str(value).strip().split("/")
    if len(parts) >= 3 and parts[0] == "stimuli":
        return parts[1]
    return ""


def load_story_events(event_path: Path) -> StoryEvents:
    match = EVENT_RE.fullmatch(event_path.name)
    if match is None:
        raise ValueError(f"Could not parse TheMoth event filename: {event_path}")

    stories: Counter[str] = Counter()
    words: list[EventWord] = []
    offsets: list[float] = []
    with event_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            story = story_from_wavile(row.get("wavile", ""))
            if story:
                stories[story] += 1
            if str(row.get("kind", "")).strip().casefold() != "word":
                continue
            text = str(row.get("segment", "")).strip()
            token = normalized_token(text)
            time_meg = finite_float(row.get("timemeg"))
            time_podcast = finite_float(row.get("timepodcast"))
            duration = finite_float(row.get("duration")) or 0.0
            if not token or time_meg is None or time_podcast is None:
                continue
            words.append(
                EventWord(
                    text=text,
                    token=token,
                    time_meg=time_meg,
                    time_podcast=time_podcast,
                    duration=max(0.0, duration),
                )
            )
            offsets.append(time_meg - time_podcast)

    if not stories:
        raise ValueError(f"No stimulus story found in {event_path}")
    story, _ = stories.most_common(1)[0]
    if not words or not offsets:
        raise ValueError(f"No aligned word events found in {event_path}")
    offset = float(np.median(np.asarray(offsets, dtype=np.float64)))
    offset_spread = float(np.max(np.abs(np.asarray(offsets, dtype=np.float64) - offset)))
    return StoryEvents(
        story=story,
        event_path=event_path,
        subject=match.group("subject"),
        session=match.group("session"),
        task=match.group("task"),
        run=match.group("run"),
        words=tuple(words),
        meg_minus_podcast_sec=offset,
        meg_minus_podcast_max_deviation_sec=offset_spread,
    )


def discover_story_events(themoth_root: Path) -> dict[str, StoryEvents]:
    event_dir = themoth_root / "derivatives" / "events"
    paths = sorted(event_dir.glob("*_events.tsv"))
    if not paths:
        raise FileNotFoundError(f"No TheMoth event TSVs under {event_dir}")
    output: dict[str, StoryEvents] = {}
    for path in paths:
        events = load_story_events(path)
        if events.story in output:
            raise ValueError(f"Duplicate TheMoth story {events.story!r}: {path} and {output[events.story].event_path}")
        output[events.story] = events
    return output


def load_oracle_rows(paths: list[Path], expected_dim: int) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in REQUIRED_ORACLE_KEYS if key not in data.files]
            if missing:
                raise KeyError(f"{path} is missing keys {missing}; available={data.files}")
            part = {key: np.asarray(data[key]) for key in REQUIRED_ORACLE_KEYS}
        n = len(part["sentence"])
        if any(len(part[key]) != n for key in REQUIRED_ORACLE_KEYS if key != "input_embeddings"):
            raise ValueError(f"Row-count mismatch in {path}")
        if part["input_embeddings"].shape != (n, expected_dim):
            raise ValueError(
                f"Expected {path} embeddings {(n, expected_dim)}, got {part['input_embeddings'].shape}"
            )
        parts.append(part)
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in REQUIRED_ORACLE_KEYS}


def candidate_matches(tokens: list[str], event_tokens: list[str]) -> list[int]:
    if not tokens or len(tokens) > len(event_tokens):
        return []
    first = tokens[0]
    width = len(tokens)
    return [
        idx
        for idx, token in enumerate(event_tokens[: len(event_tokens) - width + 1])
        if token == first and event_tokens[idx : idx + width] == tokens
    ]


def best_text_match(
    sentence: str,
    target_mid_sec: float,
    events: StoryEvents,
) -> tuple[int | None, float | None, float | None, float | None, int]:
    tokens = normalized_words(sentence)
    event_tokens = [word.token for word in events.words]
    candidates = candidate_matches(tokens, event_tokens)
    if not candidates:
        return None, None, None, None, 0

    def podcast_midpoint(index: int) -> float:
        first = events.words[index]
        last = events.words[index + len(tokens) - 1]
        return (first.time_podcast + last.time_podcast + last.duration) / 2.0

    selected = min(candidates, key=lambda idx: abs(podcast_midpoint(idx) - target_mid_sec))
    first = events.words[selected]
    last = events.words[selected + len(tokens) - 1]
    podcast_mid = podcast_midpoint(selected)
    meg_mid = (first.time_meg + last.time_meg + last.duration) / 2.0
    return selected, podcast_mid, meg_mid, podcast_mid - target_mid_sec, len(candidates)


def locally_map_podcast_to_meg(target_sec: float, events: StoryEvents) -> float:
    """Map one podcast-clock instant with the nearest word's local offset."""
    nearest = min(events.words, key=lambda word: abs(word.time_podcast - target_sec))
    return target_sec + (nearest.time_meg - nearest.time_podcast)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    if args.segment_ms <= 0:
        raise ValueError("--segment-ms must be positive")
    if not 0.0 <= args.min_exact_text_match_fraction <= 1.0:
        raise ValueError("--min-exact-text-match-fraction must be in [0, 1]")

    stories = discover_story_events(args.themoth_root)
    if len(stories) != args.expected_shared_stories:
        raise ValueError(
            f"Expected {args.expected_shared_stories} TheMoth stories, discovered {len(stories)}: {sorted(stories)}"
        )
    oracle = load_oracle_rows(
        [args.oracle_train_val, args.oracle_test],
        expected_dim=args.expected_embedding_dim,
    )
    story_values = string_array(oracle["story"])
    split_values = string_array(oracle["split"])
    sentence_values = string_array(oracle["sentence"])
    shared_mask = np.asarray([story in stories for story in story_values], dtype=bool)
    if not np.any(shared_mask):
        raise ValueError("No Tang oracle rows matched TheMoth stories")

    story_splits: dict[str, set[str]] = defaultdict(set)
    for story, split in zip(story_values[shared_mask], split_values[shared_mask]):
        story_splits[story].add(split)
    non_unique = {story: sorted(values) for story, values in story_splits.items() if len(values) != 1}
    if non_unique:
        raise ValueError(f"Tang stories cross split boundaries: {non_unique}")
    observed_val = {story for story, values in story_splits.items() if values == {"val"}}
    observed_test = {story for story, values in story_splits.items() if values == {"test"}}
    if observed_val != EXPECTED_VAL_STORIES:
        raise ValueError(f"Unexpected shared validation stories: {sorted(observed_val)}")
    if observed_test != EXPECTED_TEST_STORIES:
        raise ValueError(f"Unexpected shared test stories: {sorted(observed_test)}")
    if set(story_splits) != set(stories):
        missing = sorted(set(stories) - set(story_splits))
        extra = sorted(set(story_splits) - set(stories))
        raise ValueError(f"Tang/TheMoth story mismatch: missing={missing} extra={extra}")

    output_root = args.output_root
    summary_path = args.summary_json or output_root / "tang_themoth_apples_alignment_summary.json"
    audit_path = args.audit_csv or output_root / "tang_themoth_apples_alignment_rows.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_story: dict[str, list[int]] = defaultdict(list)
    for idx in np.flatnonzero(shared_mask).tolist():
        rows_by_story[story_values[idx]].append(int(idx))

    audit_rows: list[dict[str, Any]] = []
    exact_matches = 0
    exact_matches_by_split: Counter[str] = Counter()
    clock_errors: list[float] = []
    split_counts: Counter[str] = Counter()
    story_counts: dict[str, int] = {}
    written: list[str] = []
    half_window_sec = args.segment_ms / 2000.0

    for story, events in sorted(stories.items(), key=lambda item: int(item[1].session)):
        indices = rows_by_story[story]
        split = next(iter(story_splits[story]))
        story_counts[story] = len(indices)
        split_counts[split] += len(indices)
        start_times: list[float] = []
        end_times: list[float] = []
        row_ids: list[str] = []
        match_indices: list[int] = []
        match_errors: list[float] = []
        match_counts: list[int] = []

        for source_idx in indices:
            sentence = sentence_values[source_idx]
            tokens = normalized_words(sentence)
            if len(tokens) != args.expected_word_count:
                raise ValueError(
                    f"Expected {args.expected_word_count} normalized words for row {source_idx}, got {len(tokens)}: {sentence!r}"
                )
            target_center_sec = float(oracle["target_center_sec"][source_idx])
            target_mid_sec = float(oracle["target_window_mid_sec"][source_idx])
            match_idx, event_mid_sec, event_meg_mid_sec, error_sec, n_candidates = best_text_match(
                sentence,
                target_mid_sec,
                events,
            )
            is_exact = match_idx is not None
            if is_exact:
                exact_matches += 1
                exact_matches_by_split[split] += 1
                assert error_sec is not None
                clock_errors.append(abs(error_sec))
            requested_center_sec = target_center_sec if args.center_mode == "target_center" else target_mid_sec
            if event_meg_mid_sec is None:
                meg_center_sec = locally_map_podcast_to_meg(requested_center_sec, events)
            elif args.center_mode == "text_mid":
                meg_center_sec = event_meg_mid_sec
            else:
                # Anchor the local clock transform on this exact phrase.  This
                # remains correct when the recording inserts gaps between audio
                # chunks and the global timemeg-timepodcast offset changes.
                meg_center_sec = event_meg_mid_sec + (target_center_sec - target_mid_sec)
            meg_start_sec = meg_center_sec - half_window_sec
            meg_end_sec = meg_start_sec + args.segment_ms / 1000.0
            if meg_start_sec < 0:
                raise ValueError(
                    f"Negative MEG start for {story}:{oracle['start_tr'][source_idx]}-{oracle['stop_tr'][source_idx]}: "
                    f"{meg_start_sec:.3f}s"
                )
            row_id = f"{story}:{int(oracle['start_tr'][source_idx])}-{int(oracle['stop_tr'][source_idx])}"
            start_times.append(meg_start_sec)
            end_times.append(meg_end_sec)
            row_ids.append(row_id)
            match_indices.append(-1 if match_idx is None else int(match_idx))
            match_errors.append(float("nan") if error_sec is None else float(error_sec))
            match_counts.append(int(n_candidates))
            audit_rows.append(
                {
                    "row_id": row_id,
                    "story": story,
                    "session": events.session,
                    "split": split,
                    "start_tr": int(oracle["start_tr"][source_idx]),
                    "stop_tr": int(oracle["stop_tr"][source_idx]),
                    "target_center_sec": target_center_sec,
                    "target_window_mid_sec": target_mid_sec,
                    "meg_start_sec": meg_start_sec,
                    "meg_stop_sec": meg_end_sec,
                    "meg_center_sec": meg_center_sec,
                    "exact_text_match": int(is_exact),
                    "match_candidate_count": int(n_candidates),
                    "matched_event_word_index": -1 if match_idx is None else int(match_idx),
                    "matched_window_mid_sec": "" if event_mid_sec is None else float(event_mid_sec),
                    "clock_error_sec": "" if error_sec is None else float(error_sec),
                    "sentence": sentence,
                }
            )

        out_dir = output_root / events.task / "derivatives" / "events"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"sub-{events.subject}_ses-{events.session}_task-{events.task}_run-{events.run}_semantic_vectors.npz"
        )
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
        embeddings = np.asarray(oracle["input_embeddings"][indices], dtype=np.float32)
        sentences = sentence_values[indices]
        n = len(indices)
        meta = {
            "role": "tang_themoth_apples_to_apples_minilm384",
            "story": story,
            "subject": events.subject,
            "session": events.session,
            "task": events.task,
            "run": events.run,
            "split": split,
            "count": n,
            "segment_ms": args.segment_ms,
            "brain_window_contract": (
                f"{args.segment_ms / 1000:g}-second MEG window centered at "
                + (
                    "Tang target_center_sec on the shared podcast clock"
                    if args.center_mode == "target_center"
                    else "the exact target-text midpoint on the locally aligned MEG clock"
                )
            ),
            "center_mode": args.center_mode,
            "text_contract": "exact Tang clock-fixed contiguous 10-word content-bearing target",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": args.expected_embedding_dim,
            "meg_minus_podcast_sec": events.meg_minus_podcast_sec,
            "meg_minus_podcast_max_deviation_sec": events.meg_minus_podcast_max_deviation_sec,
            "oracle_train_val": str(args.oracle_train_val),
            "oracle_test": str(args.oracle_test),
        }
        np.savez_compressed(
            out_path,
            embeddings_minilm=embeddings,
            minilm_embeddings=embeddings,
            semantic_vectors=embeddings,
            input_embeddings=embeddings,
            start_times=np.asarray(start_times, dtype=np.float64),
            end_times=np.asarray(end_times, dtype=np.float64),
            lexical_element=np.asarray(sentences, dtype=object),
            sentence=np.asarray(sentences, dtype=object),
            set=np.asarray(["sentences"] * n, dtype=object),
            source=np.asarray(["tang_shared_middle10"] * n, dtype=object),
            source_index=np.asarray(indices, dtype=np.int64),
            sentence_id=np.asarray(row_ids, dtype=object),
            story=np.asarray([story] * n, dtype=object),
            split=np.asarray([split] * n, dtype=object),
            start_tr=np.asarray(oracle["start_tr"][indices], dtype=np.int64),
            stop_tr=np.asarray(oracle["stop_tr"][indices], dtype=np.int64),
            target_center_sec=np.asarray(oracle["target_center_sec"][indices], dtype=np.float32),
            target_window_mid_sec=np.asarray(oracle["target_window_mid_sec"][indices], dtype=np.float32),
            exact_event_match_index=np.asarray(match_indices, dtype=np.int64),
            event_match_clock_error_sec=np.asarray(match_errors, dtype=np.float32),
            event_match_candidate_count=np.asarray(match_counts, dtype=np.int16),
            meta=np.asarray([meta], dtype=object),
        )
        written.append(str(out_path))

    total_shared = int(np.sum(shared_mask))
    exact_fraction = exact_matches / max(1, total_shared)
    clock_error_p95 = percentile(clock_errors, 95)
    summary = {
        "role": "tang_themoth_apples_to_apples_alignment",
        "oracle_train_val": str(args.oracle_train_val),
        "oracle_test": str(args.oracle_test),
        "themoth_root": str(args.themoth_root),
        "output_root": str(output_root),
        "segment_ms": args.segment_ms,
        "center_mode": args.center_mode,
        "embedding_dim": args.expected_embedding_dim,
        "target_word_count": args.expected_word_count,
        "shared_story_count": len(stories),
        "shared_row_count": total_shared,
        "split_counts": dict(sorted(split_counts.items())),
        "story_counts": dict(sorted(story_counts.items())),
        "train_stories": sorted(story for story, values in story_splits.items() if values == {"train"}),
        "val_stories": sorted(observed_val),
        "test_stories": sorted(observed_test),
        "story_to_session": {story: events.session for story, events in sorted(stories.items())},
        "exact_text_matches": exact_matches,
        "exact_text_match_fraction": exact_fraction,
        "exact_text_matches_by_split": {
            split: {
                "matches": int(exact_matches_by_split[split]),
                "rows": int(split_counts[split]),
                "fraction": float(exact_matches_by_split[split] / max(1, split_counts[split])),
            }
            for split in sorted(split_counts)
        },
        "absolute_clock_error_sec": {
            "mean": float(np.mean(clock_errors)) if clock_errors else float("nan"),
            "p50": percentile(clock_errors, 50),
            "p95": clock_error_p95,
            "max": float(np.max(clock_errors)) if clock_errors else float("nan"),
        },
        "sidecar_files": written,
        "audit_csv": str(audit_path),
    }

    fieldnames = list(audit_rows[0].keys())
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if exact_fraction < args.min_exact_text_match_fraction:
        raise RuntimeError(
            f"Exact target/event text alignment {exact_fraction:.3%} is below "
            f"{args.min_exact_text_match_fraction:.3%}; see {audit_path}"
        )
    if math.isfinite(clock_error_p95) and clock_error_p95 > args.max_clock_error_p95_sec:
        raise RuntimeError(
            f"Target/event clock-error p95 {clock_error_p95:.3f}s exceeds "
            f"{args.max_clock_error_p95_sec:.3f}s; see {audit_path}"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
