#!/usr/bin/env python
"""Replace inflated MRI2SEM segment text with one centered word phrase per row.

The MRI2SEM segment cache joins ten overlapping 20-second, TR-centered text
windows.  Those repeated strings are useful neither as a 10-TR transcript nor
as an ELF target.  This exporter leaves every semantic vector and split row in
place and derives a short contiguous phrase directly from each story TextGrid.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


SPLITS = ("train", "val", "test")
BASE_ROW_KEYS = ("input_embeddings", "story", "start_tr", "stop_tr", "split")
TARGET_ROW_KEYS = (
    "sentence",
    "target_center_tr",
    "target_center_sec",
    "target_center_word_mid_sec",
    "target_center_distance_sec",
    "target_word_start_sec",
    "target_word_stop_sec",
    "target_window_mid_sec",
    "target_window_center_distance_sec",
    "target_window_boundary_spill_sec",
    "target_contains_center_word",
    "target_content_word_count",
    "target_content_words",
)

# Closed-class/function words and common conversational fillers.  This is a
# deliberately dependency-free lexical audit, not a POS tagger: words outside
# this set that contain a letter or digit count as content-bearing.
NON_CONTENT_WORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself just let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves out
    over own same shan't she she'd she'll she's should shouldn't so some such
    than that that's the their theirs them themselves then there there's these
    they they'd they'll they're they've this those through to too under until
    up very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    also anyway basically er erm huh like mhm mm hmm oh okay ok right uh um
    well yeah yep yes gonna gotta kinda sort sorta wanna y'know
    """.split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--stimulus-data-root", required=True)
    parser.add_argument("--mri2sem-src", required=True)
    parser.add_argument("--predicted-test-npz", default="")
    parser.add_argument("--predicted-test-output", default="")
    parser.add_argument(
        "--tang-start-time-sec",
        type=float,
        default=10.0,
        help=(
            "Tang simulated-TR start-time offset. MRI2SEM's Lanczos target clock uses "
            "(row + tang_trim_start) * TR - tang_start_time + TR/2."
        ),
    )
    parser.add_argument("--word-count", type=int, default=5)
    parser.add_argument(
        "--time-window-radius-sec",
        type=float,
        default=0.0,
        help=(
            "If positive, ignore --word-count and use every unique TextGrid word whose midpoint falls inside "
            "center_time +/- this radius. This creates a transcript aligned to the full semantic interval."
        ),
    )
    parser.add_argument(
        "--maximize-content-words",
        action="store_true",
        help="Among contiguous windows containing the temporal center word, choose the one with the most content words.",
    )
    parser.add_argument(
        "--min-content-words",
        type=int,
        default=0,
        help="Fail export if any selected target has fewer content words than this value.",
    )
    parser.add_argument(
        "--content-search-radius-sec",
        type=float,
        default=0.0,
        help=(
            "When maximizing content, search windows fully contained within this many seconds of the semantic "
            "center. Zero restricts candidates to windows containing the nearest center word."
        ),
    )
    parser.add_argument("--expected-dim", type=int, default=3072)
    parser.add_argument("--expected-train", type=int, default=11725)
    parser.add_argument("--expected-val", type=int, default=266)
    parser.add_argument("--expected-test", type=int, default=107)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def parse_schema(value: np.ndarray | None) -> dict[str, Any]:
    if value is None:
        return {}
    raw = str(value.tolist())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"source_schema_raw": raw}
    return parsed if isinstance(parsed, dict) else {"source_schema": parsed}


def load_textgrid_reader(mri2sem_src: Path) -> Callable[[Path], list[Any]]:
    sys.path.insert(0, str(mri2sem_src))
    from mri2sem.textgrid import read_word_intervals  # type: ignore

    return read_word_intervals


def normalized_word(word: str) -> str:
    return re.sub(r"(^[^a-z0-9']+|[^a-z0-9']+$)", "", word.lower())


def is_content_word(word: str) -> bool:
    normalized = normalized_word(word)
    if not normalized or not any(character.isalnum() for character in normalized):
        return False
    if normalized in NON_CONTENT_WORDS:
        return False
    # TextGrid tokenization can split contractions into bare suffixes.
    if normalized in {"d", "ll", "m", "re", "s", "t", "ve"}:
        return False
    return True


def textgrid_path(stimulus_root: Path, story: str, split: str) -> Path:
    candidates: list[Path] = []
    if split == "test":
        candidates.append(stimulus_root / "data_test" / "test_stimulus" / "perceived_speech" / f"{story}.TextGrid")
    candidates.append(stimulus_root / "data_train" / "train_stimulus" / f"{story}.TextGrid")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No TextGrid for story={story!r} split={split!r}; tried {candidates}")


def load_split(path: Path, expected_n: int, expected_dim: int, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in BASE_ROW_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing row keys: {missing}")
        arrays = {key: np.asarray(data[key]) for key in BASE_ROW_KEYS}
        schema = parse_schema(data["schema_json"] if "schema_json" in data.files else None)
    if arrays["input_embeddings"].shape != (expected_n, expected_dim):
        raise ValueError(
            f"Expected {split} embeddings {(expected_n, expected_dim)}, got {arrays['input_embeddings'].shape}."
        )
    if not np.isfinite(arrays["input_embeddings"]).all():
        raise ValueError(f"{split} embeddings contain non-finite values.")
    for key in ("story", "start_tr", "stop_tr", "split"):
        if arrays[key].shape != (expected_n,):
            raise ValueError(f"Expected {split} {key} shape {(expected_n,)}, got {arrays[key].shape}.")
    split_values = set(strings(arrays["split"]))
    if split_values != {split}:
        raise ValueError(f"Expected only split={split!r} in {path}, got {sorted(split_values)}.")
    arrays["input_embeddings"] = np.asarray(arrays["input_embeddings"], dtype=np.float32)
    arrays["story"] = np.asarray(strings(arrays["story"]), dtype=object)
    arrays["start_tr"] = np.asarray(arrays["start_tr"], dtype=np.int64)
    arrays["stop_tr"] = np.asarray(arrays["stop_tr"], dtype=np.int64)
    arrays["split"] = np.asarray(strings(arrays["split"]), dtype=object)
    return arrays, schema


def relabel_split(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    stimulus_root: Path,
    read_word_intervals: Callable[[Path], list[Any]],
    tr_seconds: float,
    delays: list[int],
    tang_trim_start_trs: int,
    tang_start_time_sec: float,
    word_count: int,
    maximize_content_words: bool,
    min_content_words: int,
    content_search_radius_sec: float,
    time_window_radius_sec: float,
) -> dict[str, np.ndarray]:
    if time_window_radius_sec < 0:
        raise ValueError("--time-window-radius-sec must be non-negative.")
    if not time_window_radius_sec and word_count <= 0:
        raise ValueError("--word-count must be positive.")
    if min_content_words < 0 or (not time_window_radius_sec and min_content_words > word_count):
        raise ValueError("--min-content-words must be non-negative and no larger than a fixed --word-count.")
    if min_content_words and not (maximize_content_words or time_window_radius_sec):
        raise ValueError("--min-content-words requires --maximize-content-words.")
    if content_search_radius_sec < 0:
        raise ValueError("--content-search-radius-sec must be non-negative.")
    if content_search_radius_sec and not maximize_content_words:
        raise ValueError("--content-search-radius-sec requires --maximize-content-words.")
    if time_window_radius_sec and (content_search_radius_sec or maximize_content_words):
        raise ValueError("--time-window-radius-sec cannot be combined with fixed-window content search flags.")
    mean_delay = float(np.mean(delays))
    story_cache: dict[str, list[Any]] = {}
    sentences: list[str] = []
    center_trs: list[float] = []
    center_secs: list[float] = []
    center_word_mids: list[float] = []
    center_distances: list[float] = []
    word_starts: list[float] = []
    word_stops: list[float] = []
    window_mids: list[float] = []
    window_center_distances: list[float] = []
    window_boundary_spills: list[float] = []
    contains_center_word: list[bool] = []
    content_counts: list[int] = []
    content_word_strings: list[str] = []

    for story, start_tr, stop_tr in zip(
        strings(arrays["story"]), arrays["start_tr"].tolist(), arrays["stop_tr"].tolist()
    ):
        if story not in story_cache:
            story_cache[story] = read_word_intervals(textgrid_path(stimulus_root, story, split))
        words = story_cache[story]
        if not time_window_radius_sec and len(words) < word_count:
            raise ValueError(f"Story {story!r} has only {len(words)} usable TextGrid words.")

        # Tang target block d averages base features at rows [start-d, stop-d].
        # MRI2SEM builds those base rows on Tang's trimmed simulated-TR clock:
        #   time(row) = (row + trim_start) * TR - start_time + TR/2.
        # Using row * TR here shifts the text targets about 11 seconds early
        # for the current trim_start=10, start_time=10, TR=2 dataset.
        center_tr = (float(start_tr) + float(stop_tr)) / 2.0 - mean_delay
        center_sec = (
            (center_tr + float(tang_trim_start_trs)) * tr_seconds
            - tang_start_time_sec
            + tr_seconds / 2.0
        )
        midpoints = np.fromiter(
            ((float(word.start) + float(word.end)) / 2.0 for word in words),
            dtype=np.float64,
            count=len(words),
        )
        center_idx = int(np.argmin(np.abs(midpoints - center_sec)))
        if time_window_radius_sec:
            interval_start = center_sec - time_window_radius_sec
            interval_stop = center_sec + time_window_radius_sec
            chosen_indices = np.flatnonzero((midpoints >= interval_start) & (midpoints < interval_stop))
            if not len(chosen_indices):
                raise ValueError(
                    f"No TextGrid words have midpoints inside +/-{time_window_radius_sec:g}s for "
                    f"{story}:{start_tr}-{stop_tr}."
                )
            first_idx = int(chosen_indices[0])
            selected_word_count = int(len(chosen_indices))
        else:
            centered_first_idx = min(max(0, center_idx - (word_count - 1) // 2), len(words) - word_count)
            selected_word_count = word_count
        if maximize_content_words and not time_window_radius_sec:
            if content_search_radius_sec:
                # The embedding pools a 20-second segment. Keep each target's
                # midpoint inside that span. Words may spill slightly across a
                # boundary when pauses leave fewer than word_count words inside
                # the interval; that spill is saved and audited explicitly.
                interval_start = center_sec - content_search_radius_sec
                interval_stop = center_sec + content_search_radius_sec
                left_idx = int(np.searchsorted(midpoints, interval_start, side="left"))
                right_idx = int(np.searchsorted(midpoints, interval_stop, side="right"))
                first_min = max(0, left_idx - word_count + 1)
                first_max = min(len(words) - word_count, right_idx)
            else:
                # Every candidate remains temporally anchored because it must
                # contain the TextGrid word nearest the semantic center.
                first_min = max(0, center_idx - word_count + 1)
                first_max = min(center_idx, len(words) - word_count)
            candidates: list[tuple[int, float, float, int]] = []
            for candidate_first in range(first_min, first_max + 1):
                candidate = words[candidate_first : candidate_first + word_count]
                candidate_content_count = sum(is_content_word(str(word.text)) for word in candidate)
                phrase_mid_sec = (float(candidate[0].start) + float(candidate[-1].end)) / 2.0
                if content_search_radius_sec and abs(phrase_mid_sec - center_sec) > content_search_radius_sec:
                    continue
                if content_search_radius_sec:
                    boundary_spill = max(
                        0.0,
                        interval_start - float(candidate[0].start),
                        float(candidate[-1].end) - interval_stop,
                    )
                else:
                    boundary_spill = 0.0
                candidates.append(
                    (candidate_content_count, -boundary_spill, -abs(phrase_mid_sec - center_sec), candidate_first)
                )
            if not candidates:
                raise ValueError(
                    f"No {word_count}-word window has its midpoint inside the +/-{content_search_radius_sec:g}s "
                    f"content-search span for {story}:{start_tr}-{stop_tr}."
                )
            # If ten words fit inside the semantic interval, never cross its
            # boundary merely to gain another content word. Only silent/sparse
            # intervals fall back to the least-spilling content-rich window.
            contained_candidates = [candidate for candidate in candidates if candidate[1] == 0.0]
            if contained_candidates:
                _, _, _, first_idx = max(
                    contained_candidates,
                    key=lambda candidate: (candidate[0], candidate[2], candidate[3]),
                )
            else:
                content_qualified = [candidate for candidate in candidates if candidate[0] >= min_content_words]
                selection_pool = content_qualified or candidates
                _, _, _, first_idx = max(
                    selection_pool,
                    key=lambda candidate: (candidate[1], candidate[0], candidate[2], candidate[3]),
                )
        elif not time_window_radius_sec:
            first_idx = centered_first_idx
        chosen = words[first_idx : first_idx + selected_word_count]
        sentence = " ".join(str(word.text) for word in chosen).strip()
        if not time_window_radius_sec and len(sentence.split()) != word_count:
            raise ValueError(
                f"Expected exactly {word_count} whitespace-delimited words for {story}:{start_tr}-{stop_tr}, "
                f"got {sentence!r}."
            )
        content_words = [normalized_word(str(word.text)) for word in chosen if is_content_word(str(word.text))]
        if len(content_words) < min_content_words:
            raise ValueError(
                f"Target for {story}:{start_tr}-{stop_tr} has {len(content_words)} content words, below "
                f"--min-content-words={min_content_words}: {sentence!r}."
            )

        sentences.append(sentence)
        center_trs.append(center_tr)
        center_secs.append(center_sec)
        center_word_mids.append(float(midpoints[center_idx]))
        center_distances.append(float(abs(midpoints[center_idx] - center_sec)))
        word_starts.append(float(chosen[0].start))
        word_stops.append(float(chosen[-1].end))
        window_mid = (float(chosen[0].start) + float(chosen[-1].end)) / 2.0
        window_mids.append(window_mid)
        window_center_distances.append(abs(window_mid - center_sec))
        if time_window_radius_sec:
            interval_start = center_sec - time_window_radius_sec
            interval_stop = center_sec + time_window_radius_sec
            boundary_spill = max(
                0.0,
                interval_start - float(chosen[0].start),
                float(chosen[-1].end) - interval_stop,
            )
        elif content_search_radius_sec:
            interval_start = center_sec - content_search_radius_sec
            interval_stop = center_sec + content_search_radius_sec
            boundary_spill = max(
                0.0,
                interval_start - float(chosen[0].start),
                float(chosen[-1].end) - interval_stop,
            )
        else:
            boundary_spill = 0.0
        window_boundary_spills.append(boundary_spill)
        contains_center_word.append(first_idx <= center_idx < first_idx + selected_word_count)
        content_counts.append(len(content_words))
        content_word_strings.append(" ".join(content_words))

    output = dict(arrays)
    output.update(
        {
            "sentence": np.asarray(sentences, dtype=object),
            "target_center_tr": np.asarray(center_trs, dtype=np.float32),
            "target_center_sec": np.asarray(center_secs, dtype=np.float32),
            "target_center_word_mid_sec": np.asarray(center_word_mids, dtype=np.float32),
            "target_center_distance_sec": np.asarray(center_distances, dtype=np.float32),
            "target_word_start_sec": np.asarray(word_starts, dtype=np.float32),
            "target_word_stop_sec": np.asarray(word_stops, dtype=np.float32),
            "target_window_mid_sec": np.asarray(window_mids, dtype=np.float32),
            "target_window_center_distance_sec": np.asarray(window_center_distances, dtype=np.float32),
            "target_window_boundary_spill_sec": np.asarray(window_boundary_spills, dtype=np.float32),
            "target_contains_center_word": np.asarray(contains_center_word, dtype=np.bool_),
            "target_content_word_count": np.asarray(content_counts, dtype=np.int16),
            "target_content_words": np.asarray(content_word_strings, dtype=object),
        }
    )
    return output


def target_contract(
    *,
    metadata: dict[str, Any],
    delays: list[int],
    tr_seconds: float,
    tang_trim_start_trs: int,
    tang_start_time_sec: float,
    word_count: int,
    maximize_content_words: bool,
    min_content_words: int,
    content_search_radius_sec: float,
    time_window_radius_sec: float,
) -> dict[str, Any]:
    variable_time_window = time_window_radius_sec > 0
    return {
        "mode": (
            "lag_adjusted_time_window_unique_textgrid_words"
            if variable_time_window
            else "centered_contiguous_textgrid_words"
        ),
        "word_count": None if variable_time_window else word_count,
        "exact_word_count": not variable_time_window,
        "source": "story TextGrid word intervals; inflated MRI2SEM *_text values are ignored",
        "tr_seconds": tr_seconds,
        "stimulus_delays_trs": delays,
        "mean_delay_trs": float(np.mean(delays)),
        "center_tr_formula": "(start_tr + stop_tr) / 2 - mean(stimulus_delays_trs)",
        "tang_trim_start_trs": tang_trim_start_trs,
        "tang_start_time_sec": tang_start_time_sec,
        "center_time_formula": (
            "(target_center_tr + tang_trim_start_trs) * tr_seconds "
            "- tang_start_time_sec + tr_seconds / 2"
        ),
        "time_window_radius_sec": time_window_radius_sec,
        "time_window_duration_sec": 2.0 * time_window_radius_sec if variable_time_window else None,
        "time_window_word_rule": "word midpoint in [center-radius, center+radius)" if variable_time_window else None,
        "deduplicated_by_textgrid_interval": variable_time_window,
        "window_must_contain_center_word": variable_time_window or content_search_radius_sec == 0.0,
        "maximize_content_words": maximize_content_words,
        "min_content_words": min_content_words,
        "content_search_radius_sec": content_search_radius_sec,
        "content_search_contract": (
            "target window midpoint falls inside center_time +/- content_search_radius_sec; boundary spill is audited"
            if content_search_radius_sec
            else "target contains the TextGrid word nearest center_time"
        ),
        "content_word_definition": "alphanumeric TextGrid tokens excluding the embedded function-word/filler lexicon",
        "segment_trs": metadata.get("segment_trs"),
        "rationale": (
            "unique transcript of the full lag-adjusted semantic interval"
            if variable_time_window
            else "short contiguous target anchored to the temporal center represented across all four delay blocks"
        ),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray], schema: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, schema_json=np.asarray(json.dumps(schema, indent=2), dtype=object))


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = BASE_ROW_KEYS + TARGET_ROW_KEYS
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def split_summary(arrays: dict[str, np.ndarray], sample_count: int) -> dict[str, Any]:
    distances = np.asarray(arrays["target_center_distance_sec"], dtype=np.float64)
    durations = np.asarray(arrays["target_word_stop_sec"] - arrays["target_word_start_sec"], dtype=np.float64)
    window_center_distances = np.asarray(arrays["target_window_center_distance_sec"], dtype=np.float64)
    window_boundary_spills = np.asarray(arrays["target_window_boundary_spill_sec"], dtype=np.float64)
    center_inclusions = np.asarray(arrays["target_contains_center_word"], dtype=np.bool_)
    sentences = strings(arrays["sentence"])
    counts = np.asarray([len(sentence.split()) for sentence in sentences], dtype=np.int64)
    content_counts = np.asarray(arrays["target_content_word_count"], dtype=np.int64)
    return {
        "n": int(len(sentences)),
        "embedding_shape": list(arrays["input_embeddings"].shape),
        "word_counts": sorted(int(value) for value in np.unique(counts)),
        "word_count_stats": {
            "min": int(counts.min()),
            "mean": float(counts.mean()),
            "p50": float(np.percentile(counts, 50)),
            "p95": float(np.percentile(counts, 95)),
            "max": int(counts.max()),
        },
        "content_word_counts": {
            "min": int(content_counts.min()),
            "mean": float(content_counts.mean()),
            "p50": float(np.percentile(content_counts, 50)),
            "p95": float(np.percentile(content_counts, 95)),
            "max": int(content_counts.max()),
        },
        "center_distance_seconds": {
            "mean": float(distances.mean()),
            "p95": float(np.percentile(distances, 95)),
            "max": float(distances.max()),
        },
        "selected_window_center_distance_seconds": {
            "mean": float(window_center_distances.mean()),
            "p95": float(np.percentile(window_center_distances, 95)),
            "max": float(window_center_distances.max()),
        },
        "selected_window_contains_nearest_center_word_fraction": float(center_inclusions.mean()),
        "selected_window_boundary_spill_seconds": {
            "fraction_nonzero": float((window_boundary_spills > 0).mean()),
            "mean": float(window_boundary_spills.mean()),
            "p95": float(np.percentile(window_boundary_spills, 95)),
            "max": float(window_boundary_spills.max()),
        },
        "phrase_duration_seconds": {
            "mean": float(durations.mean()),
            "p95": float(np.percentile(durations, 95)),
            "max": float(durations.max()),
        },
        "samples": [
            {
                "sentence": sentence,
                "content_words": content_words,
                "content_word_count": int(content_count),
            }
            for sentence, content_words, content_count in zip(
                sentences[:sample_count],
                strings(arrays["target_content_words"])[:sample_count],
                content_counts[:sample_count].tolist(),
            )
        ],
    }


def relabel_predicted_test(
    source_path: Path,
    output_path: Path,
    oracle_test: dict[str, np.ndarray],
    contract: dict[str, Any],
    expected_dim: int,
) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=True) as source:
        output = {key: np.asarray(source[key]) for key in source.files if key != "schema_json"}
        schema = parse_schema(source["schema_json"] if "schema_json" in source.files else None)
    expected_shape = oracle_test["input_embeddings"].shape
    if output.get("input_embeddings", np.empty(0)).shape != expected_shape or expected_shape[1] != expected_dim:
        raise ValueError(f"Predicted test shape does not match oracle test: {output.get('input_embeddings', np.empty(0)).shape} vs {expected_shape}.")
    for key in ("story", "start_tr", "stop_tr"):
        if key not in output or not np.array_equal(output[key], oracle_test[key]):
            raise ValueError(f"Predicted test rows are not aligned with oracle test key {key!r}.")
    for key in TARGET_ROW_KEYS:
        output[key] = oracle_test[key]
    output["split"] = oracle_test["split"]
    schema["target_text_contract"] = contract
    schema["paired_oracle_target_source"] = "relabelled oracle test rows"
    save_npz(output_path, output, schema)
    return {"path": str(output_path), "shape": list(output["input_embeddings"].shape), "row_alignment": True}


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    delays = [int(value) for value in metadata.get("stim_delays", [])]
    tr_seconds = float((metadata.get("args") or {}).get("tr", 0.0))
    tang_trim_start_trs = int((metadata.get("args") or {}).get("tang_trim_start", 10))
    if delays != [1, 2, 3, 4]:
        raise ValueError(f"Expected stimulus delays [1, 2, 3, 4], got {delays}.")
    if tr_seconds != 2.0:
        raise ValueError(f"Expected TR=2.0 seconds, got {tr_seconds}.")
    if metadata.get("segment_trs") != 10:
        raise ValueError(f"Expected 10-TR segments, got {metadata.get('segment_trs')}.")

    expected_counts = {"train": args.expected_train, "val": args.expected_val, "test": args.expected_test}
    read_word_intervals = load_textgrid_reader(Path(args.mri2sem_src))
    contract = target_contract(
        metadata=metadata,
        delays=delays,
        tr_seconds=tr_seconds,
        tang_trim_start_trs=tang_trim_start_trs,
        tang_start_time_sec=args.tang_start_time_sec,
        word_count=args.word_count,
        maximize_content_words=args.maximize_content_words,
        min_content_words=args.min_content_words,
        content_search_radius_sec=args.content_search_radius_sec,
        time_window_radius_sec=args.time_window_radius_sec,
    )
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    created: list[str] = []
    summaries: dict[str, Any] = {}
    for split in SPLITS:
        input_path = input_dir / f"{args.input_prefix}_{split}.npz"
        arrays, schema = load_split(input_path, expected_counts[split], args.expected_dim, split)
        relabelled = relabel_split(
            arrays,
            split=split,
            stimulus_root=Path(args.stimulus_data_root),
            read_word_intervals=read_word_intervals,
            tr_seconds=tr_seconds,
            delays=delays,
            tang_trim_start_trs=tang_trim_start_trs,
            tang_start_time_sec=args.tang_start_time_sec,
            word_count=args.word_count,
            maximize_content_words=args.maximize_content_words,
            min_content_words=args.min_content_words,
            content_search_radius_sec=args.content_search_radius_sec,
            time_window_radius_sec=args.time_window_radius_sec,
        )
        if not np.array_equal(relabelled["input_embeddings"], arrays["input_embeddings"]):
            raise AssertionError(f"{split} semantic vectors changed during text relabeling.")
        schema["target_text_contract"] = contract
        schema["source_elf_npz"] = str(input_path)
        schema["shape"] = list(relabelled["input_embeddings"].shape)
        output_path = output_dir / f"{args.output_prefix}_{split}.npz"
        save_npz(output_path, relabelled, schema)
        split_arrays[split] = relabelled
        created.append(str(output_path))
        summaries[split] = split_summary(relabelled, args.sample_count)

    train_val = concatenate([split_arrays["train"], split_arrays["val"]])
    train_val_schema = {
        "source_elf_npzs": [
            str(input_dir / f"{args.input_prefix}_train.npz"),
            str(input_dir / f"{args.input_prefix}_val.npz"),
        ],
        "target_text_contract": contract,
        "input_key": "input_embeddings",
        "sentence_key": "sentence",
        "embedding_dim": args.expected_dim,
        "split_order": ["train", "val"],
        "elf_training_contract": {
            "train_rows": args.expected_train,
            "validation_rows": args.expected_val,
            "validation_is_tail": True,
            "trainer_argument": f"--val-num-examples {args.expected_val}",
            "test_excluded": True,
        },
    }
    train_val_path = output_dir / f"{args.output_prefix}_train_val.npz"
    save_npz(train_val_path, train_val, train_val_schema)
    created.append(str(train_val_path))

    predicted_summary = None
    if args.predicted_test_npz:
        if not args.predicted_test_output:
            raise ValueError("--predicted-test-output is required with --predicted-test-npz.")
        predicted_summary = relabel_predicted_test(
            Path(args.predicted_test_npz),
            Path(args.predicted_test_output),
            split_arrays["test"],
            contract,
            args.expected_dim,
        )
        created.append(args.predicted_test_output)

    summary = {
        "created": created,
        "target_text_contract": contract,
        "splits": summaries,
        "predicted_test": predicted_summary,
        "semantic_vectors_preserved_exactly": True,
    }
    summary_path = output_dir / f"{args.output_prefix}_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
