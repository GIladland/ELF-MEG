#!/usr/bin/env python
"""Export sentence embeddings from LibriBrain-style BIDS event TSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NOISE_RE = re.compile(r"[_=<>[\]{}]|https?://|www\.", flags=re.IGNORECASE)
STOP_WORDS = set(
    """
    the a an and or but if then than that this these those of in on at to for from with
    without by as is are was were be been being am i you he she it we they his her him
    them their my your our me us not no yes do did does have has had will would could
    should can may might must into out up down over under there here
    """.split()
)
COMMON_VERBS = set(
    """
    is are was were be been being am do does did have has had say says said tell tells
    told ask asks asked answer answered reply replied know knows knew think thinks thought
    see sees saw seen look looks looked seem seems seemed come comes came go goes went
    gone get gets got make makes made take takes took taken give gives gave given find
    finds found hear hears heard feel feels felt leave leaves left turn turns turned
    stand stands stood sit sits sat lie lies lay laid run runs ran walk walks walked
    wait waits waited open opens opened close closes closed move moves moved put puts
    placed keep keeps kept hold holds held call calls called speak speaks spoke spoken
    cry cries cried smile smiles smiled laugh laughs laughed nod nods nodded begin
    begins began begun end ends ended mean means meant want wants wanted remember
    remembers remembered believe believes believed suppose supposes supposed appear
    appears appeared remain remains remained return returns returned reach reaches
    reached enter enters entered pass passes passed
    """.split()
)
VERBISH = COMMON_VERBS | set(
    """
    can could may might must shall should will would said says saying going wanted wants
    wanting tried tries try trying used uses use using need needs needed liked likes like
    liking became become becomes coming went told telling asked asking called calling put
    putting made making took taking given giving seen seeing heard hearing found finding
    left leaving moved moving started starts start starting stopped stops stop stopping
    brought bring brings bringing kept keeping felt feeling saw looked looking
    """.split()
)
NON_DECLARATIVE_INITIAL = {"who", "whom", "whose", "what", "where", "when", "why", "how", "which", "please"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-glob", action="append", default=[])
    parser.add_argument("--event-file", action="append", default=[])
    parser.add_argument("--include-path-regex", default="")
    parser.add_argument("--exclude-path-regex", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--filter-mode",
        choices=["short_content", "simple_sv", "simple_sv_no_coord"],
        default="simple_sv_no_coord",
    )
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.75)
    parser.add_argument("--min-content-words", type=int, default=2)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def fnum(value: str | None) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return None if math.isnan(parsed) else parsed
    except ValueError:
        return None


def words(sentence: str) -> list[str]:
    return WORD_RE.findall(sentence)


def has_verb(words_: list[str]) -> bool:
    return any(word.lower() in COMMON_VERBS or word.lower().endswith(("ed", "ing")) for word in words_)


def content_words(words_: list[str]) -> list[str]:
    return [word.lower() for word in words_ if word.lower() not in STOP_WORDS and len(word) > 2]


def short_content_reason(
    sentence: str,
    *,
    min_words: int,
    max_words: int,
    min_alpha_fraction: float,
    min_content_words: int,
) -> tuple[bool, str]:
    words_ = words(sentence)
    if len(words_) < min_words:
        return False, "too_short"
    if len(words_) > max_words:
        return False, "too_long"
    if NOISE_RE.search(sentence):
        return False, "noise"
    alpha_count = sum(1 for char in sentence if char.isalpha())
    nonspace_count = sum(1 for char in sentence if not char.isspace())
    if nonspace_count == 0 or alpha_count / nonspace_count < min_alpha_fraction:
        return False, "low_alpha_fraction"
    if len(content_words(words_)) < min_content_words:
        return False, "too_few_content_words"
    if not has_verb(words_):
        return False, "no_verb"
    if re.fullmatch(
        r"(yes|no|what|why|where|how|when|indeed|certainly|perhaps|maybe)(\s+\w+){0,4}",
        sentence,
        flags=re.IGNORECASE,
    ):
        return False, "generic_short_reply"
    return True, "kept"


def simple_sv(sentence: str, *, allow_coord: bool) -> bool:
    words_ = words(sentence)
    if words_ and words_[0].lower() in NON_DECLARATIVE_INITIAL:
        return False
    if not allow_coord and any(word.lower() in {"and", "or", "but"} for word in words_):
        return False
    for verb_idx, word in enumerate(words_):
        lowered = word.lower()
        if lowered not in VERBISH and not lowered.endswith(("ed", "ing")):
            continue
        before = [item for item in words_[:verb_idx] if item.lower() not in STOP_WORDS and len(item) > 2]
        after = [item for item in words_[verb_idx + 1 :] if item.lower() not in STOP_WORDS and len(item) > 2]
        if before and after:
            return True
    return False


def keep_sentence(sentence: str, args: argparse.Namespace) -> tuple[bool, str]:
    ok, reason = short_content_reason(
        sentence,
        min_words=args.min_words,
        max_words=args.max_words,
        min_alpha_fraction=args.min_alpha_fraction,
        min_content_words=args.min_content_words,
    )
    if not ok or args.filter_mode == "short_content":
        return ok, reason
    if args.filter_mode == "simple_sv" and simple_sv(sentence, allow_coord=True):
        return True, "kept"
    if args.filter_mode == "simple_sv_no_coord" and simple_sv(sentence, allow_coord=False):
        return True, "kept"
    return False, "not_simple_sv"


def metadata_for_path(path: Path) -> dict:
    match = re.search(
        r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_events.tsv$",
        path.name,
    )
    meta = {"path": str(path)}
    if match:
        meta.update(match.groupdict())
    book_match = re.search(r"/(Sherlock\d+|TIMIT|MOCHATIMIT|TheMoth)/", str(path))
    if book_match:
        meta["corpus"] = book_match.group(1)
    return meta


def collect_event_files(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.event_file]
    for pattern in args.event_glob:
        paths.extend(Path(path) for path in glob.glob(pattern))
    include_re = re.compile(args.include_path_regex) if args.include_path_regex else None
    exclude_re = re.compile(args.exclude_path_regex) if args.exclude_path_regex else None
    unique = []
    seen = set()
    for path in sorted(paths):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if include_re and not include_re.search(key):
            continue
        if exclude_re and exclude_re.search(key):
            continue
        unique.append(path)
    return unique


def collect_sentences(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        groups: dict[str, list[tuple[float, float | None, float, str]]] = {}
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if (row.get("kind") or "").strip() != "word":
                    continue
                sentence_id = row.get("sentenceidx")
                if sentence_id is None or sentence_id == "":
                    continue
                word = (row.get("segment") or "").strip()
                if not word:
                    continue
                word_idx = fnum(row.get("wordidx"))
                time_sentence = fnum(row.get("timesentence"))
                time_ds = fnum(row.get("timeds"))
                duration = fnum(row.get("duration")) or 0.0
                order = word_idx if word_idx is not None else (time_sentence if time_sentence is not None else 0.0)
                groups.setdefault(sentence_id, []).append((order, time_ds, duration, word))
        path_meta = metadata_for_path(path)
        for sentence_id, items in groups.items():
            ordered = sorted(items, key=lambda item: (item[0], item[1] if item[1] is not None else 0.0))
            sentence = " ".join(item[3] for item in ordered).strip()
            starts = [item[1] for item in ordered if item[1] is not None]
            ends = [item[1] + item[2] for item in ordered if item[1] is not None]
            rows.append(
                {
                    **path_meta,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "word_count": len(words(sentence)),
                    "span_seconds": float(max(ends) - min(starts)) if starts and ends else 0.0,
                    "word_audio_seconds": float(sum(item[2] for item in ordered)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    paths = collect_event_files(args)
    if not paths:
        raise RuntimeError("No event files matched.")
    raw_rows = collect_sentences(paths)
    kept_rows = []
    reasons = Counter()
    seen: set[str] = set()
    duplicate_count = 0
    for row in raw_rows:
        sentence = row["sentence"]
        ok, reason = keep_sentence(sentence, args)
        if ok and args.dedupe:
            key = re.sub(r"\s+", " ", sentence.strip()).casefold()
            if key in seen:
                ok = False
                reason = "duplicate"
                duplicate_count += 1
            seen.add(key)
        reasons[reason] += 1
        if ok:
            kept_rows.append(row)
    if not kept_rows:
        raise RuntimeError("No sentences kept.")

    sentences = [row["sentence"] for row in kept_rows]
    word_counts = np.asarray([row["word_count"] for row in kept_rows], dtype=np.int64)
    summary = {
        "output": args.output,
        "event_files": len(paths),
        "input_sentences": len(raw_rows),
        "kept_count": len(kept_rows),
        "kept_fraction": len(kept_rows) / max(1, len(raw_rows)),
        "unique_kept_texts": len({re.sub(r"\s+", " ", sentence.strip()).casefold() for sentence in sentences}),
        "filter_mode": args.filter_mode,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "min_alpha_fraction": args.min_alpha_fraction,
        "min_content_words": args.min_content_words,
        "dedupe": args.dedupe,
        "duplicate_count": duplicate_count,
        "drop_reasons": dict(reasons),
        "span_hours": sum(float(row["span_seconds"]) for row in kept_rows) / 3600,
        "word_audio_hours": sum(float(row["word_audio_seconds"]) for row in kept_rows) / 3600,
        "word_count_mean": float(word_counts.mean()),
        "word_count_median": float(np.median(word_counts)),
        "word_count_p10": float(np.percentile(word_counts, 10)),
        "word_count_p90": float(np.percentile(word_counts, 90)),
        "corpus_counts": dict(Counter(str(row.get("corpus", "")) for row in kept_rows)),
        "session_counts": dict(Counter(f"{row.get('corpus', '')}:ses-{row.get('session', '')}" for row in kept_rows)),
        "sample": sentences[:20],
    }
    if args.summary_only:
        print(json.dumps(summary, indent=2))
        return

    from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.embedding_model_name, local_files_only=args.local_files_only).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = encode_sentences(
            sentences=[f"{args.input_prefix}{sentence}" for sentence in sentences],
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
    summary.update(
        {
            "embedding_model_name": args.embedding_model_name,
            "embedding_shape": list(embeddings.shape),
            "pooling": args.pooling,
            "normalized": not args.no_normalize,
        }
    )
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(sentences, dtype=object),
        rows=np.asarray(kept_rows, dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
