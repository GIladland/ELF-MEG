#!/usr/bin/env python
"""Filter semantic-vector NPZ rows by sentence-shape heuristics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DEFAULT_META_RE = (
    r"\b("
    r"librivox|public domain|recording|chapter|book|sir arthur|"
    r"conan doyle|study in scarlet|copyright|end of|read by"
    r")\b"
)
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
    reached enter enters entered pass passes passed carry carries carried
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--schema-key", default="schema_json")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.75)
    parser.add_argument("--min-content-words", type=int, default=2)
    parser.add_argument("--require-verb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--filter-mode",
        choices=["short_content", "simple_sv", "simple_sv_no_coord"],
        default="short_content",
    )
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--exclude-regex", default=DEFAULT_META_RE)
    parser.add_argument("--sample-count", type=int, default=20)
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def _schema(data: np.lib.npyio.NpzFile, schema_key: str) -> Any:
    if schema_key not in data.files:
        return None
    try:
        return json.loads(str(data[schema_key].tolist()))
    except json.JSONDecodeError:
        return str(data[schema_key].tolist())


def words(sentence: str) -> list[str]:
    return WORD_RE.findall(sentence)


def has_verb(words_: list[str]) -> bool:
    lowered = [word.lower() for word in words_]
    return any(word in COMMON_VERBS or word.endswith(("ed", "ing")) for word in lowered)


def content_words(words_: list[str]) -> list[str]:
    return [word.lower() for word in words_ if word.lower() not in STOP_WORDS and len(word) > 2]


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


def keep_sentence(
    sentence: str,
    *,
    min_words: int,
    max_words: int,
    min_alpha_fraction: float,
    min_content_words: int,
    require_verb: bool,
    filter_mode: str,
    exclude_re: re.Pattern[str] | None,
) -> tuple[bool, str]:
    words_ = words(sentence)
    if len(words_) < min_words:
        return False, "too_short"
    if len(words_) > max_words:
        return False, "too_long"
    if exclude_re and exclude_re.search(sentence):
        return False, "excluded_regex"
    if NOISE_RE.search(sentence):
        return False, "noise"
    alpha_count = sum(1 for char in sentence if char.isalpha())
    nonspace_count = sum(1 for char in sentence if not char.isspace())
    if nonspace_count == 0 or alpha_count / nonspace_count < min_alpha_fraction:
        return False, "low_alpha_fraction"
    if len(content_words(words_)) < min_content_words:
        return False, "too_few_content_words"
    if require_verb and not has_verb(words_):
        return False, "no_verb"
    if re.fullmatch(
        r"(yes|no|what|why|where|how|when|indeed|certainly|perhaps|maybe)(\s+\w+){0,4}",
        sentence,
        flags=re.IGNORECASE,
    ):
        return False, "generic_short_reply"
    if filter_mode == "simple_sv" and not simple_sv(sentence, allow_coord=True):
        return False, "not_simple_sv"
    if filter_mode == "simple_sv_no_coord" and not simple_sv(sentence, allow_coord=False):
        return False, "not_simple_sv"
    return True, "kept"


def first_dim_keys(data: np.lib.npyio.NpzFile, n_rows: int) -> set[str]:
    keys = set()
    for key in data.files:
        try:
            value = data[key]
        except Exception:
            continue
        if getattr(value, "shape", None) and value.shape[0] == n_rows:
            keys.add(key)
    return keys


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE) if args.exclude_regex else None

    data = np.load(input_path, allow_pickle=True)
    sentences = np.asarray(_strings(data[args.sentence_key]), dtype=object)
    keep = []
    seen: set[str] = set()
    reasons: dict[str, int] = {}
    duplicate_count = 0
    for idx, sentence in enumerate(sentences.tolist()):
        ok, reason = keep_sentence(
            str(sentence),
            min_words=args.min_words,
            max_words=args.max_words,
            min_alpha_fraction=args.min_alpha_fraction,
            min_content_words=args.min_content_words,
            require_verb=args.require_verb,
            filter_mode=args.filter_mode,
            exclude_re=exclude_re,
        )
        if ok and args.dedupe:
            key = re.sub(r"\s+", " ", str(sentence).strip()).casefold()
            if key in seen:
                ok = False
                reason = "duplicate"
                duplicate_count += 1
            seen.add(key)
        reasons[reason] = reasons.get(reason, 0) + 1
        if ok:
            keep.append(idx)
    keep_idx = np.asarray(keep, dtype=np.int64)
    if len(keep_idx) == 0:
        raise RuntimeError(f"No rows kept from {input_path}")

    output: dict[str, np.ndarray] = {}
    row_keys = first_dim_keys(data, len(sentences))
    for key in data.files:
        value = data[key]
        if key in row_keys:
            output[key] = value[keep_idx]
        elif key != args.schema_key:
            output[key] = value

    kept_sentences = sentences[keep_idx]
    word_counts = np.asarray([len(words(str(sentence))) for sentence in kept_sentences], dtype=np.int64)
    summary = {
        "output": str(output_path),
        "input": str(input_path),
        "input_count": int(len(sentences)),
        "kept_count": int(len(keep_idx)),
        "kept_fraction": float(len(keep_idx) / len(sentences)),
        "dropped_count": int(len(sentences) - len(keep_idx)),
        "drop_reasons": reasons,
        "duplicate_count": int(duplicate_count),
        "min_words": args.min_words,
        "max_words": args.max_words,
        "min_alpha_fraction": args.min_alpha_fraction,
        "min_content_words": args.min_content_words,
        "require_verb": args.require_verb,
        "filter_mode": args.filter_mode,
        "exclude_regex": args.exclude_regex,
        "source_schema": _schema(data, args.schema_key),
        "embedding_shape": list(output[args.input_key].shape),
        "word_count_mean": float(word_counts.mean()),
        "word_count_median": float(np.median(word_counts)),
        "word_count_p10": float(np.percentile(word_counts, 10)),
        "word_count_p90": float(np.percentile(word_counts, 90)),
        "sample": kept_sentences[: args.sample_count].tolist(),
    }
    output[args.schema_key] = np.asarray(json.dumps(summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
