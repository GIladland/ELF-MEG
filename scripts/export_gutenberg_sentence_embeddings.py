#!/usr/bin/env python
"""Export Project Gutenberg sentences with local HuggingFace sentence embeddings."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

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


DEFAULT_GUTENDEX_URLS = [
    "https://gutendex.com/books?topic=detective&languages=en&copyright=false",
    "https://gutendex.com/books?topic=mystery&languages=en&copyright=false",
]

ABBREVIATIONS = [
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Rev.",
    "St.",
    "Capt.",
    "Col.",
    "Gen.",
    "Maj.",
    "Sgt.",
    "Lt.",
    "Insp.",
    "Supt.",
    "Hon.",
    "Jr.",
    "Sr.",
    "vs.",
    "etc.",
    "i.e.",
    "e.g.",
]

TEXT_FORMAT_PRIORITY = [
    "text/plain; charset=utf-8",
    "text/plain; charset=us-ascii",
    "text/plain; charset=iso-8859-1",
    "text/plain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--gutendex-url",
        action="append",
        default=[],
        help="Gutendex listing URL. May be repeated. Defaults to detective and mystery topics.",
    )
    parser.add_argument("--gutenberg-id", action="append", type=int, default=[])
    parser.add_argument(
        "--catalog-csv-url",
        default="",
        help="Project Gutenberg CSV catalog URL/path. If set, collect books from the catalog instead of Gutendex listings.",
    )
    parser.add_argument("--shuffle-books", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-books", type=int, default=0)
    parser.add_argument("--limit-sentences", type=int, default=0)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument(
        "--exclude-author-regex",
        default=r"Arthur\s+Conan\s+Doyle|Doyle,\s*Arthur\s+Conan|Conan\s+Doyle",
        help="Regex over author names to exclude. Default excludes Conan Doyle/Sherlock leakage.",
    )
    parser.add_argument(
        "--exclude-title-regex",
        default="",
        help="Regex over titles to exclude, useful for anthologies containing Sherlock/Doyle text.",
    )
    parser.add_argument(
        "--exclude-sentence-regex",
        default="",
        help="Regex over individual candidate sentences to exclude.",
    )
    parser.add_argument("--include-author-regex", default="")
    parser.add_argument("--include-title-regex", default="")
    parser.add_argument("--exclude-locc-regex", default="")
    parser.add_argument("--include-locc-regex", default="")
    parser.add_argument("--exclude-subject-regex", default="")
    parser.add_argument("--include-subject-regex", default="")
    parser.add_argument("--exclude-bookshelf-regex", default="")
    parser.add_argument("--include-bookshelf-regex", default="")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.65)
    parser.add_argument("--download-delay", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def cache_name(value: str, suffix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{digest}{suffix}"


def request_bytes(url: str, *, timeout: float, retries: int) -> bytes:
    headers = {
        "User-Agent": "BrainDiffusion research corpus builder (contact: local experiment)",
        "Accept": "text/plain, application/json;q=0.9, */*;q=0.1",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(30.0, 2.0 * attempt))
    raise RuntimeError(f"Failed to fetch {url!r} after {retries} attempts: {last_error}")


def cached_bytes(cache_dir: Path, url: str, *, suffix: str, timeout: float, retries: int) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_name(url, suffix)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    data = request_bytes(url, timeout=timeout, retries=retries)
    path.write_bytes(data)
    return data


def cached_json(cache_dir: Path, url: str, *, timeout: float, retries: int) -> dict[str, Any]:
    data = cached_bytes(cache_dir, url, suffix=".json", timeout=timeout, retries=retries)
    return json.loads(data.decode("utf-8"))


def authors_text(book: dict[str, Any]) -> str:
    authors = book.get("authors") or []
    names = [str(author.get("name", "")) for author in authors if author.get("name")]
    return "; ".join(names)


def joined_field_text(book: dict[str, Any], field: str) -> str:
    value = book.get(field) or []
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def should_keep_book(
    book: dict[str, Any],
    *,
    exclude_author: re.Pattern[str] | None,
    exclude_title: re.Pattern[str] | None,
    include_author: re.Pattern[str] | None,
    include_title: re.Pattern[str] | None,
    exclude_locc: re.Pattern[str] | None,
    include_locc: re.Pattern[str] | None,
    exclude_subject: re.Pattern[str] | None,
    include_subject: re.Pattern[str] | None,
    exclude_bookshelf: re.Pattern[str] | None,
    include_bookshelf: re.Pattern[str] | None,
) -> bool:
    title = str(book.get("title", ""))
    author = authors_text(book)
    locc = joined_field_text(book, "locc")
    subjects = joined_field_text(book, "subjects")
    bookshelves = joined_field_text(book, "bookshelves")
    if exclude_author and exclude_author.search(author):
        return False
    if exclude_title and exclude_title.search(title):
        return False
    if exclude_locc and exclude_locc.search(locc):
        return False
    if exclude_subject and exclude_subject.search(subjects):
        return False
    if exclude_bookshelf and exclude_bookshelf.search(bookshelves):
        return False
    if include_author and not include_author.search(author):
        return False
    if include_title and not include_title.search(title):
        return False
    if include_locc and not include_locc.search(locc):
        return False
    if include_subject and not include_subject.search(subjects):
        return False
    if include_bookshelf and not include_bookshelf.search(bookshelves):
        return False
    return True


def text_urls_for_book(book: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    formats = book.get("formats") or {}
    for key in TEXT_FORMAT_PRIORITY:
        value = formats.get(key)
        if isinstance(value, str) and value:
            urls.append(value)
    for key, value in formats.items():
        if str(key).startswith("text/plain") and isinstance(value, str) and value:
            urls.append(value)
    book_id = book.get("id")
    if isinstance(book_id, int):
        urls.extend(
            [
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            ]
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def best_text_url(book: dict[str, Any]) -> str | None:
    urls = text_urls_for_book(book)
    return urls[0] if urls else None


def catalog_book_from_row(row: dict[str, str]) -> dict[str, Any] | None:
    if row.get("Type", "").strip().casefold() != "text":
        return None
    if row.get("Language", "").strip().casefold() != "en":
        return None
    text_id = row.get("Text#", "").strip()
    if not text_id.isdigit():
        return None
    return {
        "id": int(text_id),
        "title": re.sub(r"\s+", " ", row.get("Title", "")).strip(),
        "authors": [{"name": re.sub(r"\s+", " ", row.get("Authors", "")).strip()}],
        "subjects": [item.strip() for item in row.get("Subjects", "").split(";") if item.strip()],
        "bookshelves": [item.strip() for item in row.get("Bookshelves", "").split(";") if item.strip()],
        "locc": [item.strip() for item in row.get("LoCC", "").split(";") if item.strip()],
        "formats": {},
        "catalog_source": "pg_catalog_csv",
    }


def collect_catalog_books(args: argparse.Namespace) -> list[dict[str, Any]]:
    catalog_arg = args.catalog_csv_url
    if re.match(r"https?://", catalog_arg):
        data = cached_bytes(
            Path(args.cache_dir) / "catalog",
            catalog_arg,
            suffix=".csv.gz" if catalog_arg.endswith(".gz") else ".csv",
            timeout=args.request_timeout,
            retries=args.retries,
        )
    else:
        data = Path(catalog_arg).read_bytes()
    if catalog_arg.endswith(".gz"):
        text = gzip.decompress(data).decode("utf-8", errors="replace")
    else:
        text = data.decode("utf-8", errors="replace")
    books: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(text)):
        book = catalog_book_from_row(row)
        if book is not None:
            books.append(book)
    if args.shuffle_books:
        random.Random(args.seed).shuffle(books)
    if args.max_books:
        books = books[: args.max_books]
    return books


def collect_books(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.catalog_csv_url:
        return collect_catalog_books(args)
    cache_dir = Path(args.cache_dir) / "gutendex"
    urls = args.gutendex_url or DEFAULT_GUTENDEX_URLS
    by_id: dict[int, dict[str, Any]] = {}
    for book_id in args.gutenberg_id:
        url = f"https://gutendex.com/books/{book_id}"
        book = cached_json(cache_dir, url, timeout=args.request_timeout, retries=args.retries)
        if isinstance(book.get("id"), int):
            by_id[int(book["id"])] = book

    for start_url in urls:
        url: str | None = start_url
        while url:
            page = cached_json(cache_dir, url, timeout=args.request_timeout, retries=args.retries)
            for book in page.get("results", []):
                book_id = book.get("id")
                if isinstance(book_id, int):
                    by_id[book_id] = book
                    if args.max_books and len(by_id) >= args.max_books:
                        url = None
                        break
            if args.max_books and len(by_id) >= args.max_books:
                break
            url = page.get("next")
    return [by_id[key] for key in sorted(by_id)]


def decode_text(data: bytes) -> str:
    text = data.decode("utf-8-sig", errors="replace")
    if text.count("\ufffd") > max(10, len(text) // 500):
        text = data.decode("latin-1", errors="replace")
    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    ]
    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    ]
    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end() :]
            break
    for pattern in end_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[: match.start()]
            break
    return text


def normalize_body(text: str) -> str:
    text = strip_gutenberg_boilerplate(text)
    text = re.sub(r"\[[^\]\n]*(?:illustration|transcriber)[^\]\n]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^\s*(contents|chapter\s+[ivxlcdm0-9]+)\s*$", line, flags=re.IGNORECASE):
            lines.append("")
            continue
        if re.search(r"project gutenberg|www\.gutenberg\.org|https?://", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)


def protect_abbreviations(text: str) -> str:
    protected = text
    for abbreviation in ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<prd>"))
    protected = re.sub(r"\b([A-Z])\.", r"\1<prd>", protected)
    return protected


def restore_abbreviations(text: str) -> str:
    return text.replace("<prd>", ".")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        protected = protect_abbreviations(paragraph)
        pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", protected)
        for piece in pieces:
            sentence = restore_abbreviations(piece).strip()
            sentence = sentence.strip(" \t\n\r\"'")
            sentence = re.sub(r"\s+", " ", sentence)
            if sentence:
                sentences.append(sentence)
    return sentences


def sentence_words(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)


def normalized_sentence_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip()).casefold()


def keep_sentence(
    sentence: str,
    *,
    min_words: int,
    max_words: int,
    min_alpha_fraction: float,
) -> bool:
    words = sentence_words(sentence)
    if len(words) < min_words or len(words) > max_words:
        return False
    alpha_count = sum(1 for char in sentence if char.isalpha())
    nonspace_count = sum(1 for char in sentence if not char.isspace())
    if nonspace_count == 0 or alpha_count / nonspace_count < min_alpha_fraction:
        return False
    if sentence.isupper():
        return False
    lowered = sentence.lower()
    blocked_fragments = [
        "project gutenberg",
        "librivox",
        "all rights reserved",
        "table of contents",
        "end of the project",
    ]
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    return True


def rows_for_book(
    book: dict[str, Any],
    *,
    cache_dir: Path,
    args: argparse.Namespace,
    exclude_sentence: re.Pattern[str] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    urls = text_urls_for_book(book)
    if not urls:
        return [], []
    data: bytes | None = None
    url = ""
    for candidate_url in urls:
        try:
            data = cached_bytes(
                cache_dir / "texts",
                candidate_url,
                suffix=".txt",
                timeout=args.request_timeout,
                retries=args.retries,
            )
            url = candidate_url
            break
        except RuntimeError as exc:
            print(
                f"warning: failed_text_url book={book.get('id')} url={candidate_url!r} error={exc}",
                flush=True,
            )
    if data is None:
        return [], []
    text = normalize_body(decode_text(data))
    sentences = split_sentences(text)
    kept_sentences: list[str] = []
    rows: list[dict[str, Any]] = []
    book_id = int(book["id"])
    title = str(book.get("title", ""))
    author = authors_text(book)
    bookshelves = [str(x) for x in book.get("bookshelves", [])]
    subjects = [str(x) for x in book.get("subjects", [])]
    locc = [str(x) for x in book.get("locc", [])]
    for idx, sentence in enumerate(sentences):
        if exclude_sentence and exclude_sentence.search(sentence):
            continue
        if not keep_sentence(
            sentence,
            min_words=args.min_words,
            max_words=args.max_words,
            min_alpha_fraction=args.min_alpha_fraction,
        ):
            continue
        kept_sentences.append(sentence)
        rows.append(
            {
                "source": "project_gutenberg",
                "gutenberg_id": book_id,
                "title": title,
                "authors": author,
                "bookshelves": bookshelves,
                "subjects": subjects,
                "locc": locc,
                "text_url": url,
                "sentence_index": idx,
            }
        )
    return kept_sentences, rows


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    exclude_author = re.compile(args.exclude_author_regex, flags=re.IGNORECASE) if args.exclude_author_regex else None
    exclude_title = re.compile(args.exclude_title_regex, flags=re.IGNORECASE) if args.exclude_title_regex else None
    exclude_sentence = re.compile(args.exclude_sentence_regex, flags=re.IGNORECASE) if args.exclude_sentence_regex else None
    include_author = re.compile(args.include_author_regex, flags=re.IGNORECASE) if args.include_author_regex else None
    include_title = re.compile(args.include_title_regex, flags=re.IGNORECASE) if args.include_title_regex else None
    exclude_locc = re.compile(args.exclude_locc_regex, flags=re.IGNORECASE) if args.exclude_locc_regex else None
    include_locc = re.compile(args.include_locc_regex, flags=re.IGNORECASE) if args.include_locc_regex else None
    exclude_subject = re.compile(args.exclude_subject_regex, flags=re.IGNORECASE) if args.exclude_subject_regex else None
    include_subject = re.compile(args.include_subject_regex, flags=re.IGNORECASE) if args.include_subject_regex else None
    exclude_bookshelf = re.compile(args.exclude_bookshelf_regex, flags=re.IGNORECASE) if args.exclude_bookshelf_regex else None
    include_bookshelf = re.compile(args.include_bookshelf_regex, flags=re.IGNORECASE) if args.include_bookshelf_regex else None

    books = collect_books(args)
    sentences: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped_books = 0
    downloaded_books = 0
    dropped_duplicates = 0
    source_counts: list[dict[str, Any]] = []
    for book in books:
        if not should_keep_book(
            book,
            exclude_author=exclude_author,
            exclude_title=exclude_title,
            include_author=include_author,
            include_title=include_title,
            exclude_locc=exclude_locc,
            include_locc=include_locc,
            exclude_subject=exclude_subject,
            include_subject=include_subject,
            exclude_bookshelf=exclude_bookshelf,
            include_bookshelf=include_bookshelf,
        ):
            skipped_books += 1
            continue
        book_sentences, book_rows = rows_for_book(
            book,
            cache_dir=cache_dir,
            args=args,
            exclude_sentence=exclude_sentence,
        )
        downloaded_books += 1
        kept_for_book = 0
        for sentence, row in zip(book_sentences, book_rows):
            key = normalized_sentence_key(sentence)
            if args.dedupe and key in seen:
                dropped_duplicates += 1
                continue
            seen.add(key)
            sentences.append(sentence)
            rows.append(row)
            kept_for_book += 1
            if args.limit_sentences and len(sentences) >= args.limit_sentences:
                break
        source_counts.append(
            {
                "gutenberg_id": int(book["id"]),
                "title": str(book.get("title", "")),
                "authors": authors_text(book),
                "kept_sentences": kept_for_book,
            }
        )
        print(
            f"book={book.get('id')} kept={kept_for_book} total_sentences={len(sentences)} "
            f"title={book.get('title', '')!r}",
            flush=True,
        )
        if args.download_delay > 0:
            time.sleep(args.download_delay)
        if args.limit_sentences and len(sentences) >= args.limit_sentences:
            break

    if not sentences:
        raise RuntimeError("No Gutenberg sentences collected.")

    embedding_inputs = [f"{args.input_prefix}{sentence}" for sentence in sentences]
    device = resolve_device(args.device)
    print(f"Using device={device}")
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
        "cache_dir": str(cache_dir),
        "gutendex_urls": args.gutendex_url or DEFAULT_GUTENDEX_URLS,
        "catalog_csv_url": args.catalog_csv_url,
        "shuffle_books": args.shuffle_books,
        "seed": args.seed,
        "gutenberg_ids": args.gutenberg_id,
        "books_seen": len(books),
        "books_downloaded": downloaded_books,
        "books_skipped": skipped_books,
        "count": len(sentences),
        "dropped_duplicates": dropped_duplicates,
        "exclude_author_regex": args.exclude_author_regex,
        "exclude_title_regex": args.exclude_title_regex,
        "exclude_sentence_regex": args.exclude_sentence_regex,
        "include_author_regex": args.include_author_regex,
        "include_title_regex": args.include_title_regex,
        "exclude_locc_regex": args.exclude_locc_regex,
        "include_locc_regex": args.include_locc_regex,
        "exclude_subject_regex": args.exclude_subject_regex,
        "include_subject_regex": args.include_subject_regex,
        "exclude_bookshelf_regex": args.exclude_bookshelf_regex,
        "include_bookshelf_regex": args.include_bookshelf_regex,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "min_alpha_fraction": args.min_alpha_fraction,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "input_prefix": args.input_prefix,
        "pooling": args.pooling,
        "normalized": not args.no_normalize,
        "source_counts": source_counts,
        "sample": sentences[:10],
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
