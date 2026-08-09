#!/usr/bin/env python
"""Generate grammatical train-derived variants with pretrained T5 span infilling."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import STOP_WORDS, keep_sentence, words


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


INFILL_RE = re.compile(r"<extra_id_0>\s*(.*?)\s*<extra_id_1>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-count", type=int, default=200000)
    parser.add_argument("--t5-model-name", default="t5-small")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--returns-per-mask", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--max-attempt-multiplier", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def canonical(sentence: str) -> str:
    return " ".join(token.lower() for token in words(sentence))


def load_blocked(paths: list[str]) -> set[str]:
    blocked = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in strings(data["sentence"]))
    return blocked


def source_recall(source: str, candidate: str) -> float:
    source_counts = Counter(token.lower() for token in words(source))
    candidate_counts = Counter(token.lower() for token in words(candidate))
    overlap = sum(min(count, candidate_counts.get(token, 0)) for token, count in source_counts.items())
    return float(overlap / max(1, sum(source_counts.values())))


def make_masked_task(source: str, rng: random.Random) -> tuple[str, int, int] | None:
    tokens = words(source)
    content_indices = [
        idx for idx, token in enumerate(tokens)
        if token.lower() not in STOP_WORDS and len(token) > 2
    ]
    if not content_indices:
        return None
    start = rng.choice(content_indices)
    span_length = rng.choices([1, 2, 3], weights=[6, 3, 1], k=1)[0]
    end = min(len(tokens), start + span_length)
    masked = tokens[:start] + ["<extra_id_0>"] + tokens[end:]
    return " ".join(masked), start, end


def parse_replacement(decoded: str) -> list[str]:
    match = INFILL_RE.search(decoded)
    if match:
        return words(match.group(1))
    cleaned = re.sub(r"</?s>|<pad>|<extra_id_\d+>", " ", decoded)
    return words(cleaned)


def build_sentence(source: str, start: int, end: int, replacement: list[str]) -> str:
    tokens = words(source)
    output = tokens[:start] + replacement + tokens[end:]
    if not output:
        return ""
    output[0] = output[0][:1].upper() + output[0][1:]
    return " ".join(output)


@torch.inference_mode()
def generate(
    *,
    sources: list[str],
    blocked: set[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(args.seed + 509)
    seen = {canonical(source) for source in sources} | blocked
    sentences: list[str] = []
    rows: list[dict[str, Any]] = []
    attempts = 0
    task_counter = 0
    dropped_invalid = 0
    dropped_duplicate = 0
    dropped_empty = 0
    max_attempts = args.target_count * args.max_attempt_multiplier

    while len(sentences) < args.target_count and attempts < max_attempts:
        tasks = []
        while len(tasks) < args.generation_batch_size:
            source_idx = task_counter % len(sources)
            task_counter += 1
            task = make_masked_task(sources[source_idx], rng)
            if task is not None:
                tasks.append((source_idx, *task))
        encoded = tokenizer(
            [task[1] for task in tasks],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **encoded,
            do_sample=True,
            num_return_sequences=args.returns_per_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=50,
            repetition_penalty=1.05,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
        for output_idx, text in enumerate(decoded):
            source_idx, masked_text, start, end = tasks[output_idx // args.returns_per_mask]
            source = sources[source_idx]
            replacement = parse_replacement(text)
            attempts += 1
            if not replacement or len(replacement) > 5:
                dropped_empty += 1
                continue
            sentence = build_sentence(source, start, end, replacement)
            key = canonical(sentence)
            if key in seen:
                dropped_duplicate += 1
                continue
            ok, _reason = keep_sentence(
                sentence,
                min_words=args.min_words,
                max_words=args.max_words,
                min_alpha_fraction=0.75,
                min_content_words=2,
                require_verb=True,
                filter_mode="simple_sv_no_coord",
                exclude_re=None,
            )
            if not ok:
                dropped_invalid += 1
                continue
            recall = source_recall(source, sentence)
            if recall < 0.55:
                dropped_invalid += 1
                continue
            seen.add(key)
            sentences.append(sentence)
            rows.append(
                {
                    "augmentation_mode": "t5_span_infill",
                    "augmentation_source_index": int(source_idx),
                    "augmentation_source_sentence": source,
                    "masked_text": masked_text,
                    "replacement": " ".join(replacement),
                    "source_word_recall": recall,
                    "sentence": sentence,
                    "word_count": len(words(sentence)),
                }
            )
            if len(sentences) >= args.target_count:
                break
        if attempts % (args.generation_batch_size * args.returns_per_mask * 10) == 0:
            print(
                f"t5_infill attempts={attempts} accepted={len(sentences)} "
                f"invalid={dropped_invalid} duplicate={dropped_duplicate} empty={dropped_empty}",
                flush=True,
            )
    if len(sentences) < args.target_count:
        raise RuntimeError(f"Generated only {len(sentences)} valid unique variants after {attempts} attempts")
    return sentences, rows, {
        "attempts": attempts,
        "dropped_invalid": dropped_invalid,
        "dropped_duplicate": dropped_duplicate,
        "dropped_empty": dropped_empty,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    source_data = np.load(args.source_npz, allow_pickle=True)
    sources = strings(source_data["sentence"])
    blocked = load_blocked(args.blocked_npz)

    tokenizer = AutoTokenizer.from_pretrained(args.t5_model_name, local_files_only=args.local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.t5_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    if device.type == "cuda":
        model.to(dtype=torch.bfloat16)
    model.eval()
    sentences, rows, stats = generate(
        sources=sources,
        blocked=blocked,
        tokenizer=tokenizer,
        model=model,
        args=args,
        device=device,
    )
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    embedding_tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    embedding_model = AutoModel.from_pretrained(
        args.embedding_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    embedding_model.eval()
    embeddings = encode_sentences(
        sentences=sentences,
        tokenizer=embedding_tokenizer,
        model=embedding_model,
        device=device,
        max_length=128,
        batch_size=args.embedding_batch_size,
        pooling="mean",
        normalize=True,
    )

    summary = {
        "output": args.output,
        "source_npz": args.source_npz,
        "blocked_npz": args.blocked_npz,
        "source_count": len(sources),
        "generated_count": len(sentences),
        "generation_stats": stats,
        "t5_model_name": args.t5_model_name,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "sample": rows[:30],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(sentences, dtype=object),
        rows=np.asarray(rows, dtype=object),
        split=np.asarray(["train"] * len(sentences), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
