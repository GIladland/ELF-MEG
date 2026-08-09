#!/usr/bin/env python
"""Fine-tune a local causal LM on Sherlock train text and sample short SVA augmentations."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_synthetic_sentence_embeddings import encode_sentences, resolve_device
from scripts.filter_semantic_npz_sentences import keep_sentence, words


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--blocked-npz", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-checkpoint-dir", required=True)
    parser.add_argument("--lm-model-name", default="gpt2-large")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--target-count", type=int, default=200000)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-max-length", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt-mode", choices=["prefix", "mixed", "span"], default="prefix")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--max-attempt-multiplier", type=int, default=20)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip() for x in array.tolist()]


def canonical(sentence: str) -> str:
    return " ".join(token.lower() for token in words(sentence))


def schema(data: np.lib.npyio.NpzFile) -> Any:
    if "schema_json" not in data.files:
        return None
    value = str(data["schema_json"].tolist())
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_blocked(paths: list[str]) -> set[str]:
    blocked = set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocked.update(canonical(sentence) for sentence in strings(data["sentence"]))
    return blocked


def fine_tune_lm(
    *,
    sentences: list[str],
    checkpoint_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if (checkpoint_dir / "config.json").exists():
        print(f"Loading existing fine-tuned LM from {checkpoint_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        [f"{sentence}{tokenizer.eos_token}" for sentence in sentences],
        padding="max_length",
        truncation=True,
        max_length=args.train_max_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    loader = DataLoader(
        TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels),
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.lm_model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(loader)

    for step in range(1, args.train_steps + 1):
        try:
            input_ids, attention_mask, batch_labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids, attention_mask, batch_labels = next(loader_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels).loss
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        if step % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0:
            print(f"lm_train step={step}/{args.train_steps} loss={float(loss):.5f}", flush=True)

    if args.train_steps % args.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    del optimizer, encoded, labels, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokenizer, model


def make_prompts(
    sentences: list[str],
    count: int,
    rng: random.Random,
    prompt_mode: str,
) -> list[tuple[str, int]]:
    prompts = []
    for _ in range(count):
        source_idx = rng.randrange(len(sentences))
        tokens = words(sentences[source_idx])
        prefix_length = rng.choices([1, 2, 3, 4, 5], weights=[1, 3, 4, 3, 1], k=1)[0]
        use_span = prompt_mode == "span" or (prompt_mode == "mixed" and rng.random() < 0.5)
        max_start = max(0, len(tokens) - prefix_length)
        start = rng.randint(0, max_start) if use_span and max_start > 0 else 0
        prompt = " ".join(tokens[start : start + min(prefix_length, len(tokens))])
        prompts.append((prompt, source_idx))
    return prompts


def clean_generated(text: str, args: argparse.Namespace) -> str:
    text = text.replace("\n", " ").strip()
    tokens = words(text)
    if len(tokens) > args.max_words:
        tokens = tokens[: args.max_words]
    if not tokens:
        return ""
    tokens[0] = tokens[0][:1].upper() + tokens[0][1:]
    return " ".join(tokens)


@torch.inference_mode()
def sample_sentences(
    *,
    train_sentences: list[str],
    blocked: set[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(args.seed + 1009)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    if device.type == "cuda":
        model.to(dtype=torch.bfloat16)
    seen = {canonical(sentence) for sentence in train_sentences} | blocked
    accepted: list[str] = []
    rows: list[dict[str, Any]] = []
    attempts = 0
    dropped_invalid = 0
    dropped_duplicate = 0
    max_attempts = args.target_count * args.max_attempt_multiplier

    while len(accepted) < args.target_count and attempts < max_attempts:
        batch_count = min(args.sample_batch_size, max_attempts - attempts)
        prompt_rows = make_prompts(train_sentences, batch_count, rng, args.prompt_mode)
        prompts = [prompt for prompt, _source_idx in prompt_rows]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=4,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for text, (prompt, source_idx) in zip(decoded, prompt_rows):
            attempts += 1
            sentence = clean_generated(text, args)
            key = canonical(sentence)
            if key in seen:
                dropped_duplicate += 1
                continue
            ok, reason = keep_sentence(
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
            seen.add(key)
            accepted.append(sentence)
            rows.append(
                {
                    "augmentation_mode": "sherlock_train_causal_lm",
                    "prompt": prompt,
                    "prompt_source_index": int(source_idx),
                    "prompt_source_sentence": train_sentences[source_idx],
                    "sentence": sentence,
                    "word_count": len(words(sentence)),
                }
            )
            if len(accepted) >= args.target_count:
                break
        if attempts == batch_count or attempts % (args.sample_batch_size * 20) == 0:
            print(
                f"lm_sample attempts={attempts} accepted={len(accepted)} "
                f"invalid={dropped_invalid} duplicate={dropped_duplicate}",
                flush=True,
            )
    if len(accepted) < args.target_count:
        raise RuntimeError(f"Generated only {len(accepted)} valid unique sentences after {attempts} attempts")
    return accepted, rows, {
        "attempts": attempts,
        "dropped_invalid": dropped_invalid,
        "dropped_duplicate": dropped_duplicate,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    train_data = np.load(args.train_npz, allow_pickle=True)
    train_sentences = strings(train_data["sentence"])
    blocked = load_blocked(args.blocked_npz)
    lm_tokenizer, lm_model = fine_tune_lm(
        sentences=train_sentences,
        checkpoint_dir=Path(args.lm_checkpoint_dir),
        args=args,
        device=device,
    )
    generated, rows, generation_stats = sample_sentences(
        train_sentences=train_sentences,
        blocked=blocked,
        tokenizer=lm_tokenizer,
        model=lm_model,
        args=args,
        device=device,
    )
    del lm_model, lm_tokenizer
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
        sentences=generated,
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
        "train_npz": args.train_npz,
        "blocked_npz": args.blocked_npz,
        "source_count": len(train_sentences),
        "generated_count": len(generated),
        "generation_stats": generation_stats,
        "lm_model_name": args.lm_model_name,
        "lm_checkpoint_dir": args.lm_checkpoint_dir,
        "train_steps": args.train_steps,
        "prompt_mode": args.prompt_mode,
        "embedding_model_name": args.embedding_model_name,
        "embedding_shape": list(embeddings.shape),
        "source_schema": schema(train_data),
        "sample": rows[:30],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        input_embeddings=embeddings.astype(np.float32),
        sentence=np.asarray(generated, dtype=object),
        rows=np.asarray(rows, dtype=object),
        split=np.asarray(["train"] * len(generated), dtype=object),
        schema_json=np.asarray(json.dumps(summary)),
    )
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
