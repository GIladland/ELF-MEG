#!/usr/bin/env python3
"""Export frozen MEG-to-T5 proposals with target-free word confidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modules.t5_generation_confidence import (
    group_sentencepiece_confidence,
    selected_token_confidence_variants,
)

from cap_generated_text_words import cap_text
from sweep_meg_t5_checkpoint_decoding import load_checkpoint
from train_fmri_minilm_t5_prefix import compact, strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--val-start", type=int, default=2652)
    parser.add_argument("--val-rows", type=int, default=110)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-output-words", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=55)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


@torch.no_grad()
def generate_confident(
    *,
    projector,
    calibrator,
    t5,
    tokenizer,
    semantic: torch.Tensor,
    batch_size: int,
    num_beams: int,
    min_new_tokens: int,
    max_new_tokens: int,
    length_penalty: float,
    max_output_words: int,
    device: torch.device,
) -> tuple[
    list[str],
    list[list[float]],
    dict[str, list[list[float]]],
    list[list[float]],
    int,
]:
    generated: list[str] = []
    word_confidence: list[list[float]] = []
    word_confidence_variants: dict[str, list[list[float]]] = {
        "teacher_probability": [],
        "teacher_margin": [],
        "teacher_inverse_entropy": [],
    }
    token_log_probabilities: list[list[float]] = []
    exact_alignment_rows = 0
    for start in range(0, len(semantic), batch_size):
        values = semantic[start : start + batch_size].to(device)
        if calibrator is not None:
            values = calibrator(values)
        context = projector(values)
        mask = torch.ones(context.shape[:2], dtype=torch.long, device=device)
        generation_args = {
            "encoder_outputs": BaseModelOutput(last_hidden_state=context),
            "attention_mask": mask,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": False,
            "early_stopping": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if min_new_tokens > 0:
            generation_args["min_new_tokens"] = min_new_tokens
        if num_beams > 1:
            generation_args["length_penalty"] = length_penalty
        output = t5.generate(**generation_args)
        beam_indices = getattr(output, "beam_indices", None)
        transition = t5.compute_transition_scores(
            output.sequences,
            output.scores,
            beam_indices,
            normalize_logits=True,
        )
        texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        sequence_tokens = output.sequences[:, 1:]
        teacher_logits = t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=context),
            attention_mask=mask,
            decoder_input_ids=output.sequences[:, :-1],
            use_cache=False,
            return_dict=True,
        ).logits
        variant_scores = selected_token_confidence_variants(
            teacher_logits, sequence_tokens
        )
        for row_index, (text, ids, log_probabilities) in enumerate(
            zip(texts, sequence_tokens, transition)
        ):
            capped = cap_text(" ".join(text.split()), max_output_words)
            expected_words = len(capped.split())
            confidence, exact = group_sentencepiece_confidence(
                ids.detach().cpu().tolist(),
                log_probabilities.detach().cpu().float().tolist(),
                tokenizer=tokenizer,
                expected_words=expected_words,
            )
            generated.append(capped)
            word_confidence.append(confidence)
            for name, scores in variant_scores.items():
                scalar_scores = (
                    scores[row_index].detach().cpu().float().clamp_min(1e-12)
                )
                grouped, _ = group_sentencepiece_confidence(
                    ids.detach().cpu().tolist(),
                    scalar_scores.log().tolist(),
                    tokenizer=tokenizer,
                    expected_words=expected_words,
                )
                word_confidence_variants[name].append(grouped)
            token_log_probabilities.append(
                log_probabilities.detach().cpu().float().tolist()
            )
            exact_alignment_rows += int(exact)
    return (
        generated,
        word_confidence,
        word_confidence_variants,
        token_log_probabilities,
        exact_alignment_rows,
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload, tokenizer, t5, projector, calibrator, semantic_dim = load_checkpoint(
        args.checkpoint, device
    )
    with np.load(args.semantic_npz, allow_pickle=True) as data:
        stop = args.val_start + args.val_rows
        values = np.asarray(data[args.embedding_key][args.val_start:stop], dtype=np.float32)
        targets = strings(data[args.sentence_key][args.val_start:stop])
    if values.shape != (args.val_rows, semantic_dim):
        raise ValueError(f"unexpected validation semantic shape: {values.shape}")
    semantic = F.normalize(torch.as_tensor(values), p=2, dim=-1)
    (
        generated,
        confidence,
        confidence_variants,
        token_log_probabilities,
        exact_rows,
    ) = generate_confident(
        projector=projector,
        calibrator=calibrator,
        t5=t5,
        tokenizer=tokenizer,
        semantic=semantic,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        max_output_words=args.max_output_words,
        device=device,
    )
    quality = compact(generated, targets)
    output = {
        "scientific_scope": (
            f"frozen MEG-to-T5 val{args.val_rows} source proposal and confidence; "
            "protected test26 not loaded"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_best": payload.get("best"),
        "generation_config": {
            "num_beams": args.num_beams,
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "length_penalty": args.length_penalty,
            "max_output_words": args.max_output_words,
        },
        "confidence_contract": {
            "token": "normalized transition probability of the selected deterministic beam",
            "word": "geometric mean of constituent SentencePiece token probabilities",
            "uses_target_text": False,
            "exact_sentencepiece_to_whitespace_alignment_rows": exact_rows,
            "total_rows": len(generated),
            "variants": {
                "teacher_probability": (
                    "geometric mean of selected-token probabilities from a "
                    "teacher-forced pass over the selected beam"
                ),
                "teacher_margin": (
                    "geometric mean of sigmoid(selected logit minus strongest "
                    "alternative logit)"
                ),
                "teacher_inverse_entropy": (
                    "geometric mean of one minus token entropy divided by log vocabulary size"
                ),
            },
        },
        "generation_quality": quality,
        "generated": generated,
        "targets": targets,
        "word_confidence": confidence,
        "word_confidence_variants": confidence_variants,
        "token_log_probabilities": token_log_probabilities,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output_json),
                "generation_quality": quality,
                "confidence_alignment": f"{exact_rows}/{len(generated)}",
                "mean_word_confidence": float(
                    np.mean([value for row in confidence for value in row])
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
