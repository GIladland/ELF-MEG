"""Target-free confidence aggregation for T5 sentence proposals."""

from __future__ import annotations

import math

import torch


def group_sentencepiece_confidence(
    token_ids: list[int],
    token_log_probabilities: list[float],
    *,
    tokenizer,
    expected_words: int,
) -> tuple[list[float], bool]:
    """Aggregate generated SentencePiece probabilities to whitespace words."""
    groups: list[list[float]] = []
    special_ids = set(tokenizer.all_special_ids)
    for token_id, log_probability in zip(token_ids, token_log_probabilities):
        if token_id == tokenizer.eos_token_id:
            break
        if token_id in special_ids:
            continue
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        starts_word = piece.startswith("▁")
        if starts_word or not groups:
            groups.append([])
        groups[-1].append(float(log_probability))
    confidence = [
        float(math.exp(sum(values) / max(1, len(values))))
        for values in groups
    ]
    exact = len(confidence) == expected_words
    if len(confidence) > expected_words:
        confidence = confidence[:expected_words]
    elif len(confidence) < expected_words:
        fallback = min(confidence) if confidence else 0.0
        confidence.extend([fallback] * (expected_words - len(confidence)))
    return confidence, exact


def selected_token_confidence_variants(
    logits: torch.Tensor,
    selected_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute aligned target-free confidence signals for selected tokens."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, length, vocabulary]")
    if selected_token_ids.shape != logits.shape[:2]:
        raise ValueError("selected_token_ids must align with logits")
    if logits.shape[-1] < 2:
        raise ValueError("confidence variants require at least two vocabulary items")
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    selected_log_probability = log_probabilities.gather(
        -1, selected_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    selected_logits = logits.float().gather(
        -1, selected_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    top_logits, top_ids = logits.float().topk(2, dim=-1)
    best_alternative = torch.where(
        top_ids[..., 0] == selected_token_ids,
        top_logits[..., 1],
        top_logits[..., 0],
    )
    margin = torch.sigmoid(selected_logits - best_alternative)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    inverse_entropy = (1.0 - entropy / math.log(logits.shape[-1])).clamp(0.0, 1.0)
    return {
        "teacher_probability": selected_log_probability.exp(),
        "teacher_margin": margin,
        "teacher_inverse_entropy": inverse_entropy,
    }
