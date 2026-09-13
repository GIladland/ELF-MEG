"""Simulated D'Ascoli sentence sources for confidence-weighted ELF refinement.

The simulator is deliberately explicit about its target-derived nature.  It is
used only to answer controlled ceiling/sensitivity questions before real
D'Ascoli predictions are available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


# The Apple contract is ten whitespace-delimited stimulus words.  Two audited
# rows contain the source transcription ``that''s``; treating apostrophe
# punctuation as a split would incorrectly turn those rows into eleven words.
WORD_RE = re.compile(r"\S+")
LEXICAL_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass
class SimulatedDAscoliSentences:
    sentences: list[str]
    word_confidence: torch.Tensor
    correct_word_mask: torch.Tensor
    row_accuracy: torch.Tensor


def external_sentence_source(
    payload: Mapping,
    *,
    expected_targets: Sequence[str],
    expected_words: int,
    default_confidence: float,
    confidence_field: str = "word_confidence",
) -> SimulatedDAscoliSentences:
    """Load aligned external sentence proposals using no target-derived confidence.

    ``confidence_field`` selects an optional row-major list. Dotted paths are
    supported, for example ``word_confidence_variants.teacher_margin``. When
    it is absent, every proposed word receives ``default_confidence``. Targets
    are used only for an alignment assertion and diagnostic correctness masks,
    not to construct or score the proposal during diffusion.
    """
    if expected_words <= 0:
        raise ValueError("expected_words must be positive")
    if not 0.0 <= float(default_confidence) <= 1.0:
        raise ValueError("default_confidence must be in [0, 1]")
    if "generated" not in payload:
        raise ValueError("external sentence-source JSON must contain 'generated'")
    generated = [str(value) for value in payload["generated"]]
    targets = [str(value) for value in expected_targets]
    if len(generated) != len(targets):
        raise ValueError(
            f"external source has {len(generated)} rows, expected {len(targets)}"
        )
    if "targets" in payload and [str(value) for value in payload["targets"]] != targets:
        raise ValueError("external sentence-source targets do not align to validation rows")

    provided_confidence = payload
    for part in confidence_field.split("."):
        if not isinstance(provided_confidence, Mapping) or part not in provided_confidence:
            provided_confidence = None
            break
        provided_confidence = provided_confidence[part]
    if provided_confidence is None and confidence_field != "word_confidence":
        raise ValueError(
            f"requested external confidence field {confidence_field!r} is absent"
        )
    if provided_confidence is not None and len(provided_confidence) != len(generated):
        raise ValueError(
            f"external {confidence_field} row count does not match generated"
        )
    # A proposal can contain a whitespace-separated punctuation token (for
    # example ``double double - i ...``) while still satisfying a ten lexical
    # word cap. Preserve it and the final lexical word instead of silently
    # truncating the proposal by whitespace count.
    capped_sentences: list[str] = []
    source_groups: list[list[str]] = []
    for row_index, sentence in enumerate(generated):
        lexical = list(LEXICAL_WORD_RE.finditer(sentence))
        if len(lexical) > expected_words:
            sentence = sentence[: lexical[expected_words - 1].end()].strip()
        groups = [match.group(0) for match in WORD_RE.finditer(sentence)]
        if not groups:
            raise ValueError(f"external source row {row_index} contains no words")
        capped_sentences.append(sentence.strip())
        source_groups.append(groups)
    confidence_width = max(expected_words, max(map(len, source_groups)))

    sentences: list[str] = []
    confidence = torch.zeros((len(generated), confidence_width), dtype=torch.float32)
    correct_mask = torch.zeros((len(generated), confidence_width), dtype=torch.bool)
    row_accuracy = torch.zeros((len(generated),), dtype=torch.float32)
    for row_index, (sentence, raw_words, target) in enumerate(
        zip(capped_sentences, source_groups, targets)
    ):
        sentences.append(sentence)
        count = len(raw_words)
        if provided_confidence is None:
            row_confidence = [float(default_confidence)] * count
        else:
            row_confidence = [float(value) for value in provided_confidence[row_index]]
            if len(row_confidence) < count:
                raise ValueError(
                    f"external confidence row {row_index} has {len(row_confidence)} "
                    f"values for {count} words"
                )
            row_confidence = row_confidence[:count]
            if any(not 0.0 <= value <= 1.0 for value in row_confidence):
                raise ValueError(f"external confidence row {row_index} is outside [0, 1]")
        confidence[row_index, :count] = torch.tensor(row_confidence)
        source_words = normalized_words(sentences[-1])
        target_words = normalized_words(target)[:expected_words]
        for position, source_word in enumerate(source_words):
            if position < len(target_words):
                correct_mask[row_index, position] = source_word == target_words[position]
        row_accuracy[row_index] = correct_mask[row_index].float().sum() / expected_words
    return SimulatedDAscoliSentences(
        sentences=sentences,
        word_confidence=confidence,
        correct_word_mask=correct_mask,
        row_accuracy=row_accuracy,
    )


def confidence_weighted_source_latents(
    source_latents: torch.Tensor,
    confidence: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Blend a sentence proposal with the diffusion prior per token."""
    if source_latents.shape != noise.shape:
        raise ValueError(
            f"source/noise shapes differ: {tuple(source_latents.shape)} vs {tuple(noise.shape)}"
        )
    if confidence.ndim == 2:
        confidence = confidence.unsqueeze(-1)
    if tuple(confidence.shape) != (*source_latents.shape[:2], 1):
        raise ValueError(
            "confidence must be [batch, length] or [batch, length, 1], got "
            f"{tuple(confidence.shape)}"
        )
    confidence = confidence.to(
        device=source_latents.device, dtype=source_latents.dtype
    ).clamp(0.0, 1.0)
    return confidence * source_latents + (1.0 - confidence) * noise


def apply_source_token_copy(
    generated_ids: torch.Tensor,
    source_ids: torch.Tensor,
    source_confidence: torch.Tensor,
    source_attention_mask: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-copy high-confidence proposal tokens after diffusion decoding."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("copy threshold must be in [0, 1]")
    if generated_ids.shape != source_ids.shape:
        raise ValueError("generated_ids and source_ids shapes do not match")
    if source_confidence.ndim == 3 and source_confidence.shape[-1] == 1:
        source_confidence = source_confidence.squeeze(-1)
    if source_confidence.shape != generated_ids.shape:
        raise ValueError("source_confidence must align with token ids")
    if source_attention_mask.shape != generated_ids.shape:
        raise ValueError("source_attention_mask must align with token ids")
    copy_mask = source_attention_mask.bool() & (source_confidence >= float(threshold))
    return torch.where(copy_mask, source_ids, generated_ids), copy_mask


def preserve_editable_source_latents(
    flow_latents: torch.Tensor,
    source_latents: torch.Tensor,
    source_confidence: torch.Tensor,
    source_attention_mask: torch.Tensor,
    *,
    copy_threshold: float,
    preservation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend source geometry into positions left editable by hard copying.

    High-confidence positions are handled exactly by ``apply_source_token_copy``.
    This operation supplies a continuous preservation control for the remaining
    positions: zero keeps the flow result, while one decodes the source latent.
    Padding and hard-copied positions are unchanged here.
    """
    if flow_latents.shape != source_latents.shape:
        raise ValueError("flow_latents and source_latents shapes do not match")
    if source_confidence.ndim == 3 and source_confidence.shape[-1] == 1:
        source_confidence = source_confidence.squeeze(-1)
    if source_confidence.shape != flow_latents.shape[:2]:
        raise ValueError("source_confidence must align with latent positions")
    if source_attention_mask.shape != flow_latents.shape[:2]:
        raise ValueError("source_attention_mask must align with latent positions")
    if not 0.0 <= float(copy_threshold) <= 1.0:
        raise ValueError("copy_threshold must be in [0, 1]")
    if not 0.0 <= float(preservation) <= 1.0:
        raise ValueError("preservation must be in [0, 1]")
    editable_mask = source_attention_mask.bool() & (
        source_confidence < float(copy_threshold)
    )
    blended = (
        float(preservation) * source_latents
        + (1.0 - float(preservation)) * flow_latents
    )
    return torch.where(editable_mask.unsqueeze(-1), blended, flow_latents), editable_mask


def transform_token_confidence(
    token_confidence: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    global_trust: float = 1.0,
    confidence_floor: float = 0.0,
) -> torch.Tensor:
    """Calibrate source trust while leaving padding at exactly zero.

    A positive floor tests whether retaining the proposal's on-manifold token
    geometry is preferable to replacing low-confidence words with Gaussian
    noise.  ``confidence_floor=1`` is the unweighted full-sentence endpoint.
    """
    if token_confidence.shape != attention_mask.shape:
        raise ValueError("token_confidence and attention_mask shapes do not match")
    if not 0.0 <= float(global_trust) <= 1.0:
        raise ValueError("global_trust must be in [0, 1]")
    if not 0.0 <= float(confidence_floor) <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    trusted = (token_confidence * float(global_trust)).clamp(0.0, 1.0)
    transformed = float(confidence_floor) + (
        1.0 - float(confidence_floor)
    ) * trusted
    return transformed * attention_mask.to(
        device=transformed.device, dtype=transformed.dtype
    )


def normalized_words(sentence: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(sentence)]


def build_position_vocabulary(
    sentences: Sequence[str],
    *,
    expected_words: int,
) -> tuple[list[list[str]], list[str]]:
    rows = [normalized_words(sentence) for sentence in sentences]
    bad = [(index, len(words)) for index, words in enumerate(rows) if len(words) != expected_words]
    if bad:
        preview = ", ".join(f"row {index}: {count}" for index, count in bad[:8])
        raise ValueError(f"Expected {expected_words} words per sentence; {preview}.")
    position_vocabulary = [
        sorted({words[position] for words in rows})
        for position in range(expected_words)
    ]
    global_vocabulary = sorted({word for words in rows for word in words})
    return position_vocabulary, global_vocabulary


def _replacement_word(
    *,
    original: str,
    candidates: Sequence[str],
    fallback: Sequence[str],
    rng: np.random.Generator,
) -> str:
    alternatives = [word for word in candidates if word != original]
    if not alternatives:
        alternatives = [word for word in fallback if word != original]
    if not alternatives:
        raise ValueError("Cannot simulate a wrong D'Ascoli word from a one-word vocabulary.")
    return alternatives[int(rng.integers(0, len(alternatives)))]


def simulate_dascoli_sentences(
    sentences: Sequence[str],
    *,
    position_vocabulary: Sequence[Sequence[str]],
    global_vocabulary: Sequence[str],
    accuracies: Sequence[float],
    seed: int,
    correct_confidence_beta: tuple[float, float] = (8.0, 2.0),
    wrong_confidence_beta: tuple[float, float] = (2.0, 8.0),
) -> SimulatedDAscoliSentences:
    """Create full ordered hypotheses with calibrated-but-overlapping confidences."""
    if not accuracies:
        raise ValueError("At least one simulated D'Ascoli accuracy is required.")
    if any(not 0.0 <= float(accuracy) <= 1.0 for accuracy in accuracies):
        raise ValueError(f"D'Ascoli accuracies must be in [0, 1], got {accuracies}.")
    expected_words = len(position_vocabulary)
    if expected_words <= 0:
        raise ValueError("Position vocabulary must not be empty.")
    rows = [normalized_words(sentence) for sentence in sentences]
    bad = [(index, len(words)) for index, words in enumerate(rows) if len(words) != expected_words]
    if bad:
        preview = ", ".join(f"row {index}: {count}" for index, count in bad[:8])
        raise ValueError(f"Expected {expected_words} words per sentence; {preview}.")

    accuracy_rng = np.random.default_rng(seed)
    correctness_rng = np.random.default_rng(seed + 1)
    replacement_rng = np.random.default_rng(seed + 2)
    confidence_rng = np.random.default_rng(seed + 3)
    output_sentences: list[str] = []
    confidence = torch.zeros((len(rows), expected_words), dtype=torch.float32)
    correct_mask = torch.zeros((len(rows), expected_words), dtype=torch.bool)
    row_accuracy = torch.zeros((len(rows),), dtype=torch.float32)
    correct_alpha, correct_beta = correct_confidence_beta
    wrong_alpha, wrong_beta = wrong_confidence_beta
    if len(accuracies) == 1:
        assigned_accuracies = np.full((len(rows),), float(accuracies[0]))
    else:
        assigned_accuracies = accuracy_rng.choice(
            np.asarray(accuracies, dtype=np.float64), size=len(rows), replace=True
        )
    correctness_draws = correctness_rng.random((len(rows), expected_words))
    correct_scores = confidence_rng.beta(
        correct_alpha, correct_beta, size=(len(rows), expected_words)
    )
    wrong_scores = confidence_rng.beta(
        wrong_alpha, wrong_beta, size=(len(rows), expected_words)
    )

    for row_index, target_words in enumerate(rows):
        accuracy = float(assigned_accuracies[row_index])
        row_accuracy[row_index] = accuracy
        hypothesis: list[str] = []
        for position, target_word in enumerate(target_words):
            is_correct = bool(correctness_draws[row_index, position] < accuracy)
            if is_correct:
                predicted_word = target_word
                score = float(correct_scores[row_index, position])
            else:
                predicted_word = _replacement_word(
                    original=target_word,
                    candidates=position_vocabulary[position],
                    fallback=global_vocabulary,
                    rng=replacement_rng,
                )
                score = float(wrong_scores[row_index, position])
            hypothesis.append(predicted_word)
            confidence[row_index, position] = score
            correct_mask[row_index, position] = predicted_word == target_word
        output_sentences.append(" ".join(hypothesis))

    return SimulatedDAscoliSentences(
        sentences=output_sentences,
        word_confidence=confidence,
        correct_word_mask=correct_mask,
        row_accuracy=row_accuracy,
    )


def align_word_confidence_to_tokens(
    *,
    sentences: Sequence[str],
    word_confidence: torch.Tensor,
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast word confidence to subwords and mean confidence to EOS.

    Padding remains at zero confidence.  A tokenizer special token that is
    inside the attention mask (T5's EOS in this project) receives the row mean:
    a complete D'Ascoli sentence provides a length/termination proposal even
    though there is no separately classified EOS word.
    """
    if word_confidence.ndim != 2:
        raise ValueError("word_confidence must have shape [rows, words]")
    if offset_mapping.shape[:2] != attention_mask.shape:
        raise ValueError("offset_mapping and attention_mask shapes do not match")
    if offset_mapping.shape[0] != len(sentences) or word_confidence.shape[0] != len(sentences):
        raise ValueError("Sentence, confidence, and token batches do not match")

    token_confidence = torch.zeros_like(attention_mask, dtype=torch.float32)
    for row_index, sentence in enumerate(sentences):
        word_spans = [match.span() for match in WORD_RE.finditer(sentence)]
        if len(word_spans) > word_confidence.shape[1]:
            raise ValueError(
                f"Row {row_index} has {len(word_spans)} words but confidence has only "
                f"{word_confidence.shape[1]} positions."
            )
        active_confidence = word_confidence[row_index, : len(word_spans)]
        for token_index, (start, end) in enumerate(offset_mapping[row_index].tolist()):
            if not bool(attention_mask[row_index, token_index]):
                continue
            if end <= start:
                token_confidence[row_index, token_index] = active_confidence.mean()
                continue
            overlaps = [
                word_index
                for word_index, (word_start, word_end) in enumerate(word_spans)
                if start < word_end and end > word_start
            ]
            if overlaps:
                token_confidence[row_index, token_index] = word_confidence[
                    row_index, overlaps
                ].max()
    return token_confidence


def force_vocabulary_token_confidence(
    *,
    sentences: Sequence[str],
    token_confidence: torch.Tensor,
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    preserved_words: set[str],
    forced_confidence: float = 1.0,
) -> torch.Tensor:
    """Force copy confidence for tokens overlapping a target-free word set."""
    if token_confidence.shape != attention_mask.shape:
        raise ValueError("token_confidence and attention_mask shapes do not match")
    if offset_mapping.shape[:2] != attention_mask.shape:
        raise ValueError("offset_mapping and attention_mask shapes do not match")
    if token_confidence.shape[0] != len(sentences):
        raise ValueError("sentence and token-confidence row counts differ")
    if not 0.0 <= float(forced_confidence) <= 1.0:
        raise ValueError("forced_confidence must be in [0, 1]")
    vocabulary = {word.lower() for word in preserved_words}
    result = token_confidence.clone()
    for row_index, sentence in enumerate(sentences):
        preserved_spans = [
            match.span()
            for match in LEXICAL_WORD_RE.finditer(sentence)
            if match.group(0).lower() in vocabulary
        ]
        for token_index, (start, end) in enumerate(offset_mapping[row_index].tolist()):
            if end <= start or not bool(attention_mask[row_index, token_index]):
                continue
            if any(start < span_end and end > span_start for span_start, span_end in preserved_spans):
                result[row_index, token_index] = float(forced_confidence)
    return result
