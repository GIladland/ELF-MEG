#!/usr/bin/env python
"""Train a frozen-T5 prefix decoder from oracle then brain-predicted semantics.

This is a low-complexity fluent control for the diffusion pipeline.  A small
projector maps exact semantic vectors to a short sequence of T5 encoder states
while T5-small remains frozen.  After oracle pretraining, the same projector is
noise-adapted using brain predictions.  The original fMRI/MiniLM path remains
the default; ``--precomputed-semantic-npz`` enables aligned predicted/exact
archives such as MEG/ADA without loading raw brain data or a brain checkpoint.
Fixed derangements quantify brain dependence and protected test rows are never
indexed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

from meg_context_overfit import _CONTENT_WORD_STOPWORDS, _WORD_RE, word_overlap_metrics
from modules.lora import LoRALinear
from modules.fmri2sem_bridge import load_mri2sem_model
from probe_fmri_supervised_content_head import ContentHead, encode_features
from train_fmri_temporal_residual import load_semantic_mapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", default="")
    parser.add_argument("--text-npz", default="")
    parser.add_argument(
        "--precomputed-semantic-npz",
        default="",
        help=(
            "Aligned train-then-validation archive containing predicted brain "
            "semantics, exact semantic targets, sentences, and grouping metadata. "
            "This bypasses raw-fMRI feature extraction."
        ),
    )
    parser.add_argument("--precomputed-input-key", default="input_embeddings")
    parser.add_argument("--precomputed-target-key", default="source_vectors")
    parser.add_argument("--precomputed-sentence-key", default="sentence")
    parser.add_argument("--precomputed-group-key", default="story")
    parser.add_argument(
        "--oracle-augmentation-npz",
        default="",
        help=(
            "Optional exact-semantic/text corpus used only to train the oracle "
            "prefix projector. Brain-stage inputs and labels still come only "
            "from the train portion of --precomputed-semantic-npz. This permits "
            "a matched known-text/x16 decoder-capacity comparison without "
            "exposing held-out predicted brain vectors."
        ),
    )
    parser.add_argument("--oracle-augmentation-input-key", default="input_embeddings")
    parser.add_argument("--oracle-augmentation-sentence-key", default="sentence")
    parser.add_argument(
        "--oracle-semantic-npz",
        default="",
        help=(
            "Optional row-aligned 384-D oracle semantic archive. Labels still "
            "come from --text-npz, enabling full-window semantics to exact-text training."
        ),
    )
    parser.add_argument("--mri-checkpoint", default="")
    parser.add_argument(
        "--oracle-init-checkpoint",
        default="",
        help=(
            "Optional oracle-stage checkpoint containing projector_state_dict. "
            "Use with --oracle-epochs 0 to share one frozen oracle prefix across "
            "raw, adapter, LoRA, and joint-training comparisons."
        ),
    )
    parser.add_argument(
        "--oof-semantic-npz",
        help=(
            "Optional leakage-safe oracle+story-OOF delayed MiniLM archive. "
            "When set, brain-stage train inputs are the 11,725 OOF rows and "
            "validation inputs are its separate 266-row prediction ensemble; "
            "the in-sample MRI forward pass is bypassed."
        ),
    )
    parser.add_argument(
        "--oof-delay-mode",
        choices=("mean", "ordered"),
        default="mean",
        help=(
            "Reduce the four OOF delay blocks to their mean before calibration, "
            "or preserve all 1536 ordered values for a learned residual mapper."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument(
        "--projector-kind", choices=("mlp", "delay_transformer"), default="mlp"
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-target-tokens", type=int, default=32)
    parser.add_argument("--max-generation-tokens", type=int, default=16)
    parser.add_argument("--min-generation-tokens", type=int, default=0)
    parser.add_argument(
        "--max-output-words", type=int, default=10,
        help="Deterministic task-contract cap; target phrases contain exactly ten words.",
    )
    parser.add_argument("--oracle-epochs", type=int, default=20)
    parser.add_argument("--brain-epochs", type=int, default=30)
    parser.add_argument("--brain-semantic-warmup-epochs", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--oracle-lr", type=float, default=3e-4)
    parser.add_argument("--brain-lr", type=float, default=3e-5)
    parser.add_argument(
        "--brain-unfreeze-projector",
        action="store_true",
        help=(
            "During the brain stage, jointly fine-tune the oracle prefix projector "
            "beside the calibrator. Oracle-preservation loss remains active."
        ),
    )
    parser.add_argument(
        "--brain-projector-lr",
        type=float,
        default=3e-6,
        help="Learning rate for the jointly unfrozen prefix projector.",
    )
    parser.add_argument("--brain-semantic-warmup-lr", type=float, default=3e-4)
    parser.add_argument(
        "--brain-calibrator-hidden-dim", type=int, default=0,
        help="If positive, freeze the oracle prefix projector and train only an identity-initialized brain calibrator.",
    )
    parser.add_argument(
        "--brain-calibrator-kind",
        choices=("residual_mlp", "delay_weighted", "identity_residual"),
        default="residual_mlp",
        help=(
            "Residual MLP calibration, or a low-capacity global mixture of the "
            "four ordered MiniLM delays followed by a diagonal affine transform. "
            "identity_residual preserves a same-dimensional semantic space such "
            "as 1536-D ADA."
        ),
    )
    parser.add_argument("--brain-semantic-alignment-weight", type=float, default=1.0)
    parser.add_argument("--brain-semantic-contrastive-weight", type=float, default=0.1)
    parser.add_argument("--content-head-checkpoint", default="")
    parser.add_argument("--content-bias-topk", type=int, default=5)
    parser.add_argument("--content-bias-strength", type=float, default=0.0)
    parser.add_argument(
        "--content-bias-once", action="store_true",
        help="Remove a lexical token's generation bias after that token has appeared.",
    )
    parser.add_argument("--content-prior-subtraction", type=float, default=0.5)
    parser.add_argument(
        "--content-idf-power",
        type=float,
        default=0.0,
        help=(
            "Train-only IDF exponent for positional content-token CE. Zero "
            "recovers uniform content weighting."
        ),
    )
    parser.add_argument(
        "--content-idf-max-multiplier",
        type=float,
        default=3.0,
        help="Cap on the train-normalized IDF multiplier per content word.",
    )
    parser.add_argument("--content-keyword-context", action="store_true")
    parser.add_argument(
        "--content-keyword-train-context", action="store_true",
        help=(
            "Also expose the frozen OOF lexical keywords during brain-stage teacher forcing. "
            "Generation-only keyword context otherwise creates a train/inference mismatch."
        ),
    )
    parser.add_argument(
        "--oracle-keyword-train-context", action="store_true",
        help=(
            "Train the oracle prefix beside keywords predicted from the corresponding "
            "story-OOF brain vector, matching the later brain-stage interface without "
            "exposing held-out target words."
        ),
    )
    parser.add_argument("--content-keyword-strength", type=float, default=1.0)
    parser.add_argument(
        "--content-keyword-template",
        choices=("labeled", "raw", "summarize", "sentence"),
        default="labeled",
    )
    parser.add_argument(
        "--content-token-ce-weight", type=float, default=1.0,
        help=(
            "Relative positional CE weight for tokenizer pieces overlapping a target content "
            "word. This preserves order and is not a set-coverage objective."
        ),
    )
    parser.add_argument(
        "--brain-pairing-margin-weight", type=float, default=0.0,
        help=(
            "Weight for matched-vs-rolled brain sequence-NLL ranking during the brain stage. "
            "This directly penalizes a decoder that ignores its brain condition."
        ),
    )
    parser.add_argument("--brain-pairing-margin", type=float, default=0.25)
    parser.add_argument("--t5-lora-rank", type=int, default=0)
    parser.add_argument("--t5-lora-alpha", type=float, default=8.0)
    parser.add_argument("--t5-lora-last-n-blocks", type=int, default=2)
    parser.add_argument("--t5-lora-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--oracle-preservation-weight", type=float, default=0.25)
    parser.add_argument("--prefix-alignment-weight", type=float, default=0.1)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--derangements", type=int, default=3)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mapper-hidden-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modality", default="fmri")
    parser.add_argument("--protected-split-name", default="test107")
    parser.add_argument(
        "--selection-objective",
        choices=("content_conditional", "word_content_mean"),
        default="content_conditional",
    )
    return parser.parse_args()


def strings(values: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in values.tolist()]


class MiniLMT5Prefix(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, prefix_length: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.prefix_length = int(prefix_length)
        self.model_dim = int(model_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prefix_length * model_dim),
        )
        nn.init.normal_(self.net[-1].weight, std=0.002)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.net(semantic).reshape(len(semantic), self.prefix_length, self.model_dim)


class OrderedDelayT5Prefix(nn.Module):
    """Preserve the four delayed MiniLM blocks as ordered conditioning tokens."""

    def __init__(
        self, *, prefix_length: int, model_dim: int, dropout: float,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if prefix_length % 4:
            raise ValueError("OrderedDelayT5Prefix requires prefix length divisible by four.")
        self.prefix_length = int(prefix_length)
        self.model_dim = int(model_dim)
        self.slots_per_delay = prefix_length // 4
        self.position = nn.Parameter(torch.zeros(1, 4, 384))
        nn.init.normal_(self.position, std=0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384, nhead=6, dim_feedforward=1024,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, self.slots_per_delay * model_dim)
        )
        nn.init.normal_(self.output[-1].weight, std=0.002)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        delayed = semantic.reshape(len(semantic), 4, 384) + self.position
        encoded = self.encoder(delayed)
        return self.output(encoded).reshape(len(semantic), self.prefix_length, self.model_dim)


class ResidualBrainCalibrator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim not in (384, 1536):
            raise ValueError(f"ResidualBrainCalibrator expects 384 or 1536 inputs, got {input_dim}.")
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 384),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        base = (
            semantic
            if self.input_dim == 384
            else semantic.reshape(len(semantic), 4, 384).mean(dim=1)
        )
        return F.normalize(base + self.net(semantic), p=2, dim=-1)


class IdentityResidualCalibrator(nn.Module):
    """Same-space identity-started residual calibration for generic semantics."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.ndim != 2 or semantic.shape[1] != self.input_dim:
            raise ValueError(
                f"IdentityResidualCalibrator expects (*, {self.input_dim}), "
                f"got {tuple(semantic.shape)}."
            )
        return F.normalize(semantic + self.net(semantic), p=2, dim=-1)


class DelayWeightedCalibrator(nn.Module):
    """Low-capacity ordered-delay calibration initialized as an exact mean."""

    def __init__(self) -> None:
        super().__init__()
        self.delay_logits = nn.Parameter(torch.zeros(4))
        self.log_scale = nn.Parameter(torch.zeros(384))
        self.bias = nn.Parameter(torch.zeros(384))

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.ndim != 2 or semantic.shape[1] != 1536:
            raise ValueError(
                "DelayWeightedCalibrator expects ordered 4 x 384 inputs, "
                f"got {tuple(semantic.shape)}."
            )
        delayed = semantic.reshape(len(semantic), 4, 384)
        weights = self.delay_logits.softmax(dim=0).reshape(1, 4, 1)
        mixed = (delayed * weights).sum(dim=1)
        calibrated = mixed * self.log_scale.clamp(-2.0, 2.0).exp() + self.bias
        return F.normalize(calibrated, p=2, dim=-1)


class StaticBatchLogitBias(LogitsProcessor):
    """Apply one fixed vocabulary bias per source row, including beam expansion."""

    def __init__(self, bias: torch.Tensor) -> None:
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] % self.bias.shape[0] != 0:
            raise ValueError(
                f"Generation rows {scores.shape[0]} are not divisible by bias rows {self.bias.shape[0]}."
            )
        repeats = scores.shape[0] // self.bias.shape[0]
        return scores + self.bias.repeat_interleave(repeats, dim=0).to(scores)


class UnseenTokenBatchLogitBias(LogitsProcessor):
    """Reward each row-specific lexical token only until it has been generated."""

    def __init__(self, bias: torch.Tensor) -> None:
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] % self.bias.shape[0] != 0:
            raise ValueError(
                f"Generation rows {scores.shape[0]} are not divisible by bias rows {self.bias.shape[0]}."
            )
        repeats = scores.shape[0] // self.bias.shape[0]
        expanded = self.bias.repeat_interleave(repeats, dim=0).to(scores)
        valid_ids = input_ids.clamp(min=0, max=scores.shape[1] - 1)
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, valid_ids, True)
        return scores + expanded.masked_fill(seen, 0.0)


def inject_t5_cross_attention_lora(
    t5: T5ForConditionalGeneration,
    *,
    rank: int,
    alpha: float,
    last_n_blocks: int,
) -> list[str]:
    if rank <= 0:
        return []
    blocks = t5.decoder.block
    start = max(0, len(blocks) - max(1, last_n_blocks))
    injected: list[str] = []
    for index in range(start, len(blocks)):
        attention = blocks[index].layer[1].EncDecAttention
        for name in ("q", "k", "v", "o"):
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected T5 cross-attention {name} to be linear.")
            setattr(attention, name, LoRALinear(base, rank=rank, alpha=alpha, dropout=0.0))
            injected.append(f"decoder.block.{index}.layer.1.EncDecAttention.{name}")
    return injected


@torch.no_grad()
def content_ranked_indices(
    head: nn.Module,
    semantic: torch.Tensor,
    prior_logit: torch.Tensor,
    *,
    topk: int,
    prior_subtraction: float,
    max_prior_probability: float = 1.0,
) -> torch.Tensor:
    adjusted = content_adjusted_logits(
        head, semantic, prior_logit,
        prior_subtraction=prior_subtraction,
        max_prior_probability=max_prior_probability,
    )
    eligible_count = int(torch.isfinite(adjusted[0]).sum().item())
    use_k = min(topk, eligible_count)
    return adjusted.topk(k=use_k, dim=1).indices


@torch.no_grad()
def content_adjusted_logits(
    head: nn.Module,
    semantic: torch.Tensor,
    prior_logit: torch.Tensor,
    *,
    prior_subtraction: float,
    max_prior_probability: float = 1.0,
) -> torch.Tensor:
    """Return train-prior-adjusted lexical evidence for each source row."""
    features = (
        semantic
        if semantic.shape[1] == 384
        else semantic.reshape(len(semantic), 4, 384).mean(dim=1)
    )
    logits = head(F.normalize(features.float(), p=2, dim=-1))
    adjusted = logits - prior_subtraction * prior_logit.to(logits)
    if not 0.0 < max_prior_probability <= 1.0:
        raise ValueError("max_prior_probability must be in (0, 1].")
    if max_prior_probability < 1.0:
        eligible = prior_logit.to(logits).sigmoid() <= max_prior_probability
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            raise ValueError(
                "max_prior_probability excludes the entire content vocabulary."
            )
        adjusted = adjusted.masked_fill(~eligible.unsqueeze(0), -torch.inf)
    else:
        eligible_count = adjusted.shape[1]
    if eligible_count == 0:
        raise ValueError("No eligible lexical words remain after prior filtering.")
    return adjusted


@torch.no_grad()
def content_token_bias(
    head: nn.Module,
    semantic: torch.Tensor,
    prior_logit: torch.Tensor,
    vocabulary_token_ids: list[list[int]],
    *,
    topk: int,
    strength: float,
    prior_subtraction: float,
    model_vocab_size: int,
    max_prior_probability: float = 1.0,
    weighting: str = "ranked",
    temperature: float = 1.0,
    max_weight_multiplier: float = 2.5,
) -> torch.Tensor:
    if weighting not in ("ranked", "softmax"):
        raise ValueError(f"Unsupported content bias weighting {weighting!r}.")
    if temperature <= 0.0:
        raise ValueError("content bias temperature must be positive.")
    adjusted = content_adjusted_logits(
        head, semantic, prior_logit,
        prior_subtraction=prior_subtraction,
        max_prior_probability=max_prior_probability,
    )
    eligible_count = int(torch.isfinite(adjusted[0]).sum().item())
    use_k = min(topk, eligible_count)
    top_values, top = adjusted.topk(k=use_k, dim=1)
    use_k = top.shape[1]
    logits_device = semantic.device
    if weighting == "ranked":
        row_weights = torch.linspace(
            1.0, 0.5, steps=use_k, device=logits_device
        ).expand(len(semantic), -1)
    else:
        row_weights = F.softmax(top_values / temperature, dim=1) * use_k
        row_weights = row_weights.clamp(max=max_weight_multiplier)
    row_weights = row_weights * strength
    bias = torch.zeros((len(semantic), model_vocab_size), device=logits_device)
    for row in range(len(semantic)):
        for rank, word_index in enumerate(top[row].tolist()):
            for token_id in vocabulary_token_ids[word_index]:
                if 0 <= token_id < model_vocab_size:
                    bias[row, token_id] = torch.maximum(
                        bias[row, token_id], row_weights[row, rank]
                    )
    return bias


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(size)
    for _ in range(10_000):
        result = rng.permutation(size)
        if np.all(result != base):
            return result
    return np.roll(base, 1)


def compact(generated: list[str], targets: list[str]) -> dict[str, float | int]:
    summary = word_overlap_metrics(generated, targets)["summary"]
    return {
        key: summary[key]
        for key in (
            "content_words_overlap", "content_words_overlap_precision",
            "content_words_overlap_recall", "words_overlap", "words_overlap_precision",
            "words_overlap_recall", "word_error_rate", "word_error_insertions",
            "word_error_deletions", "word_error_substitutions",
        )
    }


@torch.no_grad()
def semantic_retrieval(
    calibrator: nn.Module | None,
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if calibrator is None:
        calibrated = predicted
    else:
        calibrator.eval()
        calibrated = torch.cat([
            calibrator(predicted[start : start + batch_size].to(device)).cpu()
            for start in range(0, len(predicted), batch_size)
        ])
    calibrated = F.normalize(calibrated.float(), p=2, dim=-1)
    target = F.normalize(target.float(), p=2, dim=-1)
    similarity = calibrated @ target.T
    diagonal = similarity.diag()
    ranks = 1 + (similarity > diagonal[:, None]).sum(dim=1)
    return {
        "matched_cosine": float(diagonal.mean()),
        "top1": float((ranks <= 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }


def tokenized_labels_and_weights(
    tokenizer,
    sentences: list[str],
    max_length: int,
    *,
    content_weight: float = 1.0,
    content_word_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        sentences, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt", return_offsets_mapping=True,
    )
    labels = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    weights = torch.ones_like(labels, dtype=torch.float32)
    if content_weight != 1.0 or content_word_weights:
        for row, sentence in enumerate(sentences):
            content_spans = [
                (match.start(), match.end(), match.group(0).lower())
                for match in _WORD_RE.finditer(sentence)
                if match.group(0).lower() not in _CONTENT_WORD_STOPWORDS
                and any(character.isalpha() for character in match.group(0))
            ]
            for column, (start, stop) in enumerate(offsets[row].tolist()):
                if start == stop:
                    continue
                matched_words = [
                    word
                    for word_start, word_stop, word in content_spans
                    if start < word_stop and stop > word_start
                ]
                if matched_words:
                    multiplier = max(
                        (content_word_weights or {}).get(word, 1.0)
                        for word in matched_words
                    )
                    weights[row, column] = float(content_weight * multiplier)
    padding = labels == tokenizer.pad_token_id
    labels[padding] = -100
    weights[padding] = 0.0
    return labels, weights


def train_content_idf_weights(
    sentences: list[str],
    *,
    power: float,
    max_multiplier: float,
) -> dict[str, float]:
    """Return occurrence-normalized IDF weights estimated from train text only."""
    if power < 0.0:
        raise ValueError("content IDF power must be non-negative")
    if max_multiplier < 1.0:
        raise ValueError("content IDF max multiplier must be at least one")
    if power == 0.0:
        return {}
    document_frequency: Counter[str] = Counter()
    token_frequency: Counter[str] = Counter()
    for sentence in sentences:
        words = [
            match.group(0).lower()
            for match in _WORD_RE.finditer(sentence)
            if match.group(0).lower() not in _CONTENT_WORD_STOPWORDS
            and any(character.isalpha() for character in match.group(0))
        ]
        token_frequency.update(words)
        document_frequency.update(set(words))
    powered = {
        word: (
            math.log((len(sentences) + 1) / (document_frequency[word] + 1)) + 1.0
        ) ** power
        for word in document_frequency
    }
    occurrence_mean = sum(
        token_frequency[word] * value for word, value in powered.items()
    ) / max(1, sum(token_frequency.values()))
    clipped = {
        word: min(max_multiplier, value / occurrence_mean)
        for word, value in powered.items()
    }
    clipped_mean = sum(
        token_frequency[word] * value for word, value in clipped.items()
    ) / max(1, sum(token_frequency.values()))
    return {
        word: min(max_multiplier, value / clipped_mean)
        for word, value in clipped.items()
    }


def tokenized_labels(tokenizer, sentences: list[str], max_length: int) -> torch.Tensor:
    """Backward-compatible label-only helper used by smoke tests and callers."""
    return tokenized_labels_and_weights(tokenizer, sentences, max_length)[0]


def sequence_nll(
    t5: T5ForConditionalGeneration,
    context: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    label_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted mean and per-row teacher-forced NLL."""
    output = t5(
        encoder_outputs=BaseModelOutput(last_hidden_state=context),
        attention_mask=mask,
        labels=labels,
        return_dict=True,
    )
    token_loss = F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    valid = labels.ne(-100)
    weights = valid.float() if label_weights is None else label_weights.to(token_loss) * valid
    per_row = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return per_row.mean(), per_row


def format_keyword_prompts(
    ranked: torch.Tensor,
    vocabulary: list[str],
    template: str,
) -> list[str]:
    prefixes = {
        "labeled": "keywords: ",
        "raw": "",
        "summarize": "summarize: ",
        "sentence": "sentence: ",
    }
    if template not in prefixes:
        raise ValueError(f"Unsupported keyword template {template!r}.")
    return [
        prefixes[template] + " ".join(vocabulary[index] for index in row)
        for row in ranked.tolist()
    ]


@torch.no_grad()
def keyword_encoder_context(
    *,
    t5: T5ForConditionalGeneration,
    tokenizer,
    semantic: torch.Tensor,
    content_head: nn.Module,
    content_prior_logit: torch.Tensor,
    content_vocabulary: list[str],
    topk: int,
    prior_subtraction: float,
    strength: float,
    template: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranked = content_ranked_indices(
        content_head, semantic, content_prior_logit,
        topk=topk, prior_subtraction=prior_subtraction,
    )
    prompts = format_keyword_prompts(ranked, content_vocabulary, template)
    encoded = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=True).to(device)
    context = t5.encoder(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        return_dict=True,
    ).last_hidden_state
    return context * strength, encoded["attention_mask"]


def story_balanced_rows(story_pools: dict[str, np.ndarray], count: int, rng: np.random.Generator) -> np.ndarray:
    names = sorted(story_pools)
    selected = rng.integers(0, len(names), size=count)
    return np.asarray([rng.choice(story_pools[names[index]]) for index in selected], dtype=np.int64)


@torch.no_grad()
def generate(
    projector: nn.Module,
    t5: T5ForConditionalGeneration,
    tokenizer,
    semantic: torch.Tensor,
    *,
    batch_size: int,
    max_target_tokens: int,
    min_target_tokens: int = 0,
    num_beams: int,
    device: torch.device,
    calibrator: nn.Module | None = None,
    content_head: nn.Module | None = None,
    content_prior_logit: torch.Tensor | None = None,
    content_vocabulary_token_ids: list[list[int]] | None = None,
    content_bias_topk: int = 5,
    content_bias_strength: float = 0.0,
    content_bias_once: bool = False,
    content_prior_subtraction: float = 0.5,
    content_max_prior_probability: float = 1.0,
    content_bias_weighting: str = "ranked",
    content_bias_temperature: float = 1.0,
    content_vocabulary: list[str] | None = None,
    content_keyword_context: bool = False,
    content_keyword_strength: float = 1.0,
    content_keyword_template: str = "labeled",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 1.0,
    max_output_words: int = 10,
) -> list[str]:
    projector.eval()
    if calibrator is not None:
        calibrator.eval()
    outputs: list[str] = []
    for start in range(0, len(semantic), batch_size):
        values = semantic[start : start + batch_size].to(device)
        if calibrator is not None:
            values = calibrator(values)
        context = projector(values)
        mask = torch.ones(context.shape[:2], dtype=torch.long, device=device)
        raw_semantic = semantic[start : start + batch_size].to(device)
        if content_head is not None and content_keyword_context:
            if content_prior_logit is None or content_vocabulary is None:
                raise ValueError("Keyword context requested without content metadata.")
            ranked = content_ranked_indices(
                content_head, raw_semantic, content_prior_logit,
                topk=content_bias_topk,
                prior_subtraction=content_prior_subtraction,
                max_prior_probability=content_max_prior_probability,
            )
            prompts = format_keyword_prompts(
                ranked, content_vocabulary, content_keyword_template,
            )
            encoded_keywords = tokenizer(
                prompts, padding=True, return_tensors="pt", add_special_tokens=True
            ).to(device)
            keyword_context = t5.encoder(
                input_ids=encoded_keywords["input_ids"],
                attention_mask=encoded_keywords["attention_mask"],
                return_dict=True,
            ).last_hidden_state
            keyword_context = keyword_context * content_keyword_strength
            context = torch.cat([keyword_context, context], dim=1)
            mask = torch.cat([encoded_keywords["attention_mask"], mask], dim=1)
        logits_processor = None
        if content_head is not None and content_bias_strength > 0.0:
            if content_prior_logit is None or content_vocabulary_token_ids is None:
                raise ValueError("Content bias requested without prior/token metadata.")
            bias = content_token_bias(
                content_head, raw_semantic,
                content_prior_logit, content_vocabulary_token_ids,
                topk=content_bias_topk, strength=content_bias_strength,
                prior_subtraction=content_prior_subtraction,
                model_vocab_size=t5.config.vocab_size,
                max_prior_probability=content_max_prior_probability,
                weighting=content_bias_weighting,
                temperature=content_bias_temperature,
            )
            processor = (
                UnseenTokenBatchLogitBias(bias)
                if content_bias_once else StaticBatchLogitBias(bias)
            )
            logits_processor = LogitsProcessorList([processor])
        generation_limits = {}
        if min_target_tokens > 0:
            generation_limits["min_new_tokens"] = int(min_target_tokens)
        if num_beams > 1:
            generation_limits["length_penalty"] = float(length_penalty)
        ids = t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=context),
            attention_mask=mask,
            max_new_tokens=max_target_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True,
            logits_processor=logits_processor,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **generation_limits,
        )
        outputs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    compacted = [" ".join(text.split()) for text in outputs]
    if max_output_words > 0:
        compacted = [" ".join(text.split()[:max_output_words]) for text in compacted]
    return compacted


def train_epoch(
    *,
    projector: nn.Module,
    t5: T5ForConditionalGeneration,
    optimizer: torch.optim.Optimizer,
    primary_semantic: torch.Tensor,
    oracle_semantic: torch.Tensor,
    labels: torch.Tensor,
    label_weights: torch.Tensor,
    story_pools: dict[str, np.ndarray],
    count: int,
    batch_size: int,
    oracle_preservation_weight: float,
    prefix_alignment_weight: float,
    semantic_alignment_weight: float,
    semantic_contrastive_weight: float,
    rng: np.random.Generator,
    device: torch.device,
    calibrator: nn.Module | None = None,
    decoder_weight: float = 1.0,
    trainable_parameters: list[nn.Parameter] | None = None,
    pairing_margin_weight: float = 0.0,
    pairing_margin: float = 0.25,
    tokenizer=None,
    content_head: nn.Module | None = None,
    content_prior_logit: torch.Tensor | None = None,
    content_vocabulary: list[str] | None = None,
    content_bias_topk: int = 5,
    content_prior_subtraction: float = 0.5,
    content_keyword_train_context: bool = False,
    keyword_semantic: torch.Tensor | None = None,
    content_keyword_strength: float = 1.0,
    content_keyword_template: str = "labeled",
) -> float:
    if calibrator is not None:
        # The projector is the fixed oracle MiniLM-to-T5 interface in the
        # calibration stage.  Keep dropout disabled so the calibrator sees a
        # stationary target function.
        projector.eval()
        calibrator.train()
    else:
        projector.train()
    order = story_balanced_rows(story_pools, count, rng)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        primary = primary_semantic[rows].to(device)
        oracle = oracle_semantic[rows].to(device)
        target = labels[rows].to(device)
        target_weights = label_weights[rows].to(device)
        optimizer.zero_grad(set_to_none=True)
        calibrated = calibrator(primary) if calibrator is not None else primary
        primary_prefix_context = projector(calibrated)
        primary_context = primary_prefix_context
        mask = torch.ones(primary_context.shape[:2], dtype=torch.long, device=device)
        if content_keyword_train_context and decoder_weight > 0.0:
            if (
                tokenizer is None or content_head is None or content_prior_logit is None
                or content_vocabulary is None
            ):
                raise ValueError("Keyword training context requested without lexical metadata.")
            lexical_values = (
                keyword_semantic[rows].to(device)
                if keyword_semantic is not None else primary
            )
            keyword_context, keyword_mask = keyword_encoder_context(
                t5=t5, tokenizer=tokenizer, semantic=lexical_values,
                content_head=content_head, content_prior_logit=content_prior_logit,
                content_vocabulary=content_vocabulary, topk=content_bias_topk,
                prior_subtraction=content_prior_subtraction,
                strength=content_keyword_strength,
                template=content_keyword_template, device=device,
            )
            primary_context = torch.cat([keyword_context, primary_context], dim=1)
            mask = torch.cat([keyword_mask, mask], dim=1)
        loss = calibrated.sum() * 0.0
        if decoder_weight > 0.0:
            decoder_loss, matched_nll = sequence_nll(
                t5, primary_context, mask, target, target_weights,
            )
            loss = loss + decoder_weight * decoder_loss
            if pairing_margin_weight > 0.0 and len(primary_context) > 1:
                _, mismatched_nll = sequence_nll(
                    t5, primary_context.roll(shifts=1, dims=0),
                    mask.roll(shifts=1, dims=0), target, target_weights,
                )
                pairing_loss = F.relu(
                    pairing_margin + matched_nll - mismatched_nll
                ).mean()
                loss = loss + pairing_margin_weight * pairing_loss
        if decoder_weight > 0.0 and oracle_preservation_weight > 0.0:
            oracle_prefix_context = projector(oracle)
            oracle_context = oracle_prefix_context
            oracle_mask = torch.ones(
                oracle_context.shape[:2], dtype=torch.long, device=device
            )
            if content_keyword_train_context:
                oracle_context = torch.cat([keyword_context, oracle_context], dim=1)
                oracle_mask = torch.cat([keyword_mask, oracle_mask], dim=1)
            oracle_loss, _ = sequence_nll(
                t5, oracle_context, oracle_mask, target, target_weights,
            )
            loss = loss + oracle_preservation_weight * oracle_loss
            if prefix_alignment_weight > 0.0:
                alignment = 1.0 - F.cosine_similarity(
                    primary_prefix_context.float(), oracle_prefix_context.detach().float(), dim=-1
                ).mean()
                loss = loss + prefix_alignment_weight * alignment
        if calibrator is not None and semantic_alignment_weight > 0.0:
            semantic_alignment = 1.0 - F.cosine_similarity(
                calibrated.float(), oracle.float(), dim=-1
            ).mean()
            loss = loss + semantic_alignment_weight * semantic_alignment
        if calibrator is not None and semantic_contrastive_weight > 0.0:
            similarity = F.normalize(calibrated.float(), dim=-1) @ F.normalize(
                oracle.float(), dim=-1
            ).T
            labels_index = torch.arange(len(similarity), device=device)
            semantic_contrastive = 0.5 * (
                F.cross_entropy(similarity / 0.07, labels_index)
                + F.cross_entropy(similarity.T / 0.07, labels_index)
            )
            loss = loss + semantic_contrastive_weight * semantic_contrastive
        loss.backward()
        parameters_to_clip = trainable_parameters or list(
            calibrator.parameters() if calibrator is not None else projector.parameters()
        )
        torch.nn.utils.clip_grad_norm_(parameters_to_clip, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scientific_scope = (
        f"train{args.train_count}-to-val{args.val_count}; "
        f"{args.protected_split_name} sealed; modality={args.modality}; "
        f"oracle_corpus={'augmented' if args.oracle_augmentation_npz else 'train-only'}"
    )

    precomputed_mode = bool(args.precomputed_semantic_npz)
    if precomputed_mode:
        if args.oof_semantic_npz or args.oracle_semantic_npz:
            raise ValueError(
                "--precomputed-semantic-npz cannot be combined with OOF/oracle override archives."
            )
        with np.load(args.precomputed_semantic_npz, allow_pickle=True) as source:
            required = {
                args.precomputed_input_key,
                args.precomputed_target_key,
                args.precomputed_sentence_key,
                args.precomputed_group_key,
            }
            missing = required.difference(source.files)
            if missing:
                raise KeyError(
                    f"{args.precomputed_semantic_npz} missing keys: {sorted(missing)}"
                )
            predicted_values = np.asarray(
                source[args.precomputed_input_key], dtype=np.float32
            )
            oracle_values = np.asarray(
                source[args.precomputed_target_key], dtype=np.float32
            )
            sentences = strings(source[args.precomputed_sentence_key])
            groups = np.asarray(strings(source[args.precomputed_group_key]))
            if "source_split" in source.files:
                source_split = np.asarray(strings(source["source_split"]))
            elif "split" in source.files:
                source_split = np.asarray(strings(source["split"]))
            else:
                source_split = None
        expected = args.train_count + args.val_count
        if predicted_values.shape != oracle_values.shape:
            raise ValueError(
                "Predicted/exact precomputed semantic shapes differ: "
                f"{predicted_values.shape} != {oracle_values.shape}."
            )
        if predicted_values.ndim != 2 or len(predicted_values) != expected:
            raise ValueError(
                f"Expected {expected} aligned 2-D semantic rows, got {predicted_values.shape}."
            )
        if len(sentences) != expected or len(groups) != expected:
            raise ValueError("Precomputed sentence/group row count mismatch.")
        if source_split is not None:
            expected_split = np.asarray(
                ["train"] * args.train_count + ["val"] * args.val_count
            )
            if not np.array_equal(source_split.astype(str), expected_split):
                raise ValueError(
                    "Precomputed archive is not ordered as train rows followed by validation rows."
                )
        train_story = groups[: args.train_count]
        oracle = F.normalize(
            torch.as_tensor(oracle_values, dtype=torch.float32), p=2, dim=-1
        )
        train_brain = F.normalize(
            torch.as_tensor(predicted_values[: args.train_count]), p=2, dim=-1
        )
        val_brain = F.normalize(
            torch.as_tensor(predicted_values[args.train_count :]), p=2, dim=-1
        )
        text = None
        oracle_source = None
    else:
        if not args.brain_npz or not args.text_npz:
            raise ValueError(
                "The fMRI path requires --brain-npz and --text-npz; otherwise use "
                "--precomputed-semantic-npz."
            )
        brain = np.load(args.brain_npz, allow_pickle=True, mmap_mode="r")
        train_x, val_x = brain["train_x"], brain["val_x"]
        if len(train_x) != args.train_count or len(val_x) != args.val_count:
            raise ValueError("Unexpected train/validation brain counts.")
        train_story = np.asarray(strings(brain["train_story"]))
        text = np.load(args.text_npz, allow_pickle=True)
        sentences = strings(text["sentence"])
        oracle_source = (
            np.load(args.oracle_semantic_npz, allow_pickle=True)
            if args.oracle_semantic_npz
            else text
        )
        oracle = F.normalize(
            torch.as_tensor(oracle_source["input_embeddings"], dtype=torch.float32),
            p=2,
            dim=-1,
        )
        if len(sentences) != args.train_count + args.val_count:
            raise ValueError("Unexpected semantic/text target archive.")
        if len(oracle) != len(sentences):
            raise ValueError("Oracle semantic and exact-text archives have different row counts.")
        for key in ("story", "start_tr", "stop_tr"):
            if key in text.files and key in oracle_source.files:
                left = np.asarray(text[key])
                right = np.asarray(oracle_source[key])
                if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
                    aligned = np.array_equal(left.astype(str), right.astype(str))
                else:
                    aligned = np.array_equal(left, right)
                if not aligned:
                    raise ValueError(
                        f"Oracle semantic and exact-text rows are misaligned on {key}."
                    )
    brain_story_pools = {
        story: np.flatnonzero(train_story == story)
        for story in sorted(set(train_story.tolist()))
    }
    train_sentences, val_sentences = sentences[: args.train_count], sentences[args.train_count :]
    train_oracle, val_oracle = oracle[: args.train_count], oracle[args.train_count :]

    if precomputed_mode:
        pass
    elif args.oof_semantic_npz:
        with np.load(args.oof_semantic_npz, allow_pickle=True, mmap_mode="r") as oof:
            condition = np.asarray(strings(oof["condition_source"]))
            oof_train_rows = np.flatnonzero(condition == "oof_prediction")
            oof_val_rows = np.flatnonzero(condition == "validation_prediction_ensemble")
            if len(oof_train_rows) != args.train_count or len(oof_val_rows) != args.val_count:
                raise ValueError("Unexpected OOF/validation prediction counts.")
            delayed = np.asarray(oof["input_embeddings"], dtype=np.float32)
            if delayed.shape[1] != 1536:
                raise ValueError(f"Expected four delayed 384-D blocks, got {delayed.shape}.")
            oof_sentences = np.asarray(strings(oof["sentence"]))
            oof_stories = np.asarray(strings(oof["story"]))
            if not np.array_equal(oof_sentences[oof_train_rows], np.asarray(train_sentences)):
                raise ValueError("OOF train sentences are not aligned with exact targets.")
            if not np.array_equal(oof_sentences[oof_val_rows], np.asarray(val_sentences)):
                raise ValueError("OOF validation sentences are not aligned with exact targets.")
            if not np.array_equal(oof_stories[oof_train_rows], train_story):
                raise ValueError("OOF train stories are not aligned with brain rows.")
            if args.oof_delay_mode == "mean":
                train_delayed = delayed[oof_train_rows].reshape(args.train_count, 4, 384).mean(axis=1)
                val_delayed = delayed[oof_val_rows].reshape(args.val_count, 4, 384).mean(axis=1)
            else:
                train_delayed = delayed[oof_train_rows]
                val_delayed = delayed[oof_val_rows]
        train_brain = F.normalize(torch.as_tensor(train_delayed), p=2, dim=-1)
        val_brain = F.normalize(torch.as_tensor(val_delayed), p=2, dim=-1)
    else:
        if not args.mri_checkpoint:
            raise ValueError("The raw-fMRI path requires --mri-checkpoint.")
        mri = load_mri2sem_model(
            args.mri_checkpoint, input_dim=train_x.shape[1], output_dim=384,
            hidden_dim=args.mri_hidden_dim, res_blocks=args.mri_res_blocks,
        ).to(device)
        mapper = load_semantic_mapper(args.mri_checkpoint, 384, args.mapper_hidden_dim).to(device)
        for module in (mri, mapper):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        train_raw = encode_features(
            mri, train_x, feature="semantic", batch_size=args.feature_batch_size, device=device
        )
        val_raw = encode_features(
            mri, val_x, feature="semantic", batch_size=args.feature_batch_size, device=device
        )
        with torch.no_grad():
            train_brain = F.normalize(train_raw + torch.cat([
                mapper(train_raw[start : start + args.feature_batch_size].to(device)).cpu()
                for start in range(0, len(train_raw), args.feature_batch_size)
            ]), p=2, dim=-1)
            val_brain = F.normalize(val_raw + torch.cat([
                mapper(val_raw[start : start + args.feature_batch_size].to(device)).cpu()
                for start in range(0, len(val_raw), args.feature_batch_size)
            ]), p=2, dim=-1)
        del mri, mapper, train_raw, val_raw
        torch.cuda.empty_cache()

    oracle_train_sentences = list(train_sentences)
    oracle_train_semantic = train_oracle
    oracle_story_pools = brain_story_pools
    if args.oracle_augmentation_npz:
        if not precomputed_mode:
            raise ValueError(
                "--oracle-augmentation-npz currently requires --precomputed-semantic-npz."
            )
        with np.load(args.oracle_augmentation_npz, allow_pickle=True) as augmented:
            required = {
                args.oracle_augmentation_input_key,
                args.oracle_augmentation_sentence_key,
            }
            missing = required.difference(augmented.files)
            if missing:
                raise KeyError(
                    f"{args.oracle_augmentation_npz} missing keys: {sorted(missing)}"
                )
            augmented_values = np.asarray(
                augmented[args.oracle_augmentation_input_key], dtype=np.float32
            )
            oracle_train_sentences = strings(
                augmented[args.oracle_augmentation_sentence_key]
            )
        if augmented_values.ndim != 2 or len(augmented_values) != len(oracle_train_sentences):
            raise ValueError("Oracle augmentation semantic/text row count mismatch.")
        if augmented_values.shape[1] != train_oracle.shape[1]:
            raise ValueError(
                "Oracle augmentation semantic dimension differs from the brain interface: "
                f"{augmented_values.shape[1]} != {train_oracle.shape[1]}."
            )
        oracle_train_semantic = F.normalize(
            torch.as_tensor(augmented_values, dtype=torch.float32), p=2, dim=-1
        )
        oracle_story_pools = {
            "oracle_augmentation": np.arange(len(oracle_train_semantic), dtype=np.int64)
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    t5 = T5ForConditionalGeneration.from_pretrained(
        args.model_name, local_files_only=True
    ).to(device).eval()
    for parameter in t5.parameters():
        parameter.requires_grad_(False)
    t5_lora_layers = inject_t5_cross_attention_lora(
        t5, rank=args.t5_lora_rank, alpha=args.t5_lora_alpha,
        last_n_blocks=args.t5_lora_last_n_blocks,
    )
    t5_lora_parameters = [
        parameter for name, parameter in t5.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    ]
    semantic_dim = int(train_oracle.shape[1])
    if args.projector_kind == "delay_transformer":
        if args.oracle_augmentation_npz:
            raise ValueError(
                "Oracle augmentation is not implemented for delay_transformer."
            )
        if (
            semantic_dim != 384
            or train_brain.shape[1] != 1536
            or args.brain_calibrator_hidden_dim > 0
        ):
            raise ValueError(
                "delay_transformer requires 384-D oracle targets, ordered 1536-D "
                "OOF inputs, and no semantic calibrator."
            )
        projector = OrderedDelayT5Prefix(
            prefix_length=args.prefix_length, model_dim=t5.config.d_model,
            dropout=args.dropout,
        ).to(device)
        train_projector_oracle = F.normalize(train_oracle.repeat(1, 4), p=2, dim=-1)
        oracle_training_projector = train_projector_oracle
        val_projector_oracle = F.normalize(val_oracle.repeat(1, 4), p=2, dim=-1)
    else:
        projector = MiniLMT5Prefix(
            input_dim=semantic_dim, hidden_dim=args.projector_hidden_dim,
            prefix_length=args.prefix_length, model_dim=t5.config.d_model,
            dropout=args.dropout,
        ).to(device)
        train_projector_oracle = train_oracle
        oracle_training_projector = oracle_train_semantic
        val_projector_oracle = val_oracle
    if args.oracle_init_checkpoint:
        oracle_payload = torch.load(
            args.oracle_init_checkpoint, map_location="cpu", weights_only=False
        )
        projector.load_state_dict(oracle_payload["projector_state_dict"], strict=True)
        inherited_oracle_scope = oracle_payload.get("scientific_scope")
        if inherited_oracle_scope:
            scientific_scope += f"; oracle_init_scope=({inherited_oracle_scope})"
        print(
            json.dumps(
                {
                    "loaded_oracle_init": args.oracle_init_checkpoint,
                    "semantic_dim": semantic_dim,
                    "inherited_oracle_scope": inherited_oracle_scope,
                }
            ),
            flush=True,
        )
    if args.brain_calibrator_kind == "delay_weighted":
        if train_brain.shape[1] != 1536:
            raise ValueError(
                "delay_weighted calibration requires ordered 1536-D OOF inputs."
            )
        calibrator = DelayWeightedCalibrator().to(device)
    elif (
        args.brain_calibrator_kind == "identity_residual"
        and args.brain_calibrator_hidden_dim > 0
    ):
        if int(train_brain.shape[1]) != semantic_dim:
            raise ValueError(
                "identity_residual requires predicted and exact semantics in the same dimension."
            )
        calibrator = IdentityResidualCalibrator(
            semantic_dim, args.brain_calibrator_hidden_dim, args.dropout
        ).to(device)
    elif args.brain_calibrator_hidden_dim > 0:
        calibrator = ResidualBrainCalibrator(
            int(train_brain.shape[1]), args.brain_calibrator_hidden_dim, args.dropout
        ).to(device)
    else:
        calibrator = None
    content_head = None
    content_prior_logit = None
    content_vocabulary: list[str] | None = None
    content_vocabulary_token_ids: list[list[int]] | None = None
    if args.content_head_checkpoint:
        content_payload = torch.load(
            args.content_head_checkpoint, map_location="cpu", weights_only=False
        )
        content_vocabulary = [str(word) for word in content_payload["vocabulary"]]
        if int(content_payload["input_dim"]) != semantic_dim:
            raise ValueError(
                f"Content head must consume {semantic_dim}-D semantic features."
            )
        content_head = ContentHead(
            semantic_dim, int(content_payload["hidden_dim"]), len(content_vocabulary)
        ).to(device)
        content_head.load_state_dict(content_payload["head_state_dict"], strict=True)
        content_head.eval()
        for parameter in content_head.parameters():
            parameter.requires_grad_(False)
        prior = torch.as_tensor(content_payload["logit_prior"], dtype=torch.float32)
        content_prior_logit = torch.logit(prior.clamp(1e-5, 1.0 - 1e-5)).to(device)
        special_ids = set(tokenizer.all_special_ids)
        content_vocabulary_token_ids = [
            [
                token_id for token_id in tokenizer.encode(word, add_special_tokens=False)
                if token_id not in special_ids
            ]
            for word in content_vocabulary
        ]
    brain_content_word_weights = train_content_idf_weights(
        train_sentences,
        power=args.content_idf_power,
        max_multiplier=args.content_idf_max_multiplier,
    )
    brain_labels, brain_label_weights = tokenized_labels_and_weights(
        tokenizer, train_sentences, args.max_target_tokens,
        content_weight=args.content_token_ce_weight,
        content_word_weights=brain_content_word_weights,
    )
    oracle_content_word_weights = train_content_idf_weights(
        oracle_train_sentences,
        power=args.content_idf_power,
        max_multiplier=args.content_idf_max_multiplier,
    )
    oracle_labels, oracle_label_weights = tokenized_labels_and_weights(
        tokenizer, oracle_train_sentences, args.max_target_tokens,
        content_weight=args.content_token_ce_weight,
        content_word_weights=oracle_content_word_weights,
    )
    rng = np.random.default_rng(args.seed)
    derange_rng = np.random.default_rng(args.seed + 1000)
    val_derangements = [derangement(args.val_count, derange_rng) for _ in range(args.derangements)]
    history: list[dict] = []
    best_score = float("-inf")
    best_payload = None
    best_oracle_score = float("-inf")
    best_oracle_record = None

    stages = [(
        "oracle", args.oracle_epochs, args.oracle_lr,
        oracle_training_projector, 0.0, 0.0, 1.0,
    )]
    if calibrator is not None and args.brain_semantic_warmup_epochs > 0:
        stages.append((
            "semantic_warmup", args.brain_semantic_warmup_epochs,
            args.brain_semantic_warmup_lr, train_brain, 0.0, 0.0, 0.0,
        ))
    stages.append((
        "brain", args.brain_epochs, args.brain_lr, train_brain,
        args.oracle_preservation_weight, args.prefix_alignment_weight, 1.0,
    ))
    for stage, epochs, lr, primary_train, preserve_weight, alignment_weight, decoder_weight in stages:
        if stage == "oracle":
            stage_oracle = oracle_training_projector
            stage_labels = oracle_labels
            stage_label_weights = oracle_label_weights
            stage_story_pools = oracle_story_pools
            stage_count = len(oracle_training_projector)
        else:
            stage_oracle = train_projector_oracle
            stage_labels = brain_labels
            stage_label_weights = brain_label_weights
            stage_story_pools = brain_story_pools
            stage_count = args.train_count
        use_calibrator = stage in ("semantic_warmup", "brain") and calibrator is not None
        joint_projector = (
            stage == "brain" and use_calibrator and args.brain_unfreeze_projector
        )
        for parameter in projector.parameters():
            parameter.requires_grad_(not use_calibrator or joint_projector)
        if calibrator is not None:
            for parameter in calibrator.parameters():
                parameter.requires_grad_(use_calibrator)
        use_lora = stage == "brain" and bool(t5_lora_parameters)
        for parameter in t5_lora_parameters:
            parameter.requires_grad_(use_lora)
        primary_parameters = list(
            calibrator.parameters() if use_calibrator else projector.parameters()
        )
        # Optimizer groups retain the list object passed here.  Keep a copy so
        # extending the clipping list below cannot accidentally duplicate the
        # projector in both parameter groups.
        parameter_groups = [{"params": list(primary_parameters), "lr": lr}]
        if joint_projector:
            projector_parameters = list(projector.parameters())
            primary_parameters.extend(projector_parameters)
            parameter_groups.append({
                "params": projector_parameters,
                "lr": args.brain_projector_lr,
            })
        if use_lora:
            parameter_groups.append({"params": t5_lora_parameters, "lr": args.t5_lora_lr})
        trainable_parameters = primary_parameters + (t5_lora_parameters if use_lora else [])
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                projector=projector, t5=t5, optimizer=optimizer,
                primary_semantic=primary_train, oracle_semantic=stage_oracle,
                labels=stage_labels, label_weights=stage_label_weights,
                story_pools=stage_story_pools, count=stage_count,
                batch_size=args.batch_size, oracle_preservation_weight=preserve_weight,
                prefix_alignment_weight=alignment_weight,
                semantic_alignment_weight=(
                    args.brain_semantic_alignment_weight if use_calibrator else 0.0
                ),
                semantic_contrastive_weight=(
                    args.brain_semantic_contrastive_weight if use_calibrator else 0.0
                ),
                rng=rng, device=device,
                calibrator=calibrator if use_calibrator else None,
                decoder_weight=decoder_weight,
                trainable_parameters=trainable_parameters,
                pairing_margin_weight=(
                    args.brain_pairing_margin_weight if stage == "brain" else 0.0
                ),
                pairing_margin=args.brain_pairing_margin,
                tokenizer=tokenizer,
                content_head=content_head,
                content_prior_logit=content_prior_logit,
                content_vocabulary=content_vocabulary,
                content_bias_topk=args.content_bias_topk,
                content_prior_subtraction=args.content_prior_subtraction,
                content_keyword_train_context=(
                    (stage == "brain" and args.content_keyword_train_context)
                    or (stage == "oracle" and args.oracle_keyword_train_context)
                ),
                keyword_semantic=(
                    primary_train
                    if stage == "oracle" and args.oracle_keyword_train_context
                    else None
                ),
                content_keyword_strength=args.content_keyword_strength,
                content_keyword_template=args.content_keyword_template,
            )
            if epoch % args.eval_every and epoch != epochs:
                continue
            validation_semantic = val_projector_oracle if stage == "oracle" else val_brain
            generated = generate(
                projector, t5, tokenizer, validation_semantic,
                batch_size=args.eval_batch_size, max_target_tokens=args.max_generation_tokens,
                min_target_tokens=args.min_generation_tokens,
                num_beams=args.num_beams, device=device,
                calibrator=calibrator if use_calibrator else None,
                content_head=content_head,
                content_prior_logit=content_prior_logit,
                content_vocabulary_token_ids=content_vocabulary_token_ids,
                content_bias_topk=args.content_bias_topk,
                content_bias_strength=args.content_bias_strength,
                content_bias_once=args.content_bias_once,
                content_prior_subtraction=args.content_prior_subtraction,
                content_vocabulary=content_vocabulary,
                content_keyword_context=args.content_keyword_context,
                content_keyword_strength=args.content_keyword_strength,
                content_keyword_template=args.content_keyword_template,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_output_words=args.max_output_words,
            )
            matched = compact(generated, val_sentences)
            deranged_metrics = []
            if stage != "oracle":
                for permutation in val_derangements:
                    values = val_brain[torch.as_tensor(permutation, dtype=torch.long)]
                    deranged_generated = generate(
                        projector, t5, tokenizer, values,
                        batch_size=args.eval_batch_size,
                        max_target_tokens=args.max_generation_tokens,
                        min_target_tokens=args.min_generation_tokens,
                        num_beams=args.num_beams, device=device,
                        calibrator=calibrator if use_calibrator else None,
                        content_head=content_head,
                        content_prior_logit=content_prior_logit,
                        content_vocabulary_token_ids=content_vocabulary_token_ids,
                        content_bias_topk=args.content_bias_topk,
                        content_bias_strength=args.content_bias_strength,
                        content_bias_once=args.content_bias_once,
                        content_prior_subtraction=args.content_prior_subtraction,
                        content_vocabulary=content_vocabulary,
                        content_keyword_context=args.content_keyword_context,
                        content_keyword_strength=args.content_keyword_strength,
                        content_keyword_template=args.content_keyword_template,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        max_output_words=args.max_output_words,
                    )
                    deranged_metrics.append(compact(deranged_generated, val_sentences))
            deranged_mean = float(np.mean([
                row["content_words_overlap"] for row in deranged_metrics
            ])) if deranged_metrics else 0.0
            record = {
                "stage": stage, "epoch": epoch, "train_loss": train_loss,
                "matched": matched, "deranged_content_f1_mean": deranged_mean,
                "deranged_content_f1_max": float(np.max([
                    row["content_words_overlap"] for row in deranged_metrics
                ])) if deranged_metrics else 0.0,
                "conditional_content_margin": matched["content_words_overlap"] - deranged_mean,
                "semantic_retrieval": (
                    semantic_retrieval(
                        calibrator if use_calibrator else None,
                        val_brain,
                        val_oracle,
                        batch_size=args.eval_batch_size, device=device,
                    )
                    if use_calibrator or val_brain.shape[1] == val_oracle.shape[1]
                    else None
                ),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if stage == "oracle":
                oracle_score = 0.5 * (
                    matched["content_words_overlap"] + matched["words_overlap"]
                )
                if oracle_score > best_oracle_score:
                    best_oracle_score = oracle_score
                    best_oracle_record = record
                    oracle_checkpoint = {
                        "projector_state_dict": {
                            key: value.detach().cpu()
                            for key, value in projector.state_dict().items()
                        },
                        "args": vars(args),
                        "best": record,
                        "semantic_dim": semantic_dim,
                        "scientific_scope": scientific_scope,
                    }
                    torch.save(oracle_checkpoint, output_dir / "oracle_best.pt")
                    (output_dir / "oracle_best_metrics.json").write_text(
                        json.dumps(
                            {
                                "status": "oracle-best-so-far",
                                "scientific_scope": scientific_scope,
                                "args": vars(args),
                                "best": record,
                                "generated": generated,
                                "targets": val_sentences,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            else:
                if args.selection_objective == "word_content_mean":
                    score = 0.5 * (
                        matched["content_words_overlap"] + matched["words_overlap"]
                    )
                else:
                    score = (
                        matched["content_words_overlap"]
                        + record["conditional_content_margin"]
                        + 0.1 * matched["words_overlap"]
                        - 0.01 * matched["word_error_rate"]
                    )
                if score > best_score:
                    best_score = score
                    best_payload = {
                        "record": record, "generated": generated,
                        "projector_state_dict": {
                            key: value.detach().cpu() for key, value in projector.state_dict().items()
                        },
                        "calibrator_state_dict": (
                            {key: value.detach().cpu() for key, value in calibrator.state_dict().items()}
                            if calibrator is not None else None
                        ),
                        "content_head_state_dict": (
                            {key: value.detach().cpu() for key, value in content_head.state_dict().items()}
                            if content_head is not None else None
                        ),
                        "t5_lora_state_dict": {
                            key: value.detach().cpu()
                            for key, value in t5.state_dict().items()
                            if ".lora_A" in key or ".lora_B" in key
                        },
                    }
                    torch.save({
                        "projector_state_dict": best_payload["projector_state_dict"],
                        "calibrator_state_dict": best_payload["calibrator_state_dict"],
                        "content_head_state_dict": best_payload["content_head_state_dict"],
                        "content_vocabulary": content_vocabulary,
                        "content_prior_logit": (
                            content_prior_logit.detach().cpu()
                            if content_prior_logit is not None else None
                        ),
                        "t5_lora_state_dict": best_payload["t5_lora_state_dict"],
                        "t5_lora_layers": t5_lora_layers,
                        "args": vars(args), "best": record,
                        "mri_checkpoint": args.mri_checkpoint,
                        "scientific_scope": scientific_scope,
                    }, output_dir / "best.pt")
                    (output_dir / "best_metrics.json").write_text(json.dumps({
                        "status": "best-so-far", "scientific_scope": scientific_scope,
                        "args": vars(args), "best": record,
                        "generated": generated, "targets": val_sentences,
                    }, indent=2) + "\n")
            (output_dir / "history.json").write_text(json.dumps({
                "status": "running", "scientific_scope": scientific_scope,
                "args": vars(args), "history": history,
            }, indent=2) + "\n")

    (output_dir / "history.json").write_text(json.dumps({
        "status": "complete", "scientific_scope": scientific_scope,
        "args": vars(args), "history": history,
        "best_oracle": best_oracle_record,
        "best": best_payload["record"] if best_payload else None,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
