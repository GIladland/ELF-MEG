"""Oracle known-word conditioning for MEG-to-ELF ceiling experiments.

This module deliberately accepts word embeddings supplied by the caller.  In
the oracle experiment those embeddings are derived from target words, so this
path must never be presented as a brain-conditioned result.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from modules.meg_adapter import MEGAdapterOutput


ORACLE_WORD_LAYOUTS = (
    "none",
    "center1",
    "center2",
    "center4",
    "dispersed2",
    "dispersed4",
    "all",
    "random",
)


def _layout_positions(layout: str, word_count: int) -> list[int]:
    if word_count <= 0 or layout == "none":
        return []
    if layout == "all":
        return list(range(word_count))

    center_right = word_count // 2
    center_left = max(0, center_right - 1)
    if layout == "center1":
        return [center_left]
    if layout == "center2":
        return sorted({center_left, min(word_count - 1, center_right)})
    if layout == "center4":
        start = max(0, center_right - 2)
        return list(range(start, min(word_count, start + 4)))
    if layout == "dispersed2":
        return sorted({word_count // 4, min(word_count - 1, (3 * word_count) // 4)})
    if layout == "dispersed4":
        if word_count == 1:
            return [0]
        # Interior, approximately equally spaced positions.  For ten-word
        # Apple rows this is [1, 3, 6, 8].
        return sorted(
            {
                int(round((word_count - 1) * fraction))
                for fraction in (0.15, 0.35, 0.65, 0.85)
            }
        )
    raise ValueError(f"Unsupported fixed oracle-word layout: {layout!r}")


def build_oracle_word_mask(
    valid_word_mask: torch.Tensor,
    *,
    layout: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select exact word positions while respecting each row's valid words."""
    if valid_word_mask.ndim != 2:
        raise ValueError(
            "valid_word_mask must have shape [batch, words], got "
            f"{tuple(valid_word_mask.shape)}"
        )
    if layout not in ORACLE_WORD_LAYOUTS:
        raise ValueError(
            f"Unknown oracle-word layout {layout!r}; expected one of {ORACLE_WORD_LAYOUTS}."
        )

    valid = valid_word_mask.to(dtype=torch.bool)
    selected = torch.zeros_like(valid)
    if layout == "none":
        return selected

    for row_index in range(valid.shape[0]):
        valid_positions = torch.nonzero(valid[row_index], as_tuple=False).flatten()
        word_count = int(valid_positions.numel())
        if word_count == 0:
            continue
        if layout == "random":
            allowed_counts = [count for count in (1, 2, 4) if count <= word_count]
            count_index = int(
                torch.randint(len(allowed_counts), (1,), generator=generator).item()
            )
            chosen_count = allowed_counts[count_index]
            order = torch.randperm(word_count, generator=generator)[:chosen_count]
            chosen = valid_positions.index_select(0, order)
        else:
            relative_positions = _layout_positions(layout, word_count)
            chosen = valid_positions[
                torch.as_tensor(relative_positions, dtype=torch.long, device=valid.device)
            ]
        selected[row_index, chosen] = True
    return selected


class OracleKnownWordFusion(nn.Module):
    """Cross-attend semantic context tokens to an ordered known-word memory."""

    def __init__(
        self,
        *,
        context_dim: int,
        max_words: int = 10,
        attention_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if max_words <= 0:
            raise ValueError(f"max_words must be positive, got {max_words}.")
        if context_dim % attention_heads:
            raise ValueError(
                f"context_dim={context_dim} must be divisible by attention_heads={attention_heads}."
            )
        self.context_dim = int(context_dim)
        self.max_words = int(max_words)
        self.word_norm = nn.LayerNorm(context_dim)
        self.word_projection = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
        )
        self.position_embedding = nn.Parameter(torch.zeros(max_words, context_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(context_dim)
        self.cross_attention = nn.MultiheadAttention(
            context_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Zero initialization makes the no-word/before-training behavior
        # exactly equal to the selected frozen diffusion winner.
        self.residual_gate = nn.Parameter(torch.zeros(context_dim))

    def forward(
        self,
        context: torch.Tensor,
        known_word_embeddings: torch.Tensor,
        known_word_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 3 or known_word_embeddings.ndim != 3:
            raise ValueError("context and known_word_embeddings must both be rank-3 tensors")
        if known_word_mask.shape != known_word_embeddings.shape[:2]:
            raise ValueError(
                "known_word_mask must match the first two word-embedding dimensions: "
                f"mask={tuple(known_word_mask.shape)} embeddings={tuple(known_word_embeddings.shape)}"
            )
        if known_word_embeddings.shape[0] != context.shape[0]:
            raise ValueError("context and known-word batches must have the same size")
        if known_word_embeddings.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected word embedding dim {self.context_dim}, got {known_word_embeddings.shape[-1]}."
            )
        word_slots = int(known_word_embeddings.shape[1])
        if word_slots > self.max_words:
            raise ValueError(f"Received {word_slots} word slots, maximum is {self.max_words}.")

        mask = known_word_mask.to(device=context.device, dtype=torch.bool)
        row_has_words = mask.any(dim=1, keepdim=True)
        # MultiheadAttention cannot accept an all-padding row.  Give those
        # rows one zero-valued dummy memory token, then explicitly zero their
        # residual below.
        safe_mask = mask.clone()
        empty_rows = ~row_has_words.squeeze(1)
        if bool(empty_rows.any()):
            safe_mask[empty_rows, 0] = True

        words = known_word_embeddings.to(device=context.device, dtype=context.dtype)
        positions = self.position_embedding[:word_slots].to(dtype=context.dtype).unsqueeze(0)
        words = self.word_projection(self.word_norm(words)) + positions
        words = words * mask.unsqueeze(-1).to(dtype=words.dtype)
        residual, _ = self.cross_attention(
            self.query_norm(context),
            words,
            words,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        residual = residual * row_has_words.unsqueeze(-1).to(dtype=residual.dtype)
        gate = torch.tanh(self.residual_gate).reshape(1, 1, -1).to(dtype=context.dtype)
        return context + gate * residual


class OracleKnownWordMEGContextAdapter(nn.Module):
    """Wrap an existing MEG adapter with explicit oracle word conditioning."""

    def __init__(
        self,
        *,
        base_adapter: nn.Module,
        context_dim: int,
        max_words: int = 10,
        attention_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_adapter = base_adapter
        self.word_fusion = OracleKnownWordFusion(
            context_dim=context_dim,
            max_words=max_words,
            attention_heads=attention_heads,
            dropout=dropout,
        )

    @property
    def meg2sem(self):
        return self.base_adapter.meg2sem

    @property
    def semantic_projector(self):
        return self.base_adapter.semantic_projector

    @property
    def normalize_semantic_output(self) -> bool:
        return bool(getattr(self.base_adapter, "normalize_semantic_output", False))

    def semantic_projector_input(self, *args, **kwargs):
        return self.base_adapter.semantic_projector_input(*args, **kwargs)

    def context_from_projected_semantic(self, *args, **kwargs):
        method = getattr(self.base_adapter, "context_from_projected_semantic", None)
        if method is None:
            method = getattr(
                self.base_adapter.semantic_projector,
                "context_from_projected_semantic",
                self.base_adapter.semantic_projector,
            )
        return method(*args, **kwargs)

    def forward(
        self,
        meg: torch.Tensor,
        *,
        meg_lengths: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
        known_word_embeddings: Optional[torch.Tensor] = None,
        known_word_mask: Optional[torch.Tensor] = None,
    ) -> MEGAdapterOutput:
        output = self.base_adapter(
            meg,
            meg_lengths=meg_lengths,
            subjects=subjects,
        )
        if known_word_embeddings is None and known_word_mask is None:
            return output
        if known_word_embeddings is None or known_word_mask is None:
            raise ValueError(
                "known_word_embeddings and known_word_mask must be supplied together"
            )
        context = self.word_fusion(
            output.context,
            known_word_embeddings,
            known_word_mask,
        )
        return MEGAdapterOutput(
            context=context,
            context_mask=output.context_mask,
            encoded_sequence=output.encoded_sequence,
        )
