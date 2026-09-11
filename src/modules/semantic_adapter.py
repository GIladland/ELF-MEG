"""Sequence-aware semantic conditioning adapters for ELF."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DelayFusionContextProjector(nn.Module):
    """Fuse ordered delay blocks before a pretrained flat context projector.

    The fusion has one learned logit per delay and semantic feature.  At
    initialization all logits are zero, so the fused vector is the exact mean
    of the delay blocks.  The downstream ``net`` deliberately matches the
    module names and shapes of :class:`SemanticVectorContextProjector`, which
    lets a proven MiniLM-to-ELF adapter initialize it while only the small
    delay fusion is learned.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        num_delay_tokens: int = 4,
        normalize_semantic_output: bool = False,
    ) -> None:
        super().__init__()
        if num_delay_tokens <= 0:
            raise ValueError(f"num_delay_tokens must be positive, got {num_delay_tokens}.")
        if input_dim % num_delay_tokens:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_delay_tokens={num_delay_tokens}."
            )

        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.context_length = int(context_length)
        self.num_delay_tokens = int(num_delay_tokens)
        self.delay_dim = self.input_dim // self.num_delay_tokens
        self.normalize_semantic_output = bool(normalize_semantic_output)
        self.fusion_logits = nn.Parameter(torch.zeros(self.num_delay_tokens, self.delay_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.delay_dim),
            nn.Linear(self.delay_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.context_length * self.context_dim),
        )

    def _reshape_delay_tokens(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        if semantic_vectors.ndim == 2:
            if semantic_vectors.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected flat semantic dimension {self.input_dim}, got {semantic_vectors.shape[-1]}."
                )
            return semantic_vectors.reshape(
                semantic_vectors.shape[0],
                self.num_delay_tokens,
                self.delay_dim,
            )
        if semantic_vectors.ndim == 3:
            expected = (self.num_delay_tokens, self.delay_dim)
            if tuple(semantic_vectors.shape[1:]) != expected:
                raise ValueError(
                    f"Expected semantic delay-token shape [B, {expected[0]}, {expected[1]}], "
                    f"got {tuple(semantic_vectors.shape)}."
                )
            return semantic_vectors
        raise ValueError(
            f"Expected semantic vectors with shape [B, {self.input_dim}] or "
            f"[B, {self.num_delay_tokens}, {self.delay_dim}], got {tuple(semantic_vectors.shape)}."
        )

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits, dim=0)

    def project_semantic(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        delay_tokens = self._reshape_delay_tokens(semantic_vectors)
        fused = (delay_tokens * self.fusion_weights().unsqueeze(0)).sum(dim=1)
        if self.normalize_semantic_output:
            fused = F.normalize(fused.float(), p=2, dim=-1)
        return fused

    def context_from_projected_semantic(self, semantic_vectors: torch.Tensor):
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[-1] != self.delay_dim:
            raise ValueError(
                f"Expected projected semantic shape [B, {self.delay_dim}], got {tuple(semantic_vectors.shape)}."
            )
        context = self.net(semantic_vectors).reshape(
            semantic_vectors.shape[0],
            self.context_length,
            self.context_dim,
        )
        context_mask = torch.ones(
            semantic_vectors.shape[0],
            self.context_length,
            dtype=context.dtype,
            device=context.device,
        )
        return context, context_mask

    def forward(self, semantic_vectors: torch.Tensor):
        return self.context_from_projected_semantic(self.project_semantic(semantic_vectors))


class ResidualMLPFusionContextProjector(nn.Module):
    """Map delayed semantic blocks through a compact residual MLP.

    The base path is the exact mean of contiguous delay blocks.  A zero-output
    initialized MLP learns a nonlinear correction in the target semantic
    space, while the downstream ``net`` remains checkpoint-compatible with a
    pretrained flat MiniLM-to-ELF projector.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 2048,
        mapper_hidden_dim: int = 512,
        dropout: float = 0.0,
        num_delay_tokens: int = 4,
        normalize_semantic_output: bool = False,
    ) -> None:
        super().__init__()
        if num_delay_tokens <= 0:
            raise ValueError(f"num_delay_tokens must be positive, got {num_delay_tokens}.")
        if input_dim % num_delay_tokens:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_delay_tokens={num_delay_tokens}."
            )
        if mapper_hidden_dim <= 0:
            raise ValueError(f"mapper_hidden_dim must be positive, got {mapper_hidden_dim}.")
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.context_length = int(context_length)
        self.num_delay_tokens = int(num_delay_tokens)
        self.delay_dim = self.input_dim // self.num_delay_tokens
        self.normalize_semantic_output = bool(normalize_semantic_output)
        self.semantic_mapper = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, mapper_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mapper_hidden_dim, self.delay_dim),
        )
        final = self.semantic_mapper[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.LayerNorm(self.delay_dim),
            nn.Linear(self.delay_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.context_length * self.context_dim),
        )

    def project_semantic(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected flat semantic shape [B, {self.input_dim}], got {tuple(semantic_vectors.shape)}."
            )
        base = semantic_vectors.reshape(
            semantic_vectors.shape[0], self.num_delay_tokens, self.delay_dim
        ).mean(dim=1)
        projected = base + self.semantic_mapper(semantic_vectors)
        if self.normalize_semantic_output:
            projected = F.normalize(projected.float(), p=2, dim=-1)
        return projected

    def context_from_projected_semantic(self, semantic_vectors: torch.Tensor):
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[-1] != self.delay_dim:
            raise ValueError(
                f"Expected projected semantic shape [B, {self.delay_dim}], got {tuple(semantic_vectors.shape)}."
            )
        context = self.net(semantic_vectors).reshape(
            semantic_vectors.shape[0], self.context_length, self.context_dim
        )
        context_mask = torch.ones(
            semantic_vectors.shape[0],
            self.context_length,
            dtype=context.dtype,
            device=context.device,
        )
        return context, context_mask

    def forward(self, semantic_vectors: torch.Tensor):
        return self.context_from_projected_semantic(self.project_semantic(semantic_vectors))


class ResidualIdentityContextProjector(nn.Module):
    """Learn a nonlinear correction while preserving a same-width oracle interface."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 2048,
        mapper_hidden_dim: int = 1024,
        dropout: float = 0.1,
        normalize_semantic_output: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or mapper_hidden_dim <= 0:
            raise ValueError("input_dim and mapper_hidden_dim must be positive.")
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.context_length = int(context_length)
        self.normalize_semantic_output = bool(normalize_semantic_output)
        self.semantic_mapper = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, mapper_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mapper_hidden_dim, self.input_dim),
        )
        final = self.semantic_mapper[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.context_length * self.context_dim),
        )

    def project_semantic(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected semantic shape [B, {self.input_dim}], got {tuple(semantic_vectors.shape)}."
            )
        projected = semantic_vectors + self.semantic_mapper(semantic_vectors)
        if self.normalize_semantic_output:
            projected = F.normalize(projected.float(), p=2, dim=-1)
        return projected

    def context_from_projected_semantic(self, semantic_vectors: torch.Tensor):
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected projected semantic shape [B, {self.input_dim}], "
                f"got {tuple(semantic_vectors.shape)}."
            )
        context = self.net(semantic_vectors).reshape(
            semantic_vectors.shape[0], self.context_length, self.context_dim
        )
        context_mask = torch.ones(
            semantic_vectors.shape[0],
            self.context_length,
            dtype=context.dtype,
            device=context.device,
        )
        return context, context_mask

    def forward(self, semantic_vectors: torch.Tensor):
        return self.context_from_projected_semantic(self.project_semantic(semantic_vectors))


class ResidualDelayTokenFusionContextProjector(nn.Module):
    """Learn ordered delay interactions while preserving the oracle interface.

    The four delay blocks remain separate tokens inside a compact transformer.
    Its output is a residual correction to the exact arithmetic delay mean, and
    the residual output projection is initialized to zero.  Consequently this
    adapter starts identically to mean fusion and can load the proven flat
    MiniLM-to-ELF ``net`` while training only the small token mapper.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 2048,
        mapper_hidden_dim: int = 1024,
        dropout: float = 0.1,
        num_delay_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        normalize_semantic_output: bool = False,
        preserve_delay_context: bool = False,
    ) -> None:
        super().__init__()
        if num_delay_tokens <= 0:
            raise ValueError(f"num_delay_tokens must be positive, got {num_delay_tokens}.")
        if input_dim % num_delay_tokens:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_delay_tokens={num_delay_tokens}."
            )
        if mapper_hidden_dim <= 0:
            raise ValueError(f"mapper_hidden_dim must be positive, got {mapper_hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.context_length = int(context_length)
        self.num_delay_tokens = int(num_delay_tokens)
        self.delay_dim = self.input_dim // self.num_delay_tokens
        self.normalize_semantic_output = bool(normalize_semantic_output)
        self.preserve_delay_context = bool(preserve_delay_context)
        if self.delay_dim % num_heads:
            raise ValueError(
                f"delay_dim={self.delay_dim} must be divisible by num_heads={num_heads}."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.delay_dim,
            nhead=num_heads,
            dim_feedforward=mapper_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_mapper = nn.ModuleDict(
            {
                "input_norm": nn.LayerNorm(self.delay_dim),
                "delay_encoder": nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                    norm=nn.LayerNorm(self.delay_dim),
                ),
                "delay_embeddings": nn.Embedding(
                    self.num_delay_tokens,
                    self.delay_dim,
                ),
                "output_projection": nn.Linear(self.delay_dim, self.delay_dim),
            }
        )
        nn.init.normal_(
            self.semantic_mapper["delay_embeddings"].weight,
            mean=0.0,
            std=0.02,
        )
        output_projection = self.semantic_mapper["output_projection"]
        nn.init.zeros_(output_projection.weight)
        nn.init.zeros_(output_projection.bias)

        self.net = nn.Sequential(
            nn.LayerNorm(self.delay_dim),
            nn.Linear(self.delay_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.context_length * self.context_dim),
        )
        self.delay_context_projection: nn.Linear | None = None
        if self.preserve_delay_context:
            if self.context_length < self.num_delay_tokens:
                raise ValueError(
                    "context_length must be at least num_delay_tokens when preserving delay context."
                )
            self.delay_context_projection = nn.Linear(self.delay_dim, self.context_dim)
            nn.init.zeros_(self.delay_context_projection.weight)
            nn.init.zeros_(self.delay_context_projection.bias)

    def _reshape_delay_tokens(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        if semantic_vectors.ndim == 2:
            if semantic_vectors.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected flat semantic dimension {self.input_dim}, got {semantic_vectors.shape[-1]}."
                )
            return semantic_vectors.reshape(
                semantic_vectors.shape[0], self.num_delay_tokens, self.delay_dim
            )
        if semantic_vectors.ndim == 3:
            expected = (self.num_delay_tokens, self.delay_dim)
            if tuple(semantic_vectors.shape[1:]) != expected:
                raise ValueError(
                    f"Expected delay-token shape [B, {expected[0]}, {expected[1]}], "
                    f"got {tuple(semantic_vectors.shape)}."
                )
            return semantic_vectors
        raise ValueError(
            f"Expected [B, {self.input_dim}] or [B, {self.num_delay_tokens}, {self.delay_dim}], "
            f"got {tuple(semantic_vectors.shape)}."
        )

    def project_semantic(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        delay_tokens = self._reshape_delay_tokens(semantic_vectors)
        base = delay_tokens.mean(dim=1)
        encoded = self.semantic_mapper["input_norm"](delay_tokens)
        delay_embeddings = self.semantic_mapper["delay_embeddings"].weight.unsqueeze(0)
        encoded = self.semantic_mapper["delay_encoder"](encoded + delay_embeddings)
        residual = self.semantic_mapper["output_projection"](encoded.mean(dim=1))
        projected = base + residual
        if self.normalize_semantic_output:
            projected = F.normalize(projected.float(), p=2, dim=-1)
        return projected

    def context_from_projected_semantic(self, semantic_vectors: torch.Tensor):
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[-1] != self.delay_dim:
            raise ValueError(
                f"Expected projected semantic shape [B, {self.delay_dim}], "
                f"got {tuple(semantic_vectors.shape)}."
            )
        context = self.net(semantic_vectors).reshape(
            semantic_vectors.shape[0], self.context_length, self.context_dim
        )
        context_mask = torch.ones(
            semantic_vectors.shape[0],
            self.context_length,
            dtype=context.dtype,
            device=context.device,
        )
        return context, context_mask

    def condition_context(self, semantic_vectors: torch.Tensor):
        """Return fused semantics plus context with optional explicit delay slots.

        The base 64-token context remains exactly checkpoint-compatible. When
        enabled, a zero-initialized projection adds each centered delay block
        to one of the first four context tokens, so initialization is an exact
        identity while training can preserve information lost by mean fusion.
        """

        delay_tokens = self._reshape_delay_tokens(semantic_vectors)
        projected = self.project_semantic(delay_tokens)
        context, context_mask = self.context_from_projected_semantic(projected)
        if self.delay_context_projection is not None:
            centered_delays = delay_tokens - delay_tokens.mean(dim=1, keepdim=True)
            delay_context = self.delay_context_projection(centered_delays)
            context = context.clone()
            context[:, : self.num_delay_tokens] = (
                context[:, : self.num_delay_tokens] + delay_context
            )
        return projected, context, context_mask

    def forward(self, semantic_vectors: torch.Tensor):
        _, context, context_mask = self.condition_context(semantic_vectors)
        return context, context_mask


class DelayTokenContextProjector(nn.Module):
    """Project ordered delay blocks into ELF context tokens.

    MRI2SEM stores delayed Tang-GPT targets as one flat vector with contiguous
    delay blocks, for example ``[delay1, delay2, delay3, delay4]``. This
    adapter restores that structure, encodes the ordered delay tokens, and
    resamples them with learned ELF-context queries instead of averaging the
    blocks or treating the concatenation as an unstructured vector.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        num_delay_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if num_delay_tokens <= 0:
            raise ValueError(f"num_delay_tokens must be positive, got {num_delay_tokens}.")
        if input_dim % num_delay_tokens:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_delay_tokens={num_delay_tokens}."
            )
        if context_dim % num_heads:
            raise ValueError(f"context_dim={context_dim} must be divisible by num_heads={num_heads}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.context_length = int(context_length)
        self.num_delay_tokens = int(num_delay_tokens)
        self.delay_dim = self.input_dim // self.num_delay_tokens

        self.delay_norm = nn.LayerNorm(self.delay_dim)
        self.delay_projection = nn.Linear(self.delay_dim, self.context_dim)
        self.delay_embeddings = nn.Parameter(torch.empty(1, self.num_delay_tokens, self.context_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.context_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.delay_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.context_dim),
        )

        self.context_queries = nn.Parameter(torch.empty(1, self.context_length, self.context_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.context_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_resampler = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.context_dim),
        )
        nn.init.normal_(self.delay_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.context_queries, mean=0.0, std=0.02)

    def _reshape_delay_tokens(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        if semantic_vectors.ndim == 2:
            if semantic_vectors.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected flat semantic dimension {self.input_dim}, got {semantic_vectors.shape[-1]}."
                )
            return semantic_vectors.reshape(
                semantic_vectors.shape[0],
                self.num_delay_tokens,
                self.delay_dim,
            )
        if semantic_vectors.ndim == 3:
            expected = (self.num_delay_tokens, self.delay_dim)
            if tuple(semantic_vectors.shape[1:]) != expected:
                raise ValueError(
                    f"Expected semantic delay-token shape [B, {expected[0]}, {expected[1]}], "
                    f"got {tuple(semantic_vectors.shape)}."
                )
            return semantic_vectors
        raise ValueError(
            f"Expected semantic vectors with shape [B, {self.input_dim}] or "
            f"[B, {self.num_delay_tokens}, {self.delay_dim}], got {tuple(semantic_vectors.shape)}."
        )

    def forward(self, semantic_vectors: torch.Tensor):
        delay_tokens = self._reshape_delay_tokens(semantic_vectors)
        memory = self.delay_projection(self.delay_norm(delay_tokens))
        memory = self.delay_encoder(memory + self.delay_embeddings)
        queries = self.context_queries.expand(delay_tokens.shape[0], -1, -1)
        context = self.context_resampler(tgt=queries, memory=memory)
        context_mask = torch.ones(
            delay_tokens.shape[0],
            self.context_length,
            dtype=context.dtype,
            device=context.device,
        )
        return context, context_mask
