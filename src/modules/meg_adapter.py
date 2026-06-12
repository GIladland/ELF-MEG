"""MEG-to-ELF conditioning adapter.

This adapter reuses the main MEG2SEM encoder pattern:
channel projection -> 1x1 conv -> subject mixing -> dilated temporal convs
-> temporal attention -> fixed-length context pooling.

Unlike the original MEG2SEM backbone, this module returns a sequence of
conditioning latents shaped `(batch, context_length, context_dim)` so it can
be prepended as an ELF condition prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.meg_adapter_blocks import ConvSequence, SubjectLayers


@dataclass
class MEGAdapterOutput:
    context: torch.Tensor
    context_mask: torch.Tensor
    encoded_sequence: torch.Tensor


class TimeChannelLayerNorm(nn.Module):
    """Normalize channel features independently at each time step."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalSelfAttention(nn.Module):
    """Single temporal self-attention block over `[B, C, T]` features."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_time = x.transpose(1, 2)
        attn_out, _ = self.attn(x_time, x_time, x_time, key_padding_mask=key_padding_mask)
        return self.norm(attn_out + x_time).transpose(1, 2)


class MEGContextAdapter(nn.Module):
    """Encode MEG segments into a fixed-length ELF condition prefix."""

    def __init__(
        self,
        *,
        in_channels: int,
        context_dim: int,
        context_length: int,
        n_subjects: int = 1,
        merger_channels: int = 306,
        conv_channels: int = 320,
        num_conv_layers: int = 10,
        kernel_size: int = 3,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        dropout: float = 0.0,
        attention_heads: int = 8,
        use_subject_layers: bool = True,
        use_temporal_attention: bool = True,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if use_temporal_attention and context_dim % attention_heads != 0:
            raise ValueError("context_dim must be divisible by attention_heads.")

        self.context_dim = context_dim
        self.context_length = context_length
        self.n_subjects = n_subjects
        self.use_subject_layers = use_subject_layers
        self.use_temporal_attention = use_temporal_attention

        self.project = nn.Linear(in_channels, merger_channels)
        self.init_conv = nn.Conv1d(merger_channels, merger_channels, kernel_size=1)
        self.subject_layer = (
            SubjectLayers(merger_channels, merger_channels, n_subjects, init_id=False)
            if use_subject_layers
            else None
        )

        channels = [merger_channels] + [conv_channels for _ in range(num_conv_layers)]
        self.conv_blocks = ConvSequence(
            channels,
            kernel=kernel_size,
            stride=1,
            dilation_growth=dilation_growth,
            dilation_period=dilation_period,
            batch_norm=True,
            dropout=dropout,
            skip=True,
            glu_every=2,
            activation=nn.GELU,
        )

        proj_hidden = 2 * conv_channels
        if norm_type == "batch":
            norm_layer: nn.Module = nn.BatchNorm1d(proj_hidden)
        elif norm_type == "layer":
            norm_layer = TimeChannelLayerNorm(proj_hidden)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.final_convs = nn.Sequential(
            nn.Conv1d(conv_channels, proj_hidden, kernel_size=1),
            norm_layer,
            nn.GELU(),
            nn.Conv1d(proj_hidden, context_dim, kernel_size=1),
        )

        self.temporal_attention = (
            TemporalSelfAttention(context_dim, num_heads=attention_heads, dropout=dropout)
            if use_temporal_attention
            else None
        )

    @staticmethod
    def _lengths_to_padding_mask(
        lengths: Optional[torch.Tensor],
        max_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        lengths = lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_length)
        time_idx = torch.arange(max_length, device=device).unsqueeze(0)
        return time_idx >= lengths.unsqueeze(1)

    def _pool_to_context_tokens(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            pooled = F.adaptive_avg_pool1d(x, self.context_length)
            context = pooled.transpose(1, 2)
            context_mask = torch.ones(
                context.shape[0],
                self.context_length,
                dtype=torch.bool,
                device=context.device,
            )
            return context, context_mask

        valid = (~padding_mask).to(dtype=x.dtype).unsqueeze(1)
        pooled_num = F.adaptive_avg_pool1d(x * valid, self.context_length)
        pooled_den = F.adaptive_avg_pool1d(valid, self.context_length).clamp_min(1e-6)
        pooled = pooled_num / pooled_den
        pooled_valid = F.adaptive_max_pool1d(valid, self.context_length) > 0
        return pooled.transpose(1, 2), pooled_valid.squeeze(1)

    def forward(
        self,
        meg: torch.Tensor,
        *,
        meg_lengths: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
    ) -> MEGAdapterOutput:
        if meg.ndim != 3:
            raise ValueError(f"Expected MEG input with shape [B, C, T], got {tuple(meg.shape)}")

        batch_size = meg.shape[0]
        x = self.project(meg.transpose(1, 2)).transpose(1, 2)
        x = self.init_conv(x)

        if self.subject_layer is not None:
            if subjects is None:
                subjects = torch.zeros(batch_size, dtype=torch.long, device=meg.device)
            x = self.subject_layer(x, subjects)

        x = self.conv_blocks(x)
        x = self.final_convs(x)

        padding_mask = self._lengths_to_padding_mask(meg_lengths, x.shape[-1], x.device)
        if self.temporal_attention is not None:
            x = self.temporal_attention(x, key_padding_mask=padding_mask)

        context, context_mask = self._pool_to_context_tokens(x, padding_mask)
        return MEGAdapterOutput(
            context=context,
            context_mask=context_mask,
            encoded_sequence=x.transpose(1, 2),
        )
