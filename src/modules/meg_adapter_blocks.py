"""Reusable MEG adapter blocks derived from the MEG2SEM backbone.

These layers keep the core inductive bias from:
`/Users/gilad/Desktop/Projects/PNPL/MEG2SEM/brainmagick/model_utils.py`
but are trimmed to the pieces needed for the ELF conditioning adapter.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class SubjectLayers(nn.Module):
    """Per-subject channel mixer applied before the shared temporal backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            if in_channels != out_channels:
                raise ValueError("Identity init requires in_channels == out_channels.")
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= in_channels ** -0.5

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        subjects = subjects.to(device=x.device, dtype=torch.long)
        weights = self.weights.index_select(0, subjects)
        return torch.einsum("bct,bcd->bdt", x, weights)

    def extra_repr(self) -> str:
        n_subjects, in_channels, out_channels = self.weights.shape
        return f"in_channels={in_channels}, out_channels={out_channels}, n_subjects={n_subjects}"


class ConvSequence(nn.Module):
    """Dilated temporal conv stack with residual skips and optional GLU blocks."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        kernel: int = 3,
        dilation_growth: int = 1,
        dilation_period: Optional[int] = None,
        stride: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        skip: bool = False,
        glu_every: int = 0,
        glu_context: int = 0,
        activation: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least an input and output width.")
        if dilation_growth > 1 and kernel % 2 == 0:
            raise ValueError("Dilated convs require an odd kernel size.")

        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        activation = activation or nn.GELU

        dilation = 1
        for idx, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            if dilation_period and idx % dilation_period == 0:
                dilation = 1

            padding = kernel // 2 * dilation
            layers: list[nn.Module] = [nn.Conv1d(chin, chout, kernel, stride, padding, dilation=dilation)]
            dilation *= dilation_growth

            if batch_norm:
                layers.append(nn.BatchNorm1d(chout))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))

            self.sequence.append(nn.Sequential(*layers))

            if glu_every and (idx + 1) % glu_every == 0:
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, 2 * chout, 1 + 2 * glu_context, padding=glu_context),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, glu in zip(self.sequence, self.glus):
            residual = x
            x = block(x)
            if self.skip and x.shape == residual.shape:
                x = x + residual
            if glu is not None:
                x = glu(x)
        return x
