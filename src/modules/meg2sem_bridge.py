"""Adapters for routing trained MEG2SEM checkpoints into ELF."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.meg_adapter import MEGAdapterOutput, TimeChannelLayerNorm
from modules.meg_adapter_blocks import ConvSequence, SubjectLayers


NormalizationMode = Literal["auto", "never", "always"]


@dataclass
class MEG2SEMLoadInfo:
    checkpoint_format: str
    embedding_dim: int
    input_dim: int
    n_subjects: int
    normalize_output: bool
    missing_keys: list[str]
    unexpected_keys: list[str]


class MEG2SEMBrainEncoder(nn.Module):
    """MEG2SEM `BrainModel` clone with checkpoint-compatible module names."""

    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int,
        n_subjects: int,
        dropout: float = 0.0,
        aggregation: str = "attention",
        attention_heads: int = 4,
        norm_type: str = "batch",
        use_projection_head: bool = False,
    ) -> None:
        super().__init__()
        merger_channels = 306
        conv_channels = 320
        self.project = nn.Linear(input_dim, merger_channels)
        self.init_conv = nn.Conv1d(merger_channels, merger_channels, 1)
        self.n_subjects = n_subjects
        self.subject_layer = SubjectLayers(merger_channels, merger_channels, n_subjects, False)
        self.use_projection_head = use_projection_head

        channels = [merger_channels] + [conv_channels for _ in range(10)]
        self.conv_blocks = ConvSequence(
            channels,
            kernel=3,
            stride=1,
            dilation_growth=2,
            dilation_period=5,
            batch_norm=True,
            dropout=dropout,
            skip=True,
            glu_every=2,
            glu_context=1,
            activation=nn.GELU,
        )

        if norm_type == "batch":
            norm_layer: nn.Module = nn.BatchNorm1d(2 * conv_channels)
        elif norm_type == "layer":
            norm_layer = TimeChannelLayerNorm(2 * conv_channels)
        else:
            raise ValueError(f"Unsupported MEG2SEM norm_type: {norm_type}")

        self.final_convs = nn.Sequential(
            nn.Conv1d(conv_channels, 2 * conv_channels, 1),
            norm_layer,
            nn.GELU(),
            nn.ConvTranspose1d(2 * conv_channels, embedding_dim, 3, 1, 0),
        )

        aggregation = str(aggregation).lower()
        if aggregation == "avg":
            self.final_pooling = nn.AdaptiveAvgPool1d(1)
        elif aggregation == "attention":
            self.temporal_attention = MEG2SEMTemporalSelfAttention(
                embedding_dim,
                num_heads=attention_heads,
            )
            self.final_pooling = self.temporal_attention
        else:
            raise ValueError(
                "MEG2SEM checkpoint loading currently supports aggregation='avg' or 'attention', "
                f"got {aggregation!r}."
            )

        if use_projection_head:
            if norm_type == "layer":
                proj_norm = nn.LayerNorm(1024, eps=1e-6)
                proj_out_norm: nn.Module = nn.LayerNorm(embedding_dim, eps=1e-6)
            elif norm_type == "batch":
                proj_norm = nn.BatchNorm1d(1024)
                proj_out_norm = nn.Identity()
            else:
                raise ValueError(f"Unsupported MEG2SEM norm_type: {norm_type}")
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, 1024, bias=False),
                proj_norm,
                nn.ReLU(),
                nn.Linear(1024, embedding_dim, bias=False),
                proj_out_norm,
            )
        else:
            self.projection_head = nn.Identity()

    @staticmethod
    def _lengths_to_padding_mask(
        lengths: Optional[torch.Tensor],
        *,
        input_length: int,
        current_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        lengths = lengths.to(device=device, dtype=torch.long)
        length_delta = current_length - input_length
        effective = (lengths + length_delta).clamp(min=0, max=current_length)
        time_idx = torch.arange(current_length, device=device).unsqueeze(0)
        return time_idx >= effective.unsqueeze(1)

    def forward(
        self,
        meg: torch.Tensor,
        *,
        meg_lengths: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if meg.ndim != 3:
            raise ValueError(f"Expected MEG input with shape [B, C, T], got {tuple(meg.shape)}")
        if subjects is None:
            subjects = torch.zeros(meg.shape[0], dtype=torch.long, device=meg.device)

        input_length = int(meg.shape[-1])
        x = self.project(meg.transpose(1, 2)).transpose(1, 2)
        x = self.init_conv(x)
        x = self.subject_layer(x, subjects)
        x = self.conv_blocks(x)
        x = self.final_convs(x)

        if isinstance(self.final_pooling, MEG2SEMTemporalSelfAttention):
            key_padding_mask = self._lengths_to_padding_mask(
                meg_lengths,
                input_length=input_length,
                current_length=int(x.shape[-1]),
                device=x.device,
            )
            x = self.final_pooling(x, key_padding_mask=key_padding_mask)
        else:
            x = self.final_pooling(x).squeeze(-1)
        return self.projection_head(x)


class MEG2SEMTemporalSelfAttention(nn.Module):
    """MEG2SEM attention pooling block over `[B, C, T]` features."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_time = x.transpose(1, 2)
        attn_out, _ = self.attn(x_time, x_time, x_time, key_padding_mask=key_padding_mask)
        hidden = self.norm(attn_out + x_time)
        if key_padding_mask is None:
            return hidden.mean(dim=1)

        valid = (~key_padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class MinimalMEG2SEMRegressor(nn.Module):
    """Minimal flattened MEG->semantic regressor used by the PNPL smoke trainer."""

    def __init__(self, *, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(
        self,
        meg: torch.Tensor,
        *,
        meg_lengths: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del meg_lengths, subjects
        flattened = meg.reshape(meg.shape[0], -1)
        if flattened.shape[1] != self.in_features:
            raise ValueError(
                f"MEG2SEM MLP expected {self.in_features} flattened MEG features, "
                f"got {flattened.shape[1]} from input shape {tuple(meg.shape)}."
            )
        return self.net(meg)


class MEG2SEMToELFContextAdapter(nn.Module):
    """Use MEG2SEM semantic predictions as the condition source for ELF."""

    def __init__(
        self,
        *,
        meg2sem: nn.Module,
        semantic_projector: nn.Module,
        normalize_semantic_output: bool,
        residual_mean: Optional[torch.Tensor] = None,
        residual_target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.meg2sem = meg2sem
        self.semantic_projector = semantic_projector
        self.normalize_semantic_output = normalize_semantic_output
        self.residual_target_scale = float(residual_target_scale)
        if residual_mean is None:
            self.register_buffer("residual_mean", None)
        else:
            self.register_buffer("residual_mean", residual_mean.detach().float().reshape(1, -1))

    @property
    def uses_residual_reconstruction(self) -> bool:
        return self.residual_mean is not None

    def semantic_projector_input(
        self,
        meg: torch.Tensor,
        *,
        meg_lengths: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_output = self.meg2sem(meg, meg_lengths=meg_lengths, subjects=subjects)
        if self.uses_residual_reconstruction:
            residual = raw_output.float() / max(self.residual_target_scale, 1e-12)
            semantic_vectors = F.normalize(self.residual_mean.to(device=raw_output.device) + residual, p=2, dim=-1)
        else:
            semantic_vectors = raw_output
            if self.normalize_semantic_output:
                semantic_vectors = F.normalize(semantic_vectors, p=2, dim=-1)
        return semantic_vectors, raw_output

    def forward(
        self,
        meg: torch.Tensor,
        *,
        meg_lengths: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
    ) -> MEGAdapterOutput:
        semantic_vectors, _ = self.semantic_projector_input(
            meg,
            meg_lengths=meg_lengths,
            subjects=subjects,
        )
        context, context_mask = self.semantic_projector(semantic_vectors)
        return MEGAdapterOutput(
            context=context,
            context_mask=context_mask.to(dtype=torch.bool),
            encoded_sequence=semantic_vectors.unsqueeze(1),
        )


def load_residual_mean(path: str, *, key: str = "train_mean") -> torch.Tensor:
    data = np.load(path, allow_pickle=True)
    if key not in data.files:
        raise KeyError(f"{path} missing residual component key {key!r}; keys={data.files}")
    return torch.as_tensor(np.asarray(data[key], dtype=np.float32), dtype=torch.float32)


def _payload_arg(payload: Mapping[str, object], key: str, default):
    args = payload.get("args")
    if isinstance(args, Mapping) and key in args:
        return args[key]
    hparams = payload.get("hyper_parameters")
    if isinstance(hparams, Mapping) and key in hparams:
        return hparams[key]
    return default


def _resolve_normalize_output(payload: Mapping[str, object], mode: NormalizationMode) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(_payload_arg(payload, "normalize_embeddings", False))


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def _load_lightning_meg2sem(
    payload: Mapping[str, object],
    *,
    output_normalization: NormalizationMode,
    device: torch.device,
) -> tuple[nn.Module, MEG2SEMLoadInfo]:
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Lightning MEG2SEM checkpoint is missing a mapping `state_dict`.")

    target_type = str(_payload_arg(payload, "target_type", "semantic_vector")).lower()
    if target_type not in {"semantic", "semantic_vector", "semantic_residual", "residual", "ada_residual", "mean_residual"}:
        raise ValueError(
            f"MEG2SEM checkpoint target_type={target_type!r} is not a semantic-vector checkpoint."
        )

    brain_state = _strip_prefix(state_dict, "brain_model.")
    project_weight = brain_state.get("project.weight")
    final_bias = brain_state.get("final_convs.3.bias")
    subject_weights = brain_state.get("subject_layer.weights")
    input_dim = (
        int(project_weight.shape[1])
        if project_weight is not None
        else int(_payload_arg(payload, "input_dim", 306))
    )
    embedding_dim = (
        int(final_bias.shape[0])
        if final_bias is not None
        else int(_payload_arg(payload, "embedding_dim", 384))
    )
    n_subjects = (
        int(subject_weights.shape[0])
        if subject_weights is not None
        else int(_payload_arg(payload, "n_subjects", 1))
    )
    model = MEG2SEMBrainEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        n_subjects=n_subjects,
        dropout=float(_payload_arg(payload, "dropout", 0.0)),
        aggregation=str(_payload_arg(payload, "aggregation", "attention")),
        attention_heads=int(_payload_arg(payload, "attention_heads", 4)),
        norm_type=str(_payload_arg(payload, "model_norm", "batch")),
        use_projection_head=bool(_payload_arg(payload, "use_projection_head", False)),
    ).to(device)

    result = model.load_state_dict(brain_state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if any(not key.startswith("final_pooling.") for key in unexpected):
        raise ValueError(f"Unexpected MEG2SEM brain_model checkpoint keys: {unexpected}")

    normalize_output = _resolve_normalize_output(payload, output_normalization)
    return model, MEG2SEMLoadInfo(
        checkpoint_format="lightning_semantic_decoder",
        embedding_dim=embedding_dim,
        input_dim=input_dim,
        n_subjects=n_subjects,
        normalize_output=normalize_output,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def _load_minimal_meg2sem(
    payload: Mapping[str, object],
    *,
    output_normalization: NormalizationMode,
    device: torch.device,
) -> tuple[nn.Module, MEG2SEMLoadInfo]:
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Minimal MEG2SEM checkpoint is missing a mapping `model_state_dict`.")

    args = payload.get("args") if isinstance(payload.get("args"), Mapping) else {}
    first_weight = state_dict.get("net.1.weight")
    last_weight = state_dict.get("net.3.weight")
    if first_weight is None or last_weight is None:
        raise ValueError("Minimal MEG2SEM checkpoint must contain net.1.weight and net.3.weight.")
    in_features = int(first_weight.shape[1])
    hidden_dim = int(first_weight.shape[0])
    embedding_dim = int(last_weight.shape[0])
    input_dim = int(args.get("input_dim", 0) or 0)
    model = MinimalMEG2SEMRegressor(
        in_features=in_features,
        hidden_dim=hidden_dim,
        out_features=embedding_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    normalize_output = _resolve_normalize_output(payload, output_normalization)
    return model, MEG2SEMLoadInfo(
        checkpoint_format="minimal_mlp",
        embedding_dim=embedding_dim,
        input_dim=input_dim,
        n_subjects=1,
        normalize_output=normalize_output,
        missing_keys=[],
        unexpected_keys=[],
    )


def load_meg2sem_model(
    checkpoint_path: str,
    *,
    output_normalization: NormalizationMode,
    device: torch.device,
) -> tuple[nn.Module, MEG2SEMLoadInfo]:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported MEG2SEM checkpoint payload type: {type(payload)!r}")
    if "state_dict" in payload:
        return _load_lightning_meg2sem(
            payload,
            output_normalization=output_normalization,
            device=device,
        )
    if "model_state_dict" in payload:
        return _load_minimal_meg2sem(
            payload,
            output_normalization=output_normalization,
            device=device,
        )
    raise ValueError(
        f"Unsupported MEG2SEM checkpoint keys in {checkpoint_path}: {sorted(payload.keys())}"
    )
