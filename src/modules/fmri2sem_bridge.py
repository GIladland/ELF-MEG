"""Differentiable fMRI-to-semantic-to-ELF adapter components."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """Checkpoint-compatible MRI2SEM residual block."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        inner = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class MindEyeStyleMLP(nn.Module):
    """Checkpoint-compatible MRI2SEM encoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        res_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(res_blocks)],
            nn.LayerNorm(hidden_dim),
        )
        self.projector = nn.Linear(hidden_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(inputs)), p=2, dim=-1)


def load_mri2sem_model(
    checkpoint_path: str,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 2048,
    res_blocks: int = 4,
    dropout: float = 0.1,
    projector_mismatch: str = "error",
) -> MindEyeStyleMLP:
    """Load either a native MRI2SEM checkpoint or a prior E2E adapter checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported MRI2SEM checkpoint payload: {type(checkpoint).__name__}."
        )
    state = checkpoint.get("model")
    if state is None:
        adapter_state = checkpoint.get("adapter_state_dict")
        if isinstance(adapter_state, dict):
            prefix = "fmri2sem."
            state = {
                key[len(prefix):]: value
                for key, value in adapter_state.items()
                if key.startswith(prefix)
            }
    if not state:
        raise ValueError(
            f"{checkpoint_path} has neither a native 'model' state nor fmri2sem adapter weights."
        )

    state = dict(state)
    if projector_mismatch == "mean-blocks":
        projector_weight = state.get("projector.weight")
        projector_bias = state.get("projector.bias")
        if projector_weight is None or projector_bias is None:
            raise ValueError("MRI2SEM checkpoint is missing projector weights.")
        source_dim = int(projector_weight.shape[0])
        if source_dim != output_dim:
            if source_dim % output_dim:
                raise ValueError(
                    f"Cannot mean {source_dim} MRI2SEM outputs into output_dim={output_dim}."
                )
            block_count = source_dim // output_dim
            state["projector.weight"] = projector_weight.reshape(
                block_count, output_dim, projector_weight.shape[1]
            ).mean(dim=0)
            state["projector.bias"] = projector_bias.reshape(
                block_count, output_dim
            ).mean(dim=0)
    elif projector_mismatch != "error":
        raise ValueError(f"Unsupported MRI2SEM projector mismatch mode: {projector_mismatch}")

    model = MindEyeStyleMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        res_blocks=res_blocks,
        dropout=dropout,
    )
    model.load_state_dict(state, strict=True)
    return model


def configure_mri2sem_trainable(model: MindEyeStyleMLP, mode: str) -> list[str]:
    """Freeze the encoder, then expose only the requested MRI2SEM stage."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names: list[str] = []

    if mode == "frozen":
        return trainable_names
    if mode == "projector":
        modules = [("projector", model.projector)]
    elif mode == "last_block":
        residual_blocks = [
            (index, module)
            for index, module in enumerate(model.encoder)
            if isinstance(module, ResidualMLPBlock)
        ]
        if not residual_blocks:
            raise ValueError("MRI2SEM encoder has no residual block to unfreeze.")
        block_index, block = residual_blocks[-1]
        modules = [(f"encoder.{block_index}", block), ("projector", model.projector)]
    elif mode == "all":
        modules = [("encoder", model.encoder), ("projector", model.projector)]
    else:
        raise ValueError(f"Unsupported MRI2SEM unfreeze mode: {mode}")

    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(True)
            trainable_names.append(f"{prefix}.{name}")
    return trainable_names


class FMRI2SEMToELFContextAdapter(nn.Module):
    """Compose raw fMRI -> delayed semantics -> ELF context without detaching."""

    def __init__(self, fmri2sem: MindEyeStyleMLP, semantic_projector: nn.Module) -> None:
        super().__init__()
        self.fmri2sem = fmri2sem
        self.semantic_projector = semantic_projector

    @property
    def normalize_semantic_output(self) -> bool:
        return bool(getattr(self.semantic_projector, "normalize_semantic_output", False))

    def project_semantic(self, fmri_vectors: torch.Tensor) -> torch.Tensor:
        delayed_semantic = self.fmri2sem(fmri_vectors)
        return self.semantic_projector.project_semantic(delayed_semantic)

    def context_from_projected_semantic(self, semantic_vectors: torch.Tensor):
        return self.semantic_projector.context_from_projected_semantic(semantic_vectors)

    def condition_context(self, fmri_vectors: torch.Tensor):
        delayed_semantic = self.fmri2sem(fmri_vectors)
        condition_context = getattr(self.semantic_projector, "condition_context", None)
        if condition_context is not None:
            return condition_context(delayed_semantic)
        projected = self.semantic_projector.project_semantic(delayed_semantic)
        context, context_mask = self.semantic_projector.context_from_projected_semantic(projected)
        return projected, context, context_mask

    def forward(self, fmri_vectors: torch.Tensor):
        _, context, context_mask = self.condition_context(fmri_vectors)
        return context, context_mask


class ContentLogitHead(nn.Module):
    """Checkpoint-compatible MLP used by the held-out lexical probe."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if hidden_dim <= 0:
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class OrderedWordLogitHead(nn.Module):
    """Checkpoint-compatible ten-position word head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        positions: int,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.positions = int(positions)
        self.vocabulary_size = int(vocabulary_size)
        if hidden_dim > 0:
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            feature_dim = hidden_dim
        else:
            self.trunk = nn.LayerNorm(input_dim)
            feature_dim = input_dim
        self.output = nn.Linear(feature_dim, positions * vocabulary_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(inputs)).reshape(
            len(inputs), self.positions, self.vocabulary_size
        )


class FMRI2SEMLexicalToELFContextAdapter(FMRI2SEMToELFContextAdapter):
    """Add a gated content-word context without distorting semantic alignment.

    The lexical classifier and its train-only semantic word prototypes are
    frozen.  Only a zero-initialized per-context-slot gate is learned, so step
    zero is exactly the underlying fMRI -> semantic -> ELF model.
    """

    def __init__(
        self,
        fmri2sem: MindEyeStyleMLP,
        semantic_projector: nn.Module,
        *,
        content_head: nn.Module,
        lexical_prototypes: torch.Tensor,
        lexical_vocabulary: list[str] | None = None,
        lexical_logit_prior: torch.Tensor | None = None,
        lexical_token_ids: torch.Tensor | None = None,
        lexical_topk: int = 5,
        lexical_temperature: float = 0.5,
        lexical_prior_subtraction: float = 0.0,
        lexical_decode_bias_strength: float = 0.0,
        lexical_max_prior_probability: float = 1.0,
        lexical_decode_bias_once: bool = False,
        lexical_decode_bias_mode: str = "sequence",
        lexical_context_mode: str = "mixture",
        lexical_decode_bias_max_positions: int = 0,
        ordered_head: nn.Module | None = None,
        ordered_log_prior: torch.Tensor | None = None,
        ordered_token_ids: torch.Tensor | None = None,
        ordered_prototypes: torch.Tensor | None = None,
        ordered_prior_subtraction: float = 0.0,
        ordered_min_prior_probability: float = 0.0,
        ordered_min_margin: float = 0.0,
        ordered_decode_bias_strength: float = 0.0,
        ordered_decode_max_positions: int = 0,
    ) -> None:
        super().__init__(fmri2sem, semantic_projector)
        if lexical_prototypes.ndim != 2:
            raise ValueError(
                f"lexical_prototypes must be [V, D], got {tuple(lexical_prototypes.shape)}"
            )
        if lexical_topk <= 0 or lexical_topk > lexical_prototypes.shape[0]:
            raise ValueError(f"Invalid lexical_topk={lexical_topk}")
        if lexical_temperature <= 0.0:
            raise ValueError("lexical_temperature must be positive")
        context_length = int(getattr(semantic_projector, "context_length"))
        semantic_input_dim = int(
            getattr(semantic_projector, "input_dim", lexical_prototypes.shape[1])
        )
        lexical_prototype_dim = int(lexical_prototypes.shape[1])
        if semantic_input_dim % lexical_prototype_dim:
            raise ValueError(
                "The semantic projector input must be an integer number of lexical "
                f"prototype blocks: projector={semantic_input_dim} "
                f"prototype={lexical_prototype_dim}."
            )
        self.lexical_repeat_factor = semantic_input_dim // lexical_prototype_dim
        self.content_head = content_head
        for parameter in self.content_head.parameters():
            parameter.requires_grad_(False)
        self.content_head.eval()
        self.register_buffer(
            "lexical_prototypes",
            F.normalize(lexical_prototypes.detach().float(), p=2, dim=-1),
        )
        if lexical_vocabulary is not None and len(lexical_vocabulary) != lexical_prototypes.shape[0]:
            raise ValueError("lexical_vocabulary must match lexical_prototypes")
        self.lexical_vocabulary = (
            [str(word) for word in lexical_vocabulary]
            if lexical_vocabulary is not None
            else None
        )
        self.lexical_topk = int(lexical_topk)
        self.lexical_temperature = float(lexical_temperature)
        self.lexical_prior_subtraction = float(lexical_prior_subtraction)
        self.lexical_decode_bias_strength = float(lexical_decode_bias_strength)
        self.lexical_decode_bias_max_positions = int(
            lexical_decode_bias_max_positions
        )
        self.lexical_max_prior_probability = float(lexical_max_prior_probability)
        self.lexical_decode_bias_once = bool(lexical_decode_bias_once)
        if lexical_decode_bias_mode not in {"scattered", "sequence"}:
            raise ValueError(
                f"Unsupported lexical_decode_bias_mode={lexical_decode_bias_mode!r}"
            )
        self.lexical_decode_bias_mode = lexical_decode_bias_mode
        if lexical_context_mode not in {"mixture", "partitioned"}:
            raise ValueError(f"Unsupported lexical_context_mode={lexical_context_mode!r}")
        self.lexical_context_mode = lexical_context_mode
        if self.lexical_prior_subtraction < 0.0:
            raise ValueError("lexical_prior_subtraction must be non-negative")
        if self.lexical_decode_bias_strength < 0.0:
            raise ValueError("lexical_decode_bias_strength must be non-negative")
        if self.lexical_decode_bias_max_positions < 0:
            raise ValueError("lexical_decode_bias_max_positions must be non-negative")
        if not 0.0 < self.lexical_max_prior_probability <= 1.0:
            raise ValueError("lexical_max_prior_probability must be in (0, 1]")
        if lexical_logit_prior is None:
            lexical_logit_prior = torch.full(
                (lexical_prototypes.shape[0],), 0.5, dtype=torch.float32
            )
            if self.lexical_prior_subtraction:
                raise ValueError(
                    "lexical_logit_prior is required when lexical_prior_subtraction is nonzero"
                )
        if lexical_logit_prior.ndim != 1 or lexical_logit_prior.shape[0] != lexical_prototypes.shape[0]:
            raise ValueError(
                "lexical_logit_prior must be [V] and match lexical_prototypes"
            )
        self.register_buffer(
            "lexical_logit_prior",
            lexical_logit_prior.detach().float().clamp(1e-5, 1.0 - 1e-5),
        )
        if lexical_token_ids is None:
            lexical_token_ids = torch.empty(
                (lexical_prototypes.shape[0], 0), dtype=torch.long
            )
        if lexical_token_ids.ndim != 2 or lexical_token_ids.shape[0] != lexical_prototypes.shape[0]:
            raise ValueError("lexical_token_ids must be [V, T] and match lexical_prototypes")
        self.register_buffer("lexical_token_ids", lexical_token_ids.detach().long())
        self.lexical_context_gate = nn.Parameter(torch.zeros(context_length))
        self.ordered_head = ordered_head
        self.ordered_context_gate = None
        self.ordered_prior_subtraction = float(ordered_prior_subtraction)
        self.ordered_min_prior_probability = float(
            ordered_min_prior_probability
        )
        self.ordered_min_margin = float(ordered_min_margin)
        self.ordered_decode_bias_strength = float(ordered_decode_bias_strength)
        self.ordered_decode_max_positions = int(ordered_decode_max_positions)
        if self.ordered_prior_subtraction < 0.0:
            raise ValueError("ordered_prior_subtraction must be non-negative")
        if not 0.0 <= self.ordered_min_prior_probability < 1.0:
            raise ValueError(
                "ordered_min_prior_probability must be in [0, 1)"
            )
        if self.ordered_min_margin < 0.0:
            raise ValueError("ordered_min_margin must be non-negative")
        if self.ordered_decode_bias_strength < 0.0:
            raise ValueError("ordered_decode_bias_strength must be non-negative")
        if self.ordered_decode_max_positions < 0:
            raise ValueError("ordered_decode_max_positions must be non-negative")
        if ordered_head is not None:
            for parameter in ordered_head.parameters():
                parameter.requires_grad_(False)
            ordered_head.eval()
            positions = int(getattr(ordered_head, "positions"))
            vocabulary_size = int(getattr(ordered_head, "vocabulary_size"))
            if ordered_log_prior is None or tuple(ordered_log_prior.shape) != (
                positions,
                vocabulary_size,
            ):
                raise ValueError(
                    "ordered_log_prior must be [positions, vocabulary]"
                )
            if ordered_token_ids is None or ordered_token_ids.ndim != 2:
                raise ValueError("ordered_token_ids must be [vocabulary, pieces]")
            if ordered_token_ids.shape[0] != vocabulary_size:
                raise ValueError("ordered_token_ids must match ordered vocabulary")
            if ordered_prototypes is None or tuple(ordered_prototypes.shape) != (
                vocabulary_size,
                lexical_prototype_dim,
            ):
                raise ValueError(
                    "ordered_prototypes must be [ordered vocabulary, prototype dim]"
                )
            self.register_buffer(
                "ordered_log_prior", ordered_log_prior.detach().float()
            )
            self.register_buffer(
                "ordered_token_ids", ordered_token_ids.detach().long()
            )
            self.register_buffer(
                "ordered_prototypes",
                F.normalize(ordered_prototypes.detach().float(), p=2, dim=-1),
            )
            self.ordered_context_gate = nn.Parameter(torch.zeros(context_length))
        elif self.ordered_decode_bias_strength > 0.0:
            raise ValueError(
                "ordered_head is required when ordered decode bias is enabled"
            )

    def train(self, mode: bool = True):
        """Keep the frozen lexical probe deterministic while its parent trains."""

        super().train(mode)
        self.content_head.eval()
        if self.ordered_head is not None:
            self.ordered_head.eval()
        # A lexical-gate-only stage must not reactivate dropout in the frozen
        # brain or semantic paths simply because the wrapper has one trainable
        # 64-value gate.
        if not any(parameter.requires_grad for parameter in self.fmri2sem.parameters()):
            self.fmri2sem.eval()
        if not any(
            parameter.requires_grad
            for parameter in self.semantic_projector.parameters()
        ):
            self.semantic_projector.eval()
        return self

    def ordered_features(self, raw_semantic: torch.Tensor) -> torch.Tensor:
        """Match an ordered head trained on mean-pooled delayed MiniLM blocks."""

        if self.ordered_head is None:
            raise RuntimeError("No ordered head is configured")
        input_dim = int(getattr(self.ordered_head, "input_dim"))
        if raw_semantic.shape[-1] == input_dim:
            features = raw_semantic
        elif raw_semantic.shape[-1] % input_dim == 0:
            features = raw_semantic.reshape(
                len(raw_semantic), -1, input_dim
            ).mean(dim=1)
        else:
            raise ValueError(
                "Ordered-head input width is incompatible with MRI2SEM output: "
                f"ordered={input_dim} MRI2SEM={raw_semantic.shape[-1]}"
            )
        return F.normalize(features, p=2, dim=-1)

    def ordered_token_position_bias(
        self, fmri_vectors: torch.Tensor, *, vocabulary_size: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the ordered head's top word at each of ten positions."""

        if self.ordered_head is None or self.ordered_decode_bias_strength <= 0.0:
            return None
        raw_semantic = self.fmri2sem(fmri_vectors)
        logits = self.ordered_head(self.ordered_features(raw_semantic))
        if self.ordered_prior_subtraction:
            logits = logits - self.ordered_prior_subtraction * self.ordered_log_prior[
                None, :, :
            ].to(device=logits.device, dtype=logits.dtype)
        eligible = self.ordered_token_ids[:, 0] >= 0
        if self.ordered_min_prior_probability > 0.0:
            eligible = eligible[None, :] & (
                self.ordered_log_prior.exp()
                >= self.ordered_min_prior_probability
            )
        else:
            eligible = eligible[None, :].expand(logits.shape[1], -1)
        if not bool(eligible.any(dim=-1).all()):
            raise ValueError(
                "ordered_min_prior_probability leaves an output position "
                "without any tokenizable candidate"
            )
        logits = logits.masked_fill(
            ~eligible[None, :, :].to(device=logits.device), -torch.inf
        )
        top_values, top_indices = logits.topk(k=2, dim=-1)
        indices = top_indices[..., 0]
        token_ids = self.ordered_token_ids[indices].clone()
        token_ids[(token_ids < 0) | (token_ids >= vocabulary_size)] = -1
        if self.ordered_min_margin > 0.0:
            confident = (top_values[..., 0] - top_values[..., 1]) >= (
                self.ordered_min_margin
            )
            token_ids = token_ids.masked_fill(~confident[..., None], -1)
        weights = torch.full(
            indices.shape,
            self.ordered_decode_bias_strength,
            device=logits.device,
            dtype=logits.dtype,
        )
        return token_ids, weights

    def ordered_context(self, raw_semantic: torch.Tensor) -> torch.Tensor:
        """Map the ordered head's ten predicted words into ordered ELF memory slots."""

        if self.ordered_head is None or self.ordered_context_gate is None:
            raise RuntimeError("No ordered context is configured")
        logits = self.ordered_head(self.ordered_features(raw_semantic))
        if self.ordered_prior_subtraction:
            logits = logits - self.ordered_prior_subtraction * self.ordered_log_prior[
                None, :, :
            ].to(device=logits.device, dtype=logits.dtype)
        indices = logits.argmax(dim=-1)
        batch_size, positions = indices.shape
        prototypes = self.ordered_prototypes[indices].to(dtype=raw_semantic.dtype)
        flat_prototypes = prototypes.reshape(batch_size * positions, -1)
        if self.lexical_repeat_factor > 1:
            flat_prototypes = F.normalize(
                flat_prototypes.repeat(1, self.lexical_repeat_factor), p=2, dim=-1
            )
        projected = self.semantic_projector.project_semantic(flat_prototypes)
        prototype_context, _ = self.semantic_projector.context_from_projected_semantic(
            projected
        )
        context_length, context_dim = prototype_context.shape[1:]
        prototype_context = prototype_context.reshape(
            batch_size, positions, context_length, context_dim
        )
        ordered = torch.zeros_like(prototype_context[:, 0])
        for position in range(positions):
            start = position * context_length // positions
            end = (position + 1) * context_length // positions
            ordered[:, start:end] = prototype_context[:, position, start:end]
        return ordered

    def adjusted_lexical_logits(self, raw_semantic: torch.Tensor) -> torch.Tensor:
        logits = self.content_head(raw_semantic)
        if self.lexical_prior_subtraction:
            prior_logits = torch.logit(self.lexical_logit_prior).to(
                device=logits.device, dtype=logits.dtype
            )
            logits = logits - self.lexical_prior_subtraction * prior_logits[None, :]
        if self.lexical_max_prior_probability < 1.0:
            eligible = self.lexical_logit_prior.to(logits.device) <= self.lexical_max_prior_probability
            if int(eligible.sum()) < self.lexical_topk:
                raise ValueError(
                    "lexical_max_prior_probability leaves fewer eligible words than lexical_topk"
                )
            logits = logits.masked_fill(~eligible[None, :], -torch.inf)
        return logits

    def lexical_selection(
        self, raw_semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.adjusted_lexical_logits(raw_semantic)
        values, indices = logits.topk(k=self.lexical_topk, dim=1)
        weights = F.softmax(values.float() / self.lexical_temperature, dim=1).to(
            dtype=raw_semantic.dtype
        )
        return indices, weights

    def lexical_semantic(self, raw_semantic: torch.Tensor) -> torch.Tensor:
        """Lift an undelayed lexical prototype into the projector input space."""

        indices, weights = self.lexical_selection(raw_semantic)
        prototypes = self.lexical_prototypes[indices].to(dtype=raw_semantic.dtype)
        semantic = F.normalize(
            (prototypes * weights[:, :, None]).sum(dim=1).float(), p=2, dim=-1
        )
        if self.lexical_repeat_factor > 1:
            # A lexical prototype has no independent delay estimate. Repeating
            # it is the neutral four-delay representation; normalizing again
            # matches the globally normalized MRI2SEM output interface.
            semantic = F.normalize(
                semantic.repeat(1, self.lexical_repeat_factor), p=2, dim=-1
            )
        return semantic.to(dtype=raw_semantic.dtype)

    def lexical_context(self, raw_semantic: torch.Tensor) -> torch.Tensor:
        """Map selected word prototypes to one mixed or partitioned ELF memory."""

        if self.lexical_context_mode == "mixture":
            lexical_semantic = self.lexical_semantic(raw_semantic)
            lexical_projected = self.semantic_projector.project_semantic(lexical_semantic)
            lexical_context, _ = self.semantic_projector.context_from_projected_semantic(
                lexical_projected
            )
            return lexical_context

        indices, weights = self.lexical_selection(raw_semantic)
        batch_size, topk = indices.shape
        prototypes = self.lexical_prototypes[indices].to(dtype=raw_semantic.dtype)
        flat_prototypes = prototypes.reshape(batch_size * topk, -1)
        if self.lexical_repeat_factor > 1:
            flat_prototypes = F.normalize(
                flat_prototypes.repeat(1, self.lexical_repeat_factor), p=2, dim=-1
            )
        projected = self.semantic_projector.project_semantic(flat_prototypes)
        prototype_context, _ = self.semantic_projector.context_from_projected_semantic(
            projected
        )
        context_length, context_dim = prototype_context.shape[1:]
        prototype_context = prototype_context.reshape(
            batch_size, topk, context_length, context_dim
        )
        partitioned = torch.zeros_like(prototype_context[:, 0])
        for rank in range(topk):
            start = rank * context_length // topk
            end = (rank + 1) * context_length // topk
            partitioned[:, start:end] = (
                prototype_context[:, rank, start:end]
                * weights[:, rank, None, None].to(prototype_context.dtype)
            )
        return partitioned

    def lexical_token_logit_bias(
        self, fmri_vectors: torch.Tensor, *, vocabulary_size: int
    ) -> torch.Tensor | None:
        """Return a sparse per-row decoder bias from the frozen lexical head."""

        if self.lexical_decode_bias_strength <= 0.0 or self.lexical_token_ids.shape[1] == 0:
            return None
        raw_semantic = self.fmri2sem(fmri_vectors)
        indices, weights = self.lexical_selection(raw_semantic)
        token_ids = self.lexical_token_ids[indices]
        valid = (token_ids >= 0) & (token_ids < vocabulary_size)
        # Match the score-aware T5 rule: softmax weights average one rather
        # than summing to one, with a cap preventing a single lexical item
        # from overwhelming the language-model evidence.
        decode_weights = (weights * indices.shape[1]).clamp(max=2.5)
        token_weights = decode_weights[:, :, None].expand_as(token_ids).to(
            raw_semantic.dtype
        )
        bias = torch.zeros(
            (raw_semantic.shape[0], vocabulary_size),
            dtype=raw_semantic.dtype,
            device=raw_semantic.device,
        )
        rows = torch.arange(raw_semantic.shape[0], device=raw_semantic.device)
        for rank in range(token_ids.shape[1]):
            for piece in range(token_ids.shape[2]):
                ids = token_ids[:, rank, piece].clamp(0, vocabulary_size - 1)
                values = token_weights[:, rank, piece]
                current = bias[rows, ids]
                bias[rows, ids] = torch.where(
                    valid[:, rank, piece], torch.maximum(current, values), current
                )
        return bias * self.lexical_decode_bias_strength

    def lexical_token_sequence_bias(
        self, fmri_vectors: torch.Tensor, *, vocabulary_size: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return grouped word-piece sequences and score-aware row weights."""

        if self.lexical_decode_bias_strength <= 0.0 or self.lexical_token_ids.shape[1] == 0:
            return None
        raw_semantic = self.fmri2sem(fmri_vectors)
        indices, weights = self.lexical_selection(raw_semantic)
        token_ids = self.lexical_token_ids[indices].clone()
        token_ids[(token_ids < 0) | (token_ids >= vocabulary_size)] = -1
        decode_weights = (weights * indices.shape[1]).clamp(max=2.5)
        return token_ids, decode_weights * self.lexical_decode_bias_strength

    def condition_context(self, fmri_vectors: torch.Tensor):
        raw_semantic = self.fmri2sem(fmri_vectors)
        projected = self.semantic_projector.project_semantic(raw_semantic)
        context, context_mask = self.semantic_projector.context_from_projected_semantic(projected)
        lexical_context = self.lexical_context(raw_semantic)
        gate = torch.tanh(self.lexical_context_gate).reshape(1, -1, 1).to(context.dtype)
        context = context + gate * lexical_context
        if self.ordered_context_gate is not None:
            ordered_context = self.ordered_context(raw_semantic)
            ordered_gate = torch.tanh(self.ordered_context_gate).reshape(1, -1, 1)
            context = context + ordered_gate.to(context.dtype) * ordered_context
        return projected, context, context_mask
