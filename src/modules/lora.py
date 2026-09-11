"""Small, dependency-free LoRA adapters for ELF linear layers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base_output + update * self.scaling


def _replace_linear(
    parent: nn.Module,
    name: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRALinear:
    layer = getattr(parent, name)
    if isinstance(layer, LoRALinear):
        return layer
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected {name} to be nn.Linear, got {type(layer).__name__}.")
    wrapped = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
    setattr(parent, name, wrapped)
    return wrapped


def inject_elf_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    targets: Iterable[str] = ("attention",),
    last_n_blocks: int = -1,
) -> list[str]:
    """Freeze ELF and add LoRA to selected attention/MLP projections."""

    target_set = {str(target).strip().lower() for target in targets if str(target).strip()}
    unknown = target_set - {"attention", "mlp"}
    if unknown:
        raise ValueError(f"Unsupported ELF LoRA targets: {sorted(unknown)}")
    if not target_set:
        raise ValueError("At least one ELF LoRA target is required.")

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("ELF model has no blocks attribute.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    block_count = len(blocks)
    selected_count = block_count if last_n_blocks < 0 else min(block_count, max(0, last_n_blocks))
    first_index = block_count - selected_count
    injected: list[str] = []
    for index in range(first_index, block_count):
        block = blocks[index]
        if "attention" in target_set:
            for layer_name in ("qkv", "proj"):
                _replace_linear(
                    block.attn,
                    layer_name,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                injected.append(f"blocks.{index}.attn.{layer_name}")
        if "mlp" in target_set:
            for layer_name in ("w12", "w3"):
                _replace_linear(
                    block.mlp,
                    layer_name,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                injected.append(f"blocks.{index}.mlp.{layer_name}")

    if not injected:
        raise ValueError("ELF LoRA selected zero layers.")
    return injected


def lora_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    )


def remap_state_dict_for_lora(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Map an ordinary ELF state dict into a LoRA-wrapped ELF state dict.

    A non-LoRA checkpoint has ``layer.weight`` while a wrapped layer stores the
    frozen tensor as ``layer.base.weight``. Newly created low-rank tensors are
    returned as allowed missing keys so a frozen baseline can initialize LoRA.
    """

    remapped = dict(state_dict)
    allowed_missing: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            if getattr(module, "_is_zero_init_brain_cross_attention", False):
                prefix = f"{module_name}." if module_name else ""
                for key in module.state_dict():
                    full_key = f"{prefix}{key}"
                    if full_key not in remapped:
                        allowed_missing.add(full_key)
            continue
        prefix = f"{module_name}." if module_name else ""
        for suffix in ("weight", "bias"):
            source_key = f"{prefix}{suffix}"
            target_key = f"{prefix}base.{suffix}"
            if source_key in remapped and target_key not in remapped:
                remapped[target_key] = remapped.pop(source_key)
        for suffix in ("lora_A", "lora_B"):
            key = f"{prefix}{suffix}"
            if key not in remapped:
                allowed_missing.add(key)
    return remapped, allowed_missing


def load_elf_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Strictly load either a base ELF or a LoRA ELF checkpoint."""

    remapped, allowed_missing = remap_state_dict_for_lora(model, state_dict)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    disallowed_missing = set(missing) - allowed_missing
    if disallowed_missing or unexpected:
        raise ValueError(
            "Could not load ELF state dict: "
            f"missing={sorted(disallowed_missing)} unexpected={sorted(unexpected)}"
        )
