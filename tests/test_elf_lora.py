from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.lora import (  # noqa: E402
    LoRALinear,
    inject_elf_lora,
    load_elf_state_dict,
    lora_parameter_count,
)


class TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(8, 24)
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.qkv(inputs).chunk(3, dim=-1)[-1]
        return self.proj(values)


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w12 = torch.nn.Linear(8, 16)
        self.w3 = torch.nn.Linear(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = self.w12(inputs).chunk(2, dim=-1)
        return self.w3(torch.nn.functional.silu(first) * second)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = TinyAttention()
        self.mlp = TinyMLP()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attn(inputs)
        return hidden + self.mlp(hidden)


class TinyELF(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
        self.head = torch.nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return self.head(inputs)


def make_model() -> TinyELF:
    torch.manual_seed(13)
    return TinyELF().eval()


class ELFLoRATest(unittest.TestCase):
    def test_zero_initialized_lora_preserves_model_output(self) -> None:
        model = make_model()
        inputs = torch.randn(3, 7, 8)
        expected = model(inputs)

        names = inject_elf_lora(
            model,
            rank=2,
            alpha=4.0,
            targets=("attention", "mlp"),
        )
        actual = model(inputs)

        self.assertEqual(len(names), 8)
        torch.testing.assert_close(actual, expected)

    def test_only_low_rank_parameters_remain_trainable(self) -> None:
        model = make_model()
        inject_elf_lora(model, rank=2, alpha=2.0, targets=("attention",))

        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertTrue(trainable)
        self.assertTrue(all(name.endswith(("lora_A", "lora_B")) for name in trainable))
        self.assertEqual(lora_parameter_count(model), sum(model.state_dict()[name].numel() for name in trainable))
        self.assertIsInstance(model.blocks[0].attn.qkv, LoRALinear)

    def test_lora_receives_gradient(self) -> None:
        model = make_model()
        inject_elf_lora(model, rank=2, alpha=2.0, targets=("attention",))
        inputs = torch.randn(2, 6, 8)

        output = model(inputs)
        output.square().mean().backward()

        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.endswith("lora_B")
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))

    def test_base_checkpoint_loads_into_lora_wrapped_model(self) -> None:
        source = make_model()
        state = source.state_dict()
        target = make_model()
        inject_elf_lora(target, rank=2, alpha=2.0, targets=("attention",))

        load_elf_state_dict(target, state)

        inputs = torch.randn(2, 6, 8)
        expected = source(inputs)
        actual = target(inputs)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
