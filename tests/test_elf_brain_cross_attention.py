from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.lora import load_elf_state_dict  # noqa: E402
from modules.model import ELF, ZeroInitBrainCrossAttention  # noqa: E402


def make_model() -> ELF:
    torch.manual_seed(17)
    return ELF(
        text_encoder_dim=8,
        max_length=16,
        hidden_size=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        bottleneck_dim=4,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=0,
        num_model_mode_tokens=0,
        vocab_size=23,
    ).eval()


class ZeroInitBrainCrossAttentionTest(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self) -> None:
        adapter = ZeroInitBrainCrossAttention(16, 4).eval()
        target = torch.randn(2, 5, 16)
        memory = torch.randn(2, 3, 16)

        actual = adapter(target, memory)

        torch.testing.assert_close(actual, target, rtol=0.0, atol=0.0)

    def test_output_projection_receives_first_step_gradient(self) -> None:
        adapter = ZeroInitBrainCrossAttention(16, 4)
        target = torch.randn(2, 5, 16)
        memory = torch.randn(2, 3, 16)
        readout = torch.randn(16)

        loss = (adapter(target, memory) * readout).sum()
        loss.backward()

        self.assertIsNotNone(adapter.proj.weight.grad)
        self.assertGreater(float(adapter.proj.weight.grad.norm()), 0.0)

    def test_zero_memory_remains_noop_after_projection_changes(self) -> None:
        adapter = ZeroInitBrainCrossAttention(16, 4).eval()
        with torch.no_grad():
            adapter.proj.weight.normal_()
        target = torch.randn(2, 5, 16)

        actual = adapter(target, torch.zeros(2, 3, 16))

        torch.testing.assert_close(actual, target, rtol=0.0, atol=0.0)


class ELFBrainCrossAttentionTest(unittest.TestCase):
    def test_enabling_adapters_preserves_pretrained_output(self) -> None:
        model = make_model()
        inputs = torch.randn(2, 7, 8)
        attention_mask = torch.ones(2, 7)
        timesteps = torch.tensor([0.2, 0.8])

        before = model(
            inputs,
            timesteps,
            attention_mask=attention_mask,
            decoder_step_active=True,
        )
        indices = model.enable_brain_cross_attention(
            condition_length=3,
            last_n_blocks=2,
        )
        after = model(
            inputs,
            timesteps,
            attention_mask=attention_mask,
            decoder_step_active=True,
        )

        self.assertEqual(indices, [2, 3])
        self.assertEqual(set(model.brain_cross_attention), {"2", "3"})
        torch.testing.assert_close(after[0], before[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(after[1], before[1], rtol=0.0, atol=0.0)

    def test_old_checkpoint_can_initialize_new_zero_safe_modules(self) -> None:
        base = make_model()
        base_state = dict(base.state_dict())
        upgraded = make_model()
        upgraded.enable_brain_cross_attention(condition_length=3, last_n_blocks=2)

        load_elf_state_dict(upgraded, base_state)

        for adapter in upgraded.brain_cross_attention.values():
            torch.testing.assert_close(
                adapter.proj.weight,
                torch.zeros_like(adapter.proj.weight),
            )

    def test_new_adapters_inherit_existing_model_device_and_dtype(self) -> None:
        model = make_model().to(dtype=torch.float64)

        model.enable_brain_cross_attention(condition_length=3, last_n_blocks=2)

        for adapter in model.brain_cross_attention.values():
            for parameter in adapter.parameters():
                self.assertEqual(parameter.device, model.t_emb_tokens.device)
                self.assertEqual(parameter.dtype, torch.float64)

    def test_cross_attention_state_round_trip_is_strict(self) -> None:
        source = make_model()
        source.enable_brain_cross_attention(condition_length=3, last_n_blocks=2)
        with torch.no_grad():
            source.brain_cross_attention["3"].proj.weight.normal_()
        target = make_model()
        target.enable_brain_cross_attention(condition_length=3, last_n_blocks=2)

        load_elf_state_dict(target, source.state_dict())

        torch.testing.assert_close(
            target.brain_cross_attention["3"].proj.weight,
            source.brain_cross_attention["3"].proj.weight,
        )

    def test_only_target_tokens_receive_cross_attention_residual(self) -> None:
        model = make_model()
        model.enable_brain_cross_attention(condition_length=3, last_n_blocks=1)
        adapter = model.brain_cross_attention["3"]
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.eye(model.hidden_size))
        inputs = torch.randn(2, 7, 8)
        timesteps = torch.tensor([0.3, 0.6])

        _, changed = model(inputs, timesteps, decoder_step_active=True)
        with torch.no_grad():
            adapter.proj.weight.zero_()
        _, baseline = model(inputs, timesteps, decoder_step_active=True)

        # The final flow head covers the complete original sequence. Context
        # positions are untouched by the cross residual; target positions move.
        torch.testing.assert_close(changed[:, :3], baseline[:, :3])
        self.assertFalse(torch.allclose(changed[:, 3:], baseline[:, 3:]))

    def test_decoder_only_mode_preserves_flow_but_changes_decoder(self) -> None:
        model = make_model()
        model.enable_brain_cross_attention(
            condition_length=3,
            last_n_blocks=1,
            decoder_only=True,
        )
        adapter = model.brain_cross_attention["3"]
        inputs = torch.randn(2, 7, 8)
        timesteps = torch.tensor([0.3, 0.6])
        with torch.no_grad():
            adapter.proj.weight.zero_()
        baseline_flow, baseline_decoder = model(
            inputs,
            timesteps,
            decoder_step_active=True,
        )
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.eye(model.hidden_size))

        unchanged_flow, _ = model(inputs, timesteps)
        _decoder_flow, changed_decoder = model(
            inputs,
            timesteps,
            decoder_step_active=True,
        )

        torch.testing.assert_close(unchanged_flow, baseline_flow)
        self.assertFalse(torch.allclose(changed_decoder, baseline_decoder))


if __name__ == "__main__":
    unittest.main()
