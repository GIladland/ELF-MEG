from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.semantic_adapter import (  # noqa: E402
    DelayFusionContextProjector,
    DelayTokenContextProjector,
    ResidualDelayTokenFusionContextProjector,
    ResidualIdentityContextProjector,
    ResidualMLPFusionContextProjector,
)


def make_adapter() -> DelayTokenContextProjector:
    torch.manual_seed(7)
    return DelayTokenContextProjector(
        input_dim=24,
        context_dim=16,
        context_length=5,
        hidden_dim=32,
        dropout=0.0,
        num_delay_tokens=4,
        num_layers=1,
        num_heads=4,
    ).eval()


class DelayTokenContextProjectorTest(unittest.TestCase):
    def test_flat_and_explicit_delay_inputs_are_equivalent(self) -> None:
        adapter = make_adapter()
        flat = torch.randn(2, 24)

        flat_context, flat_mask = adapter(flat)
        token_context, token_mask = adapter(flat.reshape(2, 4, 6))

        torch.testing.assert_close(flat_context, token_context)
        torch.testing.assert_close(flat_mask, token_mask)
        self.assertEqual(flat_context.shape, (2, 5, 16))
        self.assertTrue(bool(torch.all(flat_mask == 1)))

    def test_delay_order_changes_the_context(self) -> None:
        adapter = make_adapter()
        delays = torch.randn(2, 4, 6)

        original, _ = adapter(delays)
        reversed_delays, _ = adapter(delays.flip(1))

        self.assertFalse(torch.allclose(original, reversed_delays))

    def test_every_delay_block_receives_gradient(self) -> None:
        adapter = make_adapter()
        delays = torch.randn(2, 4, 6, requires_grad=True)

        context, _ = adapter(delays)
        context.square().mean().backward()

        self.assertIsNotNone(delays.grad)
        self.assertTrue(bool(torch.all(delays.grad.norm(dim=-1) > 0)))

    def test_rejects_non_divisible_flat_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            DelayTokenContextProjector(
                input_dim=25,
                context_dim=16,
                context_length=5,
                num_delay_tokens=4,
                num_heads=4,
            )


class DelayFusionContextProjectorTest(unittest.TestCase):
    def make_adapter(self) -> DelayFusionContextProjector:
        torch.manual_seed(11)
        return DelayFusionContextProjector(
            input_dim=24,
            context_dim=16,
            context_length=5,
            hidden_dim=32,
            dropout=0.0,
            num_delay_tokens=4,
        ).eval()

    def test_initial_fusion_is_exact_delay_mean(self) -> None:
        adapter = self.make_adapter()
        delays = torch.randn(3, 4, 6)

        actual, actual_mask = adapter(delays)
        projected = adapter.project_semantic(delays)
        expected, expected_mask = adapter.context_from_projected_semantic(projected)

        torch.testing.assert_close(projected, delays.mean(dim=1))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_mask, expected_mask)

    def test_flat_and_explicit_delay_inputs_are_equivalent(self) -> None:
        adapter = self.make_adapter()
        delays = torch.randn(2, 4, 6)

        flat_context, _ = adapter(delays.reshape(2, 24))
        token_context, _ = adapter(delays)

        torch.testing.assert_close(flat_context, token_context)

    def test_fusion_weights_are_featurewise_probabilities(self) -> None:
        adapter = self.make_adapter()
        weights = adapter.fusion_weights()

        self.assertEqual(weights.shape, (4, 6))
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(6))

    def test_fusion_logits_receive_gradient(self) -> None:
        adapter = self.make_adapter()
        delays = torch.randn(2, 4, 6)

        context, _ = adapter(delays)
        context.square().mean().backward()

        self.assertIsNotNone(adapter.fusion_logits.grad)
        self.assertGreater(float(adapter.fusion_logits.grad.norm()), 0.0)

    def test_optional_projection_normalization_restores_unit_norm(self) -> None:
        adapter = DelayFusionContextProjector(
            input_dim=24,
            context_dim=16,
            context_length=5,
            hidden_dim=32,
            num_delay_tokens=4,
            normalize_semantic_output=True,
        ).eval()
        projected = adapter.project_semantic(torch.randn(3, 24))

        torch.testing.assert_close(projected.norm(dim=-1), torch.ones(3))


class ResidualMLPFusionContextProjectorTest(unittest.TestCase):
    def make_adapter(self, normalize: bool = False) -> ResidualMLPFusionContextProjector:
        torch.manual_seed(23)
        return ResidualMLPFusionContextProjector(
            input_dim=24,
            context_dim=16,
            context_length=5,
            hidden_dim=32,
            mapper_hidden_dim=12,
            dropout=0.0,
            num_delay_tokens=4,
            normalize_semantic_output=normalize,
        ).eval()

    def test_zero_output_initialization_is_exact_delay_mean(self) -> None:
        adapter = self.make_adapter()
        inputs = torch.randn(3, 24)

        projected = adapter.project_semantic(inputs)

        torch.testing.assert_close(projected, inputs.reshape(3, 4, 6).mean(dim=1))

    def test_mapper_receives_gradient_while_context_net_can_remain_frozen(self) -> None:
        adapter = self.make_adapter()
        for parameter in adapter.net.parameters():
            parameter.requires_grad_(False)
        inputs = torch.randn(3, 24)
        target = torch.randn(3, 6)

        (adapter.project_semantic(inputs) - target).square().mean().backward()

        self.assertIsNotNone(adapter.semantic_mapper[-1].weight.grad)
        self.assertGreater(float(adapter.semantic_mapper[-1].weight.grad.norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in adapter.net.parameters()))

    def test_optional_projection_normalization(self) -> None:
        adapter = self.make_adapter(normalize=True)
        projected = adapter.project_semantic(torch.randn(3, 24))

        torch.testing.assert_close(projected.norm(dim=-1), torch.ones(3))


class ResidualDelayTokenFusionContextProjectorTest(unittest.TestCase):
    def make_adapter(
        self, preserve_delay_context: bool = False
    ) -> ResidualDelayTokenFusionContextProjector:
        torch.manual_seed(29)
        return ResidualDelayTokenFusionContextProjector(
            input_dim=24,
            context_dim=16,
            context_length=5,
            hidden_dim=32,
            mapper_hidden_dim=24,
            dropout=0.0,
            num_delay_tokens=4,
            num_layers=1,
            num_heads=2,
            preserve_delay_context=preserve_delay_context,
        ).eval()

    def test_zero_output_initialization_is_exact_delay_mean(self) -> None:
        adapter = self.make_adapter()
        delays = torch.randn(3, 4, 6)

        token_projected = adapter.project_semantic(delays)
        flat_projected = adapter.project_semantic(delays.reshape(3, 24))

        torch.testing.assert_close(token_projected, delays.mean(dim=1))
        torch.testing.assert_close(flat_projected, token_projected)

    def test_mapper_trains_while_context_net_remains_frozen(self) -> None:
        adapter = self.make_adapter()
        for parameter in adapter.net.parameters():
            parameter.requires_grad_(False)
        inputs = torch.randn(3, 24)
        target = torch.randn(3, 6)

        (adapter.project_semantic(inputs) - target).square().mean().backward()

        output_projection = adapter.semantic_mapper["output_projection"]
        self.assertIsNotNone(output_projection.weight.grad)
        self.assertGreater(float(output_projection.weight.grad.norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in adapter.net.parameters()))

    def test_delay_embeddings_are_part_of_trainable_mapper(self) -> None:
        adapter = self.make_adapter()

        mapper_parameter_ids = {id(parameter) for parameter in adapter.semantic_mapper.parameters()}
        delay_embedding_parameter = adapter.semantic_mapper["delay_embeddings"].weight

        self.assertIn(id(delay_embedding_parameter), mapper_parameter_ids)

    def test_preserved_delay_context_starts_as_exact_baseline(self) -> None:
        adapter = self.make_adapter(preserve_delay_context=True)
        delays = torch.randn(3, 4, 6)

        projected, actual_context, actual_mask = adapter.condition_context(delays)
        expected_context, expected_mask = adapter.context_from_projected_semantic(
            projected
        )

        torch.testing.assert_close(actual_context, expected_context)
        torch.testing.assert_close(actual_mask, expected_mask)
        self.assertIsNotNone(adapter.delay_context_projection)
        torch.testing.assert_close(
            adapter.delay_context_projection.weight,
            torch.zeros_like(adapter.delay_context_projection.weight),
        )

    def test_preserved_delay_order_can_change_first_context_slots(self) -> None:
        adapter = self.make_adapter(preserve_delay_context=True)
        delays = torch.randn(2, 4, 6)
        with torch.no_grad():
            adapter.delay_context_projection.weight.normal_(mean=0.0, std=0.2)

        _, original_context, _ = adapter.condition_context(delays)
        _, reversed_context, _ = adapter.condition_context(delays.flip(1))

        self.assertFalse(
            torch.allclose(original_context[:, :4], reversed_context[:, :4])
        )
        torch.testing.assert_close(
            original_context[:, 4:], reversed_context[:, 4:]
        )


class ResidualIdentityContextProjectorTest(unittest.TestCase):
    def test_zero_output_initialization_is_identity(self) -> None:
        adapter = ResidualIdentityContextProjector(
            input_dim=12,
            context_dim=8,
            context_length=5,
            hidden_dim=16,
            mapper_hidden_dim=10,
            dropout=0.0,
        ).eval()
        inputs = torch.randn(3, 12)

        torch.testing.assert_close(adapter.project_semantic(inputs), inputs)

    def test_mapper_receives_gradient_with_context_frozen(self) -> None:
        adapter = ResidualIdentityContextProjector(
            input_dim=12,
            context_dim=8,
            context_length=5,
            hidden_dim=16,
            mapper_hidden_dim=10,
            dropout=0.0,
        ).eval()
        for parameter in adapter.net.parameters():
            parameter.requires_grad_(False)
        inputs = torch.randn(3, 12)
        target = torch.randn(3, 12)

        (adapter.project_semantic(inputs) - target).square().mean().backward()

        self.assertGreater(float(adapter.semantic_mapper[-1].weight.grad.norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in adapter.net.parameters()))


if __name__ == "__main__":
    unittest.main()
