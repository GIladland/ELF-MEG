from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.fmri2sem_bridge import (  # noqa: E402
    ContentLogitHead,
    FMRI2SEMLexicalToELFContextAdapter,
    FMRI2SEMToELFContextAdapter,
    MindEyeStyleMLP,
    OrderedWordLogitHead,
    ResidualMLPBlock,
    configure_mri2sem_trainable,
    load_mri2sem_model,
)
from modules.semantic_adapter import (  # noqa: E402
    ResidualIdentityContextProjector,
    ResidualMLPFusionContextProjector,
)


class FMRI2SEMBridgeTest(unittest.TestCase):
    def make_components(self):
        torch.manual_seed(37)
        brain = MindEyeStyleMLP(
            input_dim=12,
            output_dim=24,
            hidden_dim=16,
            res_blocks=2,
            dropout=0.0,
        )
        semantic = ResidualMLPFusionContextProjector(
            input_dim=24,
            context_dim=8,
            context_length=5,
            hidden_dim=16,
            mapper_hidden_dim=12,
            dropout=0.0,
            num_delay_tokens=4,
        )
        return brain, semantic

    def test_raw_fmri_composition_preserves_gradient(self) -> None:
        brain, semantic = self.make_components()
        configure_mri2sem_trainable(brain, "projector")
        bridge = FMRI2SEMToELFContextAdapter(brain, semantic)
        inputs = torch.randn(3, 12)

        projected = bridge.project_semantic(inputs)
        projected.square().mean().backward()

        self.assertEqual(projected.shape, (3, 6))
        self.assertIsNotNone(brain.projector.weight.grad)
        self.assertGreater(float(brain.projector.weight.grad.norm()), 0.0)

    def test_projector_mode_freezes_encoder(self) -> None:
        brain, _ = self.make_components()

        names = configure_mri2sem_trainable(brain, "projector")

        self.assertTrue(names)
        self.assertTrue(all(parameter.requires_grad for parameter in brain.projector.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in brain.encoder.parameters()))
        self.assertFalse(brain.logit_scale.requires_grad)

    def test_last_block_mode_unfreezes_only_last_residual_and_projector(self) -> None:
        brain, _ = self.make_components()

        configure_mri2sem_trainable(brain, "last_block")
        blocks = [module for module in brain.encoder if isinstance(module, ResidualMLPBlock)]

        self.assertTrue(all(not parameter.requires_grad for parameter in blocks[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in blocks[-1].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in brain.projector.parameters()))

    def test_load_can_average_contiguous_output_heads(self) -> None:
        source = MindEyeStyleMLP(
            input_dim=12,
            output_dim=24,
            hidden_dim=16,
            res_blocks=2,
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            torch.save({"model": source.state_dict()}, checkpoint)
            reduced = load_mri2sem_model(
                str(checkpoint),
                input_dim=12,
                output_dim=6,
                hidden_dim=16,
                res_blocks=2,
                dropout=0.0,
                projector_mismatch="mean-blocks",
            )

        torch.testing.assert_close(
            reduced.projector.weight,
            source.projector.weight.reshape(4, 6, 16).mean(dim=0),
        )
        torch.testing.assert_close(
            reduced.projector.bias,
            source.projector.bias.reshape(4, 6).mean(dim=0),
        )

    def test_lexical_context_gate_starts_exactly_at_zero(self) -> None:
        torch.manual_seed(41)
        brain = MindEyeStyleMLP(
            input_dim=12,
            output_dim=6,
            hidden_dim=16,
            res_blocks=1,
            dropout=0.0,
        )
        semantic = ResidualIdentityContextProjector(
            input_dim=6,
            context_dim=8,
            context_length=5,
            hidden_dim=16,
            mapper_hidden_dim=12,
            dropout=0.0,
        )
        content_head = ContentLogitHead(input_dim=6, hidden_dim=4, output_dim=7)
        bridge = FMRI2SEMLexicalToELFContextAdapter(
            brain,
            semantic,
            content_head=content_head,
            lexical_prototypes=torch.randn(7, 6),
            lexical_logit_prior=torch.linspace(0.05, 0.35, 7),
            lexical_token_ids=torch.tensor(
                [[2, -1], [3, 4], [5, -1], [6, -1], [7, 8], [9, -1], [10, -1]]
            ),
            lexical_topk=3,
            lexical_temperature=0.5,
            lexical_prior_subtraction=0.25,
            lexical_decode_bias_strength=0.4,
        )
        inputs = torch.randn(3, 12)
        raw = brain(inputs)
        expected_projected = semantic.project_semantic(raw)
        expected_context, expected_mask = semantic.context_from_projected_semantic(
            expected_projected
        )

        projected, context, mask = bridge.condition_context(inputs)

        torch.testing.assert_close(projected, expected_projected)
        torch.testing.assert_close(context, expected_context)
        torch.testing.assert_close(mask, expected_mask)
        self.assertTrue(all(not parameter.requires_grad for parameter in content_head.parameters()))
        bridge.train()
        self.assertFalse(content_head.training)
        torch.testing.assert_close(
            bridge.lexical_semantic(raw),
            bridge.lexical_semantic(raw),
        )
        token_bias = bridge.lexical_token_logit_bias(inputs, vocabulary_size=16)
        self.assertEqual(tuple(token_bias.shape), (3, 16))
        self.assertGreater(float(token_bias.abs().sum()), 0.0)
        context.sum().backward()
        self.assertIsNotNone(bridge.lexical_context_gate.grad)
        self.assertGreater(float(bridge.lexical_context_gate.grad.norm()), 0.0)

    def test_gate_only_training_keeps_frozen_paths_deterministic(self) -> None:
        brain = MindEyeStyleMLP(
            input_dim=12,
            output_dim=6,
            hidden_dim=16,
            res_blocks=1,
            dropout=0.5,
        )
        semantic = ResidualIdentityContextProjector(
            input_dim=6,
            context_dim=8,
            context_length=5,
            hidden_dim=16,
            mapper_hidden_dim=12,
            dropout=0.5,
        )
        bridge = FMRI2SEMLexicalToELFContextAdapter(
            brain,
            semantic,
            content_head=ContentLogitHead(input_dim=6, hidden_dim=4, output_dim=7),
            lexical_prototypes=torch.randn(7, 6),
            lexical_topk=3,
        )
        for parameter in bridge.parameters():
            parameter.requires_grad_(False)
        bridge.lexical_context_gate.requires_grad_(True)

        bridge.train()

        self.assertTrue(bridge.training)
        self.assertFalse(bridge.fmri2sem.training)
        self.assertFalse(bridge.semantic_projector.training)
        self.assertFalse(bridge.content_head.training)

    def test_lexical_prototypes_are_lifted_into_delayed_semantic_blocks(self) -> None:
        torch.manual_seed(43)
        brain = MindEyeStyleMLP(
            input_dim=12,
            output_dim=24,
            hidden_dim=16,
            res_blocks=1,
            dropout=0.0,
        )
        semantic = ResidualMLPFusionContextProjector(
            input_dim=24,
            context_dim=8,
            context_length=5,
            hidden_dim=16,
            mapper_hidden_dim=12,
            dropout=0.0,
            num_delay_tokens=4,
        )
        bridge = FMRI2SEMLexicalToELFContextAdapter(
            brain,
            semantic,
            content_head=ContentLogitHead(input_dim=24, hidden_dim=4, output_dim=7),
            lexical_prototypes=torch.randn(7, 6),
            lexical_topk=3,
        )
        inputs = torch.randn(3, 12)
        raw = brain(inputs)

        lifted = bridge.lexical_semantic(raw)
        blocks = lifted.reshape(3, 4, 6)

        self.assertEqual(bridge.lexical_repeat_factor, 4)
        self.assertEqual(tuple(lifted.shape), (3, 24))
        torch.testing.assert_close(blocks[:, 0], blocks[:, 1])
        torch.testing.assert_close(blocks[:, 0], blocks[:, 3])
        torch.testing.assert_close(lifted.norm(dim=-1), torch.ones(3))
        projected, context, mask = bridge.condition_context(inputs)
        self.assertEqual(tuple(projected.shape), (3, 6))
        self.assertEqual(tuple(context.shape), (3, 5, 8))
        self.assertEqual(tuple(mask.shape), (3, 5))

    def test_partitioned_lexical_context_preserves_topk_memory_regions(self) -> None:
        torch.manual_seed(47)
        brain = MindEyeStyleMLP(
            input_dim=12,
            output_dim=24,
            hidden_dim=16,
            res_blocks=1,
            dropout=0.0,
        )
        semantic = ResidualMLPFusionContextProjector(
            input_dim=24,
            context_dim=8,
            context_length=6,
            hidden_dim=16,
            mapper_hidden_dim=12,
            dropout=0.0,
            num_delay_tokens=4,
        )
        bridge = FMRI2SEMLexicalToELFContextAdapter(
            brain,
            semantic,
            content_head=ContentLogitHead(input_dim=24, hidden_dim=4, output_dim=7),
            lexical_prototypes=torch.randn(7, 6),
            lexical_topk=3,
            lexical_context_mode="partitioned",
        )
        inputs = torch.randn(2, 12)
        raw = brain(inputs)
        indices, weights = bridge.lexical_selection(raw)
        context = bridge.lexical_context(raw)

        self.assertEqual(tuple(context.shape), (2, 6, 8))
        for rank in range(3):
            prototype = bridge.lexical_prototypes[indices[:, rank]]
            lifted = F.normalize(prototype.repeat(1, 4), p=2, dim=-1)
            projected = semantic.project_semantic(lifted)
            expected, _ = semantic.context_from_projected_semantic(projected)
            start, end = rank * 2, (rank + 1) * 2
            torch.testing.assert_close(
                context[:, start:end],
                expected[:, start:end] * weights[:, rank, None, None],
            )

    def test_ordered_head_mean_pools_delays_and_returns_position_sequences(self) -> None:
        torch.manual_seed(53)
        brain = MindEyeStyleMLP(
            input_dim=12,
            output_dim=24,
            hidden_dim=16,
            res_blocks=1,
            dropout=0.0,
        )
        semantic = ResidualMLPFusionContextProjector(
            input_dim=24,
            context_dim=8,
            context_length=6,
            hidden_dim=16,
            mapper_hidden_dim=12,
            dropout=0.0,
            num_delay_tokens=4,
        )
        ordered = OrderedWordLogitHead(
            input_dim=6, hidden_dim=4, positions=3, vocabulary_size=5
        )
        with torch.no_grad():
            ordered.output.weight.zero_()
            ordered.output.bias.copy_(
                torch.tensor([0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0])
            )
        bridge = FMRI2SEMLexicalToELFContextAdapter(
            brain,
            semantic,
            content_head=ContentLogitHead(input_dim=24, hidden_dim=4, output_dim=7),
            lexical_prototypes=torch.randn(7, 6),
            lexical_topk=3,
            ordered_head=ordered,
            ordered_log_prior=torch.zeros(3, 5),
            ordered_token_ids=torch.tensor(
                [[-1, -1], [11, -1], [12, 13], [14, -1], [15, -1]]
            ),
            ordered_prototypes=torch.randn(5, 6),
            ordered_decode_bias_strength=2.5,
        )
        inputs = torch.randn(2, 12)
        raw = brain(inputs)

        expected_projected = semantic.project_semantic(raw)
        expected_context, expected_mask = semantic.context_from_projected_semantic(
            expected_projected
        )
        projected, context, context_mask = bridge.condition_context(inputs)
        torch.testing.assert_close(projected, expected_projected)
        torch.testing.assert_close(context, expected_context)
        torch.testing.assert_close(context_mask, expected_mask)
        context.sum().backward()
        self.assertIsNotNone(bridge.ordered_context_gate)
        self.assertIsNotNone(bridge.ordered_context_gate.grad)
        self.assertGreater(float(bridge.ordered_context_gate.grad.norm()), 0.0)

        features = bridge.ordered_features(raw)
        expected = F.normalize(raw.reshape(2, 4, 6).mean(dim=1), p=2, dim=-1)
        torch.testing.assert_close(features, expected)
        token_ids, weights = bridge.ordered_token_position_bias(
            inputs, vocabulary_size=16
        )
        self.assertEqual(tuple(token_ids.shape), (2, 3, 2))
        self.assertEqual(token_ids[0, 0, 0].item(), 11)
        self.assertEqual(token_ids[0, 1, 0].item(), 12)
        self.assertEqual(token_ids[0, 2, 0].item(), 14)
        torch.testing.assert_close(weights, torch.full((2, 3), 2.5))

        bridge.ordered_min_prior_probability = 0.1
        with torch.no_grad():
            bridge.ordered_log_prior.copy_(
                torch.tensor(
                    [
                        [0.01, 0.01, 0.01, 0.01, 0.20],
                        [0.01, 0.20, 0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.20, 0.01, 0.01],
                    ]
                ).log()
            )
        common_ids, _ = bridge.ordered_token_position_bias(
            inputs, vocabulary_size=16
        )
        self.assertEqual(common_ids[0, 0, 0].item(), 15)
        self.assertEqual(common_ids[0, 1, 0].item(), 11)
        self.assertEqual(common_ids[0, 2, 0].item(), 12)

        bridge.ordered_min_prior_probability = 0.0
        bridge.ordered_min_margin = 3.5
        margin_ids, _ = bridge.ordered_token_position_bias(
            inputs, vocabulary_size=16
        )
        self.assertTrue(bool((margin_ids[:, 0] == -1).all()))
        self.assertEqual(margin_ids[0, 1, 0].item(), 12)
        self.assertEqual(margin_ids[0, 2, 0].item(), 14)


if __name__ == "__main__":
    unittest.main()
