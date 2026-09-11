from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.meg_context_overfit import SemanticVectorContextProjector  # noqa: E402
from scripts.train_npz_semantic_to_elf import (  # noqa: E402
    build_balanced_index_pools,
    curriculum_loss_weights,
    learning_rate_for_step,
    sample_balanced_indices,
    semantic_interface_metrics,
    select_train_indices,
)


class SemanticAlignmentInterfaceTest(unittest.TestCase):
    def test_semantic_curriculum_switches_to_final_losses(self) -> None:
        kwargs = {
            "curriculum_epochs": 20.0,
            "final_denoiser_weight": 1.0,
            "final_decoder_weight": 1.0,
            "curriculum_denoiser_weight": 0.1,
            "curriculum_decoder_weight": 0.2,
        }

        self.assertEqual(curriculum_loss_weights(epoch=10.0, **kwargs), (0.1, 0.2))
        self.assertEqual(curriculum_loss_weights(epoch=20.01, **kwargs), (1.0, 1.0))

    def test_learning_rate_schedule_warms_up_then_cosine_decays(self) -> None:
        kwargs = {
            "base_lr": 1e-4,
            "total_steps": 100,
            "schedule": "cosine",
            "warmup_steps": 10,
            "min_lr_ratio": 0.1,
        }

        self.assertAlmostEqual(learning_rate_for_step(step=1, **kwargs), 1e-5)
        self.assertAlmostEqual(learning_rate_for_step(step=10, **kwargs), 1e-4)
        self.assertAlmostEqual(learning_rate_for_step(step=100, **kwargs), 1e-5)
        self.assertGreater(
            learning_rate_for_step(step=50, **kwargs),
            learning_rate_for_step(step=90, **kwargs),
        )

    def test_constant_schedule_stays_constant_after_warmup(self) -> None:
        value = learning_rate_for_step(
            base_lr=2e-4,
            step=25,
            total_steps=100,
            schedule="constant",
            warmup_steps=10,
            min_lr_ratio=0.0,
        )

        self.assertAlmostEqual(value, 2e-4)

    def test_condition_source_filter_never_selects_validation_tail(self) -> None:
        sources = ["oracle"] * 3 + ["oof_prediction"] * 4 + ["predicted_val"] * 2

        selected = select_train_indices(
            sources,
            train_split_n=7,
            include_sources=["oof_prediction"],
        )

        self.assertEqual(selected.tolist(), [3, 4, 5, 6])

    def test_balanced_pools_only_contain_preselected_rows(self) -> None:
        labels = ["long", "long", "short", "validation", "validation"]
        train_indices = torch.tensor([0, 1, 2])
        pools = build_balanced_index_pools(labels, train_indices)

        sampled = sample_balanced_indices(
            pools,
            batch_size=100,
            generator=torch.Generator().manual_seed(31),
        )

        self.assertTrue(set(sampled.tolist()).issubset({0, 1, 2}))
        self.assertFalse(set(sampled.tolist()).intersection({3, 4}))

    def test_flat_projection_exposes_mean_block_interface(self) -> None:
        adapter = SemanticVectorContextProjector(
            input_dim=24,
            input_projection_dim=6,
            context_dim=4,
            context_length=3,
            hidden_dim=8,
        ).eval()
        inputs = torch.randn(5, 24)
        expected = inputs.reshape(5, 4, 6).mean(dim=1)

        projected = adapter.project_semantic(inputs)
        direct_context, direct_mask = adapter(inputs)
        projected_context, projected_mask = adapter.context_from_projected_semantic(projected)

        torch.testing.assert_close(projected, expected)
        torch.testing.assert_close(direct_context, projected_context)
        torch.testing.assert_close(direct_mask, projected_mask)

    def test_semantic_interface_metrics_recovers_perfect_targets(self) -> None:
        torch.manual_seed(19)
        adapter = SemanticVectorContextProjector(
            input_dim=24,
            input_projection_dim=6,
            context_dim=4,
            context_length=3,
            hidden_dim=8,
        ).eval()
        inputs = torch.randn(5, 24)
        targets = adapter.project_semantic(inputs).detach()

        metrics = semantic_interface_metrics(
            adapter,
            inputs,
            targets,
            device=torch.device("cpu"),
            batch_size=2,
        )

        self.assertAlmostEqual(metrics["matched_cosine_mean"], 1.0, places=6)
        self.assertAlmostEqual(metrics["top1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["top5"], 1.0, places=6)

    def test_flat_projection_can_normalize_before_context_mapping(self) -> None:
        adapter = SemanticVectorContextProjector(
            input_dim=24,
            input_projection_dim=6,
            context_dim=4,
            context_length=3,
            hidden_dim=8,
            normalize_semantic_output=True,
        ).eval()
        projected = adapter.project_semantic(torch.randn(5, 24))

        torch.testing.assert_close(projected.norm(dim=-1), torch.ones(5))


if __name__ == "__main__":
    unittest.main()
