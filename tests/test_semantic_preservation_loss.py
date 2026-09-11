from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.meg_context_overfit import (  # noqa: E402
    raw_semantic_teacher,
    semantic_preservation_losses,
)


class SemanticPreservationLossTest(unittest.TestCase):
    def test_raw_teacher_matches_equal_delay_mean(self) -> None:
        values = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 24)

        actual = raw_semantic_teacher(values, output_dim=6)
        expected = values.reshape(3, 4, 6).mean(dim=1)

        torch.testing.assert_close(actual, expected)

    def test_identical_student_has_zero_preservation_losses(self) -> None:
        torch.manual_seed(7)
        inputs = torch.randn(8, 24)
        teacher = raw_semantic_teacher(inputs, output_dim=6)

        losses = semantic_preservation_losses(
            teacher.clone(),
            inputs,
            exact_targets=torch.randn(8, 6),
        )

        self.assertAlmostEqual(float(losses["anchor"]), 0.0, places=6)
        self.assertAlmostEqual(float(losses["geometry"]), 0.0, places=6)
        self.assertAlmostEqual(float(losses["rank_distill"]), 0.0, places=6)

    def test_collapsed_student_is_penalized_and_receives_gradient(self) -> None:
        torch.manual_seed(11)
        inputs = torch.randn(8, 24)
        student = torch.ones(8, 6, requires_grad=True)
        exact = torch.randn(8, 6)

        losses = semantic_preservation_losses(
            student,
            inputs,
            exact_targets=exact,
            rank_temperature=0.1,
        )
        total = losses["anchor"] + losses["geometry"] + losses["rank_distill"]
        total.backward()

        self.assertGreater(float(losses["anchor"]), 0.0)
        self.assertGreater(float(losses["geometry"]), 0.0)
        self.assertGreater(float(losses["rank_distill"]), 0.0)
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.norm()), 0.0)

    def test_invalid_delay_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not divisible"):
            raw_semantic_teacher(torch.randn(2, 10), output_dim=6)


if __name__ == "__main__":
    unittest.main()
