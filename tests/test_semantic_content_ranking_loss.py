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

from scripts.meg_context_overfit import semantic_content_ranking_loss  # noqa: E402


class SemanticContentRankingLossTest(unittest.TestCase):
    def test_target_direction_reduces_loss(self) -> None:
        directions = torch.eye(3)
        targets = torch.tensor([[True, False, False], [False, True, False]])
        wrong = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True)
        right = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)

        wrong_loss = semantic_content_ranking_loss(
            wrong, targets, directions, temperature=0.1, negative_topk=2
        )
        right_loss = semantic_content_ranking_loss(
            right, targets, directions, temperature=0.1, negative_topk=2
        )

        self.assertLess(float(right_loss), float(wrong_loss))

    def test_loss_backpropagates_to_semantic_vector(self) -> None:
        predicted = torch.randn((2, 4), requires_grad=True)
        targets = torch.tensor([[True, False, False], [False, True, False]])
        directions = torch.randn((3, 4))
        loss = semantic_content_ranking_loss(
            predicted, targets, directions, temperature=0.2, negative_topk=2
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(predicted.grad)
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
