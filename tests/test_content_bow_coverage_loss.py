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

from scripts.meg_context_overfit import content_bow_coverage_loss, content_bow_precision_loss  # noqa: E402


class ContentBowCoverageLossTest(unittest.TestCase):
    def test_content_token_anywhere_reduces_loss(self) -> None:
        target_ids = torch.tensor([[2, 3, 4]])
        target_mask = torch.ones((1, 3))
        target_weights = torch.tensor([[1.0, 2.0, 1.0]])
        missing_logits = torch.zeros((1, 3, 6))
        missing_logits[:, :, 0] = 8.0
        present_logits = missing_logits.clone()
        present_logits[:, 0, 3] = 12.0

        missing_loss = content_bow_coverage_loss(
            missing_logits, target_ids, target_mask, target_weights
        )
        present_loss = content_bow_coverage_loss(
            present_logits, target_ids, target_mask, target_weights
        )

        self.assertLess(float(present_loss), float(missing_loss))

    def test_no_marked_content_tokens_returns_differentiable_zero(self) -> None:
        logits = torch.randn((2, 3, 7), requires_grad=True)
        loss = content_bow_coverage_loss(
            logits,
            torch.tensor([[1, 2, 3], [2, 3, 4]]),
            torch.ones((2, 3)),
            torch.ones((2, 3)),
        )
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


class ContentBowPrecisionLossTest(unittest.TestCase):
    def test_absent_content_token_increases_loss(self) -> None:
        target_ids = torch.tensor([[2, 3, 4], [2, 5, 4]])
        target_mask = torch.ones((2, 3))
        target_weights = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])
        clean_logits = torch.zeros((2, 3, 7))
        clean_logits[0, :, 0] = 8.0
        clean_logits[1, :, 0] = 8.0
        hallucinated_logits = clean_logits.clone()
        hallucinated_logits[0, 0, 5] = 12.0

        clean_loss = content_bow_precision_loss(
            clean_logits, target_ids, target_mask, target_weights, negative_topk=1
        )
        hallucinated_loss = content_bow_precision_loss(
            hallucinated_logits, target_ids, target_mask, target_weights, negative_topk=1
        )

        self.assertGreater(float(hallucinated_loss), float(clean_loss))

    def test_own_content_token_is_not_penalized(self) -> None:
        target_ids = torch.tensor([[2, 3, 4], [2, 5, 4]])
        target_mask = torch.ones((2, 3))
        target_weights = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])
        logits = torch.zeros((2, 3, 7))
        baseline = content_bow_precision_loss(
            logits, target_ids, target_mask, target_weights, negative_topk=1
        )
        own_content_logits = logits.clone()
        own_content_logits[0, 0, 3] = 12.0
        own_content = content_bow_precision_loss(
            own_content_logits, target_ids, target_mask, target_weights, negative_topk=1
        )

        self.assertLessEqual(float(own_content), float(baseline) + 1e-6)

    def test_no_cross_sample_negatives_returns_differentiable_zero(self) -> None:
        logits = torch.randn((1, 3, 7), requires_grad=True)
        loss = content_bow_precision_loss(
            logits,
            torch.tensor([[1, 2, 3]]),
            torch.ones((1, 3)),
            torch.tensor([[1.0, 2.0, 1.0]]),
        )
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
