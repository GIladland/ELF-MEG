from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.generation_utils import (  # noqa: E402
    _apply_ordered_token_sequence_bias,
    _apply_position_logit_bias,
    _apply_token_logit_bias,
    _apply_token_sequence_bias,
)


class GenerationTokenBiasTest(unittest.TestCase):
    def test_position_bias_changes_only_aligned_target_slots(self) -> None:
        logits = torch.zeros(2, 6, 5)
        bias = torch.tensor(
            [[0.0, -1.0, -2.0, -3.0, -4.0], [-4.0, -3.0, -2.0, -1.0, 0.0]]
        )

        result = _apply_position_logit_bias(logits, bias, target_start=3)

        torch.testing.assert_close(result[:, :3], logits[:, :3])
        torch.testing.assert_close(result[:, 3], bias[0].expand(2, -1))
        torch.testing.assert_close(result[:, 4], bias[1].expand(2, -1))
        torch.testing.assert_close(result[:, 5], logits[:, 5])

    def test_position_bias_rejects_vocabulary_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            _apply_position_logit_bias(
                torch.zeros(1, 4, 5),
                torch.zeros(2, 6),
                target_start=1,
            )

    def test_static_bias_is_broadcast_to_every_position(self) -> None:
        logits = torch.zeros(1, 4, 6)
        bias = torch.zeros(1, 6)
        bias[0, 3] = 2.0

        result = _apply_token_logit_bias(logits, bias)

        torch.testing.assert_close(result[0, :, 3], torch.full((4,), 2.0))

    def test_once_bias_uses_distinct_compatible_target_positions(self) -> None:
        logits = torch.zeros(1, 5, 6)
        logits[0, 2, 3] = 0.8
        logits[0, 3, 4] = 0.7
        bias = torch.zeros(1, 6)
        bias[0, 3] = 2.0
        bias[0, 4] = 1.5

        result = _apply_token_logit_bias(
            logits, bias, once=True, target_start=2
        )

        self.assertEqual(float((result[0, :, 3] > logits[0, :, 3]).sum()), 1.0)
        self.assertEqual(float((result[0, :, 4] > logits[0, :, 4]).sum()), 1.0)
        self.assertGreater(float(result[0, 2, 3]), float(logits[0, 2, 3]))
        self.assertGreater(float(result[0, 3, 4]), float(logits[0, 3, 4]))
        torch.testing.assert_close(result[0, :2], logits[0, :2])

    def test_once_bias_can_be_restricted_to_early_target_positions(self) -> None:
        logits = torch.zeros(1, 7, 6)
        logits[0, 6, 3] = 5.0
        bias = torch.zeros(1, 6)
        bias[0, 3] = 2.0

        result = _apply_token_logit_bias(
            logits, bias, once=True, target_start=2, max_positions=2
        )

        changed = torch.nonzero(result[0, :, 3] > logits[0, :, 3]).flatten()
        self.assertEqual(changed.numel(), 1)
        self.assertIn(int(changed), (2, 3))

    def test_sequence_bias_keeps_word_pieces_contiguous(self) -> None:
        logits = torch.zeros(1, 7, 8)
        logits[0, 3, 4] = 0.8
        logits[0, 4, 5] = 0.7
        token_ids = torch.tensor([[[4, 5, -1], [6, -1, -1]]])
        sequence_bias = torch.tensor([[3.0, 2.0]])

        result = _apply_token_sequence_bias(
            logits, token_ids, sequence_bias, target_start=2
        )

        changed_piece4 = torch.nonzero(result[0, :, 4] > logits[0, :, 4]).flatten()
        changed_piece5 = torch.nonzero(result[0, :, 5] > logits[0, :, 5]).flatten()
        changed_piece6 = torch.nonzero(result[0, :, 6] > logits[0, :, 6]).flatten()
        self.assertEqual(int(changed_piece5), int(changed_piece4) + 1)
        self.assertNotEqual(int(changed_piece6), int(changed_piece4))
        self.assertNotEqual(int(changed_piece6), int(changed_piece5))
        torch.testing.assert_close(result[0, :2], logits[0, :2])

    def test_sequence_bias_can_be_restricted_to_early_target_positions(self) -> None:
        logits = torch.zeros(1, 8, 7)
        logits[0, 6, 4] = 5.0
        logits[0, 7, 5] = 5.0
        token_ids = torch.tensor([[[4, 5]]])
        sequence_bias = torch.tensor([[2.0]])

        result = _apply_token_sequence_bias(
            logits,
            token_ids,
            sequence_bias,
            target_start=2,
            max_positions=3,
        )

        changed_piece4 = torch.nonzero(
            result[0, :, 4] > logits[0, :, 4]
        ).flatten()
        changed_piece5 = torch.nonzero(
            result[0, :, 5] > logits[0, :, 5]
        ).flatten()
        self.assertEqual(int(changed_piece5), int(changed_piece4) + 1)
        self.assertLess(int(changed_piece5), 5)

    def test_ordered_bias_lays_words_out_in_consecutive_early_slots(self) -> None:
        logits = torch.zeros(1, 9, 10)
        token_ids = torch.tensor([[[3, -1], [4, 5], [6, -1]]])
        sequence_bias = torch.tensor([[1.0, 2.0, 3.0]])

        result = _apply_ordered_token_sequence_bias(
            logits,
            token_ids,
            sequence_bias,
            target_start=2,
            max_positions=4,
        )

        self.assertEqual(float(result[0, 2, 3]), 1.0)
        self.assertEqual(float(result[0, 3, 4]), 2.0)
        self.assertEqual(float(result[0, 4, 5]), 2.0)
        self.assertEqual(float(result[0, 5, 6]), 3.0)
        torch.testing.assert_close(result[0, :2], logits[0, :2])

    def test_ordered_bias_skips_slots_reserved_for_brain_content(self) -> None:
        logits = torch.zeros(1, 8, 10)
        token_ids = torch.tensor([[[3, -1], [4, -1]]])
        sequence_bias = torch.tensor([[1.0, 2.0]])
        blocked = torch.zeros(1, 8, dtype=torch.bool)
        blocked[0, 2] = True
        blocked[0, 4] = True

        result = _apply_ordered_token_sequence_bias(
            logits,
            token_ids,
            sequence_bias,
            target_start=2,
            max_positions=4,
            blocked_positions=blocked,
        )

        self.assertEqual(float(result[0, 3, 3]), 1.0)
        self.assertEqual(float(result[0, 5, 4]), 2.0)
        self.assertEqual(float(result[0, 2].abs().sum()), 0.0)
        self.assertEqual(float(result[0, 4].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
