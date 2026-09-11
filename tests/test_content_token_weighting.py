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

from scripts.train_npz_semantic_to_elf import build_content_token_weights  # noqa: E402


class FakeOffsetTokenizer:
    def __call__(self, sentence: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        self.asserted_sentence = sentence
        return {
            "input_ids": [10, 11, 12, 13, 1],
            "offset_mapping": [(0, 3), (4, 7), (8, 15), (16, 21), (0, 0)],
        }


class ContentTokenWeightingTest(unittest.TestCase):
    def test_only_content_word_subtokens_are_upweighted(self) -> None:
        weights = build_content_token_weights(
            FakeOffsetTokenizer(),
            ["the red bicycle moved"],
            torch.tensor([[10, 11, 12, 13, 1]]),
            torch.ones((1, 5)),
            content_weight=3.0,
        )

        torch.testing.assert_close(weights, torch.tensor([[1.0, 3.0, 3.0, 3.0, 1.0]]))

    def test_weight_one_is_identity_without_offset_lookup(self) -> None:
        weights = build_content_token_weights(
            None,
            ["anything"],
            torch.tensor([[4, 1]]),
            torch.ones((1, 2)),
            content_weight=1.0,
        )

        torch.testing.assert_close(weights, torch.ones((1, 2)))

    def test_rejects_downweighting_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1.0"):
            build_content_token_weights(
                None,
                ["anything"],
                torch.tensor([[4, 1]]),
                torch.ones((1, 2)),
                content_weight=0.5,
            )


if __name__ == "__main__":
    unittest.main()
