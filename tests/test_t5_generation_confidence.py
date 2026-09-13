import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.t5_generation_confidence import (
    group_sentencepiece_confidence,
    selected_token_confidence_variants,
)


class FakeTokenizer:
    all_special_ids = [0, 1]
    eos_token_id = 1

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return {2: "▁one", 3: "▁multi", 4: "piece", 5: "▁three"}[token_id]


def test_group_sentencepiece_confidence_uses_geometric_mean() -> None:
    confidence, exact = group_sentencepiece_confidence(
        [2, 3, 4, 5, 1],
        [0.0, -0.2, -0.4, -0.1, -0.3],
        tokenizer=FakeTokenizer(),
        expected_words=3,
    )
    assert exact
    assert abs(confidence[0] - 1.0) < 1e-6
    assert abs(confidence[1] - __import__("math").exp(-0.3)) < 1e-6
    assert abs(confidence[2] - __import__("math").exp(-0.1)) < 1e-6


def test_selected_token_confidence_variants_are_aligned_and_bounded() -> None:
    logits = torch.tensor([[[4.0, 1.0, 0.0], [0.0, 1.0, 3.0]]])
    selected = torch.tensor([[0, 1]])
    variants = selected_token_confidence_variants(logits, selected)
    assert set(variants) == {
        "teacher_probability",
        "teacher_margin",
        "teacher_inverse_entropy",
    }
    assert all(value.shape == selected.shape for value in variants.values())
    assert all(
        bool(((value >= 0.0) & (value <= 1.0)).all())
        for value in variants.values()
    )
    assert variants["teacher_margin"][0, 0] > 0.5
    assert variants["teacher_margin"][0, 1] < 0.5
