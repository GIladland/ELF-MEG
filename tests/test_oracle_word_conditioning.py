import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.meg_adapter import MEGAdapterOutput
from modules.oracle_word_conditioning import (
    OracleKnownWordFusion,
    OracleKnownWordMEGContextAdapter,
    build_oracle_word_mask,
)


def test_fixed_ten_word_layouts_are_explicit():
    valid = torch.ones((1, 10), dtype=torch.bool)

    expected = {
        "none": [],
        "center1": [4],
        "center2": [4, 5],
        "center4": [3, 4, 5, 6],
        "dispersed2": [2, 7],
        "dispersed4": [1, 3, 6, 8],
        "all": list(range(10)),
    }
    for layout, positions in expected.items():
        mask = build_oracle_word_mask(valid, layout=layout)
        selected = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
        assert selected == positions


def test_random_layout_uses_only_valid_positions_and_allowed_counts():
    valid = torch.tensor(
        [
            [True, False, True, False, True, False, True, False, True, False],
            [True, True, False, False, False, False, False, False, False, False],
        ]
    )
    generator = torch.Generator().manual_seed(19)
    for _ in range(20):
        selected = build_oracle_word_mask(valid, layout="random", generator=generator)
        assert not bool((selected & ~valid).any())
        assert int(selected[0].sum()) in {1, 2, 4}
        assert int(selected[1].sum()) in {1, 2}


def test_zero_gate_and_no_word_rows_preserve_base_context_exactly():
    fusion = OracleKnownWordFusion(context_dim=8, max_words=10, attention_heads=2)
    context = torch.randn(2, 5, 8)
    words = torch.randn(2, 10, 8)
    some_words = torch.zeros(2, 10, dtype=torch.bool)
    some_words[0, [4, 5]] = True

    initial = fusion(context, words, some_words)
    assert torch.equal(initial, context)

    with torch.no_grad():
        fusion.residual_gate.fill_(0.5)
    changed = fusion(context, words, some_words)
    assert not torch.equal(changed[0], context[0])
    assert torch.equal(changed[1], context[1])


def test_wrapper_preserves_semantic_output_and_trains_gate():
    class BaseAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.meg2sem = nn.Linear(8, 8)
            self.semantic_projector = nn.Linear(8, 8)
            self.normalize_semantic_output = False

        def forward(self, meg, *, meg_lengths=None, subjects=None):
            del meg_lengths, subjects
            semantic = meg.mean(dim=1)
            context = semantic[:, None, :].expand(-1, 4, -1)
            return MEGAdapterOutput(
                context=context,
                context_mask=torch.ones(context.shape[:2], dtype=torch.bool),
                encoded_sequence=semantic[:, None, :],
            )

    wrapper = OracleKnownWordMEGContextAdapter(
        base_adapter=BaseAdapter(),
        context_dim=8,
        max_words=10,
        attention_heads=2,
    )
    meg = torch.randn(3, 2, 8)
    words = torch.randn(3, 10, 8)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[:, 4:6] = True

    base_output = wrapper.base_adapter(meg)
    output = wrapper(meg, known_word_embeddings=words, known_word_mask=mask)
    assert torch.equal(output.context, base_output.context)
    assert torch.equal(output.encoded_sequence, base_output.encoded_sequence)

    output.context.square().mean().backward()
    assert wrapper.word_fusion.residual_gate.grad is not None
