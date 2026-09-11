from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_fmri_temporal_residual import TemporalResidualEncoder
from scripts.train_fmri_temporal_semantic import TemporalLagFMRIEncoder
from scripts.probe_fmri_ordered_word_head import OrderedWordHead


def _model(unique: bool) -> TemporalResidualEncoder:
    return TemporalResidualEncoder(
        lagged_dim=12,
        response_lags=4,
        segment_trs=10,
        token_dim=8,
        output_dim=5,
        layers=1,
        heads=2,
        dropout=0.0,
        unique_response_trs=unique,
    )


def test_temporal_residual_is_exact_identity_at_initialization() -> None:
    inputs = torch.randn(3, 10, 12)
    base = torch.nn.functional.normalize(torch.randn(3, 5), p=2, dim=-1)
    for unique in (False, True):
        exact, full, exact_residual, full_residual = _model(unique)(inputs, base)
        torch.testing.assert_close(exact_residual, torch.zeros_like(exact_residual))
        torch.testing.assert_close(full_residual, torch.zeros_like(full_residual))
        torch.testing.assert_close(exact, base)
        torch.testing.assert_close(full, base)


def test_temporal_residual_backpropagates_for_both_tokenizations() -> None:
    inputs = torch.randn(3, 10, 12)
    base = torch.nn.functional.normalize(torch.randn(3, 5), p=2, dim=-1)
    target = torch.nn.functional.normalize(torch.randn(3, 5), p=2, dim=-1)
    for unique in (False, True):
        model = _model(unique)
        exact, _, _, _ = model(inputs, base)
        (1.0 - torch.nn.functional.cosine_similarity(exact, target).mean()).backward()
        assert model.exact_residual.weight.grad is not None
        assert model.exact_residual.weight.grad.abs().sum() > 0


def test_scratch_temporal_encoder_supports_both_tokenizations() -> None:
    inputs = torch.randn(3, 10, 12)
    for unique, sequence_length in ((False, 40), (True, 13)):
        model = TemporalLagFMRIEncoder(
            lagged_dim=12,
            response_lags=4,
            segment_trs=10,
            token_dim=8,
            output_dim=5,
            layers=1,
            heads=2,
            dropout=0.0,
            unique_response_trs=unique,
        )
        exact, full = model(inputs)
        assert exact.shape == full.shape == (3, 5)
        assert model.sequence_length == sequence_length


def test_ordered_word_head_starts_as_position_prior_and_learns_residual() -> None:
    prior = torch.log_softmax(torch.randn(10, 7), dim=-1)
    model = OrderedWordHead(
        input_dim=5,
        hidden_dim=8,
        positions=10,
        vocabulary_size=7,
        log_prior=prior,
    )
    inputs = torch.randn(3, 5)
    logits = model(inputs)
    torch.testing.assert_close(logits, prior[None, :, :].expand(3, -1, -1))
    targets = torch.randint(0, 7, (3, 10))
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    assert model.output.weight.grad is not None
    assert model.output.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    test_temporal_residual_is_exact_identity_at_initialization()
    test_temporal_residual_backpropagates_for_both_tokenizations()
    test_scratch_temporal_encoder_supports_both_tokenizations()
    test_ordered_word_head_starts_as_position_prior_and_learns_residual()
    print("temporal and ordered-word identity/gradient checks passed")
