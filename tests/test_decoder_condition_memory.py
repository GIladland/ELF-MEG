from __future__ import annotations

import unittest

import torch

from scripts.meg_context_overfit import build_decoder_training_latent


class DecoderConditionMemoryTest(unittest.TestCase):
    def test_clean_prefix_contract_preserves_only_condition_positions(self) -> None:
        x0 = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        noise = torch.zeros_like(x0)
        decoder_lambda = torch.full((2, 5, 1), 0.25)
        condition_mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
            dtype=torch.float32,
        )

        legacy = build_decoder_training_latent(
            x0,
            noise,
            decoder_lambda,
            condition_mask,
            preserve_condition_prefix=False,
        )
        decoder_only = build_decoder_training_latent(
            x0,
            noise,
            decoder_lambda,
            condition_mask,
            preserve_condition_prefix=True,
        )

        torch.testing.assert_close(legacy, 0.25 * x0)
        torch.testing.assert_close(decoder_only[:, :2], x0[:, :2])
        torch.testing.assert_close(decoder_only[:, 2:], legacy[:, 2:])


if __name__ == "__main__":
    unittest.main()
