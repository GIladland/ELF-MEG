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

from scripts.train_fmri_minilm_t5_prefix import DelayWeightedCalibrator  # noqa: E402


class DelayWeightedCalibratorTest(unittest.TestCase):
    def test_initialization_is_normalized_equal_delay_mean(self) -> None:
        torch.manual_seed(13)
        values = torch.nn.functional.normalize(torch.randn(5, 1536), dim=-1)
        expected = torch.nn.functional.normalize(
            values.reshape(5, 4, 384).mean(dim=1), dim=-1
        )

        actual = DelayWeightedCalibrator()(values)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_delay_weights_and_affine_parameters_receive_gradients(self) -> None:
        torch.manual_seed(17)
        values = torch.randn(3, 1536)
        module = DelayWeightedCalibrator()

        module(values).square().sum(dim=0).mean().backward()

        self.assertIsNotNone(module.delay_logits.grad)
        self.assertIsNotNone(module.log_scale.grad)
        self.assertIsNotNone(module.bias.grad)

    def test_rejects_nonordered_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "4 x 384"):
            DelayWeightedCalibrator()(torch.randn(2, 384))


if __name__ == "__main__":
    unittest.main()
