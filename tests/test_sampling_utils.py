import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.sampling_utils import truncate_sampling_steps


def test_truncate_sampling_steps_keeps_prefix_and_exact_endpoint() -> None:
    steps = torch.tensor([0.0, 0.1, 0.4, 0.8, 1.0])
    truncated = truncate_sampling_steps(steps, 0.5)
    assert torch.allclose(truncated, torch.tensor([0.0, 0.1, 0.4, 0.5]))


def test_truncate_sampling_steps_has_identity_and_full_endpoints() -> None:
    steps = torch.tensor([0.0, 0.3, 1.0])
    assert torch.equal(truncate_sampling_steps(steps, 0.0), steps[:1])
    assert truncate_sampling_steps(steps, 1.0) is steps
