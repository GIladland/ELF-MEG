import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "interpolate_e2e_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("interpolate_e2e_checkpoints", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_interpolates_float_and_preserves_identical_integer_tensor() -> None:
    baseline = {"weight": torch.tensor([0.0, 2.0]), "count": torch.tensor(4)}
    candidate = {"weight": torch.tensor([2.0, 6.0]), "count": torch.tensor(4)}
    result = MODULE.interpolate_state_dict(baseline, candidate, 0.25)
    assert torch.equal(result["weight"], torch.tensor([0.5, 3.0]))
    assert torch.equal(result["count"], torch.tensor(4))


def test_rejects_changed_integer_tensor() -> None:
    with pytest.raises(ValueError, match="non-floating checkpoint tensor changed"):
        MODULE.interpolate_state_dict({"x": torch.tensor(1)}, {"x": torch.tensor(2)}, 0.5)
