import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "merge_lora_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("merge_lora_checkpoint", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_merge_matches_scaled_low_rank_update() -> None:
    state = {
        "layer.base.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer.base.bias": torch.tensor([0.5, -0.5]),
        "layer.lora_A": torch.tensor([[1.0, 2.0]]),
        "layer.lora_B": torch.tensor([[3.0], [4.0]]),
        "other": torch.tensor([7.0]),
    }
    result = MODULE.merge_lora_state_dict(state, rank=1, alpha=0.5)
    expected = state["layer.base.weight"] + 0.5 * (state["layer.lora_B"] @ state["layer.lora_A"])
    assert torch.equal(result["layer.weight"], expected)
    assert torch.equal(result["layer.bias"], state["layer.base.bias"])
    assert torch.equal(result["other"], state["other"])


def test_merge_rejects_checkpoint_without_lora() -> None:
    with pytest.raises(ValueError, match="no LoRA tensors"):
        MODULE.merge_lora_state_dict({"weight": torch.ones(1)}, rank=1, alpha=1.0)
