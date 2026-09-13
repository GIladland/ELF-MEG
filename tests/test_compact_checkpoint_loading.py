import sys
import tempfile
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_npz_semantic_to_elf import load_e2e_initialization


def test_compact_checkpoint_restores_parent_then_trainable_delta():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        parent_path = directory / "parent.pt"
        delta_path = directory / "delta.pt"

        parent_model = nn.Linear(3, 2)
        parent_adapter = nn.Linear(4, 3)
        with torch.no_grad():
            parent_model.weight.fill_(1.0)
            parent_model.bias.fill_(2.0)
            parent_adapter.weight.fill_(3.0)
            parent_adapter.bias.fill_(4.0)
        torch.save(
            {
                "model_state_dict": parent_model.state_dict(),
                "adapter_state_dict": parent_adapter.state_dict(),
            },
            parent_path,
        )

        delta_weight = torch.full_like(parent_model.weight, 9.0)
        torch.save(
            {
                "checkpoint_format": "trainable_elf_delta_v1",
                "parent_init_e2e_checkpoint": str(parent_path),
                "trainable_model_state_dict": {"weight": delta_weight},
            },
            delta_path,
        )

        restored_model = nn.Linear(3, 2)
        restored_adapter = nn.Linear(4, 3)
        summary = load_e2e_initialization(
            restored_model,
            restored_adapter,
            str(delta_path),
            torch.device("cpu"),
        )
        assert torch.equal(restored_model.weight, delta_weight)
        assert torch.equal(restored_model.bias, parent_model.bias)
        assert torch.equal(restored_adapter.weight, parent_adapter.weight)
        assert torch.equal(restored_adapter.bias, parent_adapter.bias)
        assert "parent_init_e2e_checkpoint" in summary["loaded"]
        assert "trainable_model_state_dict" in summary["loaded"]


def test_compact_checkpoint_restores_trainable_adapter_delta():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        parent_path = directory / "parent.pt"
        delta_path = directory / "delta.pt"

        parent_model = nn.Linear(3, 2)
        parent_adapter = nn.Linear(4, 3)
        with torch.no_grad():
            parent_model.weight.fill_(1.0)
            parent_model.bias.fill_(2.0)
            parent_adapter.weight.fill_(3.0)
            parent_adapter.bias.fill_(4.0)
        torch.save(
            {
                "model_state_dict": parent_model.state_dict(),
                "adapter_state_dict": parent_adapter.state_dict(),
            },
            parent_path,
        )

        delta_bias = torch.full_like(parent_adapter.bias, 8.0)
        torch.save(
            {
                "checkpoint_format": "trainable_e2e_delta_v2",
                "parent_init_e2e_checkpoint": str(parent_path),
                "trainable_adapter_state_dict": {"bias": delta_bias},
            },
            delta_path,
        )

        restored_model = nn.Linear(3, 2)
        restored_adapter = nn.Linear(4, 3)
        summary = load_e2e_initialization(
            restored_model,
            restored_adapter,
            str(delta_path),
            torch.device("cpu"),
        )
        assert torch.equal(restored_model.weight, parent_model.weight)
        assert torch.equal(restored_model.bias, parent_model.bias)
        assert torch.equal(restored_adapter.weight, parent_adapter.weight)
        assert torch.equal(restored_adapter.bias, delta_bias)
        assert "parent_init_e2e_checkpoint" in summary["loaded"]
        assert "trainable_adapter_state_dict" in summary["loaded"]


if __name__ == "__main__":
    test_compact_checkpoint_restores_parent_then_trainable_delta()
    test_compact_checkpoint_restores_trainable_adapter_delta()
    print("compact checkpoint round-trip passed")
