from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_npz_semantic_to_elf as training  # noqa: E402


class _Adapter(nn.Module):
    def __init__(self, context_length: int, latent_dim: int):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim

    def forward(self, semantic: torch.Tensor):
        context = semantic[:, :1, None].expand(
            -1, self.context_length, self.latent_dim
        ).contiguous()
        mask = torch.ones(
            (semantic.shape[0], self.context_length), dtype=context.dtype
        )
        return context, mask


def test_frozen_flow_cache_is_target_only_and_reused():
    calls = []

    def fake_sampling_steps(**_kwargs):
        return torch.tensor([1.0, 0.0])

    def fake_generate(*, z, cond_seq, **_kwargs):
        calls.append(int(z.shape[0]))
        return cond_seq + 3.0

    with tempfile.TemporaryDirectory() as temporary_directory:
        semantic = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        cache_path = Path(temporary_directory) / "flow.npy"
        kwargs = dict(
            cache_path=cache_path,
            model=nn.Identity(),
            adapter=_Adapter(context_length=2, latent_dim=5),
            semantic_vectors=semantic,
            target_length=3,
            context_length=2,
            latent_dim=5,
            config=SimpleNamespace(
                time_schedule="uniform",
                denoiser_p_mean=0.0,
                denoiser_p_std=1.0,
                denoiser_noise_scale=1.0,
            ),
            sampling_steps=2,
            cfg_scale=1.0,
            self_cond_cfg_scale=1.0,
            device=torch.device("cpu"),
            batch_size=2,
            seed=7,
        )

        with patch.object(training, "get_sampling_steps", fake_sampling_steps), patch.object(
            training, "_generate_samples_single_batch", fake_generate
        ):
            cached = training.frozen_flow_target_latent_memmap(**kwargs)
            assert cached.shape == (3, 3, 5)
            np.testing.assert_allclose(cached, 3.0)
            assert calls == [2, 1]

            calls.clear()
            reused = training.frozen_flow_target_latent_memmap(**kwargs)
            assert reused.shape == (3, 3, 5)
            assert calls == []


if __name__ == "__main__":
    test_frozen_flow_cache_is_target_only_and_reused()
    print("frozen-flow latent cache test passed")
