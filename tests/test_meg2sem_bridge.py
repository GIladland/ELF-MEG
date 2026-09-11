import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.meg2sem_bridge import (
    MEG2SEMToELFContextAdapter,
    MinimalMEG2SEMRegressor,
    configure_meg2sem_trainable,
)


def test_partial_training_keeps_frozen_submodules_in_eval_mode():
    meg2sem = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
    projector = nn.Sequential(nn.Dropout(0.5), nn.Linear(4, 4))
    for parameter in meg2sem.parameters():
        parameter.requires_grad_(False)

    adapter = MEG2SEMToELFContextAdapter(
        meg2sem=meg2sem,
        semantic_projector=projector,
        normalize_semantic_output=False,
    )
    adapter.train()

    assert not adapter.meg2sem.training
    assert adapter.semantic_projector.training


def test_fully_frozen_bridge_stays_deterministic_in_train_mode():
    meg2sem = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
    projector = nn.Sequential(nn.Dropout(0.5), nn.Linear(4, 4))
    adapter_parameters = list(meg2sem.parameters()) + list(projector.parameters())
    for parameter in adapter_parameters:
        parameter.requires_grad_(False)

    adapter = MEG2SEMToELFContextAdapter(
        meg2sem=meg2sem,
        semantic_projector=projector,
        normalize_semantic_output=False,
    )
    adapter.train()

    assert adapter_parameters
    assert not adapter.meg2sem.training
    assert not adapter.semantic_projector.training


def test_output_only_scope_for_minimal_regressor_freezes_backbone():
    model = MinimalMEG2SEMRegressor(in_features=8, hidden_dim=6, out_features=4)
    names = configure_meg2sem_trainable(model, trainable=True, scope="output")

    assert names == ["net.3.weight", "net.3.bias"]
    assert not model.net[1].weight.requires_grad
    assert model.net[3].weight.requires_grad


def test_disabled_meg2sem_training_freezes_every_scope():
    model = MinimalMEG2SEMRegressor(in_features=8, hidden_dim=6, out_features=4)
    names = configure_meg2sem_trainable(model, trainable=False, scope="output")

    assert names == []
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_trainable_meg2sem_can_keep_batchnorm_statistics_frozen():
    meg2sem = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
    projector = nn.Linear(4, 4)
    adapter = MEG2SEMToELFContextAdapter(
        meg2sem=meg2sem,
        semantic_projector=projector,
        normalize_semantic_output=False,
        freeze_meg2sem_batchnorm_stats=True,
    )
    adapter.train()

    assert adapter.meg2sem.training
    assert not adapter.meg2sem[0].training
    assert adapter.meg2sem[1].training
    assert adapter.meg2sem[0].weight.requires_grad


def test_detached_context_routes_gradients_away_from_meg2sem():
    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, values):
            context = self.linear(values).unsqueeze(1)
            mask = torch.ones(context.shape[:2], dtype=torch.bool)
            return context, mask

    meg2sem = MinimalMEG2SEMRegressor(in_features=4, hidden_dim=5, out_features=4)
    projector = Projector()
    adapter = MEG2SEMToELFContextAdapter(
        meg2sem=meg2sem,
        semantic_projector=projector,
        normalize_semantic_output=False,
        detach_meg2sem_for_context=True,
    )

    output = adapter(torch.randn(2, 4))
    output.context.sum().backward()

    assert all(parameter.grad is None for parameter in meg2sem.parameters())
    assert projector.linear.weight.grad is not None
    assert output.encoded_sequence.requires_grad
