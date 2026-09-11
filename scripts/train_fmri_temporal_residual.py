#!/usr/bin/env python
"""Learn ordered-fMRI residuals on top of a frozen direct MiniLM solution.

This validation-only trainer is deliberately conservative: the established
MRI2SEM/direct-384 prediction and its residual semantic mapper are frozen.
The temporal branch starts with exactly zero output, so epoch zero is the
existing semantic model.  It can consume either all 40 time-by-lag tokens or
the 13 unique chronological response TRs represented by those overlapping
lag blocks.  Only train11725 and val266 members are indexed; test keys are
never read.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modules.fmri2sem_bridge import load_mri2sem_model
try:
    from train_fmri_temporal_semantic import (
        assert_target_alignment,
        retrieval_metrics,
        strings,
        symmetric_contrastive,
        temporal_row_indices,
    )
except ModuleNotFoundError:  # Imported as scripts.train_fmri_temporal_residual in tests.
    from scripts.train_fmri_temporal_semantic import (
        assert_target_alignment,
        retrieval_metrics,
        strings,
        symmetric_contrastive,
        temporal_row_indices,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tr-npz", required=True)
    parser.add_argument("--segment-npz", required=True)
    parser.add_argument("--exact-target-npz", required=True)
    parser.add_argument("--full-target-npz", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--segment-trs", type=int, default=10)
    parser.add_argument("--response-lags", type=int, default=4)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unique-response-trs", action="store_true")
    parser.add_argument("--exact-weight", type=float, default=1.0)
    parser.add_argument("--exact-weight-start", type=float, default=None)
    parser.add_argument("--full-weight", type=float, default=0.5)
    parser.add_argument("--full-weight-end", type=float, default=None)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--residual-l2-weight", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mapper-hidden-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_semantic_mapper(checkpoint_path: str, dim: int, hidden_dim: int) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("adapter_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Base checkpoint has no adapter_state_dict.")
    mapper = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, dim),
    )
    for prefix in ("semantic_projector.semantic_mapper.", "semantic_mapper."):
        mapper_state = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if mapper_state:
            mapper.load_state_dict(mapper_state, strict=True)
            return mapper
    raise ValueError("Base checkpoint has no residual semantic mapper weights.")


class TemporalResidualEncoder(nn.Module):
    def __init__(
        self,
        *,
        lagged_dim: int,
        response_lags: int,
        segment_trs: int,
        token_dim: int,
        output_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        unique_response_trs: bool,
    ) -> None:
        super().__init__()
        if lagged_dim % response_lags:
            raise ValueError("Lagged input dimension is not divisible by response_lags.")
        self.response_lags = int(response_lags)
        self.segment_trs = int(segment_trs)
        self.voxel_dim = lagged_dim // response_lags
        self.unique_response_trs = bool(unique_response_trs)
        sequence_length = (
            segment_trs + response_lags - 1
            if unique_response_trs
            else segment_trs * response_lags
        )
        self.voxel_norm = nn.LayerNorm(self.voxel_dim)
        self.voxel_projection = nn.Linear(self.voxel_dim, token_dim)
        self.time_embedding = nn.Parameter(
            torch.zeros(sequence_length if unique_response_trs else segment_trs, token_dim)
        )
        self.lag_embedding = (
            None
            if unique_response_trs
            else nn.Parameter(torch.zeros(response_lags, token_dim))
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_query = nn.Parameter(torch.randn(token_dim) / math.sqrt(token_dim))
        self.final_norm = nn.LayerNorm(token_dim)
        self.exact_residual = nn.Linear(token_dim, output_dim)
        self.full_residual = nn.Linear(token_dim, output_dim)
        nn.init.normal_(self.time_embedding, std=0.02)
        if self.lag_embedding is not None:
            nn.init.normal_(self.lag_embedding, std=0.02)
        # The complete temporal branch is an identity correction at step zero.
        nn.init.zeros_(self.exact_residual.weight)
        nn.init.zeros_(self.exact_residual.bias)
        nn.init.zeros_(self.full_residual.weight)
        nn.init.zeros_(self.full_residual.bias)

    def forward(
        self, inputs: torch.Tensor, base_semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = inputs.shape[0]
        tokens = inputs.reshape(
            batch, self.segment_trs, self.response_lags, self.voxel_dim
        )
        tokens = self.voxel_projection(self.voxel_norm(tokens))
        if self.unique_response_trs:
            prefix = tokens[:, 0, 1:, :].flip(dims=(1,))
            tokens = torch.cat((prefix, tokens[:, :, 0, :]), dim=1)
            tokens = tokens + self.time_embedding[None, :, :]
        else:
            tokens = tokens + self.time_embedding[None, :, None, :]
            tokens = tokens + self.lag_embedding[None, None, :, :]
            tokens = tokens.reshape(batch, self.segment_trs * self.response_lags, -1)
        tokens = self.transformer(tokens)
        scores = (tokens * self.pool_query[None, None, :]).sum(dim=-1)
        scores = scores / math.sqrt(tokens.shape[-1])
        pooled = (tokens * F.softmax(scores, dim=1)[:, :, None]).sum(dim=1)
        pooled = self.final_norm(pooled)
        exact_residual = self.exact_residual(pooled)
        full_residual = self.full_residual(pooled)
        return (
            F.normalize(base_semantic + exact_residual, p=2, dim=-1),
            F.normalize(base_semantic + full_residual, p=2, dim=-1),
            exact_residual,
            full_residual,
        )


@torch.no_grad()
def encode_frozen_base(
    mri: nn.Module,
    mapper: nn.Module,
    rows: np.ndarray,
    temporal_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    mri.eval()
    mapper.eval()
    for start in range(0, len(temporal_indices), batch_size):
        indices = temporal_indices[start : start + batch_size]
        mean_input = torch.as_tensor(
            rows[indices].mean(axis=1), dtype=torch.float32, device=device
        )
        raw = mri(mean_input)
        outputs.append(F.normalize(raw + mapper(raw), p=2, dim=-1).cpu())
    return torch.cat(outputs)


@torch.no_grad()
def encode_temporal(
    model: TemporalResidualEncoder,
    rows: np.ndarray,
    temporal_indices: np.ndarray,
    base: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    exact_chunks: list[torch.Tensor] = []
    full_chunks: list[torch.Tensor] = []
    for start in range(0, len(temporal_indices), batch_size):
        indices = temporal_indices[start : start + batch_size]
        inputs = torch.as_tensor(rows[indices], dtype=torch.float32, device=device)
        exact, full, _, _ = model(inputs, base[start : start + len(indices)].to(device))
        exact_chunks.append(exact.cpu())
        full_chunks.append(full.cpu())
    return torch.cat(exact_chunks), torch.cat(full_chunks)


def interpolated(start: float, end: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return float(end)
    fraction = (epoch - 1) / (epochs - 1)
    return float(start + fraction * (end - start))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Access only train/validation members, even though the source archives
    # also contain a test split.
    per_tr = np.load(args.per_tr_npz, allow_pickle=True)
    train_rows = np.asarray(per_tr["train_x"], dtype=np.float32)
    val_rows = np.asarray(per_tr["val_x"], dtype=np.float32)
    segments = np.load(args.segment_npz, allow_pickle=True)
    train_story = strings(segments["train_story"])
    val_story = strings(segments["val_story"])
    train_start = np.asarray(segments["train_start_tr"], dtype=np.int64)
    val_start = np.asarray(segments["val_start_tr"], dtype=np.int64)
    train_stop = np.asarray(segments["train_stop_tr"], dtype=np.int64)
    val_stop = np.asarray(segments["val_stop_tr"], dtype=np.int64)
    if len(train_start) != args.train_count or len(val_start) != args.val_count:
        raise ValueError("Unexpected train/validation segment counts.")
    train_temporal = temporal_row_indices(
        row_story=per_tr["train_story"], row_tr=per_tr["train_tr"],
        segment_story=segments["train_story"], segment_start=train_start,
        segment_stop=train_stop, segment_trs=args.segment_trs,
    )
    val_temporal = temporal_row_indices(
        row_story=per_tr["val_story"], row_tr=per_tr["val_tr"],
        segment_story=segments["val_story"], segment_start=val_start,
        segment_stop=val_stop, segment_trs=args.segment_trs,
    )

    exact_archive = np.load(args.exact_target_npz, allow_pickle=True)
    full_archive = np.load(args.full_target_npz, allow_pickle=True)
    for archive, label in ((exact_archive, "exact"), (full_archive, "full-window")):
        assert_target_alignment(
            archive, train_story=train_story, val_story=val_story,
            train_start=train_start, val_start=val_start,
            train_stop=train_stop, val_stop=val_stop, label=label,
        )
    exact = F.normalize(
        torch.as_tensor(exact_archive["input_embeddings"], dtype=torch.float32),
        p=2, dim=-1,
    )
    full = F.normalize(
        torch.as_tensor(full_archive["input_embeddings"], dtype=torch.float32),
        p=2, dim=-1,
    )
    train_exact, val_exact = exact[: args.train_count], exact[args.train_count :]
    train_full, val_full = full[: args.train_count], full[args.train_count :]

    mri = load_mri2sem_model(
        args.base_checkpoint, input_dim=train_rows.shape[1], output_dim=384,
        hidden_dim=args.mri_hidden_dim, res_blocks=args.mri_res_blocks,
    ).to(device)
    mapper = load_semantic_mapper(
        args.base_checkpoint, dim=384, hidden_dim=args.mapper_hidden_dim
    ).to(device)
    for module in (mri, mapper):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    train_base = encode_frozen_base(
        mri, mapper, train_rows, train_temporal,
        batch_size=args.base_batch_size, device=device,
    )
    val_base = encode_frozen_base(
        mri, mapper, val_rows, val_temporal,
        batch_size=args.base_batch_size, device=device,
    )
    del mri, mapper

    base_exact_metrics = retrieval_metrics(val_base, val_exact)
    base_full_metrics = retrieval_metrics(val_base, val_full)
    print(json.dumps({"epoch": 0, "exact": base_exact_metrics, "full": base_full_metrics}), flush=True)

    model = TemporalResidualEncoder(
        lagged_dim=train_rows.shape[1], response_lags=args.response_lags,
        segment_trs=args.segment_trs, token_dim=args.token_dim,
        output_dim=384, layers=args.layers, heads=args.heads,
        dropout=args.dropout, unique_response_trs=args.unique_response_trs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_story_array = np.asarray(train_story)
    story_pools = {
        story: np.flatnonzero(train_story_array == story) for story in sorted(set(train_story))
    }
    story_names = sorted(story_pools)
    steps_per_epoch = math.ceil(args.train_count / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))
    generator = np.random.default_rng(args.seed)
    history: list[dict] = []
    best_score = base_exact_metrics["top5"] + 0.25 * base_full_metrics["top5"]
    best_record = {"epoch": 0, "exact": base_exact_metrics, "full": base_full_metrics,
                   "selection_score": best_score}

    for epoch in range(1, args.epochs + 1):
        exact_weight = interpolated(
            args.exact_weight if args.exact_weight_start is None else args.exact_weight_start,
            args.exact_weight, epoch, args.epochs,
        )
        full_weight = interpolated(
            args.full_weight,
            args.full_weight if args.full_weight_end is None else args.full_weight_end,
            epoch, args.epochs,
        )
        model.train()
        losses: list[float] = []
        for step_in_epoch in range(steps_per_epoch):
            selected_stories = generator.integers(0, len(story_names), size=args.batch_size)
            selected_rows = np.asarray([
                generator.choice(story_pools[story_names[index]])
                for index in selected_stories
            ], dtype=np.int64)
            inputs = torch.as_tensor(
                train_rows[train_temporal[selected_rows]], dtype=torch.float32, device=device
            )
            base = train_base[selected_rows].to(device)
            exact_target = train_exact[selected_rows].to(device)
            full_target = train_full[selected_rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            exact_prediction, full_prediction, exact_residual, full_residual = model(inputs, base)
            exact_loss = 1.0 - F.cosine_similarity(exact_prediction, exact_target, dim=-1).mean()
            full_loss = 1.0 - F.cosine_similarity(full_prediction, full_target, dim=-1).mean()
            contrastive = symmetric_contrastive(
                exact_prediction, exact_target, args.contrastive_temperature
            )
            if full_weight > 0.0:
                contrastive = contrastive + symmetric_contrastive(
                    full_prediction, full_target, args.contrastive_temperature
                )
            residual_l2 = exact_residual.square().mean()
            if full_weight > 0.0:
                residual_l2 = residual_l2 + full_residual.square().mean()
            loss = (
                exact_weight * exact_loss + full_weight * full_loss
                + args.contrastive_weight * contrastive
                + args.residual_l2_weight * residual_l2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            global_step = (epoch - 1) * steps_per_epoch + step_in_epoch + 1
            if warmup_steps and global_step <= warmup_steps:
                lr = args.lr * global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr = args.lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_exact_prediction, val_full_prediction = encode_temporal(
            model, val_rows, val_temporal, val_base,
            batch_size=args.eval_batch_size, device=device,
        )
        exact_metrics = retrieval_metrics(val_exact_prediction, val_exact)
        full_metrics = retrieval_metrics(val_full_prediction, val_full)
        score = exact_metrics["top5"] + 0.25 * full_metrics["top5"]
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "lr": optimizer.param_groups[0]["lr"],
            "exact_weight": exact_weight, "full_weight": full_weight,
            "exact": exact_metrics, "full": full_metrics, "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_record = record
            torch.save({
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "args": vars(args), "best": best_record,
                "base_checkpoint": args.base_checkpoint,
                "scientific_scope": "train11725-to-val266; no test key indexed",
            }, output_dir / "best.pt")
            np.savez_compressed(
                output_dir / "best_val_predictions.npz",
                predicted_exact=val_exact_prediction.numpy(), target_exact=val_exact.numpy(),
                predicted_full=val_full_prediction.numpy(), target_full=val_full.numpy(),
                base_prediction=val_base.numpy(), story=np.asarray(val_story, dtype=object),
                start_tr=val_start, stop_tr=val_stop,
            )
        (output_dir / "metrics.json").write_text(json.dumps({
            "status": "running" if epoch < args.epochs else "complete",
            "scientific_scope": "train11725-to-val266; no test key indexed",
            "args": vars(args), "base_exact": base_exact_metrics,
            "base_full": base_full_metrics, "best": best_record, "history": history,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
