#!/usr/bin/env python
"""Train a leakage-safe temporal fMRI encoder on train11725 -> val266.

The current MRI2SEM baseline mean-pools ten TRs before its MLP.  This script
reconstructs the ten row-aligned lagged response vectors from the existing
per-TR train/validation cache and keeps time explicit.  It never indexes any
test key.  A shared voxel projection produces 40 time-by-HRF-lag tokens, a
small Transformer integrates them, and separate heads predict the exact
10-word and complete 20-second MiniLM targets.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tr-npz", required=True)
    parser.add_argument("--segment-npz", required=True)
    parser.add_argument("--exact-target-npz", required=True)
    parser.add_argument("--full-target-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--segment-trs", type=int, default=10)
    parser.add_argument("--response-lags", type=int, default=4)
    parser.add_argument(
        "--unique-response-trs",
        action="store_true",
        help=(
            "Collapse the overlapping lag representation to the 13 unique "
            "chronological response TRs (t-3..t+9) instead of retaining 40 "
            "time-by-lag tokens with duplicated fMRI vectors."
        ),
    )
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--exact-weight", type=float, default=1.0)
    parser.add_argument("--full-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in array.tolist()]


def temporal_row_indices(
    *,
    row_story: np.ndarray,
    row_tr: np.ndarray,
    segment_story: np.ndarray,
    segment_start: np.ndarray,
    segment_stop: np.ndarray,
    segment_trs: int,
) -> np.ndarray:
    lookup = {
        (story, int(tr)): index
        for index, (story, tr) in enumerate(zip(strings(row_story), row_tr.tolist()))
    }
    result = np.empty((len(segment_start), segment_trs), dtype=np.int64)
    for row, (story, start, stop) in enumerate(
        zip(strings(segment_story), segment_start.tolist(), segment_stop.tolist())
    ):
        start = int(start)
        stop = int(stop)
        # build_segment_cache records stop_tr as the final included TR.
        # Preserve all ten TRs instead of silently interpreting stop_tr as an
        # exclusive Python-slice endpoint.
        inclusive_length = stop - start + 1
        if inclusive_length != segment_trs:
            raise ValueError(
                f"Segment {row} has inclusive [{start}, {stop}] length "
                f"{inclusive_length}, expected {segment_trs}."
            )
        try:
            result[row] = [lookup[(story, tr)] for tr in range(start, stop + 1)]
        except KeyError as error:
            raise KeyError(f"Missing per-TR row for segment {row}: {error}") from error
    return result


def assert_target_alignment(
    target: np.lib.npyio.NpzFile,
    *,
    train_story: list[str],
    val_story: list[str],
    train_start: np.ndarray,
    val_start: np.ndarray,
    train_stop: np.ndarray,
    val_stop: np.ndarray,
    label: str,
) -> None:
    expected_story = train_story + val_story
    if "story" in target and strings(target["story"]) != expected_story:
        raise ValueError(f"{label} story metadata is not aligned to train11725+val266.")
    expected_start = np.concatenate([train_start, val_start])
    expected_stop = np.concatenate([train_stop, val_stop])
    if "start_tr" in target and not np.array_equal(target["start_tr"], expected_start):
        raise ValueError(f"{label} start_tr metadata is misaligned.")
    if "stop_tr" in target and not np.array_equal(target["stop_tr"], expected_stop):
        raise ValueError(f"{label} stop_tr metadata is misaligned.")


class TemporalLagFMRIEncoder(nn.Module):
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
        unique_response_trs: bool = False,
    ) -> None:
        super().__init__()
        if lagged_dim % response_lags:
            raise ValueError(
                f"lagged_dim={lagged_dim} is not divisible by response_lags={response_lags}."
            )
        self.response_lags = int(response_lags)
        self.segment_trs = int(segment_trs)
        self.voxel_dim = lagged_dim // response_lags
        self.unique_response_trs = bool(unique_response_trs)
        self.sequence_length = (
            segment_trs + response_lags - 1
            if self.unique_response_trs
            else segment_trs * response_lags
        )
        self.voxel_norm = nn.LayerNorm(self.voxel_dim)
        self.voxel_projection = nn.Linear(self.voxel_dim, token_dim)
        self.time_embedding = nn.Parameter(
            torch.zeros(
                self.sequence_length if self.unique_response_trs else segment_trs,
                token_dim,
            )
        )
        self.lag_embedding = (
            None
            if self.unique_response_trs
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
        self.exact_head = nn.Linear(token_dim, output_dim)
        self.full_head = nn.Linear(token_dim, output_dim)
        nn.init.normal_(self.time_embedding, std=0.02)
        if self.lag_embedding is not None:
            nn.init.normal_(self.lag_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.segment_trs:
            raise ValueError(
                f"Expected [B, {self.segment_trs}, lagged_dim], got {tuple(inputs.shape)}"
            )
        batch = inputs.shape[0]
        tokens = inputs.reshape(
            batch, self.segment_trs, self.response_lags, self.voxel_dim
        )
        tokens = self.voxel_projection(self.voxel_norm(tokens))
        if self.unique_response_trs:
            # Each lagged row j contains responses at j, j-1, ..., j-(L-1).
            # The chronological union for a segment is therefore -(L-1)..T-1.
            # Earlier responses come from the first row's reversed nonzero
            # lag blocks; segment responses come from lag zero of every row.
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
        return (
            F.normalize(self.exact_head(pooled), p=2, dim=-1),
            F.normalize(self.full_head(pooled), p=2, dim=-1),
        )


def symmetric_contrastive(
    predicted: torch.Tensor, target: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = F.normalize(predicted, p=2, dim=-1) @ F.normalize(target, p=2, dim=-1).T
    logits = logits / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def retrieval_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = F.normalize(predicted.float().cpu(), p=2, dim=-1)
    target = F.normalize(target.float().cpu(), p=2, dim=-1)
    similarity = predicted @ target.T
    matched = similarity.diag()
    ranks = 1 + (similarity > matched[:, None]).sum(dim=1)
    return {
        "cosine": float(matched.mean()),
        "top1": float((ranks == 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }


def encode_split(
    model: nn.Module,
    source: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    exact_chunks: list[torch.Tensor] = []
    full_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = indices[start : start + batch_size]
            inputs = torch.as_tensor(source[rows], dtype=torch.float32, device=device)
            exact, full = model(inputs)
            exact_chunks.append(exact.float().cpu())
            full_chunks.append(full.float().cpu())
    return torch.cat(exact_chunks), torch.cat(full_chunks)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately access only train/validation members.  The archive may
    # contain test members, but they are neither indexed nor materialized.
    per_tr = np.load(args.per_tr_npz, allow_pickle=True)
    train_rows = np.asarray(per_tr["train_x"], dtype=np.float32)
    val_rows = np.asarray(per_tr["val_x"], dtype=np.float32)
    train_row_story = per_tr["train_story"]
    val_row_story = per_tr["val_story"]
    train_row_tr = per_tr["train_tr"]
    val_row_tr = per_tr["val_tr"]

    segments = np.load(args.segment_npz, allow_pickle=True)
    train_story = strings(segments["train_story"])
    val_story = strings(segments["val_story"])
    train_start = np.asarray(segments["train_start_tr"], dtype=np.int64)
    val_start = np.asarray(segments["val_start_tr"], dtype=np.int64)
    train_stop = np.asarray(segments["train_stop_tr"], dtype=np.int64)
    val_stop = np.asarray(segments["val_stop_tr"], dtype=np.int64)
    if len(train_start) != args.train_count or len(val_start) != args.val_count:
        raise ValueError(
            f"Expected train/val {args.train_count}/{args.val_count}, "
            f"got {len(train_start)}/{len(val_start)}."
        )
    train_temporal = temporal_row_indices(
        row_story=train_row_story,
        row_tr=train_row_tr,
        segment_story=segments["train_story"],
        segment_start=train_start,
        segment_stop=train_stop,
        segment_trs=args.segment_trs,
    )
    val_temporal = temporal_row_indices(
        row_story=val_row_story,
        row_tr=val_row_tr,
        segment_story=segments["val_story"],
        segment_start=val_start,
        segment_stop=val_stop,
        segment_trs=args.segment_trs,
    )

    exact_archive = np.load(args.exact_target_npz, allow_pickle=True)
    full_archive = np.load(args.full_target_npz, allow_pickle=True)
    assert_target_alignment(
        exact_archive,
        train_story=train_story,
        val_story=val_story,
        train_start=train_start,
        val_start=val_start,
        train_stop=train_stop,
        val_stop=val_stop,
        label="exact target",
    )
    assert_target_alignment(
        full_archive,
        train_story=train_story,
        val_story=val_story,
        train_start=train_start,
        val_start=val_start,
        train_stop=train_stop,
        val_stop=val_stop,
        label="full-window target",
    )
    exact = torch.as_tensor(exact_archive["input_embeddings"], dtype=torch.float32)
    full = torch.as_tensor(full_archive["input_embeddings"], dtype=torch.float32)
    expected_total = args.train_count + args.val_count
    if exact.shape != (expected_total, 384) or full.shape != (expected_total, 384):
        raise ValueError(f"Unexpected target shapes exact={exact.shape} full={full.shape}")
    exact = F.normalize(exact, p=2, dim=-1)
    full = F.normalize(full, p=2, dim=-1)
    train_exact, val_exact = exact[: args.train_count], exact[args.train_count :]
    train_full, val_full = full[: args.train_count], full[args.train_count :]

    model = TemporalLagFMRIEncoder(
        lagged_dim=train_rows.shape[1],
        response_lags=args.response_lags,
        segment_trs=args.segment_trs,
        token_dim=args.token_dim,
        output_dim=384,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        unique_response_trs=args.unique_response_trs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    story_pools: dict[str, np.ndarray] = {}
    train_story_array = np.asarray(train_story)
    for story in sorted(set(train_story)):
        story_pools[story] = np.flatnonzero(train_story_array == story)
    story_names = sorted(story_pools)
    steps_per_epoch = math.ceil(args.train_count / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))
    generator = np.random.default_rng(args.seed)
    history: list[dict] = []
    best_score = float("-inf")
    best_record: dict | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for step_in_epoch in range(steps_per_epoch):
            selected_stories = generator.integers(0, len(story_names), size=args.batch_size)
            selected_rows = np.asarray(
                [
                    generator.choice(story_pools[story_names[index]])
                    for index in selected_stories
                ],
                dtype=np.int64,
            )
            inputs = torch.as_tensor(
                train_rows[train_temporal[selected_rows]], dtype=torch.float32, device=device
            )
            exact_target = train_exact[selected_rows].to(device)
            full_target = train_full[selected_rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            exact_prediction, full_prediction = model(inputs)
            exact_loss = 1.0 - F.cosine_similarity(
                exact_prediction, exact_target, dim=-1
            ).mean()
            full_loss = 1.0 - F.cosine_similarity(
                full_prediction, full_target, dim=-1
            ).mean()
            contrastive = exact_prediction.sum() * 0.0
            if args.contrastive_weight > 0.0:
                contrastive = symmetric_contrastive(
                    exact_prediction, exact_target, args.contrastive_temperature
                )
                if args.full_weight > 0.0:
                    contrastive = contrastive + symmetric_contrastive(
                        full_prediction, full_target, args.contrastive_temperature
                    )
            loss = (
                args.exact_weight * exact_loss
                + args.full_weight * full_loss
                + args.contrastive_weight * contrastive
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            global_step = (epoch - 1) * steps_per_epoch + step_in_epoch + 1
            if warmup_steps and global_step <= warmup_steps:
                lr = args.lr * global_step / warmup_steps
            else:
                decay_progress = (global_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                lr = args.lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * decay_progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_exact_prediction, val_full_prediction = encode_split(
            model,
            val_rows,
            val_temporal,
            batch_size=args.eval_batch_size,
            device=device,
        )
        exact_metrics = retrieval_metrics(val_exact_prediction, val_exact)
        full_metrics = retrieval_metrics(val_full_prediction, val_full)
        score = exact_metrics["top5"] + (
            0.25 * full_metrics["top5"] if args.full_weight > 0.0 else 0.0
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "lr": optimizer.param_groups[0]["lr"],
            "exact": exact_metrics,
            "full": full_metrics,
            "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_record = record
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "args": vars(args),
                    "best": best_record,
                    "scientific_scope": "train11725-to-val266; no test key indexed",
                },
                output_dir / "best.pt",
            )
            np.savez_compressed(
                output_dir / "best_val_predictions.npz",
                predicted_exact=val_exact_prediction.numpy(),
                target_exact=val_exact.numpy(),
                predicted_full=val_full_prediction.numpy(),
                target_full=val_full.numpy(),
                story=np.asarray(val_story, dtype=object),
                start_tr=val_start,
                stop_tr=val_stop,
            )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "status": "running" if epoch < args.epochs else "complete",
                    "scientific_scope": "train11725-to-val266; no test key indexed",
                    "args": vars(args),
                    "train_shape": [args.train_count, args.segment_trs, train_rows.shape[1]],
                    "val_shape": [args.val_count, args.segment_trs, val_rows.shape[1]],
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "best": best_record,
                    "history": history,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
