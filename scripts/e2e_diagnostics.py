#!/usr/bin/env python
"""E2E diagnostics for MEG2SEM ADA vectors before and after ELF tuning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F

from modules.meg2sem_bridge import load_meg2sem_model


@dataclass
class PackedMEGExamples:
    meg: torch.Tensor
    meg_lengths: torch.Tensor
    semantic_vectors: torch.Tensor
    sentences: list[str]
    subject_labels: list[str]
    source_path: str


class SemanticVectorContextProjector(torch.nn.Module):
    """Project one semantic embedding per sentence into ELF context tokens."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.context_dim = int(context_dim)
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden_dim, context_length * context_dim),
        )

    def forward(self, semantic_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.net(semantic_vectors).reshape(
            semantic_vectors.shape[0],
            self.context_length,
            self.context_dim,
        )
        context_mask = torch.ones(
            semantic_vectors.shape[0],
            self.context_length,
            dtype=context.dtype,
            device=semantic_vectors.device,
        )
        return context, context_mask


def strings(array) -> list[str]:  # noqa: ANN001
    out = []
    for item in array:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def limit_count(total: int, limit: int) -> int:
    return total if limit <= 0 else min(total, limit)


def load_packed_npz(args: argparse.Namespace, path: str, *, limit: int = 0) -> PackedMEGExamples:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    for key in (args.meg_key, args.semantic_key, args.sentence_key):
        if key not in keys:
            raise KeyError(f"{path} missing required key {key!r}; available keys={sorted(keys)}")

    total_n = int(data[args.semantic_key].shape[0])
    n = limit_count(total_n, limit)
    meg_np = np.asarray(data[args.meg_key][:n])
    semantic_np = np.asarray(data[args.semantic_key][:n], dtype=np.float32)
    sentences = strings(data[args.sentence_key][:n])

    if args.meg_lengths_key in keys:
        meg_lengths_np = np.asarray(data[args.meg_lengths_key][:n], dtype=np.int64)
    elif args.meg_mask_key in keys:
        meg_lengths_np = np.asarray(data[args.meg_mask_key][:n]).sum(axis=1).astype(np.int64)
    else:
        meg_lengths_np = np.full((n,), meg_np.shape[-1], dtype=np.int64)

    if args.subject_key in keys:
        subject_labels = strings(data[args.subject_key][:n])
    else:
        subject_labels = ["0"] * n

    return PackedMEGExamples(
        meg=torch.as_tensor(meg_np),
        meg_lengths=torch.as_tensor(meg_lengths_np, dtype=torch.long),
        semantic_vectors=torch.as_tensor(semantic_np, dtype=torch.float32),
        sentences=sentences,
        subject_labels=subject_labels,
        source_path=path,
    )


def load_train_subjects_only(args: argparse.Namespace, path: str) -> PackedMEGExamples:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    if args.subject_key in keys:
        subject_labels = strings(data[args.subject_key])
    else:
        subject_labels = ["0"]
    n = len(subject_labels)
    return PackedMEGExamples(
        meg=torch.empty((n, 0, 0), dtype=torch.float16),
        meg_lengths=torch.zeros((n,), dtype=torch.long),
        semantic_vectors=torch.empty((n, 0), dtype=torch.float32),
        sentences=[""] * n,
        subject_labels=subject_labels,
        source_path=path,
    )


def meg_valid_mask(meg: torch.Tensor, meg_lengths: torch.Tensor) -> torch.Tensor:
    time = torch.arange(meg.shape[-1], dtype=torch.long)
    return time.unsqueeze(0) < meg_lengths.to(dtype=torch.long).clamp(min=0).unsqueeze(1)


def compute_train_meg_stats(
    meg: torch.Tensor,
    meg_lengths: torch.Tensor,
    *,
    eps: float,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(meg.shape[1])
    sums = torch.zeros((channels,), dtype=torch.float64)
    sq_sums = torch.zeros((channels,), dtype=torch.float64)
    counts = torch.zeros((channels,), dtype=torch.float64)

    for start in range(0, meg.shape[0], chunk_size):
        end = min(start + chunk_size, meg.shape[0])
        chunk = meg[start:end].to(dtype=torch.float32)
        mask = meg_valid_mask(chunk, meg_lengths[start:end]).to(dtype=torch.float32)
        sums += (chunk * mask[:, None, :]).sum(dim=(0, 2), dtype=torch.float64)
        sq_sums += ((chunk * chunk) * mask[:, None, :]).sum(dim=(0, 2), dtype=torch.float64)
        counts += mask.sum(dim=(0, 1)).to(dtype=torch.float64).expand(channels)

    counts = counts.clamp_min(1.0)
    mean = sums / counts
    var = (sq_sums / counts) - (mean * mean)
    std = torch.sqrt(var.clamp_min(float(eps) ** 2))
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)


def apply_meg_standardization(
    examples: PackedMEGExamples,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    clip_boundary: float,
    chunk_size: int = 128,
) -> PackedMEGExamples:
    out = torch.empty_like(examples.meg, dtype=torch.float16)
    mean = mean.to(dtype=torch.float32).view(1, -1, 1)
    std = std.to(dtype=torch.float32).view(1, -1, 1)
    do_clip = clip_boundary > 0.0

    for start in range(0, examples.meg.shape[0], chunk_size):
        end = min(start + chunk_size, examples.meg.shape[0])
        chunk = (examples.meg[start:end].to(dtype=torch.float32) - mean) / std
        if do_clip:
            chunk = chunk.clamp(min=-clip_boundary, max=clip_boundary)
        out[start:end] = chunk.to(dtype=torch.float16)

    return PackedMEGExamples(
        meg=out,
        meg_lengths=examples.meg_lengths,
        semantic_vectors=examples.semantic_vectors,
        sentences=examples.sentences,
        subject_labels=examples.subject_labels,
        source_path=examples.source_path,
    )


def maybe_standardize_meg(
    args: argparse.Namespace,
    train_examples: PackedMEGExamples,
    eval_examples: PackedMEGExamples,
) -> tuple[PackedMEGExamples, PackedMEGExamples, dict[str, object]]:
    if args.meg_standardization == "none":
        return train_examples, eval_examples, {
            "meg_standardization": "none",
            "meg_dtype_after_load": str(train_examples.meg.dtype),
        }

    mean, std = compute_train_meg_stats(
        train_examples.meg,
        train_examples.meg_lengths,
        eps=args.meg_standardize_eps,
    )
    clip_boundary = float(args.meg_clip_boundary)
    train_standardized = apply_meg_standardization(
        train_examples,
        mean=mean,
        std=std,
        clip_boundary=clip_boundary,
    )
    eval_standardized = apply_meg_standardization(
        eval_examples,
        mean=mean,
        std=std,
        clip_boundary=clip_boundary,
    )
    stats = {
        "meg_standardization": args.meg_standardization,
        "meg_stats_source": "train_split",
        "meg_mean_abs_mean": float(mean.abs().mean().item()),
        "meg_std_mean": float(std.mean().item()),
        "meg_std_min": float(std.min().item()),
        "meg_std_max": float(std.max().item()),
        "meg_clip_boundary": None if clip_boundary <= 0.0 else clip_boundary,
        "meg_dtype_after_standardization": str(train_standardized.meg.dtype),
    }
    return train_standardized, eval_standardized, stats


def assign_subject_ids(
    train: PackedMEGExamples,
    eval_examples: PackedMEGExamples,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    labels = sorted(set(train.subject_labels) | set(eval_examples.subject_labels))
    subject_to_id = {subject: idx for idx, subject in enumerate(labels)}
    train_ids = torch.tensor([subject_to_id[label] for label in train.subject_labels], dtype=torch.long)
    eval_ids = torch.tensor([subject_to_id[label] for label in eval_examples.subject_labels], dtype=torch.long)
    return train_ids, eval_ids, subject_to_id


def load_teacher_projector(
    *,
    checkpoint_path: str,
    input_dim: int,
    context_dim: int,
    context_length: int,
    hidden_dim: int,
    device: torch.device,
) -> SemanticVectorContextProjector:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict) or "adapter_state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} must contain adapter_state_dict.")
    projector = SemanticVectorContextProjector(
        input_dim=input_dim,
        context_dim=context_dim,
        context_length=context_length,
        hidden_dim=hidden_dim,
    ).to(device)
    projector.load_state_dict(ckpt["adapter_state_dict"])
    projector.eval()
    for param in projector.parameters():
        param.requires_grad_(False)
    return projector


@dataclass
class VectorBundle:
    true: torch.Tensor
    pre_raw: torch.Tensor
    pre_projector_input: torch.Tensor
    pre_residual_output: torch.Tensor | None
    true_residual_target: torch.Tensor | None
    post_raw: torch.Tensor | None
    post_projector_input: torch.Tensor | None
    true_projector_input: torch.Tensor
    sentences: list[str]


def cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T


def finite_mean(tensor: torch.Tensor) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().cpu())


def finite_std(tensor: torch.Tensor) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.std(unbiased=False).cpu())


def paired_metrics(pred: torch.Tensor, target: torch.Tensor, *, prefix: str = "") -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    cosine = F.cosine_similarity(pred, target, dim=-1)
    diff = pred - target
    sim = cosine_matrix(pred, target)
    ids = torch.arange(sim.shape[0], device=sim.device)
    true_similarity = sim[ids, ids]
    ranks = (sim > true_similarity[:, None]).sum(dim=1) + 1
    out = {
        "cosine_mean": float(cosine.mean().cpu()),
        "cosine_std": float(cosine.std(unbiased=False).cpu()),
        "cosine_min": float(cosine.min().cpu()),
        "cosine_max": float(cosine.max().cpu()),
        "mse": float(F.mse_loss(pred, target).cpu()),
        "rmse_mean": float(diff.pow(2).mean(dim=-1).sqrt().mean().cpu()),
        "l2_mean": float(diff.norm(dim=-1).mean().cpu()),
        "l2_std": float(diff.norm(dim=-1).std(unbiased=False).cpu()),
        "pred_norm_mean": float(pred.norm(dim=-1).mean().cpu()),
        "target_norm_mean": float(target.norm(dim=-1).mean().cpu()),
        "pred_dim_std_mean": float(pred.std(dim=0, unbiased=False).mean().cpu()),
        "target_dim_std_mean": float(target.std(dim=0, unbiased=False).mean().cpu()),
        "retrieval_top1": float((ranks == 1).float().mean().cpu()),
        "retrieval_top5": float((ranks <= min(5, sim.shape[1])).float().mean().cpu()),
        "retrieval_mean_rank": float(ranks.float().mean().cpu()),
        "retrieval_median_rank": float(ranks.float().median().cpu()),
    }
    if prefix:
        return {f"{prefix}_{key}": value for key, value in out.items()}
    return out


def retrieval_geometry(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred = pred.float()
    target = target.float()
    n = int(target.shape[0])
    ids = torch.arange(n, device=target.device)
    sim = cosine_matrix(pred, target)
    true_similarity = sim[ids, ids]
    ranks = (sim > true_similarity[:, None]).sum(dim=1) + 1

    if n <= 1:
        nearest_wrong_index = torch.full((n,), -1, dtype=torch.long, device=target.device)
        nearest_wrong_similarity = torch.full((n,), float("nan"), dtype=sim.dtype, device=target.device)
        random_wrong_index = nearest_wrong_index.clone()
        random_wrong_similarity = nearest_wrong_similarity.clone()
    else:
        wrong_sim = sim.clone()
        wrong_sim[ids, ids] = -float("inf")
        nearest_wrong_similarity, nearest_wrong_index = wrong_sim.max(dim=1)
        random_wrong_index = (ids * 37 + 17) % n
        random_wrong_index = torch.where(random_wrong_index == ids, (ids + 1) % n, random_wrong_index)
        random_wrong_similarity = sim[ids, random_wrong_index]

    return {
        "similarity_matrix": sim,
        "true_similarity": true_similarity,
        "true_rank": ranks,
        "nearest_wrong_index": nearest_wrong_index,
        "nearest_wrong_similarity": nearest_wrong_similarity,
        "random_wrong_index": random_wrong_index,
        "random_wrong_similarity": random_wrong_similarity,
    }


def semantic_error_geometry(pred: torch.Tensor, target: torch.Tensor, *, prefix: str = "") -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    pred = pred.float()
    target = target.float()
    n = int(target.shape[0])
    retrieval = retrieval_geometry(pred, target)
    true_similarity = retrieval["true_similarity"]
    true_rank = retrieval["true_rank"]
    nearest_wrong_index = retrieval["nearest_wrong_index"]
    nearest_wrong_similarity = retrieval["nearest_wrong_similarity"]
    random_wrong_index = retrieval["random_wrong_index"]
    random_wrong_similarity = retrieval["random_wrong_similarity"]
    margin = true_similarity - nearest_wrong_similarity

    error = pred - target
    if n <= 1:
        error_to_nearest_wrong_cosine = torch.full((n,), float("nan"), dtype=pred.dtype, device=pred.device)
        error_to_random_wrong_cosine = torch.full((n,), float("nan"), dtype=pred.dtype, device=pred.device)
        projection_to_nearest_wrong_fraction = torch.full((n,), float("nan"), dtype=pred.dtype, device=pred.device)
        projection_to_random_wrong_fraction = torch.full((n,), float("nan"), dtype=pred.dtype, device=pred.device)
    else:
        nearest_wrong_direction = target[nearest_wrong_index] - target
        random_wrong_direction = target[random_wrong_index] - target
        error_to_nearest_wrong_cosine = F.cosine_similarity(error, nearest_wrong_direction, dim=-1)
        error_to_random_wrong_cosine = F.cosine_similarity(error, random_wrong_direction, dim=-1)
        projection_to_nearest_wrong_fraction = (
            (error * nearest_wrong_direction).sum(dim=-1) / nearest_wrong_direction.pow(2).sum(dim=-1).clamp_min(1e-12)
        )
        projection_to_random_wrong_fraction = (
            (error * random_wrong_direction).sum(dim=-1) / random_wrong_direction.pow(2).sum(dim=-1).clamp_min(1e-12)
        )

    delta_wrong_alignment = error_to_nearest_wrong_cosine - error_to_random_wrong_cosine
    out = {
        "true_similarity_mean": float(true_similarity.mean().cpu()),
        "true_similarity_std": float(true_similarity.std(unbiased=False).cpu()),
        "nearest_wrong_similarity_mean": finite_mean(nearest_wrong_similarity),
        "random_wrong_similarity_mean": finite_mean(random_wrong_similarity),
        "true_minus_nearest_wrong_margin_mean": finite_mean(margin),
        "true_minus_nearest_wrong_margin_min": float(margin.nan_to_num(nan=0.0).min().cpu()),
        "nearest_wrong_beats_true_fraction": float((margin < 0).float().mean().cpu()),
        "true_top1_fraction": float((true_rank == 1).float().mean().cpu()),
        "true_top5_fraction": float((true_rank <= min(5, n)).float().mean().cpu()),
        "true_mean_rank": float(true_rank.float().mean().cpu()),
        "error_to_nearest_wrong_cosine_mean": finite_mean(error_to_nearest_wrong_cosine),
        "error_to_nearest_wrong_cosine_std": finite_std(error_to_nearest_wrong_cosine),
        "error_to_random_wrong_cosine_mean": finite_mean(error_to_random_wrong_cosine),
        "error_to_random_wrong_cosine_std": finite_std(error_to_random_wrong_cosine),
        "error_to_nearest_minus_random_cosine_mean": finite_mean(delta_wrong_alignment),
        "projection_to_nearest_wrong_fraction_mean": finite_mean(projection_to_nearest_wrong_fraction),
        "projection_to_random_wrong_fraction_mean": finite_mean(projection_to_random_wrong_fraction),
        "positive_nearest_wrong_direction_fraction": float((error_to_nearest_wrong_cosine > 0).float().mean().cpu()),
    }
    if prefix:
        out = {f"{prefix}_{key}": value for key, value in out.items()}
    details = {
        **retrieval,
        "true_minus_nearest_wrong_margin": margin,
        "error_to_nearest_wrong_cosine": error_to_nearest_wrong_cosine,
        "error_to_random_wrong_cosine": error_to_random_wrong_cosine,
        "error_to_nearest_minus_random_cosine": delta_wrong_alignment,
        "projection_to_nearest_wrong_fraction": projection_to_nearest_wrong_fraction,
        "projection_to_random_wrong_fraction": projection_to_random_wrong_fraction,
    }
    return out, details


def pca_components(vectors: torch.Tensor, num_components: int) -> torch.Tensor:
    if num_components <= 0:
        return torch.empty((0, vectors.shape[-1]), dtype=torch.float32)
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    q = min(max(num_components + 8, num_components), centered.shape[0], centered.shape[1])
    _, _, vh = torch.pca_lowrank(centered, q=q, center=False)
    return vh.T[:num_components].contiguous()


def remove_components(vectors: torch.Tensor, basis: torch.Tensor, *, center: torch.Tensor) -> torch.Tensor:
    residual = vectors.float() - center.float()
    if basis.numel() == 0:
        return residual
    return residual - (residual @ basis.T) @ basis


def component_energy_report(vectors: torch.Tensor, basis: torch.Tensor, *, center: torch.Tensor, prefix: str) -> dict[str, float]:
    centered = vectors.float() - center.float()
    total_energy = centered.pow(2).sum(dim=-1).clamp_min(1e-12)
    if basis.numel() == 0:
        removed = torch.zeros_like(centered)
    else:
        removed = (centered @ basis.T) @ basis
    residual = centered - removed
    removed_energy = removed.pow(2).sum(dim=-1)
    residual_energy = residual.pow(2).sum(dim=-1)
    return {
        f"{prefix}_centered_norm_mean": float(centered.norm(dim=-1).mean().cpu()),
        f"{prefix}_removed_norm_mean": float(removed.norm(dim=-1).mean().cpu()),
        f"{prefix}_residual_norm_mean": float(residual.norm(dim=-1).mean().cpu()),
        f"{prefix}_removed_energy_fraction_mean": float((removed_energy / total_energy).mean().cpu()),
        f"{prefix}_residual_energy_fraction_mean": float((residual_energy / total_energy).mean().cpu()),
    }


def error_component_report(
    pred: torch.Tensor,
    target: torch.Tensor,
    basis: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    error = pred.float() - target.float()
    total_energy = error.pow(2).sum(dim=-1).clamp_min(1e-12)
    if basis.numel() == 0:
        component_error = torch.zeros_like(error)
    else:
        component_error = (error @ basis.T) @ basis
    residual_error = error - component_error
    return {
        f"{prefix}_error_norm_mean": float(error.norm(dim=-1).mean().cpu()),
        f"{prefix}_component_error_norm_mean": float(component_error.norm(dim=-1).mean().cpu()),
        f"{prefix}_residual_error_norm_mean": float(residual_error.norm(dim=-1).mean().cpu()),
        f"{prefix}_component_error_energy_fraction_mean": float(
            (component_error.pow(2).sum(dim=-1) / total_energy).mean().cpu()
        ),
        f"{prefix}_residual_error_energy_fraction_mean": float(
            (residual_error.pow(2).sum(dim=-1) / total_energy).mean().cpu()
        ),
    }


def principal_component_report(vectors: torch.Tensor, *, max_components: int = 10) -> dict[str, object]:
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    q = min(max_components, centered.shape[0], centered.shape[1])
    _, singular, _ = torch.pca_lowrank(centered, q=q, center=False)
    variance = singular.pow(2)
    explained = variance / variance.sum().clamp_min(1e-12)
    return {
        "top_explained_variance": [float(x) for x in explained[:max_components].cpu().tolist()],
        "cumulative_top_explained_variance": [float(x) for x in explained[:max_components].cumsum(0).cpu().tolist()],
    }


def load_mean_components(path: str, *, semantic_key: str, max_rows: int, num_components: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = np.load(path, allow_pickle=True)
    if semantic_key not in data.files:
        raise KeyError(f"{path} missing semantic key {semantic_key!r}; keys={data.files}")
    array = np.asarray(data[semantic_key], dtype=np.float32)
    if max_rows > 0:
        array = array[:max_rows]
    vectors = torch.as_tensor(array, dtype=torch.float32)
    center = vectors.mean(dim=0, keepdim=True)
    basis = pca_components(vectors, num_components)
    report = {
        "component_source": path,
        "component_rows": int(vectors.shape[0]),
        "component_dim": int(vectors.shape[1]),
        "num_removed_components": int(basis.shape[0]),
    }
    report.update(principal_component_report(vectors))
    return center, basis, report


def load_component_json(path: str, *, num_components: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    center = torch.as_tensor(payload["center"], dtype=torch.float32).unsqueeze(0)
    basis = torch.as_tensor(payload["basis"], dtype=torch.float32)
    if num_components is not None and num_components >= 0:
        basis = basis[:num_components]
    report = {
        "component_source": path,
        "component_rows": payload.get("component_rows"),
        "component_dim": int(center.shape[1]),
        "num_removed_components": int(basis.shape[0]),
    }
    return center, basis, report


def save_component_json(path: str | Path, center: torch.Tensor, basis: torch.Tensor, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **report,
        "center": center.squeeze(0).cpu().tolist(),
        "basis": basis.cpu().tolist(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_component_center(path: str, *, semantic_key: str, max_rows: int) -> tuple[torch.Tensor, dict]:
    data = np.load(path, allow_pickle=True)
    if "train_mean" in data.files:
        center = torch.as_tensor(np.asarray(data["train_mean"], dtype=np.float32), dtype=torch.float32).reshape(1, -1)
        return center, {
            "component_source": path,
            "component_rows": None,
            "component_dim": int(center.shape[1]),
            "component_key": "train_mean",
            "residual_reconstruction": "normalize(train_mean + predicted_residual)",
        }
    if semantic_key not in data.files:
        raise KeyError(f"{path} missing semantic key {semantic_key!r}; keys={data.files}")
    array = np.asarray(data[semantic_key], dtype=np.float32)
    if max_rows > 0:
        array = array[:max_rows]
    vectors = torch.as_tensor(array, dtype=torch.float32)
    vectors = F.normalize(vectors, p=2, dim=-1)
    center = vectors.mean(dim=0, keepdim=True)
    return center, {
        "component_source": path,
        "component_rows": int(vectors.shape[0]),
        "component_dim": int(vectors.shape[1]),
        "residual_reconstruction": "normalize(train_mean + predicted_residual)",
    }


def _make_args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def load_examples(args: argparse.Namespace):
    loader_args = _make_args(
        meg_key=args.meg_key,
        meg_lengths_key=args.meg_lengths_key,
        meg_mask_key=args.meg_mask_key,
        semantic_key=args.semantic_key,
        sentence_key=args.sentence_key,
        subject_key=args.subject_key,
        meg_standardization=args.meg_standardization,
        meg_standardize_eps=args.meg_standardize_eps,
        meg_clip_boundary=args.meg_clip_boundary,
    )
    eval_examples = load_packed_npz(loader_args, args.val_npz, limit=args.eval_num_examples)
    if args.meg_standardization == "none":
        train = load_train_subjects_only(loader_args, args.train_npz)
        meg_norm = {
            "meg_standardization": "none",
            "meg_dtype_after_load": str(eval_examples.meg.dtype),
            "train_meg_load": "skipped",
        }
    else:
        train = load_packed_npz(loader_args, args.train_npz, limit=args.train_limit)
        train, eval_examples, meg_norm = maybe_standardize_meg(loader_args, train, eval_examples)
    train_subject_ids, eval_subject_ids, subject_to_id = assign_subject_ids(train, eval_examples)
    return train, eval_examples, train_subject_ids, eval_subject_ids, subject_to_id, meg_norm


@torch.no_grad()
def collect_vectors(args: argparse.Namespace, device: torch.device) -> tuple[VectorBundle, dict]:
    train, eval_examples, _, eval_subject_ids, subject_to_id, meg_norm = load_examples(args)
    del train
    meg = eval_examples.meg.to(device=device, dtype=torch.float32)
    meg_lengths = eval_examples.meg_lengths.to(device=device, dtype=torch.long)
    subject_ids = eval_subject_ids.to(device=device, dtype=torch.long)
    true = eval_examples.semantic_vectors.to(device=device, dtype=torch.float32)

    pre_meg2sem, load_info = load_meg2sem_model(
        args.meg2sem_checkpoint,
        output_normalization=args.meg2sem_output_normalization,
        device=device,
    )
    pre_meg2sem.eval()
    pre_raw = pre_meg2sem(meg, meg_lengths=meg_lengths, subjects=subject_ids)
    normalize = bool(load_info.normalize_output)
    true_projector_input = F.normalize(true.float(), p=2, dim=-1) if normalize else true.float()
    pre_residual_output = None
    true_residual_target = None
    residual_meta = None
    if args.meg2sem_prediction_mode == "mean_residual":
        if not args.residual_mean_npz:
            raise ValueError("--residual-mean-npz is required with --meg2sem-prediction-mode mean_residual.")
        residual_center, residual_meta = load_component_center(
            args.residual_mean_npz,
            semantic_key=args.semantic_key,
            max_rows=args.component_max_rows,
        )
        residual_center = residual_center.to(device=device, dtype=torch.float32)
        pre_residual_output = pre_raw.float()
        residual_scale = float(args.residual_target_scale)
        if residual_scale <= 0.0:
            raise ValueError("--residual-target-scale must be positive.")
        residual_for_reconstruction = pre_residual_output / residual_scale
        true_residual_target = true_projector_input - residual_center
        pre_projector_input = F.normalize(residual_center + residual_for_reconstruction, p=2, dim=-1)
    else:
        pre_projector_input = F.normalize(pre_raw.float(), p=2, dim=-1) if normalize else pre_raw.float()

    post_raw = None
    post_projector_input = None
    if args.e2e_checkpoint:
        if args.meg2sem_prediction_mode != "full_ada":
            raise NotImplementedError("E2E post-checkpoint diagnostics currently support only full_ada MEG2SEM outputs.")
        if not args.semantic_projector_checkpoint:
            raise ValueError("--semantic-projector-checkpoint is required with --e2e-checkpoint.")
        semantic_projector = load_teacher_projector(
            checkpoint_path=args.semantic_projector_checkpoint,
            input_dim=int(true.shape[-1]),
            context_dim=args.context_dim,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            device=device,
        )
        post_meg2sem, _ = load_meg2sem_model(
            args.meg2sem_checkpoint,
            output_normalization=args.meg2sem_output_normalization,
            device=device,
        )
        from modules.meg2sem_bridge import MEG2SEMToELFContextAdapter

        adapter = MEG2SEMToELFContextAdapter(
            meg2sem=post_meg2sem,
            semantic_projector=semantic_projector,
            normalize_semantic_output=normalize,
        ).to(device)

        class _NoopModel(torch.nn.Module):
            def state_dict(self, *a, **k):  # noqa: ANN001, ANN002
                return {}

            def load_state_dict(self, state_dict, strict=True):  # noqa: ANN001
                if state_dict:
                    raise RuntimeError("E2E checkpoint contains model_state_dict; pass --ignore-model-state or load via trainer.")
                return torch.nn.modules.module._IncompatibleKeys([], [])

        ckpt = torch.load(args.e2e_checkpoint, map_location=device)
        if not isinstance(ckpt, dict) or "adapter_state_dict" not in ckpt:
            raise ValueError(f"{args.e2e_checkpoint} missing adapter_state_dict")
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter.eval()
        post_raw = adapter.meg2sem(meg, meg_lengths=meg_lengths, subjects=subject_ids)
        post_projector_input = F.normalize(post_raw.float(), p=2, dim=-1) if normalize else post_raw.float()

    meta = {
        "meg2sem_load_info": {
            "checkpoint_format": load_info.checkpoint_format,
            "embedding_dim": load_info.embedding_dim,
            "input_dim": load_info.input_dim,
            "n_subjects": load_info.n_subjects,
            "normalize_output": load_info.normalize_output,
            "missing_keys": load_info.missing_keys,
            "unexpected_keys": load_info.unexpected_keys,
        },
        "meg_norm": meg_norm,
        "subject_to_id": subject_to_id,
        "eval_source_path": eval_examples.source_path,
        "eval_num_examples": len(eval_examples.sentences),
        "meg2sem_prediction_mode": args.meg2sem_prediction_mode,
        "residual_reconstruction": residual_meta,
        "residual_target_scale": float(args.residual_target_scale),
    }
    return (
        VectorBundle(
            true=true.detach().cpu(),
            pre_raw=pre_raw.detach().cpu(),
            pre_projector_input=pre_projector_input.detach().cpu(),
            pre_residual_output=pre_residual_output.detach().cpu() if pre_residual_output is not None else None,
            true_residual_target=true_residual_target.detach().cpu() if true_residual_target is not None else None,
            post_raw=post_raw.detach().cpu() if post_raw is not None else None,
            post_projector_input=post_projector_input.detach().cpu() if post_projector_input is not None else None,
            true_projector_input=true_projector_input.detach().cpu(),
            sentences=eval_examples.sentences,
        ),
        meta,
    )


def per_row(bundle: VectorBundle, *, residual_center: torch.Tensor | None, residual_basis: torch.Tensor | None) -> list[dict]:
    true = bundle.true_projector_input.float()
    pre = bundle.pre_projector_input.float()
    post = bundle.post_projector_input.float() if bundle.post_projector_input is not None else None
    predicted_residual = bundle.pre_residual_output.float() if bundle.pre_residual_output is not None else None
    target_residual = bundle.true_residual_target.float() if bundle.true_residual_target is not None else None
    rows = []
    pre_to_true_cos = F.cosine_similarity(pre, true, dim=-1)
    pre_to_true_l2 = (pre - true).norm(dim=-1)
    _, pre_geometry = semantic_error_geometry(pre, true, prefix="pre")
    if predicted_residual is not None and target_residual is not None:
        residual_output_cos = F.cosine_similarity(predicted_residual, target_residual, dim=-1)
        residual_output_l2 = (predicted_residual - target_residual).norm(dim=-1)
        _, residual_output_geometry = semantic_error_geometry(
            predicted_residual,
            target_residual,
            prefix="residual_output",
        )
    else:
        residual_output_cos = residual_output_l2 = residual_output_geometry = None
    if post is not None:
        post_to_true_cos = F.cosine_similarity(post, true, dim=-1)
        post_to_true_l2 = (post - true).norm(dim=-1)
        _, post_geometry = semantic_error_geometry(post, true, prefix="post")
        delta = post - pre
        target_delta = true - pre
        movement_cos = F.cosine_similarity(delta, target_delta, dim=-1)
        target_dist = target_delta.norm(dim=-1)
        progress = ((delta * target_delta).sum(dim=-1) / target_dist.pow(2).clamp_min(1e-12)).cpu()
        delta_norm = delta.norm(dim=-1)
    else:
        post_to_true_cos = post_to_true_l2 = post_geometry = movement_cos = progress = delta_norm = None

    if residual_center is not None and residual_basis is not None:
        true_res = remove_components(true, residual_basis, center=residual_center)
        pre_res = remove_components(pre, residual_basis, center=residual_center)
        pre_res_cos = F.cosine_similarity(pre_res, true_res, dim=-1)
        _, pre_res_geometry = semantic_error_geometry(pre_res, true_res, prefix="pre_residual")
        if post is not None:
            post_res = remove_components(post, residual_basis, center=residual_center)
            post_res_cos = F.cosine_similarity(post_res, true_res, dim=-1)
            _, post_res_geometry = semantic_error_geometry(post_res, true_res, prefix="post_residual")
        else:
            post_res_cos = None
            post_res_geometry = None
    else:
        pre_res_cos = post_res_cos = None
        pre_res_geometry = post_res_geometry = None

    for idx, sentence in enumerate(bundle.sentences):
        pre_nearest_wrong_index = int(pre_geometry["nearest_wrong_index"][idx])
        row = {
            "index": idx,
            "sentence": sentence,
            "pre_cosine": float(pre_to_true_cos[idx]),
            "pre_l2": float(pre_to_true_l2[idx]),
            "pre_true_rank": int(pre_geometry["true_rank"][idx]),
            "pre_true_similarity": float(pre_geometry["true_similarity"][idx]),
            "pre_nearest_wrong_index": pre_nearest_wrong_index,
            "pre_nearest_wrong_sentence": bundle.sentences[pre_nearest_wrong_index] if pre_nearest_wrong_index >= 0 else "",
            "pre_nearest_wrong_similarity": float(pre_geometry["nearest_wrong_similarity"][idx]),
            "pre_true_minus_nearest_wrong_margin": float(pre_geometry["true_minus_nearest_wrong_margin"][idx]),
            "pre_error_to_nearest_wrong_cosine": float(pre_geometry["error_to_nearest_wrong_cosine"][idx]),
            "pre_error_to_random_wrong_cosine": float(pre_geometry["error_to_random_wrong_cosine"][idx]),
            "pre_projection_to_nearest_wrong_fraction": float(pre_geometry["projection_to_nearest_wrong_fraction"][idx]),
        }
        if residual_output_cos is not None and residual_output_geometry is not None:
            residual_output_nearest_wrong_index = int(residual_output_geometry["nearest_wrong_index"][idx])
            row.update(
                {
                    "residual_output_cosine": float(residual_output_cos[idx]),
                    "residual_output_l2": float(residual_output_l2[idx]),
                    "residual_output_true_rank": int(residual_output_geometry["true_rank"][idx]),
                    "residual_output_nearest_wrong_index": residual_output_nearest_wrong_index,
                    "residual_output_nearest_wrong_sentence": (
                        bundle.sentences[residual_output_nearest_wrong_index]
                        if residual_output_nearest_wrong_index >= 0
                        else ""
                    ),
                    "residual_output_true_minus_nearest_wrong_margin": float(
                        residual_output_geometry["true_minus_nearest_wrong_margin"][idx]
                    ),
                    "residual_output_error_to_nearest_wrong_cosine": float(
                        residual_output_geometry["error_to_nearest_wrong_cosine"][idx]
                    ),
                }
            )
        if post is not None:
            post_nearest_wrong_index = int(post_geometry["nearest_wrong_index"][idx])
            row.update(
                {
                    "post_cosine": float(post_to_true_cos[idx]),
                    "post_l2": float(post_to_true_l2[idx]),
                    "post_true_rank": int(post_geometry["true_rank"][idx]),
                    "post_true_similarity": float(post_geometry["true_similarity"][idx]),
                    "post_nearest_wrong_index": post_nearest_wrong_index,
                    "post_nearest_wrong_sentence": bundle.sentences[post_nearest_wrong_index] if post_nearest_wrong_index >= 0 else "",
                    "post_nearest_wrong_similarity": float(post_geometry["nearest_wrong_similarity"][idx]),
                    "post_true_minus_nearest_wrong_margin": float(post_geometry["true_minus_nearest_wrong_margin"][idx]),
                    "post_error_to_nearest_wrong_cosine": float(post_geometry["error_to_nearest_wrong_cosine"][idx]),
                    "post_error_to_random_wrong_cosine": float(post_geometry["error_to_random_wrong_cosine"][idx]),
                    "post_projection_to_nearest_wrong_fraction": float(
                        post_geometry["projection_to_nearest_wrong_fraction"][idx]
                    ),
                    "delta_norm": float(delta_norm[idx]),
                    "movement_cosine_to_target_delta": float(movement_cos[idx]),
                    "fractional_progress_to_target": float(progress[idx]),
                    "cosine_change": float(post_to_true_cos[idx] - pre_to_true_cos[idx]),
                    "l2_change": float(post_to_true_l2[idx] - pre_to_true_l2[idx]),
                }
            )
        if pre_res_cos is not None:
            pre_res_nearest_wrong_index = int(pre_res_geometry["nearest_wrong_index"][idx])
            row["pre_residual_cosine"] = float(pre_res_cos[idx])
            row["pre_residual_true_rank"] = int(pre_res_geometry["true_rank"][idx])
            row["pre_residual_nearest_wrong_index"] = pre_res_nearest_wrong_index
            row["pre_residual_nearest_wrong_sentence"] = (
                bundle.sentences[pre_res_nearest_wrong_index] if pre_res_nearest_wrong_index >= 0 else ""
            )
            row["pre_residual_true_minus_nearest_wrong_margin"] = float(
                pre_res_geometry["true_minus_nearest_wrong_margin"][idx]
            )
            row["pre_residual_error_to_nearest_wrong_cosine"] = float(
                pre_res_geometry["error_to_nearest_wrong_cosine"][idx]
            )
            if post_res_cos is not None:
                post_res_nearest_wrong_index = int(post_res_geometry["nearest_wrong_index"][idx])
                row["post_residual_cosine"] = float(post_res_cos[idx])
                row["post_residual_true_rank"] = int(post_res_geometry["true_rank"][idx])
                row["post_residual_nearest_wrong_index"] = post_res_nearest_wrong_index
                row["post_residual_nearest_wrong_sentence"] = (
                    bundle.sentences[post_res_nearest_wrong_index] if post_res_nearest_wrong_index >= 0 else ""
                )
                row["post_residual_true_minus_nearest_wrong_margin"] = float(
                    post_res_geometry["true_minus_nearest_wrong_margin"][idx]
                )
                row["post_residual_error_to_nearest_wrong_cosine"] = float(
                    post_res_geometry["error_to_nearest_wrong_cosine"][idx]
                )
                row["residual_cosine_change"] = float(post_res_cos[idx] - pre_res_cos[idx])
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(bundle: VectorBundle, args: argparse.Namespace) -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
    summary: dict[str, object] = {}
    summary["true_ada_pca"] = principal_component_report(bundle.true_projector_input)
    summary["pre_vs_true"] = paired_metrics(bundle.pre_projector_input, bundle.true_projector_input, prefix="pre")
    pre_error_geometry, _ = semantic_error_geometry(
        bundle.pre_projector_input,
        bundle.true_projector_input,
        prefix="pre",
    )
    summary["pre_error_geometry"] = pre_error_geometry
    if bundle.pre_residual_output is not None and bundle.true_residual_target is not None:
        summary["residual_output_vs_true_residual"] = paired_metrics(
            bundle.pre_residual_output,
            bundle.true_residual_target,
            prefix="residual_output",
        )
        residual_output_error_geometry, _ = semantic_error_geometry(
            bundle.pre_residual_output,
            bundle.true_residual_target,
            prefix="residual_output",
        )
        summary["residual_output_error_geometry"] = residual_output_error_geometry

    residual_center = None
    residual_basis = None
    residual_report = None
    if args.component_json:
        residual_center, residual_basis, residual_report = load_component_json(
            args.component_json,
            num_components=args.remove_components,
        )
    elif args.component_npz:
        residual_center, residual_basis, residual_report = load_mean_components(
            args.component_npz,
            semantic_key=args.semantic_key,
            max_rows=args.component_max_rows,
            num_components=args.remove_components,
        )
        if args.save_component_json:
            save_component_json(args.save_component_json, residual_center, residual_basis, residual_report)
    elif args.remove_components > 0:
        residual_center = bundle.true_projector_input.float().mean(dim=0, keepdim=True)
        residual_basis = pca_components(bundle.true_projector_input, args.remove_components)
        residual_report = {
            "component_source": "eval_true_ada",
            "component_rows": int(bundle.true_projector_input.shape[0]),
            "component_dim": int(bundle.true_projector_input.shape[1]),
            "num_removed_components": int(residual_basis.shape[0]),
        }
    if residual_center is not None and residual_basis is not None:
        summary["residual_component_report"] = residual_report
        summary["component_energy"] = {}
        summary["component_energy"].update(
            component_energy_report(
                bundle.true_projector_input,
                residual_basis,
                center=residual_center,
                prefix="true",
            )
        )
        summary["component_energy"].update(
            component_energy_report(
                bundle.pre_projector_input,
                residual_basis,
                center=residual_center,
                prefix="pre",
            )
        )
        summary["component_error"] = error_component_report(
            bundle.pre_projector_input,
            bundle.true_projector_input,
            residual_basis,
            prefix="pre_vs_true",
        )
        true_res = remove_components(bundle.true_projector_input, residual_basis, center=residual_center)
        pre_res = remove_components(bundle.pre_projector_input, residual_basis, center=residual_center)
        summary["pre_residual_vs_true_residual"] = paired_metrics(pre_res, true_res, prefix="pre_residual")
        pre_res_error_geometry, _ = semantic_error_geometry(pre_res, true_res, prefix="pre_residual")
        summary["pre_residual_error_geometry"] = pre_res_error_geometry

    if bundle.post_projector_input is not None:
        post = bundle.post_projector_input
        true = bundle.true_projector_input
        pre = bundle.pre_projector_input
        summary["post_vs_true"] = paired_metrics(post, true, prefix="post")
        summary["post_vs_pre"] = paired_metrics(post, pre, prefix="post_pre")
        post_error_geometry, _ = semantic_error_geometry(post, true, prefix="post")
        summary["post_error_geometry"] = post_error_geometry
        delta = post.float() - pre.float()
        target_delta = true.float() - pre.float()
        movement_cos = F.cosine_similarity(delta, target_delta, dim=-1)
        progress = (delta * target_delta).sum(dim=-1) / target_delta.norm(dim=-1).pow(2).clamp_min(1e-12)
        summary["e2e_movement"] = {
            "delta_norm_mean": float(delta.norm(dim=-1).mean()),
            "delta_norm_std": float(delta.norm(dim=-1).std(unbiased=False)),
            "target_delta_norm_mean": float(target_delta.norm(dim=-1).mean()),
            "movement_cosine_to_target_delta_mean": float(movement_cos.mean()),
            "movement_cosine_to_target_delta_std": float(movement_cos.std(unbiased=False)),
            "fractional_progress_to_target_mean": float(progress.mean()),
            "fractional_progress_to_target_std": float(progress.std(unbiased=False)),
            "cosine_improved_fraction": float(
                (
                    F.cosine_similarity(post.float(), true.float(), dim=-1)
                    > F.cosine_similarity(pre.float(), true.float(), dim=-1)
                )
                .float()
                .mean()
            ),
            "l2_improved_fraction": float(((post.float() - true.float()).norm(dim=-1) < target_delta.norm(dim=-1)).float().mean()),
        }
        if residual_center is not None and residual_basis is not None:
            summary["component_energy"].update(
                component_energy_report(
                    post,
                    residual_basis,
                    center=residual_center,
                    prefix="post",
                )
            )
            summary["component_error"].update(
                error_component_report(
                    post,
                    true,
                    residual_basis,
                    prefix="post_vs_true",
                )
            )
            true_res = remove_components(true, residual_basis, center=residual_center)
            pre_res = remove_components(pre, residual_basis, center=residual_center)
            post_res = remove_components(post, residual_basis, center=residual_center)
            summary["post_residual_vs_true_residual"] = paired_metrics(post_res, true_res, prefix="post_residual")
            summary["post_residual_vs_pre_residual"] = paired_metrics(post_res, pre_res, prefix="post_pre_residual")
            post_res_error_geometry, _ = semantic_error_geometry(post_res, true_res, prefix="post_residual")
            summary["post_residual_error_geometry"] = post_res_error_geometry

    return summary, residual_center, residual_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--meg2sem-checkpoint", required=True)
    parser.add_argument("--semantic-projector-checkpoint", default="")
    parser.add_argument("--e2e-checkpoint", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-num-examples", type=int, default=75)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--meg-key", default="meg")
    parser.add_argument("--meg-lengths-key", default="meg_lengths")
    parser.add_argument("--meg-mask-key", default="meg_time_mask")
    parser.add_argument("--semantic-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--subject-key", default="subject")
    parser.add_argument("--meg-standardization", choices=["none", "train"], default="none")
    parser.add_argument("--meg-standardize-eps", type=float, default=1e-6)
    parser.add_argument("--meg-clip-boundary", type=float, default=-1.0)
    parser.add_argument("--meg2sem-output-normalization", choices=["auto", "never", "always"], default="auto")
    parser.add_argument("--meg2sem-prediction-mode", choices=["full_ada", "mean_residual"], default="full_ada")
    parser.add_argument("--residual-mean-npz", default="")
    parser.add_argument("--residual-target-scale", type=float, default=1.0)
    parser.add_argument("--context-dim", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--semantic-hidden-dim", type=int, default=4096)
    parser.add_argument("--component-npz", default="")
    parser.add_argument("--component-json", default="")
    parser.add_argument("--save-component-json", default="")
    parser.add_argument("--component-max-rows", type=int, default=0)
    parser.add_argument("--remove-components", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, meta = collect_vectors(args, device)
    summary, residual_center, residual_basis = summarize(bundle, args)
    rows = per_row(bundle, residual_center=residual_center, residual_basis=residual_basis)
    rows.sort(key=lambda row: (row.get("post_cosine", row["pre_cosine"]), row["pre_cosine"]), reverse=True)

    payload = {
        "diagnostic_name": "E2E Diagnostics",
        "args": vars(args),
        "meta": meta,
        "summary": summary,
    }
    with (output_dir / "e2e_diagnostics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_rows(output_dir / "e2e_diagnostics_per_sentence.tsv", rows)
    print(json.dumps(payload, indent=2))
    print(f"rows_tsv={output_dir / 'e2e_diagnostics_per_sentence.tsv'}")


if __name__ == "__main__":
    main()
