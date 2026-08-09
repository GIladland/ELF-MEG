#!/usr/bin/env python
"""Train a MEG adapter into a frozen ELF checkpoint from packed NPZ windows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from configs.config import SamplingConfig
from modules.meg_adapter import MEGContextAdapter
from modules.t5_encoder import get_encoder
from scripts.meg_context_overfit import (
    OverfitBatch,
    SemanticVectorContextProjector,
    build_config,
    encode_text_batched,
    eval_checkpoint_scores,
    evaluate_generation,
    evaluate_retrieval,
    freeze_for_toy_tuning,
    load_pretrained_model,
    save_best_results,
    save_eval_checkpoint,
    save_results,
    tokenize_sentences,
    train_step,
)
from utils.checkpoint_utils import _download_hf_checkpoint, _restore_checkpoint
from utils.logging_utils import log_for_0

try:
    import wandb
except ImportError:
    wandb = None


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)


@dataclass
class PackedMEGExamples:
    meg: torch.Tensor
    meg_lengths: torch.Tensor
    semantic_vectors: torch.Tensor
    sentences: list[str]
    subject_labels: list[str]
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--meg-key", default="meg")
    parser.add_argument("--meg-lengths-key", default="meg_lengths")
    parser.add_argument("--meg-mask-key", default="meg_time_mask")
    parser.add_argument("--semantic-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--subject-key", default="subject")
    parser.add_argument(
        "--meg-standardization",
        choices=["none", "train"],
        default="none",
        help=(
            "Apply MEG channel standardization inside this trainer. Use 'train' for raw "
            "MEG NPZs so train/eval are normalized with training-split statistics only."
        ),
    )
    parser.add_argument("--meg-standardize-eps", type=float, default=1e-6)
    parser.add_argument(
        "--meg-clip-boundary",
        type=float,
        default=-1.0,
        help="Optional post-standardization clipping boundary. <=0 disables clipping.",
    )
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument(
        "--val-num-examples",
        type=int,
        default=512,
        help="If --val-npz is unset, hold out this many examples from the end of train-npz.",
    )
    parser.add_argument("--eval-num-examples", type=int, default=64)
    parser.add_argument("--retrieval-num-examples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--retrieval-eval-every", type=int, default=500)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--retrieval-t", type=float, default=0.5)
    parser.add_argument("--target-encode-batch-size", type=int, default=128)
    parser.add_argument("--num_sampling_steps", type=int, default=32)
    parser.add_argument(
        "--train-timestep-mode",
        choices=["logit_normal", "sampling_schedule", "sampling_schedule_all"],
        default="sampling_schedule",
    )
    parser.add_argument("--train-timestep-steps", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--semantic-hidden-dim", type=int, default=4096)
    parser.add_argument(
        "--teacher-context-loss-weight",
        type=float,
        default=0.0,
        help=(
            "If >0, load the semantic projector from --checkpoint_path and train MEG context "
            "tokens to match frozen MiniLM->ELF context tokens."
        ),
    )
    parser.add_argument(
        "--init-adapter-checkpoint",
        default="",
        help="Optional checkpoint containing adapter_state_dict used to initialize the MEG adapter.",
    )
    parser.add_argument(
        "--unfreeze-elf",
        action="store_true",
        help="Update selected ELF parameters during end-to-end MEG->text training.",
    )
    parser.add_argument(
        "--unfreeze-last-n-blocks",
        type=int,
        default=0,
        help=(
            "When --unfreeze-elf is set, unfreeze text_proj/final_layer plus this many "
            "final transformer blocks. Use -1 to unfreeze the whole ELF."
        ),
    )
    parser.add_argument(
        "--elf-lr",
        type=float,
        default=None,
        help="Learning rate for unfrozen ELF params. Defaults to --lr if unset.",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--denoiser_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument(
        "--train-target-mask-mode",
        choices=["valid", "full"],
        default="full",
        help="Use full to match the best full-mask SVA ELF run.",
    )
    parser.add_argument("--merger-channels", type=int, default=306)
    parser.add_argument("--conv-channels", type=int, default=320)
    parser.add_argument("--num-conv-layers", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilation-growth", type=int, default=2)
    parser.add_argument("--dilation-period", type=int, default=5)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--disable-subject-layers", action="store_true")
    parser.add_argument("--disable-temporal-attention", action="store_true")
    parser.add_argument("--norm-type", choices=["batch", "layer"], default="batch")
    parser.add_argument("--generation-t5-retrieval", action="store_true")
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--eval-checkpoint-top-k", type=int, default=3)
    parser.add_argument("--eval-checkpoint-every", type=int, default=0)
    parser.add_argument("--include-optimizer-in-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="BrainDiffusion")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default="npz-meg-to-frozen-elf")
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--wandb_sample_examples", type=int, default=16)
    parser.add_argument("--no-wandb-samples", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def _limit_count(n: int, limit: int) -> int:
    return n if limit <= 0 else min(n, limit)


def load_packed_npz(args: argparse.Namespace, path: str, *, limit: int = 0) -> PackedMEGExamples:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    for key in (args.meg_key, args.semantic_key, args.sentence_key):
        if key not in keys:
            raise KeyError(f"{path} missing required key {key!r}; available keys={sorted(keys)}")

    total_n = int(data[args.semantic_key].shape[0])
    n = _limit_count(total_n, limit)
    meg_np = np.asarray(data[args.meg_key][:n])
    semantic_np = np.asarray(data[args.semantic_key][:n], dtype=np.float32)
    sentences = _strings(data[args.sentence_key][:n])

    if args.meg_lengths_key in keys:
        meg_lengths_np = np.asarray(data[args.meg_lengths_key][:n], dtype=np.int64)
    elif args.meg_mask_key in keys:
        meg_lengths_np = np.asarray(data[args.meg_mask_key][:n]).sum(axis=1).astype(np.int64)
    else:
        meg_lengths_np = np.full((n,), meg_np.shape[-1], dtype=np.int64)

    if args.subject_key in keys:
        subject_labels = _strings(data[args.subject_key][:n])
    else:
        subject_labels = ["0"] * n

    examples = PackedMEGExamples(
        meg=torch.as_tensor(meg_np),
        meg_lengths=torch.as_tensor(meg_lengths_np, dtype=torch.long),
        semantic_vectors=torch.as_tensor(semantic_np, dtype=torch.float32),
        sentences=sentences,
        subject_labels=subject_labels,
        source_path=path,
    )
    log_for_0(
        f"Loaded {n}/{total_n} rows from {path}; "
        f"meg_shape={tuple(examples.meg.shape)} meg_dtype={examples.meg.dtype} "
        f"semantic_shape={tuple(examples.semantic_vectors.shape)}"
    )
    return examples


def take_examples(examples: PackedMEGExamples, start: int, end: int) -> PackedMEGExamples:
    indices = torch.arange(start, end, dtype=torch.long)
    return PackedMEGExamples(
        meg=examples.meg.index_select(0, indices),
        meg_lengths=examples.meg_lengths.index_select(0, indices),
        semantic_vectors=examples.semantic_vectors.index_select(0, indices),
        sentences=examples.sentences[start:end],
        subject_labels=examples.subject_labels[start:end],
        source_path=examples.source_path,
    )


def _meg_valid_mask(meg: torch.Tensor, meg_lengths: torch.Tensor) -> torch.Tensor:
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
        mask = _meg_valid_mask(chunk, meg_lengths[start:end]).to(dtype=torch.float32)
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


def select_strings(items: Sequence[str], indices: torch.Tensor) -> list[str]:
    return [items[int(index)] for index in indices.detach().cpu().tolist()]


def sample_indices(num_examples: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(low=0, high=num_examples, size=(batch_size,), generator=generator, dtype=torch.long)


def maybe_init_wandb(args: argparse.Namespace, config, *, train_n: int, eval_n: int, subject_count: int):
    if not args.use_wandb or wandb is None:
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name,
        notes=args.wandb_notes,
        config={
            **vars(args),
            "num_train_examples": train_n,
            "num_eval_examples_available": eval_n,
            "subject_count": subject_count,
            "max_length": config.max_length,
            "frozen_elf": True,
        },
    )
    wandb.define_metric("train/epoch")
    wandb.define_metric("eval/epoch")
    wandb.define_metric("retrieval/epoch")
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("eval/*", step_metric="eval/epoch")
    wandb.define_metric("retrieval/*", step_metric="retrieval/epoch")
    wandb.define_metric("generation_t5_retrieval/*", step_metric="eval/epoch")
    wandb.define_metric("generation_quality/*", step_metric="eval/epoch")
    return run


def load_teacher_projector(
    *,
    checkpoint_path: str,
    input_dim: int,
    context_dim: int,
    context_length: int,
    hidden_dim: int,
    device: torch.device,
) -> SemanticVectorContextProjector:
    ckpt_root = _download_hf_checkpoint(checkpoint_path) or checkpoint_path
    ckpt = _restore_checkpoint(ckpt_root)
    if ckpt is None or "adapter_state_dict" not in ckpt:
        raise ValueError(
            f"Teacher checkpoint {checkpoint_path} must contain adapter_state_dict; "
            f"got keys={sorted(ckpt.keys()) if ckpt is not None else None}"
        )
    projector = SemanticVectorContextProjector(
        input_dim=input_dim,
        context_dim=context_dim,
        context_length=context_length,
        hidden_dim=hidden_dim,
        dropout=0.0,
    ).to(device)
    projector.load_state_dict(ckpt["adapter_state_dict"])
    projector.eval()
    for param in projector.parameters():
        param.requires_grad_(False)
    return projector


def load_adapter_initialization(adapter: MEGContextAdapter, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "adapter_state_dict" in ckpt:
        state_dict = ckpt["adapter_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported adapter checkpoint payload in {checkpoint_path}")
    adapter.load_state_dict(state_dict)
    log_for_0(f"Initialized MEG adapter from {checkpoint_path}")


def teacher_context_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_flat = predicted.detach().float().reshape(predicted.shape[0], -1)
    target_flat = target.detach().float().reshape(target.shape[0], -1)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    return {
        "teacher_context_mse": float(F.mse_loss(predicted.detach().float(), target.detach().float()).cpu()),
        "teacher_context_cosine": float(cosine.mean().cpu()),
    }


def teacher_train_step(
    *,
    adapter: MEGContextAdapter,
    teacher: SemanticVectorContextProjector,
    batch: OverfitBatch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: float,
) -> dict[str, float]:
    adapter.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    meg = batch.meg.to(device=device, dtype=torch.float32)
    meg_lengths = batch.meg_lengths.to(device=device, dtype=torch.long)
    subject_ids = batch.subject_ids.to(device=device, dtype=torch.long)
    semantic_vectors = batch.semantic_vectors.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        target_context, _ = teacher(semantic_vectors)
    predicted = adapter(meg, meg_lengths=meg_lengths, subjects=subject_ids).context
    context_loss = F.mse_loss(predicted.float(), target_context.float())
    loss = loss_weight * context_loss
    loss.backward()
    optimizer.step()

    metrics = teacher_context_metrics(predicted, target_context)
    metrics.update({
        "loss": float(loss.detach().cpu()),
        "denoiser_loss": 0.0,
        "decoder_loss": 0.0,
    })
    return metrics


@torch.no_grad()
def evaluate_teacher_context(
    *,
    adapter: MEGContextAdapter,
    teacher: SemanticVectorContextProjector,
    meg: torch.Tensor,
    meg_lengths: torch.Tensor,
    semantic_vectors: torch.Tensor,
    subject_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    teacher.eval()
    target_context, _ = teacher(semantic_vectors.to(device=device, dtype=torch.float32))
    predicted = adapter(
        meg.to(device=device, dtype=torch.float32),
        meg_lengths=meg_lengths.to(device=device, dtype=torch.long),
        subjects=subject_ids.to(device=device, dtype=torch.long),
    ).context
    return teacher_context_metrics(predicted, target_context)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_for_0(f"Using device: {device}")

    all_train = load_packed_npz(args, args.train_npz, limit=args.num_examples)
    if args.val_npz:
        train_examples = all_train
        eval_examples = load_packed_npz(args, args.val_npz, limit=0)
        eval_split = "val_npz"
    else:
        total_n = len(all_train.sentences)
        val_n = min(max(0, args.val_num_examples), max(0, total_n - 1))
        train_n = total_n - val_n
        train_examples = take_examples(all_train, 0, train_n)
        eval_examples = take_examples(all_train, train_n, total_n) if val_n > 0 else take_examples(all_train, 0, total_n)
        eval_split = "heldout_tail" if val_n > 0 else "train"

    train_examples, eval_examples, meg_norm_summary = maybe_standardize_meg(args, train_examples, eval_examples)
    log_for_0(f"MEG normalization: {json.dumps(meg_norm_summary, sort_keys=True)}")

    train_subject_ids, eval_subject_ids, subject_to_id = assign_subject_ids(train_examples, eval_examples)
    train_n = len(train_examples.sentences)
    eval_available = len(eval_examples.sentences)
    steps_per_epoch = int(np.ceil(train_n / args.batch_size))
    if args.steps <= 0:
        args.steps = int(np.ceil(steps_per_epoch * args.epochs))
    else:
        args.epochs = args.steps / max(1, steps_per_epoch)

    log_for_0(
        f"Training schedule: train_n={train_n} eval_n={eval_available} eval_split={eval_split} "
        f"batch_size={args.batch_size} steps_per_epoch={steps_per_epoch} steps={args.steps} "
        f"epochs={args.steps / max(1, steps_per_epoch):.3f}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = tokenize_sentences(tokenizer, train_examples.sentences)
    eval_input_ids, eval_attention_mask = tokenize_sentences(tokenizer, eval_examples.sentences)
    target_length = int(input_ids.shape[1])
    eval_target_length = int(eval_input_ids.shape[1])
    config = build_config(args, max_length=args.context_length + max(target_length, eval_target_length))

    encoder_config, encoder = get_encoder(args.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    model = load_pretrained_model(
        args=args,
        config=config,
        encoder_dim=encoder_config.d_model,
        vocab_size=len(tokenizer),
        device=device,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_elf_names: list[str] = []
    if args.unfreeze_elf:
        if args.teacher_context_loss_weight > 0.0:
            raise ValueError(
                "--unfreeze-elf has no effect with the teacher-context-only objective. "
                "Set --teacher-context-loss-weight 0 for end-to-end ELF training."
            )
        trainable_elf_names = freeze_for_toy_tuning(model, args.unfreeze_last_n_blocks)
        log_for_0(
            f"Unfrozen ELF params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
            f"across {len(trainable_elf_names)} tensors; unfreeze_last_n_blocks={args.unfreeze_last_n_blocks}"
        )
    else:
        log_for_0(f"Frozen ELF parameters: {sum(p.numel() for p in model.parameters()):,}")

    adapter = MEGContextAdapter(
        in_channels=int(train_examples.meg.shape[1]),
        context_dim=encoder_config.d_model,
        context_length=args.context_length,
        n_subjects=max(1, len(subject_to_id)),
        merger_channels=args.merger_channels,
        conv_channels=args.conv_channels,
        num_conv_layers=args.num_conv_layers,
        kernel_size=args.kernel_size,
        dilation_growth=args.dilation_growth,
        dilation_period=args.dilation_period,
        dropout=args.adapter_dropout,
        attention_heads=args.attention_heads,
        use_subject_layers=not args.disable_subject_layers,
        use_temporal_attention=not args.disable_temporal_attention,
        norm_type=args.norm_type,
    ).to(device)
    if args.init_adapter_checkpoint:
        load_adapter_initialization(adapter, args.init_adapter_checkpoint, device)
    log_for_0(f"Trainable MEG adapter parameters: {sum(p.numel() for p in adapter.parameters() if p.requires_grad):,}")

    teacher_adapter = None
    if args.teacher_context_loss_weight > 0.0:
        teacher_adapter = load_teacher_projector(
            checkpoint_path=args.checkpoint_path,
            input_dim=int(train_examples.semantic_vectors.shape[-1]),
            context_dim=encoder_config.d_model,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            device=device,
        )
        log_for_0(
            f"Loaded frozen semantic teacher projector from {args.checkpoint_path}; "
            f"teacher_context_loss_weight={args.teacher_context_loss_weight}"
        )

    log_for_0(f"Encoding train target T5 latents in batches of {args.target_encode_batch_size}")
    target_latents = encode_text_batched(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        device=device,
        batch_size=args.target_encode_batch_size,
    )
    log_for_0(f"Encoding eval target T5 latents in batches of {args.target_encode_batch_size}")
    eval_target_latents = encode_text_batched(
        input_ids=eval_input_ids,
        attention_mask=eval_attention_mask,
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        device=device,
        batch_size=args.target_encode_batch_size,
    )

    target_ids = input_ids.detach().cpu()
    valid_target_mask = attention_mask.detach().cpu().to(torch.float32)
    if args.train_target_mask_mode == "full":
        target_mask = torch.ones_like(valid_target_mask)
    else:
        target_mask = valid_target_mask
    eval_target_ids = eval_input_ids.detach().cpu()
    eval_target_mask = eval_attention_mask.detach().cpu().to(torch.float32)

    param_groups = [
        {
            "params": [param for param in adapter.parameters() if param.requires_grad],
            "lr": args.lr,
            "name": "meg_adapter",
        }
    ]
    if args.unfreeze_elf:
        elf_lr = args.lr if args.elf_lr is None else args.elf_lr
        param_groups.append(
            {
                "params": [param for param in model.parameters() if param.requires_grad],
                "lr": elf_lr,
                "name": "elf",
            }
        )
        log_for_0(f"Optimizer param groups: adapter_lr={args.lr:g} elf_lr={elf_lr:g}")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )
    noise_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(args.seed + 17)
    batch_generator = torch.Generator().manual_seed(args.seed + 29)

    eval_count = eval_available if args.eval_num_examples <= 0 else min(args.eval_num_examples, eval_available)
    retrieval_count = (
        eval_count
        if args.retrieval_num_examples <= 0
        else min(args.retrieval_num_examples, eval_available)
    )
    eval_indices = torch.arange(0, eval_count, dtype=torch.long)
    retrieval_indices = torch.arange(0, retrieval_count, dtype=torch.long)

    config_payload = {
        **vars(args),
        "num_train_examples": train_n,
        "num_eval_examples_available": eval_available,
        "eval_split": eval_split,
        "subject_to_id": subject_to_id,
        "meg_norm_summary": meg_norm_summary,
        "target_length": target_length,
        "eval_target_length": eval_target_length,
        "train_target_mask_density": float(target_mask.mean().item()),
        "trainable_elf_tensor_count": len(trainable_elf_names),
        "trainable_elf_tensors": trainable_elf_names[:200],
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    run = maybe_init_wandb(args, config, train_n=train_n, eval_n=eval_available, subject_count=len(subject_to_id))
    best_metrics = None
    best_eval_score = None
    saved_checkpoints: list[dict] = []
    eval_index = 0

    for step in range(1, args.steps + 1):
        indices = sample_indices(train_n, args.batch_size, batch_generator)
        batch = OverfitBatch(
            indices=indices,
            meg=train_examples.meg.index_select(0, indices),
            meg_lengths=train_examples.meg_lengths.index_select(0, indices),
            semantic_vectors=train_examples.semantic_vectors.index_select(0, indices),
            subject_ids=train_subject_ids.index_select(0, indices),
            target_latents=target_latents.index_select(0, indices),
            target_ids=target_ids.index_select(0, indices),
            target_mask=target_mask.index_select(0, indices),
        )
        epoch = step * args.batch_size / max(1, train_n)
        if teacher_adapter is not None:
            metrics = teacher_train_step(
                adapter=adapter,
                teacher=teacher_adapter,
                batch=batch,
                optimizer=optimizer,
                device=device,
                loss_weight=args.teacher_context_loss_weight,
            )
        else:
            metrics = train_step(
                model=model,
                adapter=adapter,
                batch=batch,
                optimizer=optimizer,
                config=config,
                device=device,
                noise_generator=noise_generator,
                condition_source="meg",
                cond_dropout_prob=args.cond_dropout_prob,
                train_timestep_mode=args.train_timestep_mode,
                train_timestep_steps=args.train_timestep_steps,
            )

        if step == 1 or step % 10 == 0:
            log_for_0(
                f"step={step} epoch={epoch:.4f} loss={metrics['loss']:.6f} "
                f"denoiser={metrics['denoiser_loss']:.6f} decoder={metrics['decoder_loss']:.6f}"
                + (
                    f" teacher_mse={metrics['teacher_context_mse']:.6f} "
                    f"teacher_cos={metrics['teacher_context_cosine']:.3f}"
                    if "teacher_context_mse" in metrics
                    else ""
                )
            )
        if run is not None:
            payload = {
                "train/loss": metrics["loss"],
                "train/denoiser_loss": metrics["denoiser_loss"],
                "train/decoder_loss": metrics["decoder_loss"],
                "train/step": step,
                "train/epoch": epoch,
                "train/examples_seen": step * args.batch_size,
            }
            if "teacher_context_mse" in metrics:
                payload.update({
                    "train/teacher_context_mse": metrics["teacher_context_mse"],
                    "train/teacher_context_cosine": metrics["teacher_context_cosine"],
                })
            wandb.log(payload, step=step)

        run_retrieval = args.retrieval_eval_every > 0 and (
            step % args.retrieval_eval_every == 0 or step == args.steps
        )
        run_generation = step % args.eval_every == 0 or step == args.steps

        if run_retrieval:
            retrieval_metrics = evaluate_retrieval(
                model=model,
                adapter=adapter,
                meg=eval_examples.meg.index_select(0, retrieval_indices),
                meg_lengths=eval_examples.meg_lengths.index_select(0, retrieval_indices),
                semantic_vectors=eval_examples.semantic_vectors.index_select(0, retrieval_indices),
                subject_ids=eval_subject_ids.index_select(0, retrieval_indices),
                target_latents=eval_target_latents.index_select(0, retrieval_indices),
                target_ids=eval_target_ids.index_select(0, retrieval_indices),
                target_mask=eval_target_mask.index_select(0, retrieval_indices),
                target_sentences=select_strings(eval_examples.sentences, retrieval_indices),
                config=config,
                device=device,
                condition_source="meg",
                retrieval_batch_size=args.retrieval_batch_size,
                retrieval_t=args.retrieval_t,
            )
            retrieval_metrics["step"] = step
            retrieval_metrics["epoch"] = epoch
            retrieval_metrics["eval_num_examples"] = retrieval_count
            with (output_dir / f"retrieval_step_{step:06d}.json").open("w", encoding="utf-8") as handle:
                json.dump(retrieval_metrics, handle, ensure_ascii=False, indent=2)
            log_for_0(
                f"retrieval step={step} combined_top1={retrieval_metrics['combined']['top1']:.3f} "
                f"combined_top5={retrieval_metrics['combined']['top5']:.3f} "
                f"mean_rank={retrieval_metrics['combined']['mean_rank']:.2f}"
            )
            if run is not None:
                wandb.log(
                    {
                        "retrieval/combined_top1": retrieval_metrics["combined"]["top1"],
                        "retrieval/combined_top5": retrieval_metrics["combined"]["top5"],
                        "retrieval/combined_mean_rank": retrieval_metrics["combined"]["mean_rank"],
                        "retrieval/denoiser_top1": retrieval_metrics["denoiser"]["top1"],
                        "retrieval/decoder_top1": retrieval_metrics["decoder"]["top1"],
                        "retrieval/epoch": epoch,
                    },
                    step=step,
                )

        if not run_generation:
            continue

        eval_metrics = evaluate_generation(
            model=model,
            adapter=adapter,
            meg=eval_examples.meg.index_select(0, eval_indices),
            meg_lengths=eval_examples.meg_lengths.index_select(0, eval_indices),
            semantic_vectors=eval_examples.semantic_vectors.index_select(0, eval_indices),
            subject_ids=eval_subject_ids.index_select(0, eval_indices),
            tokenizer=tokenizer,
            encoder=encoder if args.generation_t5_retrieval else None,
            target_sentences=select_strings(eval_examples.sentences, eval_indices),
            target_latents=eval_target_latents.index_select(0, eval_indices) if args.generation_t5_retrieval else None,
            target_mask=eval_target_mask.index_select(0, eval_indices) if args.generation_t5_retrieval else None,
            target_length=eval_target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=noise_generator,
            condition_source="meg",
        )
        eval_metrics["step"] = step
        eval_metrics["epoch"] = epoch
        eval_metrics["eval_num_examples"] = eval_count
        eval_metrics["eval_split"] = eval_split
        if teacher_adapter is not None:
            eval_metrics["teacher_context"] = evaluate_teacher_context(
                adapter=adapter,
                teacher=teacher_adapter,
                meg=eval_examples.meg.index_select(0, eval_indices),
                meg_lengths=eval_examples.meg_lengths.index_select(0, eval_indices),
                semantic_vectors=eval_examples.semantic_vectors.index_select(0, eval_indices),
                subject_ids=eval_subject_ids.index_select(0, eval_indices),
                device=device,
            )
        eval_metrics["eval_checkpoint_scores"] = eval_checkpoint_scores(eval_metrics)
        save_results(args.output_dir, step, eval_metrics)
        eval_index += 1
        save_eval_checkpoint(
            args=args,
            model=model,
            adapter=adapter,
            optimizer=optimizer,
            config=config,
            metrics=eval_metrics,
            eval_index=eval_index,
            saved_checkpoints=saved_checkpoints,
        )
        score = eval_metrics["eval_checkpoint_scores"]["structured_rank"]
        if best_eval_score is None or score > best_eval_score:
            best_eval_score = score
            best_metrics = eval_metrics
            save_best_results(args.output_dir, eval_metrics)
            best_payload = {
                "step": step,
                "epoch": epoch,
                "score": score,
                "metrics": eval_metrics,
                "adapter_state_dict": adapter.state_dict(),
                "args": vars(args),
                "subject_to_id": subject_to_id,
            }
            if args.unfreeze_elf:
                best_payload["model_state_dict"] = model.state_dict()
            torch.save(best_payload, output_dir / "best_adapter.pt")
            log_for_0(f"New best structured_rank={score:.6f}; saved best_adapter.pt")

        quality = eval_metrics.get("generation_quality", {})
        retrieval = eval_metrics.get("generation_t5_retrieval", {})
        log_for_0(
            f"eval step={step} exact={eval_metrics['exact_match']:.3f} "
            f"words_overlap={quality.get('words_overlap', float('nan')):.3f} "
            f"well_structured={quality.get('well_structured_sentence', float('nan')):.3f} "
            f"gen_t5_top5={retrieval.get('top5')}"
            + (
                f" teacher_mse={eval_metrics['teacher_context']['teacher_context_mse']:.6f} "
                f"teacher_cos={eval_metrics['teacher_context']['teacher_context_cosine']:.3f}"
                if "teacher_context" in eval_metrics
                else ""
            )
        )
        for idx in range(min(5, len(eval_metrics["generated"]))):
            log_for_0(
                f"[{idx}] target={eval_metrics['targets'][idx]!r} "
                f"generated={eval_metrics['generated'][idx]!r}"
            )
        if run is not None:
            payload = {
                "eval/exact_match": eval_metrics["exact_match"],
                "eval/epoch": epoch,
            }
            payload.update(
                {
                    f"generation_quality/{key}": value
                    for key, value in quality.items()
                    if isinstance(value, (int, float))
                }
            )
            if retrieval:
                payload.update(
                    {
                        "generation_t5_retrieval/top1": retrieval["top1"],
                        "generation_t5_retrieval/top5": retrieval["top5"],
                        "generation_t5_retrieval/mean_rank": retrieval["mean_rank"],
                        "generation_t5_retrieval/median_rank": retrieval["median_rank"],
                    }
                )
            if "teacher_context" in eval_metrics:
                payload.update({
                    "teacher_context/mse": eval_metrics["teacher_context"]["teacher_context_mse"],
                    "teacher_context/cosine": eval_metrics["teacher_context"]["teacher_context_cosine"],
                })
            if not args.no_wandb_samples and args.wandb_sample_examples > 0:
                sample_count = min(args.wandb_sample_examples, len(eval_metrics["generated"]))
                overlap_per_sample = eval_metrics.get("word_overlap", {}).get("per_sample", [])
                payload["eval/samples"] = wandb.Table(
                    columns=["index", "target", "generated", "exact", "words_overlap"],
                    data=[
                        [
                            idx,
                            eval_metrics["targets"][idx],
                            eval_metrics["generated"][idx],
                            int(eval_metrics["exact"][idx]),
                            float(overlap_per_sample[idx]) if idx < len(overlap_per_sample) else None,
                        ]
                        for idx in range(sample_count)
                    ],
                )
            wandb.log(payload, step=step)

    if run is not None and best_metrics is not None:
        run.summary["best_exact_match"] = best_metrics["exact_match"]
        run.summary["best_structured_rank"] = best_eval_score
        run.finish()
    log_for_0(f"Finished. best_structured_rank={best_eval_score}")


if __name__ == "__main__":
    main()
