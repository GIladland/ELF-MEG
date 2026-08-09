#!/usr/bin/env python
"""Train a bridge from exported T5 latent sequences into an ELF condition prefix."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from configs.config import SamplingConfig
from modules.t5_encoder import get_encoder

from scripts.meg_context_overfit import (
    OverfitBatch,
    build_config,
    evaluate_generation,
    evaluate_retrieval,
    load_pretrained_model,
    train_step,
)

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
logger = logging.getLogger(__name__)


if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


class T5LatentBridge(nn.Module):
    """Resample a source latent token sequence into an ELF condition prefix."""

    def __init__(
        self,
        input_dim: int = 512,
        dim: int = 512,
        context_length: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.query = nn.Parameter(torch.randn(context_length, dim) * 0.02)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Identity() if input_dim == dim else nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList()
        hidden = int(dim * mlp_ratio)
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "q_norm": nn.LayerNorm(dim),
                "kv_norm": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                "ffn": nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, dim),
                ),
            }))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, semantic_vectors: torch.Tensor):
        # semantic_vectors is [B, T, D] for this script.
        source = self.input_proj(self.input_norm(semantic_vectors))
        batch_size = source.shape[0]
        x = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            q = layer["q_norm"](x)
            kv = layer["kv_norm"](source)
            attn_out, _ = layer["attn"](q, kv, kv, need_weights=False)
            x = x + attn_out
            x = x + layer["ffn"](x)
        context = self.out_norm(x)
        context_mask = torch.ones(
            batch_size,
            self.context_length,
            dtype=context.dtype,
            device=context.device,
        )
        return context, context_mask


class PrecomputedT5ContextAdapter(nn.Module):
    """Return precomputed T5 contexts selected by example index."""

    def __init__(self, contexts: torch.Tensor, masks: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("contexts", contexts)
        self.register_buffer("masks", masks)

    def forward(self, semantic_vectors: torch.Tensor):
        indices = semantic_vectors.reshape(-1).to(dtype=torch.long, device=self.contexts.device)
        context = self.contexts.index_select(0, indices)
        context_mask = self.masks.index_select(0, indices).to(dtype=context.dtype)
        return context, context_mask


class OrderPreservingContextAdapter(nn.Module):
    """Identity-initialized per-token adapter over stored context sequences."""

    def __init__(
        self,
        contexts: torch.Tensor,
        masks: torch.Tensor,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(contexts.shape[-1]) != dim:
            raise ValueError(f"Order-preserving adapter requires input dim {dim}; got {contexts.shape[-1]}.")
        self.register_buffer("contexts", contexts)
        self.register_buffer("masks", masks)
        hidden = int(dim * mlp_ratio)
        self.residual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        final_linear = self.residual[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, semantic_vectors: torch.Tensor):
        indices = semantic_vectors.reshape(-1).to(dtype=torch.long, device=self.contexts.device)
        context = self.contexts.index_select(0, indices)
        context_mask = self.masks.index_select(0, indices).to(dtype=context.dtype)
        residual = self.residual(context) * context_mask.unsqueeze(-1)
        return context + residual, context_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input-key", default="target_t5_latents")
    parser.add_argument("--input-mask-key", default="")
    parser.add_argument("--target-latents-key", default="target_t5_latents")
    parser.add_argument("--target-mask-key", default="t5_attention_mask")
    parser.add_argument("--target-ids-key", default="t5_input_ids")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--bridge-type", choices=["resampler", "identity", "order_preserving"], default="resampler")
    parser.add_argument("--bridge-layers", type=int, default=2)
    parser.add_argument("--bridge-heads", type=int, default=8)
    parser.add_argument("--bridge-dropout", type=float, default=0.0)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument(
        "--val-num-examples",
        type=int,
        default=0,
        help="Hold out this many examples from the end of the NPZ for validation. Default 0 keeps legacy in-sample eval.",
    )
    parser.add_argument("--eval-num-examples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--epochs", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--retrieval-eval-every", type=int, default=0)
    parser.add_argument("--retrieval-num-examples", type=int, default=0)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--retrieval-t", type=float, default=0.5)
    parser.add_argument("--num_sampling_steps", type=int, default=64)
    parser.add_argument("--train-timestep-mode", default="sampling_schedule", choices=["logit_normal", "sampling_schedule", "sampling_schedule_all"])
    parser.add_argument("--train-timestep-steps", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=0.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.1)
    parser.add_argument("--denoiser_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument("--freeze-elf", action="store_true")
    parser.add_argument("--last_n_blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="BrainDiffusion")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default="npz-t5-bridge-elf")
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--wandb_sample_examples", type=int, default=16)
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def make_batches(num_examples: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(num_examples, generator=generator)
    return [perm[start:start + batch_size] for start in range(0, num_examples, batch_size)]


def select_strings(items: list[str], indices: torch.Tensor) -> list[str]:
    return [items[int(index)] for index in indices.detach().cpu().tolist()]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    input_latents = torch.as_tensor(data[args.input_key], dtype=torch.float32)
    if args.input_mask_key:
        input_mask = torch.as_tensor(data[args.input_mask_key], dtype=torch.float32)
    elif args.input_key == args.target_latents_key:
        input_mask = torch.as_tensor(data[args.target_mask_key], dtype=torch.float32)
    else:
        input_mask = torch.ones(input_latents.shape[:2], dtype=torch.float32)
    target_latents = torch.as_tensor(data[args.target_latents_key], dtype=torch.float32)
    target_mask = torch.as_tensor(data[args.target_mask_key], dtype=torch.float32)
    target_ids = torch.as_tensor(data[args.target_ids_key], dtype=torch.long)
    sentences = _strings(data[args.sentence_key])
    total_n = input_latents.shape[0] if args.num_examples <= 0 else min(args.num_examples, input_latents.shape[0])
    if args.val_num_examples < 0:
        raise ValueError("--val-num-examples must be non-negative.")
    val_n = min(args.val_num_examples, max(0, total_n - 1))
    train_n = total_n - val_n
    if train_n <= 0:
        raise ValueError(f"Need at least one training example after holdout; got total_n={total_n}, val_n={val_n}.")
    input_latents = input_latents[:total_n]
    input_mask = input_mask[:total_n]
    target_latents = target_latents[:total_n]
    target_mask = target_mask[:total_n]
    target_ids = target_ids[:total_n]
    sentences = sentences[:total_n]
    train_indices = torch.arange(0, train_n, dtype=torch.long)
    eval_pool_indices = (
        torch.arange(train_n, total_n, dtype=torch.long)
        if val_n > 0
        else train_indices
    )
    eval_split = "val" if val_n > 0 else "train"
    indexed_bridge = args.bridge_type in {"identity", "order_preserving"}
    if not indexed_bridge:
        input_latents = input_latents * input_mask.unsqueeze(-1)
    target_length = int(target_latents.shape[1])
    context_length = int(input_latents.shape[1]) if indexed_bridge else args.context_length

    steps_per_epoch = int(np.ceil(train_n / args.batch_size))
    if args.epochs > 0:
        args.steps = int(np.ceil(steps_per_epoch * args.epochs))
    logger.info(
        "Loaded total_n=%d train_n=%d val_n=%d eval_split=%s input=%s target=%s bridge=%s context_length=%d batch_size=%d steps=%d epochs=%.2f",
        total_n, train_n, val_n, eval_split,
        tuple(input_latents.shape), tuple(target_latents.shape), args.bridge_type,
        context_length, args.batch_size, args.steps, args.steps / max(1, steps_per_epoch),
    )

    config = build_config(args, max_length=context_length + target_length)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
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
    if args.freeze_elf:
        for param in model.parameters():
            param.requires_grad_(False)
    logger.info("Trainable ELF parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.bridge_type == "identity":
        bridge = PrecomputedT5ContextAdapter(
            contexts=input_latents.to(device),
            masks=input_mask.to(device),
        )
    elif args.bridge_type == "order_preserving":
        bridge = OrderPreservingContextAdapter(
            contexts=input_latents.to(device),
            masks=input_mask.to(device),
            dim=encoder_config.d_model,
            dropout=args.bridge_dropout,
        ).to(device)
    else:
        bridge = T5LatentBridge(
            input_dim=int(input_latents.shape[-1]),
            dim=encoder_config.d_model,
            context_length=context_length,
            num_layers=args.bridge_layers,
            num_heads=args.bridge_heads,
            dropout=args.bridge_dropout,
        ).to(device)
    logger.info("Trainable bridge parameters: %d", sum(p.numel() for p in bridge.parameters() if p.requires_grad))

    params = [p for p in model.parameters() if p.requires_grad] + list(bridge.parameters())
    if not params:
        raise ValueError("No trainable parameters. Do not combine --freeze-elf with --bridge-type identity.")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    order_generator = torch.Generator().manual_seed(args.seed + 2)
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )

    run = None
    if args.use_wandb and wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            config={
                **vars(args),
                "max_length": config.max_length,
                "target_length": target_length,
                "num_total_examples": total_n,
                "num_train_examples": train_n,
                "num_val_examples": val_n,
                "eval_split": eval_split,
                "steps_per_epoch": steps_per_epoch,
            },
        )
        wandb.define_metric("train/epoch")
        wandb.define_metric("eval/epoch")
        wandb.define_metric("retrieval/epoch")
        wandb.define_metric("train/*", step_metric="train/epoch")
        wandb.define_metric("eval/*", step_metric="eval/epoch")
        wandb.define_metric("retrieval/*", step_metric="retrieval/epoch")
        wandb.define_metric("generation_quality/*", step_metric="eval/epoch")
        wandb.define_metric("generation_t5_retrieval/*", step_metric="eval/epoch")
    eval_available = int(eval_pool_indices.numel())
    eval_n = eval_available if args.eval_num_examples <= 0 else min(args.eval_num_examples, eval_available)
    eval_indices = eval_pool_indices[:eval_n]
    eval_semantic = (
        eval_indices.to(dtype=torch.float32).reshape(-1, 1)
        if indexed_bridge
        else input_latents.index_select(0, eval_indices)
    )
    retrieval_n = (
        eval_n
        if args.retrieval_num_examples <= 0
        else min(args.retrieval_num_examples, eval_available)
    )
    retrieval_indices = eval_pool_indices[:retrieval_n]
    retrieval_semantic = (
        retrieval_indices.to(dtype=torch.float32).reshape(-1, 1)
        if indexed_bridge
        else input_latents.index_select(0, retrieval_indices)
    )
    best_score = -1.0
    step = 0
    while step < args.steps:
        for batch_indices in make_batches(train_n, args.batch_size, order_generator):
            if step >= args.steps:
                break
            indices = train_indices.index_select(0, batch_indices)
            batch = OverfitBatch(
                indices=indices,
                meg=torch.zeros((indices.numel(), 1, 1), dtype=torch.float32),
                meg_lengths=torch.ones((indices.numel(),), dtype=torch.long),
                semantic_vectors=(
                    indices.to(dtype=torch.float32).reshape(-1, 1)
                    if indexed_bridge
                    else input_latents.index_select(0, indices)
                ),
                subject_ids=torch.zeros((indices.numel(),), dtype=torch.long),
                target_latents=target_latents.index_select(0, indices),
                target_ids=target_ids.index_select(0, indices),
                target_mask=target_mask.index_select(0, indices),
            )
            step += 1
            metrics = train_step(
                model=model,
                adapter=bridge,
                batch=batch,
                optimizer=optimizer,
                config=config,
                device=device,
                noise_generator=noise_generator,
                condition_source="semantic",
                cond_dropout_prob=args.cond_dropout_prob,
                train_timestep_mode=args.train_timestep_mode,
                train_timestep_steps=args.train_timestep_steps,
            )
            epoch = step / max(1, steps_per_epoch)
            if step == 1 or step % 10 == 0:
                logger.info(
                    "step=%d epoch=%.2f loss=%.6f denoiser=%.6f decoder=%.6f",
                    step, epoch, metrics["loss"], metrics["denoiser_loss"], metrics["decoder_loss"],
                )
            if run is not None:
                wandb.log({
                    "train/loss": metrics["loss"],
                    "train/denoiser_loss": metrics["denoiser_loss"],
                    "train/decoder_loss": metrics["decoder_loss"],
                    "train/epoch": epoch,
                    "train/step": step,
                }, step=step)

            run_retrieval = args.retrieval_eval_every > 0 and (
                step % args.retrieval_eval_every == 0 or step == args.steps
            )
            if run_retrieval:
                retrieval_metrics = evaluate_retrieval(
                    model=model,
                    adapter=bridge,
                    meg=torch.zeros((retrieval_n, 1, 1), dtype=torch.float32),
                    meg_lengths=torch.ones((retrieval_n,), dtype=torch.long),
                    semantic_vectors=retrieval_semantic,
                    subject_ids=torch.zeros((retrieval_n,), dtype=torch.long),
                    target_latents=target_latents.index_select(0, retrieval_indices),
                    target_ids=target_ids.index_select(0, retrieval_indices),
                    target_mask=target_mask.index_select(0, retrieval_indices),
                    target_sentences=select_strings(sentences, retrieval_indices),
                    config=config,
                    device=device,
                    condition_source="semantic",
                    retrieval_batch_size=args.retrieval_batch_size,
                    retrieval_t=args.retrieval_t,
                )
                retrieval_metrics["step"] = step
                retrieval_metrics["epoch"] = epoch
                retrieval_metrics["split"] = eval_split
                retrieval_metrics["num_train_examples"] = train_n
                retrieval_metrics["eval_num_examples"] = retrieval_n
                retrieval_path = output_dir / f"retrieval_step_{step:06d}.json"
                with retrieval_path.open("w", encoding="utf-8") as f:
                    json.dump(retrieval_metrics, f, ensure_ascii=False, indent=2)
                logger.info(
                    "retrieval step=%d combined_top1=%.3f combined_top5=%.3f mean_rank=%.2f",
                    step,
                    retrieval_metrics["combined"]["top1"],
                    retrieval_metrics["combined"]["top5"],
                    retrieval_metrics["combined"]["mean_rank"],
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

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                eval_metrics = evaluate_generation(
                    model=model,
                    adapter=bridge,
                    meg=torch.zeros((eval_n, 1, 1), dtype=torch.float32),
                    meg_lengths=torch.ones((eval_n,), dtype=torch.long),
                    semantic_vectors=eval_semantic,
                    subject_ids=torch.zeros((eval_n,), dtype=torch.long),
                    tokenizer=tokenizer,
                    encoder=encoder,
                    target_sentences=select_strings(sentences, eval_indices),
                    target_latents=target_latents.index_select(0, eval_indices),
                    target_mask=target_mask.index_select(0, eval_indices),
                    target_length=target_length,
                    context_length=context_length,
                    config=config,
                    sampling_config=sampling_config,
                    device=device,
                    generator=noise_generator,
                    condition_source="semantic",
                )
                out = {
                    "step": step,
                    "epoch": epoch,
                    "split": eval_split,
                    "num_train_examples": train_n,
                    "num_eval_examples": eval_n,
                    **eval_metrics,
                }
                out_path = output_dir / f"eval_step_{step:06d}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                quality = eval_metrics["generation_quality"]
                gen_retr = eval_metrics.get("generation_t5_retrieval", {})
                logger.info(
                    "eval step=%d exact=%.4f well_structured=%.4f words_overlap=%.4f gen_t5_top1=%.4f",
                    step,
                    eval_metrics["exact_match"],
                    quality.get("well_structured_sentence", float("nan")),
                    quality.get("words_overlap", float("nan")),
                    gen_retr.get("top1", float("nan")),
                )
                for i in range(min(5, eval_n)):
                    logger.info("[%d] target=%r generated=%r", i, eval_metrics["targets"][i], eval_metrics["generated"][i])
                if run is not None:
                    payload = {
                        "eval/exact_match": eval_metrics["exact_match"],
                        "eval/epoch": epoch,
                        **{f"generation_quality/{k}": v for k, v in quality.items() if isinstance(v, (int, float))},
                    }
                    if gen_retr:
                        payload.update({
                            "generation_t5_retrieval/top1": gen_retr["top1"],
                            "generation_t5_retrieval/top5": gen_retr["top5"],
                            "generation_t5_retrieval/mean_rank": gen_retr["mean_rank"],
                        })
                    if args.wandb_sample_examples > 0:
                        sample_count = min(args.wandb_sample_examples, len(eval_metrics["generated"]))
                        overlap_per_sample = eval_metrics.get("word_overlap", {}).get("per_sample", [])
                        payload["eval/samples"] = wandb.Table(
                            columns=["index", "target", "generated", "exact", "words_overlap"],
                            data=[
                                [
                                    i,
                                    eval_metrics["targets"][i],
                                    eval_metrics["generated"][i],
                                    int(eval_metrics["exact"][i]),
                                    float(overlap_per_sample[i]) if i < len(overlap_per_sample) else None,
                                ]
                                for i in range(sample_count)
                            ],
                        )
                    wandb.log(payload, step=step)
                score = gen_retr.get("top1", eval_metrics["exact_match"])
                if score > best_score:
                    best_score = score
                    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)
                    torch.save(
                        {
                            "step": step,
                            "epoch": epoch,
                            "score": score,
                            "bridge": bridge.state_dict(),
                            "model": model.state_dict(),
                            "args": vars(args),
                        },
                        output_dir / "best.pt",
                    )

    if run is not None:
        run.summary["best_score"] = best_score
        run.finish()
    logger.info("Finished. best_score=%.4f", best_score)


if __name__ == "__main__":
    main()
