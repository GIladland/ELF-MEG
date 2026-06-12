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
    """Resample a T5 latent token sequence into an ELF condition prefix."""

    def __init__(
        self,
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
        self.input_norm = nn.LayerNorm(dim)
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
        source = self.input_norm(semantic_vectors)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input-key", default="target_t5_latents")
    parser.add_argument("--target-latents-key", default="target_t5_latents")
    parser.add_argument("--target-mask-key", default="t5_attention_mask")
    parser.add_argument("--target-ids-key", default="t5_input_ids")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--bridge-layers", type=int, default=2)
    parser.add_argument("--bridge-heads", type=int, default=8)
    parser.add_argument("--bridge-dropout", type=float, default=0.0)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--eval-num-examples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--epochs", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=1000)
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
    parser.add_argument("--wandb_run_name", default="npz-t5-bridge-elf")
    parser.add_argument("--wandb_notes", default=None)
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def make_batches(num_examples: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(num_examples, generator=generator)
    return [perm[start:start + batch_size] for start in range(0, num_examples, batch_size)]


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
    target_latents = torch.as_tensor(data[args.target_latents_key], dtype=torch.float32)
    target_mask = torch.as_tensor(data[args.target_mask_key], dtype=torch.float32)
    target_ids = torch.as_tensor(data[args.target_ids_key], dtype=torch.long)
    sentences = _strings(data[args.sentence_key])
    n = input_latents.shape[0] if args.num_examples <= 0 else min(args.num_examples, input_latents.shape[0])
    input_latents = input_latents[:n]
    target_latents = target_latents[:n]
    target_mask = target_mask[:n]
    target_ids = target_ids[:n]
    sentences = sentences[:n]
    target_length = int(target_latents.shape[1])

    steps_per_epoch = int(np.ceil(n / args.batch_size))
    if args.epochs > 0:
        args.steps = int(np.ceil(steps_per_epoch * args.epochs))
    logger.info(
        "Loaded n=%d input=%s target=%s batch_size=%d steps=%d epochs=%.2f",
        n, tuple(input_latents.shape), tuple(target_latents.shape), args.batch_size,
        args.steps, args.steps / max(1, steps_per_epoch),
    )

    config = build_config(args, max_length=args.context_length + target_length)
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

    bridge = T5LatentBridge(
        dim=encoder_config.d_model,
        context_length=args.context_length,
        num_layers=args.bridge_layers,
        num_heads=args.bridge_heads,
        dropout=args.bridge_dropout,
    ).to(device)
    logger.info("Trainable bridge parameters: %d", sum(p.numel() for p in bridge.parameters() if p.requires_grad))

    params = [p for p in model.parameters() if p.requires_grad] + list(bridge.parameters())
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
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            config={
                **vars(args),
                "max_length": config.max_length,
                "target_length": target_length,
                "num_train_examples": n,
                "steps_per_epoch": steps_per_epoch,
            },
        )
    eval_n = n if args.eval_num_examples <= 0 else min(args.eval_num_examples, n)
    eval_slice = slice(0, eval_n)
    best_exact = -1.0
    step = 0
    while step < args.steps:
        for indices in make_batches(n, args.batch_size, order_generator):
            if step >= args.steps:
                break
            batch = OverfitBatch(
                indices=indices,
                meg=torch.zeros((indices.numel(), 1, 1), dtype=torch.float32),
                meg_lengths=torch.ones((indices.numel(),), dtype=torch.long),
                semantic_vectors=input_latents.index_select(0, indices),
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
                }, step=step)

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                eval_metrics = evaluate_generation(
                    model=model,
                    adapter=bridge,
                    meg=torch.zeros((eval_n, 1, 1), dtype=torch.float32),
                    meg_lengths=torch.ones((eval_n,), dtype=torch.long),
                    semantic_vectors=input_latents[eval_slice],
                    subject_ids=torch.zeros((eval_n,), dtype=torch.long),
                    tokenizer=tokenizer,
                    encoder=encoder,
                    target_sentences=sentences[:eval_n],
                    target_latents=target_latents[eval_slice],
                    target_mask=target_mask[eval_slice],
                    target_length=target_length,
                    context_length=args.context_length,
                    config=config,
                    sampling_config=sampling_config,
                    device=device,
                    generator=noise_generator,
                    condition_source="semantic",
                )
                out = {
                    "step": step,
                    "epoch": epoch,
                    **eval_metrics,
                }
                out_path = output_dir / f"eval_step_{step:06d}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                quality = eval_metrics["generation_quality"]
                gen_retr = eval_metrics.get("generation_t5_retrieval", {})
                logger.info(
                    "eval step=%d exact=%.4f well_structured=%.4f gen_t5_top1=%.4f",
                    step,
                    eval_metrics["exact_match"],
                    quality.get("well_structured_sentence", float("nan")),
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
                    wandb.log(payload, step=step)
                if eval_metrics["exact_match"] > best_exact:
                    best_exact = eval_metrics["exact_match"]
                    torch.save(
                        {
                            "step": step,
                            "bridge": bridge.state_dict(),
                            "model": model.state_dict(),
                            "args": vars(args),
                        },
                        output_dir / "best.pt",
                    )

    if run is not None:
        run.summary["best_exact_match"] = best_exact
        run.finish()
    logger.info("Finished. best_exact=%.4f", best_exact)


if __name__ == "__main__":
    main()
