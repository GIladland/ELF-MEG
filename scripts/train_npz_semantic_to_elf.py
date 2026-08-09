#!/usr/bin/env python
"""Train ELF text generation from fixed semantic vectors stored in an NPZ."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from transformers import AutoTokenizer

from configs.config import SamplingConfig
from modules.t5_encoder import get_encoder
from scripts.meg_context_overfit import (
    OverfitBatch,
    SemanticVectorContextProjector,
    build_config,
    encode_text_batched,
    evaluate_generation,
    evaluate_retrieval,
    freeze_for_toy_tuning,
    load_pretrained_model,
    tokenize_sentences,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--context_length", type=int, default=16)
    parser.add_argument("--semantic-hidden-dim", type=int, default=2048)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument(
        "--val-num-examples",
        type=int,
        default=0,
        help="Hold out this many examples from the end of the NPZ for validation. Default 0 keeps legacy in-sample eval.",
    )
    parser.add_argument("--eval-num-examples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--target-latents-mode",
        choices=["precompute", "cache", "lazy"],
        default="precompute",
        help=(
            "How to provide T5 target latents. precompute keeps legacy all-in-RAM behavior; "
            "cache writes/reads a disk-backed .npy memmap; lazy encodes each requested batch."
        ),
    )
    parser.add_argument(
        "--target-latents-cache",
        default="",
        help="Path to a .npy target-latent cache when --target-latents-mode=cache.",
    )
    parser.add_argument(
        "--target-latents-cache-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for disk target-latent cache. Values are cast to float32 for training.",
    )
    parser.add_argument("--target-latent-cache-chunk-size", type=int, default=4096)
    parser.add_argument("--target-latent-encode-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--retrieval-eval-every", type=int, default=0)
    parser.add_argument("--retrieval-num-examples", type=int, default=0)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--retrieval-t", type=float, default=0.5)
    parser.add_argument("--num_sampling_steps", type=int, default=32)
    parser.add_argument(
        "--train-timestep-mode",
        default="sampling_schedule",
        choices=["logit_normal", "sampling_schedule", "sampling_schedule_all"],
    )
    parser.add_argument("--train-timestep-steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--denoiser_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument(
        "--train-target-mask-mode",
        choices=["valid", "full"],
        default="valid",
        help=(
            "Which target positions receive train loss. valid matches the T5 attention mask; "
            "full also trains post-EOS/pad positions, testing whether stop/length errors come from masked pads."
        ),
    )
    parser.add_argument("--freeze-elf", action="store_true")
    parser.add_argument("--last_n_blocks", type=int, default=1)
    parser.add_argument("--generation-t5-retrieval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="BrainDiffusion")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default="npz-semantic-to-elf")
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--wandb_sample_examples", type=int, default=16)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def make_batches(num_examples: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(num_examples, generator=generator)
    return [perm[start:start + batch_size] for start in range(0, num_examples, batch_size)]


def select_strings(items: list[str], indices: torch.Tensor) -> list[str]:
    return [items[int(index)] for index in indices.detach().cpu().tolist()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def encode_target_latent_memmap(
    *,
    cache_path: Path,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder: torch.nn.Module,
    latent_mean: float,
    latent_std: float,
    device: torch.device,
    encode_batch_size: int,
    chunk_size: int,
    dtype: str,
    metadata: dict,
) -> np.memmap:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    storage_dtype = np.float16 if dtype == "float16" else np.float32
    expected_shape = (
        int(input_ids.shape[0]),
        int(input_ids.shape[1]),
        int(getattr(encoder.config, "d_model")),
    )
    meta_path = cache_meta_path(cache_path)
    if cache_path.exists() and meta_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if tuple(cached.shape) == expected_shape and meta.get("metadata") == metadata:
            logger.info("Using target latent cache %s shape=%s dtype=%s", cache_path, cached.shape, cached.dtype)
            return cached
        logger.warning(
            "Ignoring stale target latent cache %s shape=%s metadata_match=%s; expected shape=%s",
            cache_path,
            tuple(cached.shape),
            meta.get("metadata") == metadata,
            expected_shape,
        )

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    logger.info(
        "Encoding target T5 latents to cache %s shape=%s dtype=%s",
        cache_path,
        expected_shape,
        np.dtype(storage_dtype).name,
    )
    mmap = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=storage_dtype, shape=expected_shape)
    chunk_size = max(1, chunk_size)
    for start in range(0, input_ids.shape[0], chunk_size):
        end = min(start + chunk_size, input_ids.shape[0])
        chunk = encode_text_batched(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            encoder=encoder,
            latent_mean=latent_mean,
            latent_std=latent_std,
            device=device,
            batch_size=encode_batch_size,
        )
        mmap[start:end] = chunk.numpy().astype(storage_dtype, copy=False)
        mmap.flush()
        logger.info("cached target latents rows %d:%d / %d", start, end, input_ids.shape[0])
    del mmap
    os.replace(tmp_path, cache_path)
    meta_path.write_text(
        json.dumps(
            {
                "path": str(cache_path),
                "shape": expected_shape,
                "dtype": np.dtype(storage_dtype).name,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return np.load(cache_path, mmap_mode="r")


def select_memmap_rows(mmap: np.memmap, indices: torch.Tensor) -> torch.Tensor:
    index_np = indices.detach().cpu().numpy()
    return torch.as_tensor(np.asarray(mmap[index_np]), dtype=torch.float32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    semantic_vectors = torch.as_tensor(data[args.input_key], dtype=torch.float32)
    sentences = _strings(data[args.sentence_key])
    total_n = semantic_vectors.shape[0] if args.num_examples <= 0 else min(args.num_examples, semantic_vectors.shape[0])
    if args.val_num_examples < 0:
        raise ValueError("--val-num-examples must be non-negative.")
    val_n = min(args.val_num_examples, max(0, total_n - 1))
    train_n = total_n - val_n
    if train_n <= 0:
        raise ValueError(f"Need at least one training example after holdout; got total_n={total_n}, val_n={val_n}.")
    semantic_vectors = semantic_vectors[:total_n]
    sentences = sentences[:total_n]
    train_indices = torch.arange(0, train_n, dtype=torch.long)
    eval_pool_indices = (
        torch.arange(train_n, total_n, dtype=torch.long)
        if val_n > 0
        else train_indices
    )
    eval_split = "val" if val_n > 0 else "train"

    steps_per_epoch = int(np.ceil(train_n / args.batch_size))
    if args.epochs > 0:
        args.steps = int(np.ceil(steps_per_epoch * args.epochs))
    logger.info(
        "Loaded total_n=%d train_n=%d val_n=%d eval_split=%s semantic=%s batch_size=%d steps=%d epochs=%.2f",
        total_n,
        train_n,
        val_n,
        eval_split,
        tuple(semantic_vectors.shape),
        args.batch_size,
        args.steps,
        args.steps / max(1, steps_per_epoch),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask = tokenize_sentences(tokenizer, sentences)
    target_length = input_ids.shape[1]
    config = build_config(args, max_length=args.context_length + target_length)

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
        trainable_names = []
    else:
        trainable_names = freeze_for_toy_tuning(model, args.last_n_blocks)
    logger.info("Trainable ELF parameter groups: %d", len(trainable_names))
    logger.info("Trainable ELF parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    adapter = SemanticVectorContextProjector(
        input_dim=int(semantic_vectors.shape[-1]),
        context_dim=encoder_config.d_model,
        context_length=args.context_length,
        hidden_dim=args.semantic_hidden_dim,
        dropout=args.adapter_dropout,
    ).to(device)
    logger.info("Trainable adapter parameters: %d", sum(p.numel() for p in adapter.parameters() if p.requires_grad))

    target_latents: torch.Tensor | np.memmap | None
    if args.target_latents_mode == "precompute":
        logger.info("Encoding target T5 latents in memory")
        target_latents = encode_text_batched(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            device=device,
            batch_size=args.batch_size,
        )
    elif args.target_latents_mode == "cache":
        if not args.target_latents_cache:
            raise ValueError("--target-latents-cache is required when --target-latents-mode=cache.")
        target_latents = encode_target_latent_memmap(
            cache_path=Path(args.target_latents_cache),
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            device=device,
            encode_batch_size=args.target_latent_encode_batch_size,
            chunk_size=args.target_latent_cache_chunk_size,
            dtype=args.target_latents_cache_dtype,
            metadata={
                "npz_path": str(Path(args.npz_path).resolve()),
                "input_key": args.input_key,
                "sentence_key": args.sentence_key,
                "total_n": int(total_n),
                "encoder_model_name": args.encoder_model_name,
                "latent_mean": float(config.latent_mean),
                "latent_std": float(config.latent_std),
                "target_length": int(target_length),
            },
        )
    else:
        logger.info("Using lazy per-batch target T5 latent encoding")
        target_latents = None

    def get_target_latents(indices: torch.Tensor) -> torch.Tensor:
        if isinstance(target_latents, torch.Tensor):
            return target_latents.index_select(0, indices)
        if isinstance(target_latents, np.memmap):
            return select_memmap_rows(target_latents, indices)
        return encode_text_batched(
            input_ids=input_ids.index_select(0, indices),
            attention_mask=attention_mask.index_select(0, indices),
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            device=device,
            batch_size=args.target_latent_encode_batch_size,
        )

    target_ids = input_ids.detach().cpu()
    target_mask = attention_mask.detach().cpu().to(torch.float32)
    if args.train_target_mask_mode == "valid":
        train_target_mask = target_mask
    elif args.train_target_mask_mode == "full":
        train_target_mask = torch.ones_like(target_mask)
    else:
        raise ValueError(f"Unsupported train target mask mode: {args.train_target_mask_mode}")
    train_target_mask_density = float(train_target_mask.mean().item())
    meg = torch.zeros((total_n, 1, 1), dtype=torch.float32)
    meg_lengths = torch.ones((total_n,), dtype=torch.long)
    subject_ids = torch.zeros((total_n,), dtype=torch.long)

    params = [p for p in model.parameters() if p.requires_grad] + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    noise_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(args.seed + 17)
    order_generator = torch.Generator().manual_seed(args.seed + 29)
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "target_length": target_length,
                "num_total_examples": total_n,
                "num_train_examples": train_n,
                "num_val_examples": val_n,
                "eval_split": eval_split,
                "train_target_mask_density": train_target_mask_density,
            },
            f,
            indent=2,
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
                "target_length": target_length,
                "num_total_examples": total_n,
                "num_train_examples": train_n,
                "num_val_examples": val_n,
                "eval_split": eval_split,
                "steps_per_epoch": steps_per_epoch,
                "semantic_dim": int(semantic_vectors.shape[-1]),
                "train_target_mask_density": train_target_mask_density,
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
    retrieval_n = (
        eval_n
        if args.retrieval_num_examples <= 0
        else min(args.retrieval_num_examples, eval_available)
    )
    retrieval_indices = eval_pool_indices[:retrieval_n]
    eval_target_latents = get_target_latents(eval_indices)
    retrieval_target_latents = get_target_latents(retrieval_indices) if retrieval_n > 0 else eval_target_latents
    best_score = None

    for step in range(1, args.steps + 1):
        epoch = step * args.batch_size / max(1, train_n)
        batches = make_batches(train_n, args.batch_size, order_generator)
        indices = train_indices.index_select(0, batches[(step - 1) % len(batches)])
        batch = OverfitBatch(
            indices=indices,
            meg=meg.index_select(0, indices),
            meg_lengths=meg_lengths.index_select(0, indices),
            semantic_vectors=semantic_vectors.index_select(0, indices),
            subject_ids=subject_ids.index_select(0, indices),
            target_latents=get_target_latents(indices),
            target_ids=target_ids.index_select(0, indices),
            target_mask=train_target_mask.index_select(0, indices),
        )
        metrics = train_step(
            model=model,
            adapter=adapter,
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
        if step == 1 or step % 10 == 0:
            logger.info(
                "step=%d epoch=%.2f loss=%.6f denoiser=%.6f decoder=%.6f",
                step,
                epoch,
                metrics["loss"],
                metrics["denoiser_loss"],
                metrics["decoder_loss"],
            )
        if run is not None:
            wandb.log(
                {
                    "train/loss": metrics["loss"],
                    "train/denoiser_loss": metrics["denoiser_loss"],
                    "train/decoder_loss": metrics["decoder_loss"],
                    "train/epoch": epoch,
                    "train/step": step,
                },
                step=step,
            )

        run_retrieval = args.retrieval_eval_every > 0 and (
            step % args.retrieval_eval_every == 0 or step == args.steps
        )
        if run_retrieval:
            retrieval_metrics = evaluate_retrieval(
                model=model,
                adapter=adapter,
                meg=meg.index_select(0, retrieval_indices),
                meg_lengths=meg_lengths.index_select(0, retrieval_indices),
                semantic_vectors=semantic_vectors.index_select(0, retrieval_indices),
                subject_ids=subject_ids.index_select(0, retrieval_indices),
                target_latents=retrieval_target_latents,
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
            retrieval_metrics["num_eval_examples"] = retrieval_n
            with (output_dir / f"retrieval_step_{step:06d}.json").open("w", encoding="utf-8") as f:
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

        if step % args.eval_every != 0 and step != args.steps:
            continue
        eval_metrics = evaluate_generation(
            model=model,
            adapter=adapter,
            meg=meg.index_select(0, eval_indices),
            meg_lengths=meg_lengths.index_select(0, eval_indices),
            semantic_vectors=semantic_vectors.index_select(0, eval_indices),
            subject_ids=subject_ids.index_select(0, eval_indices),
            tokenizer=tokenizer,
            encoder=encoder if args.generation_t5_retrieval else None,
            target_sentences=select_strings(sentences, eval_indices),
            target_latents=eval_target_latents if args.generation_t5_retrieval else None,
            target_mask=target_mask.index_select(0, eval_indices) if args.generation_t5_retrieval else None,
            target_length=target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=noise_generator,
            condition_source="semantic",
        )
        eval_metrics["step"] = step
        eval_metrics["epoch"] = epoch
        eval_metrics["split"] = eval_split
        eval_metrics["num_train_examples"] = train_n
        eval_metrics["num_eval_examples"] = eval_n
        with (output_dir / f"eval_step_{step:06d}.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
        retrieval = eval_metrics.get("generation_t5_retrieval", {})
        score = retrieval.get("top1", eval_metrics["exact_match"])
        if best_score is None or score > best_score:
            best_score = score
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
            if args.save_best_checkpoint:
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "score": score,
                        "model_state_dict": model.state_dict(),
                        "adapter_state_dict": adapter.state_dict(),
                        "args": vars(args),
                        "config": {
                            key: value
                            for key, value in vars(config).items()
                            if isinstance(value, (str, int, float, bool, type(None), list, tuple))
                        },
                    },
                    output_dir / "best.pt",
                )
        quality = eval_metrics.get("generation_quality", {})
        logger.info(
            "eval step=%d exact=%.3f well_structured=%.3f words_overlap=%.3f gen_t5_top1=%s",
            step,
            eval_metrics["exact_match"],
            quality.get("well_structured_sentence", float("nan")),
            quality.get("words_overlap", float("nan")),
            retrieval.get("top1"),
        )
        if run is not None:
            payload = {
                "eval/exact_match": eval_metrics["exact_match"],
                "eval/epoch": epoch,
            }
            if retrieval:
                payload.update(
                    {
                        "generation_t5_retrieval/top1": retrieval["top1"],
                        "generation_t5_retrieval/top5": retrieval["top5"],
                        "generation_t5_retrieval/mean_rank": retrieval["mean_rank"],
                        "generation_t5_retrieval/median_rank": retrieval["median_rank"],
                    }
                )
            payload.update(
                {
                    f"generation_quality/{key}": value
                    for key, value in quality.items()
                    if isinstance(value, (int, float))
                }
            )
            if args.wandb_sample_examples > 0:
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
        for idx in range(min(5, eval_n)):
            logger.info("[%d] target=%r generated=%r", idx, eval_metrics["targets"][idx], eval_metrics["generated"][idx])

    if run is not None:
        run.summary["best_score"] = best_score
        run.finish()
    logger.info("Finished. best_score=%s", best_score)


if __name__ == "__main__":
    main()
