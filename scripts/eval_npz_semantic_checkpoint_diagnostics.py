#!/usr/bin/env python
"""Diagnostics for semantic-vector-to-ELF checkpoints trained from an NPZ."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from transformers import AutoTokenizer

from configs.config import Config, SamplingConfig
from modules.model import ELF_models
from modules.semantic_adapter import DelayTokenContextProjector
from modules.t5_encoder import get_encoder
from scripts.meg_context_overfit import (
    SemanticVectorContextProjector,
    generation_repetition_metrics,
    mean_pool_latents,
    rank_true_targets_by_similarity,
    tokenize_sentences,
    word_overlap_metrics,
)
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    _dlm_decode_batch,
    _generate_samples_single_batch,
    mask_after_eos,
    shift_left,
)
from utils.sampling_utils import get_sampling_steps


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-latents-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--encoder-model-name", default="t5-small")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--semantic-hidden-dim", type=int, default=0)
    parser.add_argument(
        "--semantic-adapter-kind",
        choices=["flat", "delay_tokens"],
        default=None,
        help="Override the checkpoint adapter type; by default this is inferred from checkpoint args.",
    )
    parser.add_argument("--semantic-delay-tokens", type=int, default=0)
    parser.add_argument("--semantic-delay-layers", type=int, default=0)
    parser.add_argument("--semantic-delay-heads", type=int, default=0)
    parser.add_argument(
        "--semantic-input-projection-dim",
        type=int,
        default=-1,
        help="Override the checkpoint's semantic input projection dimension; -1 infers it and 0 disables it.",
    )
    parser.add_argument("--num-sampling-steps", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument("--direct-decode-t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def load_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload: {type(checkpoint).__name__}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain model_state_dict")
    if "adapter_state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain adapter_state_dict")
    return checkpoint


def config_from_checkpoint(
    checkpoint: dict,
    *,
    model_name: str,
    encoder_model_name: str,
    max_length: int,
) -> Config:
    config = Config()
    config.model = model_name
    config.encoder_model_name = encoder_model_name
    config.max_length = max_length
    saved = checkpoint.get("config") or {}
    for key, value in saved.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.model = model_name
    config.encoder_model_name = encoder_model_name
    config.max_length = max_length
    config.latent_mean = float(getattr(config, "latent_mean", 0.0))
    config.latent_std = float(getattr(config, "latent_std", 0.2))
    config.num_time_tokens = int(getattr(config, "num_time_tokens", 4))
    config.num_self_cond_cfg_tokens = int(getattr(config, "num_self_cond_cfg_tokens", 4))
    config.num_model_mode_tokens = int(getattr(config, "num_model_mode_tokens", 4))
    config.denoiser_p_mean = float(getattr(config, "denoiser_p_mean", -1.5))
    config.denoiser_p_std = float(getattr(config, "denoiser_p_std", 0.8))
    config.denoiser_noise_scale = float(getattr(config, "denoiser_noise_scale", 2.0))
    config.time_schedule = str(getattr(config, "time_schedule", "logit_normal"))
    config.self_cond_prob = float(getattr(config, "self_cond_prob", 0.5))
    config.use_bf16 = bool(getattr(config, "use_bf16", True))
    return config


def build_model(
    *,
    config: Config,
    encoder_dim: int,
    vocab_size: int,
    checkpoint: dict,
    device: torch.device,
):
    model = ELF_models[config.model](
        text_encoder_dim=encoder_dim,
        max_length=config.max_length,
        attn_drop=float(getattr(config, "attn_dropout", 0.0)),
        proj_drop=float(getattr(config, "proj_dropout", 0.0)),
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        vocab_size=vocab_size,
        bottleneck_dim=int(getattr(config, "bottleneck_dim", 128)),
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_adapter(
    *,
    input_dim: int,
    context_dim: int,
    context_length: int,
    hidden_dim: int,
    input_projection_dim: int,
    adapter_kind: str,
    delay_tokens: int,
    delay_layers: int,
    delay_heads: int,
    checkpoint: dict,
    device: torch.device,
):
    if adapter_kind == "flat":
        adapter = SemanticVectorContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=context_length,
            hidden_dim=hidden_dim,
            dropout=0.0,
            input_projection_dim=input_projection_dim or None,
        ).to(device)
    else:
        if input_projection_dim:
            raise ValueError("Delay-token checkpoints cannot use semantic_input_projection_dim.")
        adapter = DelayTokenContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=context_length,
            hidden_dim=hidden_dim,
            dropout=0.0,
            num_delay_tokens=delay_tokens,
            num_layers=delay_layers,
            num_heads=delay_heads,
        ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    return adapter


def text_metrics(
    *,
    name: str,
    generated: Sequence[str],
    targets: Sequence[str],
    tokenizer,
    encoder,
    target_latents: torch.Tensor,
    target_mask: torch.Tensor,
    config: Config,
    device: torch.device,
) -> dict:
    normalized_targets = [target.strip() for target in targets]
    exact = [int(gen.strip() == target) for gen, target in zip(generated, normalized_targets)]
    overlap = word_overlap_metrics(generated, normalized_targets)
    quality = generation_repetition_metrics(generated)
    quality.update(overlap["summary"])
    safe_generated = [text if text.strip() else tokenizer.eos_token or "." for text in generated]
    generated_ids, generated_mask = tokenize_sentences(tokenizer, safe_generated)
    generated_latents = encode_text(
        input_ids=generated_ids.to(device),
        attention_mask=generated_mask.to(device),
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        use_bf16=False,
    )
    generated_pooled = mean_pool_latents(generated_latents, generated_mask.to(device))
    target_pooled = mean_pool_latents(target_latents.to(device), target_mask.to(device))
    similarity = generated_pooled @ target_pooled.T
    return {
        "name": name,
        "exact_match": float(sum(exact) / max(1, len(exact))),
        "generation_quality": quality,
        "word_overlap": overlap,
        "generation_t5_retrieval": rank_true_targets_by_similarity(similarity),
        "targets": list(normalized_targets),
        "generated": list(generated),
        "exact": exact,
    }


@torch.no_grad()
def direct_decode_target_latents(
    *,
    model,
    tokenizer,
    target_latents: torch.Tensor,
    config: Config,
    t_value: float,
    self_cond_cfg_scale: float,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    generated: list[str] = []
    for start in range(0, target_latents.shape[0], batch_size):
        end = min(start + batch_size, target_latents.shape[0])
        latents = target_latents[start:end].to(device)
        predicted_ids = _dlm_decode_batch(
            z=latents,
            model=model,
            t_final_val=t_value,
            config=config,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
        predicted_ids = mask_after_eos(
            predicted_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        generated.extend(
            tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
            for row in predicted_ids
        )
        logger.info("oracle direct decoded %d/%d", end, target_latents.shape[0])
    return generated


@torch.no_grad()
def sample_from_semantic_vectors(
    *,
    model,
    adapter,
    tokenizer,
    semantic_vectors: torch.Tensor,
    target_latents: torch.Tensor,
    target_mask: torch.Tensor,
    context_length: int,
    config: Config,
    sampling_config: SamplingConfig,
    generator: torch.Generator,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, dict]:
    generated: list[str] = []
    sampled_target_latents: list[torch.Tensor] = []
    for start in range(0, semantic_vectors.shape[0], batch_size):
        end = min(start + batch_size, semantic_vectors.shape[0])
        vectors = semantic_vectors[start:end].to(device=device, dtype=torch.float32)
        context, context_mask = adapter(vectors)
        context_mask = context_mask.to(device=device, dtype=context.dtype)
        zeros_target = torch.zeros(
            (context.shape[0], target_latents.shape[1], context.shape[-1]),
            dtype=context.dtype,
            device=device,
        )
        cond_seq = torch.cat([context, zeros_target], dim=1)
        cond_mask = torch.cat(
            [
                context_mask,
                torch.zeros((context.shape[0], target_latents.shape[1]), dtype=context.dtype, device=device),
            ],
            dim=1,
        )
        t_steps = get_sampling_steps(
            n_steps=sampling_config.num_sampling_steps[0],
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean,
            P_std=config.denoiser_p_std,
            device=device,
            dtype=context.dtype,
        )
        z = torch.randn(cond_seq.shape, generator=generator, device=device, dtype=context.dtype)
        z = z * config.denoiser_noise_scale
        latent = _generate_samples_single_batch(
            model=model,
            generator=generator,
            z=z,
            t_steps=t_steps,
            cond_seq=cond_seq,
            cond_seq_mask=cond_mask,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=sampling_config.cfgs[0],
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
        )
        sampled_target_latents.append(latent[:, context_length:].detach().cpu().float())
        predicted_ids = _dlm_decode_batch(
            z=latent,
            model=model,
            t_final_val=t_steps[-1].item(),
            config=config,
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
        )
        shift = torch.full((context.shape[0],), context_length, dtype=torch.long, device=device)
        predicted_ids = shift_left(predicted_ids, shift, 0)[:, : target_latents.shape[1]]
        predicted_ids = mask_after_eos(
            predicted_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        generated.extend(
            tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
            for row in predicted_ids
        )
        logger.info("semantic sampled %d/%d", end, semantic_vectors.shape[0])

    sampled = torch.cat(sampled_target_latents, dim=0)
    sampled_pooled = mean_pool_latents(sampled.to(device), target_mask.to(device))
    target_pooled = mean_pool_latents(target_latents.to(device), target_mask.to(device))
    similarity = sampled_pooled @ target_pooled.T
    latent_retrieval = rank_true_targets_by_similarity(similarity)
    diag_cos = similarity.diag()
    latent_retrieval["diag_cosine_mean"] = float(diag_cos.mean().cpu())
    latent_retrieval["diag_cosine_median"] = float(diag_cos.median().cpu())
    latent_retrieval["sampled_target_mse"] = float(
        ((sampled.to(device) - target_latents.to(device)) ** 2 * target_mask.to(device).unsqueeze(-1))
        .sum()
        .div(target_mask.to(device).sum().clamp_min(1.0) * target_latents.shape[-1])
        .cpu()
    )
    return generated, sampled, latent_retrieval


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint)
    saved_args = SimpleNamespace(**(checkpoint.get("args") or {}))
    context_length = args.context_length or int(getattr(saved_args, "context_length", 64))
    semantic_hidden_dim = args.semantic_hidden_dim or int(getattr(saved_args, "semantic_hidden_dim", 4096))
    semantic_adapter_kind = args.semantic_adapter_kind or str(
        getattr(saved_args, "semantic_adapter_kind", "flat")
    )
    semantic_delay_tokens = args.semantic_delay_tokens or int(
        getattr(saved_args, "semantic_delay_tokens", 4)
    )
    semantic_delay_layers = args.semantic_delay_layers or int(
        getattr(saved_args, "semantic_delay_layers", 2)
    )
    semantic_delay_heads = args.semantic_delay_heads or int(
        getattr(saved_args, "semantic_delay_heads", 8)
    )
    semantic_input_projection_dim = (
        args.semantic_input_projection_dim
        if args.semantic_input_projection_dim >= 0
        else int(getattr(saved_args, "semantic_input_projection_dim", 0))
    )

    data = np.load(args.npz_path, allow_pickle=True)
    semantic_vectors_np = data[args.input_key].astype(np.float32)
    sentences_all = strings(data[args.sentence_key])
    total_n = semantic_vectors_np.shape[0]
    n = total_n if args.num_examples <= 0 else min(args.num_examples, total_n)
    semantic_vectors = torch.as_tensor(semantic_vectors_np[:n], dtype=torch.float32)
    sentences = sentences_all[:n]

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask = tokenize_sentences(tokenizer, sentences_all[:total_n])
    target_length = input_ids.shape[1]

    target_latents_all = np.load(args.target_latents_cache, mmap_mode="r")
    if target_latents_all.shape[0] < total_n:
        raise ValueError(
            f"Target latent cache has {target_latents_all.shape[0]} rows, but NPZ has {total_n} rows."
        )
    target_latents = torch.as_tensor(np.asarray(target_latents_all[:n]), dtype=torch.float32)
    target_mask = attention_mask[:n].to(torch.float32)
    targets = sentences_all[:n]

    encoder_config, encoder = get_encoder(args.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    config = config_from_checkpoint(
        checkpoint,
        model_name=args.model,
        encoder_model_name=args.encoder_model_name,
        max_length=context_length + target_length,
    )
    model = build_model(
        config=config,
        encoder_dim=encoder_config.d_model,
        vocab_size=len(tokenizer),
        checkpoint=checkpoint,
        device=device,
    )
    adapter = build_adapter(
        input_dim=int(semantic_vectors.shape[-1]),
        context_dim=encoder_config.d_model,
        context_length=context_length,
        hidden_dim=semantic_hidden_dim,
        input_projection_dim=semantic_input_projection_dim,
        adapter_kind=semantic_adapter_kind,
        delay_tokens=semantic_delay_tokens,
        delay_layers=semantic_delay_layers,
        delay_heads=semantic_delay_heads,
        checkpoint=checkpoint,
        device=device,
    )
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(args.seed + 17)

    semantic_generated, _sampled_latents, latent_retrieval = sample_from_semantic_vectors(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        semantic_vectors=semantic_vectors,
        target_latents=target_latents,
        target_mask=target_mask,
        context_length=context_length,
        config=config,
        sampling_config=sampling_config,
        generator=generator,
        batch_size=args.batch_size,
        device=device,
    )
    semantic_metrics = text_metrics(
        name="semantic_sample",
        generated=semantic_generated,
        targets=targets,
        tokenizer=tokenizer,
        encoder=encoder,
        target_latents=target_latents,
        target_mask=target_mask,
        config=config,
        device=device,
    )
    semantic_metrics["sampled_latent_retrieval"] = latent_retrieval

    oracle_generated = direct_decode_target_latents(
        model=model,
        tokenizer=tokenizer,
        target_latents=target_latents,
        config=config,
        t_value=args.direct_decode_t,
        self_cond_cfg_scale=args.self_cond_cfg_scale,
        batch_size=args.batch_size,
        device=device,
    )
    oracle_metrics = text_metrics(
        name="oracle_target_direct_decode",
        generated=oracle_generated,
        targets=targets,
        tokenizer=tokenizer,
        encoder=encoder,
        target_latents=target_latents,
        target_mask=target_mask,
        config=config,
        device=device,
    )

    output = {
        "npz_path": args.npz_path,
        "checkpoint": args.checkpoint,
        "target_latents_cache": args.target_latents_cache,
        "n": n,
        "total_n": total_n,
        "context_length": context_length,
        "target_length": target_length,
        "num_sampling_steps": args.num_sampling_steps,
        "cfg_scale": args.cfg_scale,
        "self_cond_cfg_scale": args.self_cond_cfg_scale,
        "results": [semantic_metrics, oracle_metrics],
    }
    out_path = output_dir / "npz_semantic_checkpoint_diagnostics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    for metrics in output["results"]:
        quality = metrics["generation_quality"]
        retrieval = metrics["generation_t5_retrieval"]
        logger.info(
            "%s exact=%.4f words=%.4f content=%.4f t5_top1=%.4f t5_top5=%.4f mean_rank=%.2f",
            metrics["name"],
            metrics["exact_match"],
            quality.get("words_overlap", float("nan")),
            quality.get("content_words_overlap", float("nan")),
            retrieval["top1"],
            retrieval["top5"],
            retrieval["mean_rank"],
        )
        if "sampled_latent_retrieval" in metrics:
            latent = metrics["sampled_latent_retrieval"]
            logger.info(
                "%s sampled_latent_top1=%.4f top5=%.4f mean_rank=%.2f diag_cos=%.4f mse=%.6f",
                metrics["name"],
                latent["top1"],
                latent["top5"],
                latent["mean_rank"],
                latent["diag_cosine_mean"],
                latent["sampled_target_mse"],
            )
        for idx in range(min(5, len(metrics["generated"]))):
            logger.info(
                "[%s:%d] target=%r generated=%r",
                metrics["name"],
                idx,
                metrics["targets"][idx],
                metrics["generated"][idx],
            )
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
