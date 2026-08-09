#!/usr/bin/env python
"""Decode exported ELF context latents from an NPZ with a frozen ELF checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import _download_hf_checkpoint, _restore_checkpoint
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    _dlm_decode_batch,
    _generate_samples_single_batch,
    mask_after_eos,
    shift_left,
)
from utils.sampling_utils import get_sampling_steps

from scripts.meg_context_overfit import (
    evaluate_retrieval,
    generation_repetition_metrics,
    mean_pool_latents,
    rank_true_targets_by_similarity,
    tokenize_sentences,
    word_overlap_metrics,
)


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

# Some MEG2SEM exports are pickled with NumPy 2.x object-array module names.
# ARC jobs may run NumPy 1.x, where these aliases do not exist.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-key", default="context")
    parser.add_argument("--context-mask-key", default="context_mask")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--target-latents-key", default="target_t5_latents")
    parser.add_argument("--target-mask-key", default="t5_attention_mask")
    parser.add_argument("--target-ids-key", default="t5_input_ids")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--num-examples", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--num-sampling-steps", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--self-cond-cfg-scale", type=float, default=0.0)
    parser.add_argument("--direct-decode", action="store_true", help="Decode latent sequences directly with ELF's decoder head, without sampling.")
    parser.add_argument("--direct-decode-t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-oracle-target", action="store_true")
    parser.add_argument("--retrieval-num-examples", type=int, default=0, help="0 disables conditional retrieval.")
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--retrieval-t", type=float, default=0.5)
    parser.add_argument("--denoiser-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace, max_length: int) -> Config:
    config = Config()
    config.encoder_model_name = args.encoder_model_name
    config.model = args.model
    config.max_length = max_length
    config.bottleneck_dim = 128
    config.num_time_tokens = 4
    config.num_self_cond_cfg_tokens = 4
    config.num_model_mode_tokens = 4
    config.latent_mean = 0.0
    config.latent_std = 0.2
    config.denoiser_p_mean = -1.5
    config.denoiser_p_std = 0.8
    config.denoiser_noise_scale = 2.0
    config.time_schedule = "logit_normal"
    config.denoiser_loss_weight = args.denoiser_loss_weight
    config.decoder_loss_weight = args.decoder_loss_weight
    return config


def load_model(args: argparse.Namespace, config: Config, encoder_dim: int, vocab_size: int, device: torch.device):
    model = ELF_models[config.model](
        text_encoder_dim=encoder_dim,
        max_length=config.max_length,
        bottleneck_dim=config.bottleneck_dim,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        vocab_size=vocab_size,
    ).to(device)
    ckpt_root = _download_hf_checkpoint(args.checkpoint_path) or args.checkpoint_path
    ckpt = _restore_checkpoint(ckpt_root)
    if ckpt is None or "params" not in ckpt:
        raise ValueError(f"Could not restore checkpoint from {args.checkpoint_path}")
    model.load_state_dict(ckpt["params"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def decode_context_batch(
    *,
    model,
    tokenizer,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    target_length: int,
    config: Config,
    sampling_config: SamplingConfig,
    generator: torch.Generator,
) -> list[str]:
    device = context.device
    zeros_target = torch.zeros(
        (context.shape[0], target_length, context.shape[-1]),
        dtype=context.dtype,
        device=device,
    )
    cond_seq = torch.cat([context, zeros_target], dim=1)
    cond_mask = torch.cat(
        [
            context_mask,
            torch.zeros((context.shape[0], target_length), dtype=context.dtype, device=device),
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
    predicted_ids = _dlm_decode_batch(
        z=latent,
        model=model,
        t_final_val=t_steps[-1].item(),
        config=config,
        self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
    )
    shift = torch.full((context.shape[0],), context.shape[1], dtype=torch.long, device=device)
    predicted_ids = shift_left(predicted_ids, shift, 0)[:, :target_length]
    predicted_ids = mask_after_eos(
        predicted_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    return [
        tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
        for row in predicted_ids
    ]


def _as_string_list(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


class StoredContextAdapter(torch.nn.Module):
    """Expose precomputed contexts through the semantic adapter interface."""

    def __init__(self, contexts: torch.Tensor, masks: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("contexts", contexts)
        self.register_buffer("masks", masks)

    def forward(self, semantic_vectors: torch.Tensor):
        indices = semantic_vectors.reshape(-1).to(dtype=torch.long, device=self.contexts.device)
        contexts = self.contexts.index_select(0, indices)
        masks = self.masks.index_select(0, indices).to(dtype=contexts.dtype)
        return contexts, masks


@torch.no_grad()
def direct_decode_batch(
    *,
    model,
    tokenizer,
    latents: torch.Tensor,
    config: Config,
    t_value: float,
    self_cond_cfg_scale: float,
) -> list[str]:
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
    return [
        tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
        for row in predicted_ids
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    logger.info("Loaded keys: %s", sorted(data.files))
    context_all_np = data[args.context_key]
    context_mask_all_np = data[args.context_mask_key]
    sentences_all = _as_string_list(data[args.sentence_key])

    total_n = context_all_np.shape[0]
    n = total_n if args.num_examples <= 0 else min(args.num_examples, total_n)
    context_np = context_all_np[:n]
    context_mask_np = context_mask_all_np[:n]
    sentences = sentences_all[:n]

    target_latents_all_np = data[args.target_latents_key] if args.target_latents_key in data.files else None
    target_mask_all_np = data[args.target_mask_key] if args.target_mask_key in data.files else None
    target_ids_all_np = data[args.target_ids_key] if args.target_ids_key in data.files else None
    if target_latents_all_np is not None:
        target_latents_all_np = target_latents_all_np.astype(np.float32)
    if target_mask_all_np is not None:
        target_mask_all_np = target_mask_all_np.astype(np.float32)
    target_latents_np = target_latents_all_np[:n] if target_latents_all_np is not None else None
    target_mask_np = target_mask_all_np[:n] if target_mask_all_np is not None else None

    context_length = int(context_np.shape[1])
    target_length = int(args.target_length)
    max_length = context_length + target_length
    config = build_config(args, max_length=max_length)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder_config, encoder = get_encoder(args.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    model = load_model(args, config, encoder_config.d_model, len(tokenizer), device)
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def build_metrics(name: str, generated: list[str]) -> dict:
        exact = [int(g.strip() == t.strip()) for g, t in zip(generated, sentences)]
        overlap = word_overlap_metrics(generated, sentences)
        generation_quality = generation_repetition_metrics(generated)
        generation_quality.update(overlap["summary"])
        metrics = {
            "name": name,
            "npz_path": args.npz_path,
            "context_key": args.context_key,
            "n": n,
            "total_n": total_n,
            "context_length": context_length,
            "target_length": target_length,
            "num_sampling_steps": args.num_sampling_steps,
            "cfg_scale": args.cfg_scale,
            "direct_decode": args.direct_decode,
            "retrieval_num_examples": args.retrieval_num_examples,
            "exact_match": float(sum(exact) / max(1, len(exact))),
            "generation_quality": generation_quality,
            "word_overlap": overlap,
            "targets": sentences,
            "generated": generated,
            "exact": exact,
        }
        if target_latents_np is not None and target_mask_np is not None:
            safe_generated = [text if text.strip() else tokenizer.eos_token or "." for text in generated]
            generated_ids, generated_mask = tokenize_sentences(tokenizer, safe_generated)
            generated_ids = generated_ids.to(device)
            generated_mask = generated_mask.to(device)
            generated_latents = encode_text(
                input_ids=generated_ids,
                attention_mask=generated_mask,
                encoder=encoder,
                latent_mean=config.latent_mean,
                latent_std=config.latent_std,
                use_bf16=False,
            )
            generated_pooled = mean_pool_latents(generated_latents, generated_mask)
            target_latents = torch.as_tensor(target_latents_np, dtype=torch.float32, device=device)
            target_mask = torch.as_tensor(target_mask_np, dtype=torch.float32, device=device)
            target_pooled = mean_pool_latents(target_latents, target_mask)
            similarity = generated_pooled @ target_pooled.T
            metrics["generation_t5_retrieval"] = rank_true_targets_by_similarity(similarity)
        return metrics

    def retrieval_for_context(name: str, ctx_np: np.ndarray, mask_np: np.ndarray) -> dict | None:
        if args.retrieval_num_examples <= 0:
            return None
        if target_latents_all_np is None or target_mask_all_np is None or target_ids_all_np is None:
            logger.warning("%s retrieval skipped; target latents, mask, or ids are missing.", name)
            return None
        retrieval_n = min(args.retrieval_num_examples, total_n)
        contexts = torch.as_tensor(ctx_np[:retrieval_n], dtype=torch.float32, device=device)
        masks = torch.as_tensor(mask_np[:retrieval_n], dtype=torch.float32, device=device)
        adapter = StoredContextAdapter(contexts, masks).to(device).eval()
        metrics = evaluate_retrieval(
            model=model,
            adapter=adapter,
            meg=torch.zeros((retrieval_n, 1, 1), dtype=torch.float32),
            meg_lengths=torch.ones((retrieval_n,), dtype=torch.long),
            semantic_vectors=torch.arange(retrieval_n, dtype=torch.float32).reshape(-1, 1),
            subject_ids=torch.zeros((retrieval_n,), dtype=torch.long),
            target_latents=torch.as_tensor(target_latents_all_np[:retrieval_n], dtype=torch.float32),
            target_ids=torch.as_tensor(target_ids_all_np[:retrieval_n], dtype=torch.long),
            target_mask=torch.as_tensor(target_mask_all_np[:retrieval_n], dtype=torch.float32),
            target_sentences=sentences_all[:retrieval_n],
            config=config,
            device=device,
            condition_source="semantic",
            retrieval_batch_size=args.retrieval_batch_size,
            retrieval_t=args.retrieval_t,
        )
        logger.info(
            "%s retrieval combined_top1=%.3f combined_top5=%.3f mean_rank=%.2f",
            name,
            metrics["combined"]["top1"],
            metrics["combined"]["top5"],
            metrics["combined"]["mean_rank"],
        )
        return metrics

    def run_sample_decode(name: str, ctx_np: np.ndarray, mask_np: np.ndarray) -> dict:
        generated: list[str] = []
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            context = torch.as_tensor(ctx_np[start:end], dtype=torch.float32, device=device)
            context_mask = torch.as_tensor(mask_np[start:end], dtype=torch.float32, device=device)
            generated.extend(
                decode_context_batch(
                    model=model,
                    tokenizer=tokenizer,
                    context=context,
                    context_mask=context_mask,
                    target_length=target_length,
                    config=config,
                    sampling_config=sampling_config,
                    generator=generator,
                )
            )
            logger.info("%s decoded %d/%d", name, end, n)

        metrics = build_metrics(name, generated)
        retrieval_metrics = retrieval_for_context(name, ctx_np, mask_np)
        if retrieval_metrics is not None:
            metrics["retrieval"] = retrieval_metrics
        return metrics

    def run_direct_decode(name: str, latents_np: np.ndarray) -> dict:
        generated: list[str] = []
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            latents = torch.as_tensor(latents_np[start:end], dtype=torch.float32, device=device)
            generated.extend(
                direct_decode_batch(
                    model=model,
                    tokenizer=tokenizer,
                    latents=latents,
                    config=config,
                    t_value=args.direct_decode_t,
                    self_cond_cfg_scale=args.self_cond_cfg_scale,
                )
            )
            logger.info("%s direct-decoded %d/%d", name, end, n)
        return build_metrics(name, generated)

    if args.direct_decode:
        results = [run_direct_decode(f"direct_{args.context_key}", context_np.astype(np.float32))]
        if args.decode_oracle_target and target_latents_np is not None:
            results.append(run_direct_decode(f"direct_{args.target_latents_key}", target_latents_np))
    else:
        results = [run_sample_decode(args.context_key, context_all_np.astype(np.float32), context_mask_all_np.astype(np.float32))]
        if args.decode_oracle_target and target_latents_all_np is not None and target_mask_all_np is not None:
            results.append(run_sample_decode(args.target_latents_key, target_latents_all_np, target_mask_all_np))

    output = {"results": results}
    out_path = out_dir / "decode_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    for result in results:
        logger.info(
            "%s exact=%.4f well_structured=%.4f",
            result["name"],
            result["exact_match"],
            result["generation_quality"]["well_structured_sentence"],
        )
        for idx in range(min(8, len(result["generated"]))):
            logger.info("[%s:%d] target=%r generated=%r", result["name"], idx, result["targets"][idx], result["generated"][idx])
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
