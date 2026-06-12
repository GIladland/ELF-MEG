#!/usr/bin/env python
"""Overfit ELF on a tiny learned-context conditional generation task.

This is the smallest experiment for validating the conditioning pathway before
plugging in MEG. Each training example gets its own learned context prefix; the
model must generate the matching sentence from that prefix.
"""

from __future__ import annotations

import argparse
import difflib
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
import torch.nn as nn
import torch.nn.functional as F
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
from utils.logging_utils import log_for_0
from utils.sampling_utils import add_noise, get_sampling_steps, sample_timesteps

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


DEFAULT_SENTENCES = [
    "The red fox jumped over the sleeping dog.",
    "A glass of water sat beside the notebook.",
    "The train arrived just before the rain started.",
    "She folded the letter and locked it in a drawer.",
    "Bright lamps reflected across the empty station floor.",
    "The musician tuned the violin in complete silence.",
    "A small boat drifted near the edge of the harbor.",
    "Fresh coffee filled the kitchen before sunrise.",
    "The mountain path curved behind a wall of pine trees.",
    "He placed the old photograph back in the box.",
]


DEFAULT_RANDOM_INIT_TEXTS = [
    (
        "Clouds gathered above the quiet museum while visitors studied maps, "
        "marble statues, old letters, and a row of brass clocks near the western gallery."
    ),
    (
        "The carpenter measured the wooden frame twice before choosing a narrow chisel, "
        "a clean pencil line, and a stack of boards from the back of the workshop."
    ),
    (
        "Several candles flickered beside the stone window as the librarian sorted blue "
        "folders, repaired torn labels, and counted the keys in a small ceramic bowl."
    ),
    (
        "A patient gardener carried soil across the courtyard, watered the herbs, trimmed "
        "the vines, and wrote careful notes about weather changes in a green notebook."
    ),
    (
        "During the afternoon meeting, the engineer reviewed circuit diagrams, battery "
        "tests, safety forms, and a schedule for shipping replacement sensors overseas."
    ),
    (
        "The baker mixed cinnamon, lemon peel, and warm milk before arranging trays of "
        "dough near the oven and sweeping flour from the long steel counter."
    ),
    (
        "A geography teacher unfolded a large paper map, pointed toward distant islands, "
        "and described rivers, deserts, railways, and ports along the northern coast."
    ),
    (
        "The theater crew adjusted heavy curtains, checked the balcony lights, painted "
        "two wooden chairs, and tested microphones before the evening rehearsal began."
    ),
    (
        "At the repair shop, a mechanic cleaned the bicycle chain, replaced a cracked "
        "pedal, tightened several bolts, and stored the tools in a red cabinet."
    ),
    (
        "The astronomer compared telescope images, marked faint stars with white ink, "
        "and prepared a short report about the clear sky above the desert observatory."
    ),
]


TARGET_SUBJECTS = [
    "red fox", "glass cup", "silver train", "quiet musician", "small boat",
    "green notebook", "old photograph", "bright lamp", "mountain path", "wooden drawer",
    "yellow bicycle", "stone bridge", "paper map", "blue curtain", "garden gate",
    "library clock", "copper kettle", "winter coat", "market stall", "harbor bell",
]

TARGET_VERBS = [
    "rested beside", "moved past", "waited near", "turned toward", "leaned against",
    "crossed over", "stood behind", "glowed above", "slipped under", "circled around",
    "balanced on", "drifted beyond", "paused before", "rolled across", "settled inside",
]

TARGET_OBJECTS = [
    "the sleeping dog", "the empty station", "a narrow window", "the kitchen table",
    "a quiet harbor", "the folded letter", "a row of pine trees", "the marble floor",
    "a dusty cabinet", "the river bank", "a broken fence", "the western gallery",
    "a stack of books", "the garden wall", "a brass lantern",
]

TARGET_ENDINGS = [
    "before sunrise", "during the rain", "after the meeting", "in complete silence",
    "near the old museum", "beside the notebook", "under a pale sky", "while bells rang",
    "at the edge of town", "behind the blue house",
]

SOURCE_SUBJECTS = [
    "careful baker", "patient gardener", "young astronomer", "local mechanic",
    "history teacher", "theater crew", "quiet librarian", "software engineer",
    "museum visitor", "carpenter", "field researcher", "radio operator",
    "ceramic artist", "weather reporter", "music student", "village doctor",
]

SOURCE_ACTIONS = [
    "checked", "sorted", "measured", "carried", "cleaned", "reviewed", "painted",
    "counted", "folded", "recorded", "arranged", "repaired", "compared", "packed",
]

SOURCE_OBJECTS = [
    "several brass clocks", "a stack of blue folders", "three wooden chairs",
    "a box of old maps", "the bicycle tools", "warm trays of bread",
    "the balcony lights", "a ceramic bowl", "weather notes", "telescope images",
    "fresh garden soil", "replacement sensors", "paper labels", "a narrow chisel",
]


@dataclass
class ToyBatch:
    indices: torch.Tensor
    target_latents: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor


class LearnedContextTable(nn.Module):
    def __init__(
        self,
        num_examples: int,
        context_length: int,
        context_dim: int,
        init_context: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if init_context is None:
            context = torch.randn(num_examples, context_length, context_dim) * 0.02
        else:
            expected_shape = (num_examples, context_length, context_dim)
            if tuple(init_context.shape) != expected_shape:
                raise ValueError(
                    f"init_context shape {tuple(init_context.shape)} does not match {expected_shape}"
                )
            context = init_context.detach().clone()
        self.context = nn.Parameter(context)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.context.index_select(0, indices)


def build_target_latent_context(
    target_latents: torch.Tensor,
    context_length: int,
) -> torch.Tensor:
    if context_length <= target_latents.shape[1]:
        return target_latents[:, :context_length]

    pad = torch.zeros(
        target_latents.shape[0],
        context_length - target_latents.shape[1],
        target_latents.shape[2],
        dtype=target_latents.dtype,
        device=target_latents.device,
    )
    return torch.cat([target_latents, pad], dim=1)


def _make_sentences_from_templates(
    *,
    count: int,
    seed: int,
    subjects: Sequence[str],
    verbs: Sequence[str],
    objects: Sequence[str],
    endings: Sequence[str],
) -> list[str]:
    all_sentences = []
    for subject in subjects:
        for verb in verbs:
            for obj in objects:
                for ending in endings:
                    all_sentences.append(f"The {subject} {verb} {obj} {ending}.")
    rng = random.Random(seed)
    rng.shuffle(all_sentences)
    if count > len(all_sentences):
        raise ValueError(f"Requested {count} sentences but only {len(all_sentences)} templates are available.")
    return all_sentences[:count]


def build_sentence_dataset(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.dataset_mode == "default10":
        if args.dataset_size != len(DEFAULT_SENTENCES):
            raise ValueError("--dataset_size must be 10 when --dataset_mode default10.")
        return list(DEFAULT_SENTENCES), list(DEFAULT_RANDOM_INIT_TEXTS)

    targets = _make_sentences_from_templates(
        count=args.dataset_size,
        seed=args.seed + 1000,
        subjects=TARGET_SUBJECTS,
        verbs=TARGET_VERBS,
        objects=TARGET_OBJECTS,
        endings=TARGET_ENDINGS,
    )
    sources = _make_sentences_from_templates(
        count=args.dataset_size,
        seed=args.seed + 2000,
        subjects=SOURCE_SUBJECTS,
        verbs=SOURCE_ACTIONS,
        objects=SOURCE_OBJECTS,
        endings=[
            "before lunch", "near the workshop", "during the afternoon",
            "beside the courtyard", "after the rehearsal", "under the clear sky",
            "while the room was quiet", "next to the storage cabinet",
        ],
    )

    perm = list(range(args.dataset_size))
    random.Random(args.seed + 3000).shuffle(perm)
    return [targets[i] for i in perm], sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--output_dir", default="outputs/toy_context_overfit")
    parser.add_argument(
        "--context_mode",
        choices=[
            "learned",
            "learned_from_target",
            "learned_from_shuffled_target",
            "learned_from_random_text",
            "target_latents",
        ],
        default="learned",
    )
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument(
        "--dataset_mode",
        choices=["default10", "random_pairs"],
        default="default10",
        help="default10 keeps the original hand-written toy set; random_pairs uses unrelated source/target text pairs.",
    )
    parser.add_argument("--dataset_size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--num_sampling_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument("--last_n_blocks", type=int, default=-1)
    parser.add_argument(
        "--freeze_elf",
        action="store_true",
        help="Freeze all ELF weights and train only learned context parameters.",
    )
    parser.add_argument("--denoiser_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument(
        "--cond_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of zeroing the condition prefix for a training example.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="BrainDiffusion")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="toy-context-overfit")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave unset for auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_config(args: argparse.Namespace, max_length: int) -> Config:
    config = Config()
    config.encoder_model_name = args.encoder_model_name
    config.model = args.model
    config.max_length = max_length
    config.latent_mean = 0.0
    config.latent_std = 0.2
    config.num_time_tokens = 4
    config.num_self_cond_cfg_tokens = 4
    config.num_model_mode_tokens = 4
    config.denoiser_p_mean = -1.5
    config.denoiser_p_std = 0.8
    config.denoiser_noise_scale = 2.0
    config.decoder_p_mean = 0.8
    config.decoder_p_std = 0.8
    config.decoder_noise_scale = args.decoder_noise_scale
    config.t_eps = 0.05
    config.time_schedule = "logit_normal"
    config.self_cond_prob = 0.5
    config.use_bf16 = True
    config.output_dir = args.output_dir
    config.denoiser_loss_weight = args.denoiser_loss_weight
    config.decoder_loss_weight = args.decoder_loss_weight
    config.label_drop_prob = args.cond_dropout_prob
    return config


def tokenize_sentences(tokenizer, sentences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids_list = [tokenizer(sentence, add_special_tokens=False)["input_ids"] for sentence in sentences]
    max_len = max(len(ids) for ids in ids_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or eos token.")

    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    for row, ids in enumerate(ids_list):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


def load_pretrained_model(
    args: argparse.Namespace,
    config: Config,
    encoder_dim: int,
    vocab_size: int,
    device: torch.device,
) -> nn.Module:
    model = ELF_models[config.model](
        text_encoder_dim=encoder_dim,
        max_length=config.max_length,
        attn_drop=0.0,
        proj_drop=0.0,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        vocab_size=vocab_size,
        bottleneck_dim=128,
        gradient_checkpointing=False,
    ).to(device)

    ckpt_root = _download_hf_checkpoint(args.checkpoint_path) or args.checkpoint_path
    ckpt = _restore_checkpoint(ckpt_root)
    if ckpt is None or "params" not in ckpt:
        raise ValueError(f"Could not restore checkpoint from {args.checkpoint_path}")
    model.load_state_dict(ckpt["params"])
    return model


def freeze_for_toy_tuning(model: nn.Module, last_n_blocks: int) -> list[str]:
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_names = []

    def _mark(module: nn.Module, prefix: str) -> None:
        for name, param in module.named_parameters():
            param.requires_grad_(True)
            trainable_names.append(f"{prefix}.{name}" if prefix else name)

    if last_n_blocks < 0:
        _mark(model, "")
        return trainable_names

    _mark(model.text_proj, "text_proj")
    _mark(model.final_layer, "final_layer")

    last_n_blocks = max(0, min(last_n_blocks, len(model.blocks)))
    for idx in range(len(model.blocks) - last_n_blocks, len(model.blocks)):
        if idx < 0:
            continue
        _mark(model.blocks[idx], f"blocks.{idx}")

    return trainable_names


def build_batches(
    target_latents: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> list[ToyBatch]:
    num_examples = target_latents.shape[0]
    perm = torch.randperm(num_examples, generator=generator)
    batches = []
    for start in range(0, num_examples, batch_size):
        indices = perm[start:start + batch_size]
        batches.append(
            ToyBatch(
                indices=indices,
                target_latents=target_latents.index_select(0, indices),
                target_ids=target_ids.index_select(0, indices),
                target_mask=target_mask.index_select(0, indices),
            )
        )
    return batches


def train_step_toy(
    *,
    model: nn.Module,
    context_table: LearnedContextTable | None,
    batch: ToyBatch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    noise_generator: torch.Generator,
    context_mode: str,
    context_length: int,
    cond_dropout_prob: float,
) -> dict[str, float]:
    model.train()
    if context_table is not None:
        context_table.train()

    indices = batch.indices.to(device)
    target_latents = batch.target_latents.to(device)
    target_ids = batch.target_ids.to(device)
    target_token_mask = batch.target_mask.to(device, dtype=torch.float32)
    if context_mode in {
        "learned",
        "learned_from_target",
        "learned_from_shuffled_target",
        "learned_from_random_text",
    }:
        if context_table is None:
            raise ValueError("context_table is required for learned context mode.")
        context = context_table(indices)
    elif context_mode == "target_latents":
        context = build_target_latent_context(target_latents, context_length)
    else:
        raise ValueError(f"Unsupported context_mode: {context_mode}")
    if cond_dropout_prob > 0.0:
        drop_shape = (context.shape[0], 1, 1)
        drop = torch.rand(drop_shape, generator=noise_generator, device=device) < cond_dropout_prob
        context = torch.where(drop, torch.zeros_like(context), context)

    x0 = torch.cat([context, target_latents], dim=1)
    cond_seq_mask = torch.zeros(x0.shape[:2], dtype=torch.float32, device=device)
    cond_seq_mask[:, : context.shape[1]] = 1.0
    attention_mask = torch.ones_like(cond_seq_mask)

    t = sample_timesteps(
        batch_size=x0.shape[0],
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device,
        dtype=x0.dtype,
    )
    noise = torch.randn(x0.shape, generator=noise_generator, device=device, dtype=x0.dtype)
    z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask.unsqueeze(-1))
    decoder_lambda = torch.sigmoid(
        torch.randn(x0.shape[:2], generator=noise_generator, device=device, dtype=x0.dtype)
        * config.decoder_p_std + config.decoder_p_mean
    ).unsqueeze(-1)
    decoder_noise = (
        torch.randn(x0.shape, generator=noise_generator, device=device, dtype=x0.dtype)
        * config.decoder_noise_scale
    )
    decoder_z = decoder_lambda * x0 + (1.0 - decoder_lambda) * decoder_noise
    sc_scale = torch.ones((x0.shape[0],), dtype=x0.dtype, device=device)

    optimizer.zero_grad(set_to_none=True)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        pred, _ = model(
            z,
            t,
            attention_mask=attention_mask,
            deterministic=False,
            self_cond_cfg_scale=sc_scale,
        )
        latent_target_mask = (1.0 - cond_seq_mask).unsqueeze(-1)
        denoiser_loss = (
            ((pred - x0) ** 2 * latent_target_mask).sum()
            / latent_target_mask.sum().clamp_min(1.0)
            / x0.shape[-1]
        )

        decoder_input = (
            torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
            if config.self_cond_prob > 0
            else decoder_z
        )
        _, decoder_logits = model(
            decoder_input,
            torch.ones((x0.shape[0],), dtype=x0.dtype, device=device),
            attention_mask=attention_mask,
            deterministic=False,
            self_cond_cfg_scale=sc_scale,
            decoder_step_active=True,
        )
        target_logits = decoder_logits[:, context.shape[1]:]
        ce_per_token = F.cross_entropy(
            target_logits.transpose(1, 2).to(torch.float32),
            target_ids,
            reduction="none",
        )
        decoder_loss = (ce_per_token * target_token_mask).sum() / target_token_mask.sum().clamp_min(1.0)
        loss = (
            config.denoiser_loss_weight * denoiser_loss
            + config.decoder_loss_weight * decoder_loss
        )

    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "denoiser_loss": float(denoiser_loss.detach().cpu()),
        "decoder_loss": float(decoder_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate_generation(
    *,
    model: nn.Module,
    context_table: LearnedContextTable | None,
    tokenizer,
    target_sentences: Sequence[str],
    target_latents: torch.Tensor,
    target_length: int,
    context_length: int,
    config: Config,
    sampling_config: SamplingConfig,
    device: torch.device,
    generator: torch.Generator,
    context_mode: str,
) -> dict:
    model.eval()
    if context_table is not None:
        context_table.eval()

    num_examples = len(target_sentences)
    target_latents = target_latents.to(device)
    if context_mode in {
        "learned",
        "learned_from_target",
        "learned_from_shuffled_target",
        "learned_from_random_text",
    }:
        if context_table is None:
            raise ValueError("context_table is required for learned context mode.")
        context = context_table.context.to(device)
    elif context_mode == "target_latents":
        context = build_target_latent_context(target_latents, context_length)
    else:
        raise ValueError(f"Unsupported context_mode: {context_mode}")
    zeros_target = torch.zeros((num_examples, target_length, context.shape[-1]), dtype=context.dtype, device=device)
    cond_seq = torch.cat([context, zeros_target], dim=1)
    cond_mask = torch.zeros((num_examples, cond_seq.shape[1]), dtype=context.dtype, device=device)
    cond_mask[:, :context_length] = 1.0

    t_steps = get_sampling_steps(
        n_steps=sampling_config.num_sampling_steps[0],
        time_schedule=sampling_config.time_schedule,
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        device=device,
        dtype=context.dtype,
    )
    z = torch.randn(cond_seq.shape, generator=generator, device=device, dtype=context.dtype) * config.denoiser_noise_scale

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
    shift = torch.full((num_examples,), context_length, dtype=torch.long, device=device)
    predicted_ids = shift_left(predicted_ids, shift, 0)[:, :target_length]
    predicted_ids = mask_after_eos(
        predicted_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    generated = [
        tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
        for row in predicted_ids
    ]
    normalized_targets = [sentence.strip() for sentence in target_sentences]
    exact = [int(gen == tgt) for gen, tgt in zip(generated, normalized_targets)]
    nearest_target_indices = []
    nearest_target_scores = []
    nearest_target_exact = []
    for row, gen in enumerate(generated):
        scores = [
            difflib.SequenceMatcher(None, gen.lower(), target.lower()).ratio()
            for target in normalized_targets
        ]
        nearest_idx = max(range(len(scores)), key=scores.__getitem__)
        nearest_target_indices.append(nearest_idx)
        nearest_target_scores.append(scores[nearest_idx])
        nearest_target_exact.append(int(nearest_idx == row))
    return {
        "generated": generated,
        "targets": normalized_targets,
        "exact": exact,
        "exact_match": float(sum(exact) / len(exact)),
        "nearest_target_indices": nearest_target_indices,
        "nearest_target_scores": nearest_target_scores,
        "nearest_target_accuracy": float(sum(nearest_target_exact) / len(nearest_target_exact)),
    }


def maybe_init_wandb(args: argparse.Namespace, config: Config):
    if not args.use_wandb or wandb is None:
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "checkpoint_path": args.checkpoint_path,
            "context_mode": args.context_mode,
            "context_length": args.context_length,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "last_n_blocks": args.last_n_blocks,
            "decoder_loss_weight": args.decoder_loss_weight,
            "denoiser_loss_weight": args.denoiser_loss_weight,
            "cond_dropout_prob": args.cond_dropout_prob,
            "dataset_mode": args.dataset_mode,
            "dataset_size": args.dataset_size,
            "max_length": config.max_length,
        },
    )


def save_results(output_dir: str, step: int, metrics: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"eval_step_{step:04d}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    log_for_0(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    target_sentences, init_texts = build_sentence_dataset(args)
    log_for_0(f"Dataset mode: {args.dataset_mode}; examples: {len(target_sentences)}")

    input_ids, attention_mask = tokenize_sentences(tokenizer, target_sentences)
    target_length = input_ids.shape[1]
    max_length = args.context_length + target_length
    config = build_config(args, max_length=max_length)

    encoder_config, encoder = get_encoder(args.encoder_model_name, dtype=torch.float32)
    encoder = encoder.to(device)
    encoder.eval()
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
    log_for_0(f"Trainable ELF parameter groups: {len(trainable_names)}")
    log_for_0(f"Trainable ELF parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    context_table = None
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_latents = encode_text(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        use_bf16=False,
    ).detach()
    target_ids = input_ids.detach().cpu()
    target_mask = attention_mask.detach().cpu().to(torch.float32)

    init_permutation = None
    if args.context_mode in {
        "learned",
        "learned_from_target",
        "learned_from_shuffled_target",
        "learned_from_random_text",
    }:
        init_context = None
        if args.context_mode in {"learned_from_target", "learned_from_shuffled_target"}:
            init_context = build_target_latent_context(target_latents, args.context_length).detach().cpu()
            if args.context_mode == "learned_from_shuffled_target":
                perm_generator = torch.Generator().manual_seed(args.seed + 101)
                init_permutation = torch.randperm(init_context.shape[0], generator=perm_generator)
                init_context = init_context.index_select(0, init_permutation)
                log_for_0(f"Shuffled init permutation: {init_permutation.tolist()}")
        elif args.context_mode == "learned_from_random_text":
            init_ids, init_attention_mask = tokenize_sentences(tokenizer, init_texts)
            init_latents = encode_text(
                input_ids=init_ids.to(device),
                attention_mask=init_attention_mask.to(device),
                encoder=encoder,
                latent_mean=config.latent_mean,
                latent_std=config.latent_std,
                use_bf16=False,
            ).detach()
            init_context = build_target_latent_context(init_latents, args.context_length).detach().cpu()
            log_for_0("Initialized learned context from unrelated random English text.")
        context_table = LearnedContextTable(
            num_examples=len(target_sentences),
            context_length=args.context_length,
            context_dim=encoder_config.d_model,
            init_context=init_context,
        ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if context_table is not None:
        params.extend(list(context_table.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )

    run = maybe_init_wandb(args, config)
    noise_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    noise_generator.manual_seed(args.seed + 17)
    batch_generator = torch.Generator().manual_seed(args.seed + 29)

    best_metrics = None
    for step in range(1, args.steps + 1):
        batches = build_batches(
            target_latents.cpu(),
            target_ids,
            target_mask,
            args.batch_size,
            batch_generator,
        )
        batch = batches[(step - 1) % len(batches)]
        batch = ToyBatch(
            indices=batch.indices,
            target_latents=batch.target_latents.to(device),
            target_ids=batch.target_ids,
            target_mask=batch.target_mask,
        )
        loss_metrics = train_step_toy(
            model=model,
            context_table=context_table,
            batch=batch,
            optimizer=optimizer,
            config=config,
            device=device,
            noise_generator=noise_generator,
            context_mode=args.context_mode,
            context_length=args.context_length,
            cond_dropout_prob=args.cond_dropout_prob,
        )

        if step % 10 == 0 or step == 1:
            log_for_0(
                f"step={step} loss={loss_metrics['loss']:.6f} "
                f"denoiser={loss_metrics['denoiser_loss']:.6f} "
                f"decoder={loss_metrics['decoder_loss']:.6f}"
            )
        if run is not None:
            wandb.log(
                {
                    "train/loss": loss_metrics["loss"],
                    "train/denoiser_loss": loss_metrics["denoiser_loss"],
                    "train/decoder_loss": loss_metrics["decoder_loss"],
                    "train/step": step,
                },
                step=step,
            )

        if step % args.eval_every != 0 and step != args.steps:
            continue

        metrics = evaluate_generation(
            model=model,
            context_table=context_table,
            tokenizer=tokenizer,
            target_sentences=target_sentences,
            target_latents=target_latents,
            target_length=target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=noise_generator,
            context_mode=args.context_mode,
        )
        metrics["step"] = step
        if init_permutation is not None:
            metrics["init_permutation"] = init_permutation.tolist()
        if args.context_mode == "learned_from_random_text":
            metrics["init_texts"] = init_texts
        save_results(args.output_dir, step, metrics)
        best_metrics = metrics if best_metrics is None or metrics["exact_match"] >= best_metrics["exact_match"] else best_metrics

        log_for_0(
            f"eval step={step} exact_match={metrics['exact_match']:.3f} "
            f"nearest_target_accuracy={metrics['nearest_target_accuracy']:.3f}"
        )
        for idx, (target, generated, exact) in enumerate(
            zip(metrics["targets"], metrics["generated"], metrics["exact"])
        ):
            log_for_0(f"[{idx}] exact={exact} target={target!r} generated={generated!r}")

        if run is not None:
            table = wandb.Table(
                columns=[
                    "epoch", "step", "id", "target", "generated", "exact",
                    "nearest_target_id", "nearest_target_score",
                ]
            )
            for idx, (target, generated, exact) in enumerate(
                zip(metrics["targets"], metrics["generated"], metrics["exact"])
            ):
                table.add_data(
                    step,
                    step,
                    idx,
                    target,
                    generated,
                    exact,
                    metrics["nearest_target_indices"][idx],
                    metrics["nearest_target_scores"][idx],
                )
            wandb.log(
                {
                    "eval/exact_match": metrics["exact_match"],
                    "eval/nearest_target_accuracy": metrics["nearest_target_accuracy"],
                    "eval/samples": table,
                },
                step=step,
            )

        if metrics["exact_match"] == 1.0:
            log_for_0("Reached perfect exact match; stopping early.")
            break

    if run is not None and best_metrics is not None:
        run.summary["best_exact_match"] = best_metrics["exact_match"]
        run.finish()

    if best_metrics is not None:
        final_path = Path(args.output_dir) / "best_metrics.json"
        with final_path.open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        log_for_0(f"Best exact match: {best_metrics['exact_match']:.3f}")
        log_for_0(f"Saved best metrics to {final_path}")


if __name__ == "__main__":
    main()
