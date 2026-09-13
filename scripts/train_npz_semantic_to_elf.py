#!/usr/bin/env python
"""Train ELF text generation from fixed semantic vectors stored in an NPZ."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

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

from configs.config import SamplingConfig
from modules.semantic_adapter import (
    DelayFusionContextProjector,
    DelayTokenContextProjector,
    ResidualDelayTokenFusionContextProjector,
    ResidualIdentityContextProjector,
    ResidualMLPFusionContextProjector,
)
from modules.fmri2sem_bridge import (
    ContentLogitHead,
    FMRI2SEMLexicalToELFContextAdapter,
    FMRI2SEMToELFContextAdapter,
    OrderedWordLogitHead,
    configure_mri2sem_trainable,
    load_mri2sem_model,
)
from modules.lora import inject_elf_lora, load_elf_state_dict, lora_parameter_count
from modules.dascoli_sentence_source import (
    SimulatedDAscoliSentences,
    align_word_confidence_to_tokens,
    build_position_vocabulary,
    external_sentence_source,
    force_vocabulary_token_confidence,
    normalized_words,
    simulate_dascoli_sentences,
    transform_token_confidence,
)
from modules.t5_encoder import get_encoder
from utils.generation_utils import _generate_samples_single_batch
from utils.sampling_utils import get_sampling_steps, restore_cond
from scripts.meg_context_overfit import (
    _CONTENT_WORD_STOPWORDS,
    _WORD_RE,
    OverfitBatch,
    SemanticVectorContextProjector,
    build_config,
    encode_text_batched,
    eval_checkpoint_scores,
    evaluate_generation,
    evaluate_retrieval,
    freeze_for_toy_tuning,
    load_pretrained_model,
    save_eval_checkpoint,
    semantic_content_ranking_loss,
    semantic_preservation_losses,
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


def safe_wandb_log(payload: dict, *, step: int) -> None:
    """Keep an ARC training run alive when optional W&B telemetry is unavailable."""

    try:
        wandb.log(payload, step=step)
    except Exception as exc:  # telemetry must never invalidate a saved validation result
        logger.warning("W&B logging failed at step %d; continuing locally: %s", step, exc)


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint beside its destination, then atomically replace it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    parser.add_argument(
        "--brain-input-npz",
        default="",
        help=(
            "Optional MRI2SEM segment NPZ. When set, concatenate its train/validation fMRI arrays "
            "and use those raw vectors as the adapter input while keeping text rows from --npz-path."
        ),
    )
    parser.add_argument("--brain-train-key", default="train_x")
    parser.add_argument("--brain-val-key", default="val_x")
    parser.add_argument(
        "--brain-val-roll",
        type=int,
        default=0,
        help=(
            "Deterministically roll only the held-out brain rows after alignment checks. "
            "Nonzero values are evaluation controls for matched-versus-deranged conditioning; "
            "text targets and train rows remain fixed."
        ),
    )
    parser.add_argument(
        "--semantic-val-roll",
        type=int,
        default=0,
        help=(
            "Deterministically roll only the held-out condition vectors after the "
            "train/validation boundary is established. Nonzero values are leakage-safe "
            "matched-versus-deranged controls; targets and train rows remain fixed."
        ),
    )
    parser.add_argument("--brain-model-checkpoint", default="")
    parser.add_argument("--brain-semantic-output-dim", type=int, default=1536)
    parser.add_argument(
        "--brain-projector-mismatch",
        choices=["error", "mean-blocks"],
        default="error",
        help="Optionally initialize a smaller MRI2SEM output head by averaging contiguous checkpoint heads.",
    )
    parser.add_argument("--brain-hidden-dim", type=int, default=2048)
    parser.add_argument("--brain-res-blocks", type=int, default=4)
    parser.add_argument("--brain-dropout", type=float, default=0.1)
    parser.add_argument(
        "--brain-unfreeze",
        choices=["frozen", "projector", "last_block", "all"],
        default="frozen",
    )
    parser.add_argument("--brain-lr", type=float, default=1e-5)
    parser.add_argument(
        "--brain-lexical-head-checkpoint",
        default="",
        help=(
            "Optional train11725 lexical-probe checkpoint. Its frozen top-k word prototypes "
            "are injected through a zero-initialized ELF context gate."
        ),
    )
    parser.add_argument("--brain-lexical-topk", type=int, default=5)
    parser.add_argument("--brain-lexical-temperature", type=float, default=0.5)
    parser.add_argument(
        "--brain-lexical-context-mode",
        choices=["mixture", "partitioned"],
        default="mixture",
        help=(
            "Mix selected word prototypes before projection, or preserve them in disjoint "
            "ELF context-memory partitions."
        ),
    )
    parser.add_argument(
        "--brain-lexical-prior-subtraction",
        type=float,
        default=0.0,
        help="Subtract this multiple of the train content-word logit prior before lexical top-k.",
    )
    parser.add_argument(
        "--brain-lexical-decode-bias-strength",
        type=float,
        default=0.0,
        help="Add a sparse frozen-head token bias to ELF decoder logits during generation.",
    )
    parser.add_argument("--brain-lexical-max-prior-probability", type=float, default=1.0)
    parser.add_argument("--brain-lexical-decode-bias-once", action="store_true")
    parser.add_argument(
        "--brain-lexical-decode-bias-max-positions",
        type=int,
        default=0,
        help="Restrict one-shot content forcing to this many early target-token slots; zero uses all.",
    )
    parser.add_argument(
        "--brain-lexical-decode-bias-mode",
        choices=["scattered", "sequence"],
        default="sequence",
        help=(
            "When one-shot lexical bias is enabled, place individual word pieces at "
            "their best distinct positions or keep each candidate word contiguous."
        ),
    )
    parser.add_argument(
        "--brain-ordered-head-checkpoint",
        default="",
        help="Optional train11725 ordered ten-word head used as a soft position-aware decoder prior.",
    )
    parser.add_argument("--brain-ordered-prior-subtraction", type=float, default=0.0)
    parser.add_argument(
        "--brain-ordered-min-prior-probability",
        type=float,
        default=0.0,
        help=(
            "Restrict the ordered head to words whose train-only positional "
            "frequency is at least this value; useful for function-word repair."
        ),
    )
    parser.add_argument(
        "--brain-ordered-min-margin",
        type=float,
        default=0.0,
        help="Skip ordered-head positions whose adjusted top-1/top-2 logit margin is smaller.",
    )
    parser.add_argument("--brain-ordered-decode-bias-strength", type=float, default=0.0)

    parser.add_argument(
        "--semantic-lexical-head-checkpoint",
        default="",
        help=(
            "Optional train-only word head whose input is the NPZ semantic vector itself. "
            "Its top-k words are injected only as a soft ELF decoder-logit bias; it never "
            "provides a generated sentence or a second inference condition. The existing "
            "--brain-lexical-* controls configure selection and bias strength."
        ),
    )
    parser.add_argument(
        "--semantic-ordered-head-checkpoint",
        default="",
        help=(
            "Optional train-only position-aware word head paired with "
            "--semantic-lexical-head-checkpoint. Its predictions are soft ELF "
            "decoder biases, never a generated prefix."
        ),
    )
    parser.add_argument(
        "--brain-ordered-decode-max-positions",
        type=int,
        default=0,
        help="Maximum early target-token slots occupied by the ordered soft bias; zero uses all.",
    )
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--context_length", type=int, default=16)
    parser.add_argument("--semantic-hidden-dim", type=int, default=2048)
    parser.add_argument(
        "--semantic-adapter-kind",
        choices=[
            "flat",
            "delay_fusion",
            "residual_identity",
            "residual_mlp",
            "residual_delay_tokens",
            "delay_tokens",
        ],
        default="flat",
        help=(
            "Use the legacy flat-vector MLP, a small learned fusion in front of the pretrained flat MLP, "
            "or a transformer that preserves contiguous semantic delay blocks as ordered tokens."
        ),
    )
    parser.add_argument("--semantic-delay-tokens", type=int, default=4)
    parser.add_argument("--semantic-delay-layers", type=int, default=2)
    parser.add_argument("--semantic-delay-heads", type=int, default=8)
    parser.add_argument(
        "--semantic-preserve-delay-context",
        action="store_true",
        help="Inject four zero-initialized centered delay residuals into the first ELF context tokens.",
    )
    parser.add_argument("--semantic-mapper-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--semantic-input-projection-dim",
        type=int,
        default=0,
        help=(
            "Optional trainable input projection before the semantic adapter. If it evenly divides the NPZ "
            "dimension, it is initialized to the exact mean of contiguous blocks (for example 3072->768 "
            "for four Tang delay blocks). Zero disables it."
        ),
    )
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument(
        "--normalize-semantic-projection",
        action="store_true",
        help="L2-normalize the fused/projected semantic vector before the pretrained context MLP.",
    )
    parser.add_argument(
        "--train-semantic-input-only",
        action="store_true",
        help=(
            "Freeze the pretrained semantic adapter backbone and train only its input projection or "
            "delay-fusion logits. Intended for leakage-safe adaptation of noisy brain predictions."
        ),
    )
    parser.add_argument(
        "--freeze-semantic-adapter",
        action="store_true",
        help=(
            "Freeze the complete semantic/raw-brain adapter after checkpoint restoration. "
            "Useful for isolating a decoder-side conditioning intervention."
        ),
    )
    parser.add_argument(
        "--train-lexical-context-gate-only",
        action="store_true",
        help=(
            "Freeze the complete semantic/raw-brain adapter except the zero-initialized "
            "lexical_context_gate. Requires --brain-lexical-head-checkpoint."
        ),
    )
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument(
        "--eval-target-length",
        type=int,
        default=0,
        help=(
            "Optional validation generation canvas shorter than the training tokenizer "
            "canvas. Zero uses the full canvas; 22 reproduces the published Apple "
            "ELF-M validation decode while retaining its 27-token model configuration."
        ),
    )
    parser.add_argument(
        "--val-num-examples",
        type=int,
        default=0,
        help="Hold out this many examples from the end of the NPZ for validation. Default 0 keeps legacy in-sample eval.",
    )
    parser.add_argument(
        "--condition-source-key",
        default="condition_source",
        help="NPZ key used by --train-condition-sources.",
    )
    parser.add_argument(
        "--train-condition-sources",
        nargs="+",
        default=None,
        help=(
            "Optionally train only on pre-validation rows whose condition-source label is in this list. "
            "The held-out validation tail is unchanged. This supports distribution-matched OOF-only training "
            "without rewriting the NPZ or its row-aligned latent cache."
        ),
    )
    parser.add_argument(
        "--train-balance-key",
        default="",
        help=(
            "Optional categorical NPZ key (for example story). When set, each training draw first samples "
            "a category uniformly and then samples one selected training row from that category."
        ),
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
    parser.add_argument(
        "--elf-lr",
        type=float,
        default=0.0,
        help="ELF or ELF-LoRA learning rate. Zero reuses --lr.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="Learning-rate schedule across the complete training run.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Final/base LR ratio for cosine decay.",
    )
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--retrieval-eval-every", type=int, default=0)
    parser.add_argument(
        "--retrieval-num-examples",
        type=int,
        default=0,
        help="Rows for latent retrieval: 0 means all evaluation rows; -1 disables it.",
    )
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
    parser.add_argument(
        "--decoder-content-token-weight",
        type=float,
        default=1.0,
        help="Relative decoder-CE weight for tokens overlapping target content words.",
    )
    parser.add_argument(
        "--decoder-content-bow-loss-weight",
        type=float,
        default=0.0,
        help="Weight for order-tolerant content-token presence loss; requires content-token weight > 1.",
    )
    parser.add_argument(
        "--decoder-content-precision-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for penalizing likely in-batch content tokens absent from each target; "
            "requires content-token weight > 1."
        ),
    )
    parser.add_argument(
        "--decoder-content-precision-topk",
        type=int,
        default=32,
        help="Number of highest-loss absent content candidates retained per sample; zero uses all.",
    )
    parser.add_argument(
        "--decoder-condition-pairing-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for matched-vs-rolled condition sequence-NLL ranking. This directly "
            "penalizes a decoder that ignores its brain-derived context."
        ),
    )
    parser.add_argument(
        "--decoder-condition-pairing-margin",
        type=float,
        default=0.1,
    )
    parser.add_argument("--decoder_prob", type=float, default=0.5)
    parser.add_argument(
        "--decoder_p_mean",
        type=float,
        default=0.8,
        help=(
            "Mean of the logistic-normal target-latent mixing coefficient used for "
            "decoder training. Lower values remove more target signal and force greater "
            "use of the semantic condition."
        ),
    )
    parser.add_argument(
        "--decoder_p_std",
        type=float,
        default=0.8,
        help="Standard deviation of the decoder logistic-normal mixing coefficient.",
    )
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument(
        "--decoder-position-prior-strength",
        type=float,
        default=0.0,
        help=(
            "Scale a target-position token prior estimated exclusively from selected "
            "training rows and added to ELF decoder logits at evaluation."
        ),
    )
    parser.add_argument("--decoder-position-prior-smoothing", type=float, default=0.25)
    parser.add_argument("--decoder-position-prior-clip", type=float, default=5.0)
    parser.add_argument(
        "--decoder-flow-latent-cache",
        default="",
        help=(
            "Optional NPY cache of target latents sampled from noise by the frozen "
            "semantic-conditioned flow. When set, these deployment-matched latents "
            "replace sentence-source latents for decoder training only."
        ),
    )
    parser.add_argument("--decoder-flow-latent-batch-size", type=int, default=16)
    parser.add_argument("--decoder-flow-latent-seed", type=int, default=66)
    parser.add_argument(
        "--decoder-flow-latent-direct",
        action="store_true",
        help=(
            "Feed cached frozen-flow target latents directly to the decoder objective "
            "instead of mixing them with ground-truth target latents."
        ),
    )
    parser.add_argument(
        "--semantic-alignment-target-npz",
        default="",
        help="Optional row-aligned NPZ containing exact semantic targets for noisy-input alignment.",
    )
    parser.add_argument("--semantic-alignment-target-key", default="input_embeddings")
    parser.add_argument("--semantic-alignment-target-sentence-key", default="sentence")
    parser.add_argument(
        "--allow-semantic-alignment-sentence-mismatch",
        action="store_true",
        help=(
            "Allow row-aligned auxiliary semantics to describe a different temporal text window. "
            "Use only when row identity/timing has been audited independently."
        ),
    )
    parser.add_argument("--semantic-alignment-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic-alignment-loss-type",
        choices=["cosine", "mse"],
        default="cosine",
    )
    parser.add_argument("--semantic-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--semantic-context-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-content-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-content-min-frequency", type=int, default=5)
    parser.add_argument("--semantic-content-max-vocabulary", type=int, default=3000)
    parser.add_argument("--semantic-content-negative-topk", type=int, default=32)
    parser.add_argument("--semantic-content-temperature", type=float, default=0.1)
    parser.add_argument("--semantic-raw-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-geometry-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-rank-distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-rank-distill-temperature", type=float, default=0.07)
    parser.add_argument(
        "--semantic-objective-only",
        action="store_true",
        help="Skip ELF denoiser/decoder updates and train only the semantic interface objectives.",
    )
    parser.add_argument(
        "--curriculum-semantic-only-epochs",
        type=float,
        default=0.0,
        help=(
            "For this many initial epochs, down-weight diffusion and decoder losses while "
            "retaining the configured semantic losses. Zero disables the curriculum."
        ),
    )
    parser.add_argument("--curriculum-denoiser-loss-weight", type=float, default=0.1)
    parser.add_argument("--curriculum-decoder-loss-weight", type=float, default=0.1)
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
    parser.add_argument(
        "--elf-lora-rank",
        type=int,
        default=0,
        help="Positive rank freezes ELF base weights and enables dependency-free LoRA.",
    )
    parser.add_argument("--elf-lora-alpha", type=float, default=16.0)
    parser.add_argument("--elf-lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--elf-lora-targets",
        default="attention",
        help="Comma-separated ELF module families: attention and optionally mlp.",
    )
    parser.add_argument(
        "--elf-lora-last-n-blocks",
        type=int,
        default=-1,
        help="Apply LoRA to all ELF blocks when negative, otherwise only the final N blocks.",
    )
    parser.add_argument(
        "--elf-brain-cross-attention-last-n-blocks",
        type=int,
        default=0,
        help=(
            "Enable zero-initialized target-to-condition cross-attention in the final N ELF "
            "blocks. Zero disables it. The first --context_length sequence positions are "
            "used as clean brain memory."
        ),
    )
    parser.add_argument(
        "--elf-brain-cross-attention-heads",
        type=int,
        default=0,
        help="Cross-attention head count; zero reuses the ELF model head count.",
    )
    parser.add_argument(
        "--elf-brain-cross-attention-dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--elf-brain-cross-attention-only",
        action="store_true",
        help=(
            "Freeze all other ELF tensors, including restored LoRA weights, and train only "
            "the newly enabled brain cross-attention modules. Adapter/brain learning rates "
            "remain independently controlled."
        ),
    )
    parser.add_argument(
        "--elf-brain-cross-attention-decoder-only",
        action="store_true",
        help=(
            "Apply the new brain cross-attention only when the discrete decoder head is "
            "active, preserving the pretrained diffusion/flow trajectory exactly."
        ),
    )
    parser.add_argument("--eval-only", action="store_true", help="Run one retrieval/generation eval pass and exit.")
    parser.add_argument(
        "--init-e2e-checkpoint",
        default="",
        help=(
            "Optional checkpoint containing model_state_dict and/or adapter_state_dict. "
            "For MEG2SEM e2e checkpoints, semantic_projector.* keys are stripped before "
            "loading into the semantic-only projector."
        ),
    )
    parser.add_argument(
        "--init-e2e-adapter-mismatch",
        choices=[
            "error",
            "skip",
            "mean-blocks",
            "delay-fusion",
            "residual-mlp",
            "delay-context",
            "direct-identity",
        ],
        default="error",
        help=(
            "What to do if --init-e2e-checkpoint adapter weights do not fit this NPZ input shape. "
            "mean-blocks loads a pretrained flat adapter behind a new mean-block initialized projection; "
            "delay-fusion loads it behind newly initialized feature-wise delay weights; residual-mlp "
            "loads it behind a zero-output initialized nonlinear residual mapper."
        ),
    )
    parser.add_argument("--last_n_blocks", type=int, default=1)
    parser.add_argument("--generation-t5-retrieval", action="store_true")
    parser.add_argument(
        "--dascoli-simulated-source",
        action="store_true",
        help=(
            "Train/evaluate confidence-weighted transport from a target-derived simulated "
            "full D'Ascoli sentence. This is an explicitly labelled validation ceiling, not "
            "a brain-decoding result."
        ),
    )
    parser.add_argument(
        "--sentence-source-eval-json",
        default="",
        help=(
            "Optional aligned validation metrics JSON whose generated sentences replace the "
            "simulated validation proposals. Training proposals remain target-derived simulated "
            "corruptions; this enables validation-only T5-to-flow editing without target leakage."
        ),
    )
    parser.add_argument(
        "--sentence-source-eval-confidence",
        type=float,
        default=0.5,
        help=(
            "Uniform proposed-word confidence when --sentence-source-eval-json has no "
            "word_confidence field."
        ),
    )
    parser.add_argument(
        "--sentence-source-eval-confidence-field",
        default="word_confidence",
        help=(
            "Field or dotted field path containing aligned external word confidence, "
            "for example word_confidence_variants.teacher_margin."
        ),
    )
    parser.add_argument("--dascoli-expected-words", type=int, default=10)
    parser.add_argument(
        "--dascoli-train-accuracies",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 0.90],
        help="Per-row word-correctness probabilities mixed during training.",
    )
    parser.add_argument("--dascoli-eval-accuracy", type=float, default=0.50)
    parser.add_argument(
        "--dascoli-eval-mode",
        choices=[
            "semantic_only",
            "sentence_only",
            "joint_unweighted",
            "joint_weighted",
            "joint_shuffled",
        ],
        default="joint_weighted",
    )
    parser.add_argument(
        "--dascoli-global-trust",
        type=float,
        default=1.0,
        help="Multiply calibrated per-token confidence by this value before clipping to [0,1].",
    )
    parser.add_argument(
        "--dascoli-confidence-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum non-padding source trust after calibration; 1.0 is the "
            "unweighted full-sentence endpoint."
        ),
    )
    parser.add_argument(
        "--dascoli-eval-semantic-scale",
        type=float,
        default=1.0,
        help="Scale the semantic context during D'Ascoli evaluation (0=sentence only, 1=full).",
    )
    parser.add_argument(
        "--sentence-source-flow-end-time",
        type=float,
        default=1.0,
        help=(
            "Stop source-to-target ODE editing at this time in [0,1]. Zero decodes the "
            "source latent directly; one applies the full learned transport."
        ),
    )
    parser.add_argument(
        "--sentence-source-copy-confidence-threshold",
        type=float,
        default=-1.0,
        help=(
            "If in [0,1], hard-copy external source tokens at or above this raw "
            "confidence after diffusion decoding. Negative disables hard copying."
        ),
    )
    parser.add_argument(
        "--sentence-source-edit-latent-preservation",
        type=float,
        default=0.0,
        help=(
            "For source positions below the hard-copy threshold, blend this fraction "
            "of the original source latent back immediately before decoding. Zero is "
            "the existing full-edit behavior; one is latent-preserving at editable tokens."
        ),
    )
    parser.add_argument(
        "--sentence-source-copy-function-words",
        action="store_true",
        help=(
            "Always hard-copy proposal tokens in the evaluator function-word list, "
            "leaving the confidence gate to edit content words."
        ),
    )
    parser.add_argument(
        "--dascoli-source-dropout-prob",
        type=float,
        default=0.10,
        help=(
            "Training-only probability of setting a complete source row's confidence to zero; "
            "preserves a semantic-only branch and prevents compulsory copying."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="BrainDiffusion")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default="npz-semantic-to-elf")
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--wandb_sample_examples", type=int, default=16)
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "generation_t5_top1",
            "word_overlap",
            "content_word_overlap",
            "word_content_mean",
            "negative_wer",
            "semantic_top5",
        ],
        default="generation_t5_top1",
        help="Validation metric used to select best.pt. negative_wer directly minimizes corpus WER.",
    )
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument(
        "--save-trainable-only-checkpoint",
        action="store_true",
        help=(
            "Store only trainable ELF parameters plus a parent initialization reference. "
            "Requires a frozen adapter and --init-e2e-checkpoint; useful on constrained ARC storage."
        ),
    )
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument(
        "--save-eval-checkpoints",
        action="store_true",
        help="Retain Pareto candidates across all reported validation criteria.",
    )
    parser.add_argument("--eval-checkpoint-top-k", type=int, default=1)
    parser.add_argument("--eval-checkpoint-every", type=int, default=0)
    parser.add_argument("--include-optimizer-in-checkpoints", action="store_true")
    return parser.parse_args()


def _strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def select_train_indices(
    condition_sources: list[str] | None,
    *,
    train_split_n: int,
    include_sources: list[str] | None,
) -> torch.Tensor:
    """Select training rows without ever admitting rows from the validation tail."""
    indices = torch.arange(0, train_split_n, dtype=torch.long)
    if not include_sources:
        return indices
    if condition_sources is None:
        raise ValueError("--train-condition-sources requires the configured condition-source NPZ key.")
    if len(condition_sources) < train_split_n:
        raise ValueError(
            "Condition-source labels do not cover the training split: "
            f"labels={len(condition_sources)}, train_split_n={train_split_n}."
        )
    requested = set(include_sources)
    selected = [index for index, source in enumerate(condition_sources[:train_split_n]) if source in requested]
    if not selected:
        available = sorted(set(condition_sources[:train_split_n]))
        raise ValueError(
            f"No training rows matched condition sources {sorted(requested)}; available={available}."
        )
    return torch.as_tensor(selected, dtype=torch.long)


def build_balanced_index_pools(labels: list[str], train_indices: torch.Tensor) -> list[torch.Tensor]:
    """Group already-selected training rows without admitting any outside index."""
    grouped: dict[str, list[int]] = {}
    for index in train_indices.tolist():
        grouped.setdefault(labels[index], []).append(index)
    if not grouped:
        raise ValueError("Cannot construct balanced sampling pools from an empty training selection.")
    return [torch.as_tensor(grouped[key], dtype=torch.long) for key in sorted(grouped)]


def sample_balanced_indices(
    pools: list[torch.Tensor],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniformly sample groups, then uniformly sample one row within each chosen group."""
    group_ids = torch.randint(len(pools), (batch_size,), generator=generator)
    selected = []
    for group_id in group_ids.tolist():
        pool = pools[group_id]
        row_offset = int(torch.randint(len(pool), (1,), generator=generator).item())
        selected.append(int(pool[row_offset]))
    return torch.as_tensor(selected, dtype=torch.long)


def format_overlap_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return ""
    return ", ".join(f"{token}x{count}" if count > 1 else token for token, count in counts.items())


def make_batches(num_examples: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(num_examples, generator=generator)
    return [perm[start:start + batch_size] for start in range(0, num_examples, batch_size)]


def build_decoder_position_logit_bias(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    vocabulary_size: int,
    strength: float,
    smoothing: float,
    clip: float,
) -> torch.Tensor | None:
    """Estimate a leakage-safe positional token prior from selected train rows."""

    if strength < 0.0:
        raise ValueError("decoder position-prior strength must be non-negative")
    if strength == 0.0:
        return None
    if smoothing <= 0.0:
        raise ValueError("decoder position-prior smoothing must be positive")
    if clip <= 0.0:
        raise ValueError("decoder position-prior clip must be positive")
    selected_ids = input_ids.index_select(0, train_indices.detach().cpu()).long()
    selected_mask = attention_mask.index_select(
        0, train_indices.detach().cpu()
    ).float()
    if selected_ids.ndim != 2 or selected_mask.shape != selected_ids.shape:
        raise ValueError("token IDs and masks must be aligned two-dimensional tensors")
    counts = torch.full(
        (selected_ids.shape[1], vocabulary_size),
        float(smoothing),
        dtype=torch.float32,
    )
    counts.scatter_add_(1, selected_ids.T, selected_mask.T)
    log_probability = counts.log() - counts.sum(dim=1, keepdim=True).log()
    # Center each slot at its most common token. The prior only penalizes less
    # plausible options and therefore cannot manufacture a positive logit spike.
    centered = log_probability - log_probability.max(dim=1, keepdim=True).values
    return centered.clamp(min=-float(clip), max=0.0) * float(strength)


def select_strings(items: list[str], indices: torch.Tensor) -> list[str]:
    return [items[int(index)] for index in indices.detach().cpu().tolist()]


def build_content_token_weights(
    tokenizer,
    sentences: list[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    content_weight: float,
) -> torch.Tensor:
    """Weight subword tokens that overlap non-stopword target spans."""

    if content_weight < 1.0:
        raise ValueError("decoder content-token weight must be at least 1.0")
    weights = torch.ones_like(attention_mask, dtype=torch.float32)
    if content_weight == 1.0:
        return weights

    for row, sentence in enumerate(sentences):
        encoded = tokenizer(
            sentence,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("Content-token weighting requires a fast tokenizer with offset mappings.")
        expected_ids = input_ids[row, : len(ids)].tolist()
        if list(ids) != expected_ids:
            raise ValueError(f"Tokenizer IDs changed while building content weights for row {row}.")
        content_spans = [
            match.span()
            for match in _WORD_RE.finditer(sentence)
            if match.group(0).lower() not in _CONTENT_WORD_STOPWORDS
            and any(character.isalpha() for character in match.group(0))
        ]
        for column, (start, end) in enumerate(offsets):
            if end <= start or not bool(attention_mask[row, column]):
                continue
            if any(start < span_end and end > span_start for span_start, span_end in content_spans):
                weights[row, column] = float(content_weight)
    return weights


def content_words(sentence: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _WORD_RE.finditer(sentence)
        if match.group(0).lower() not in _CONTENT_WORD_STOPWORDS
        and any(character.isalpha() for character in match.group(0))
    }


def build_semantic_content_prototypes(
    sentences: list[str],
    exact_semantic: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    min_frequency: int,
    max_vocabulary: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Build train-only word directions and row-level multi-label targets."""

    if min_frequency <= 0 or max_vocabulary <= 0:
        raise ValueError("semantic content vocabulary limits must be positive")
    train_rows = train_indices.detach().cpu().tolist()
    document_frequency = Counter(
        token
        for row in train_rows
        for token in content_words(sentences[row])
    )
    vocabulary = [
        token
        for token, count in sorted(document_frequency.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_frequency
    ][:max_vocabulary]
    if len(vocabulary) < 2:
        raise ValueError(f"Semantic content vocabulary is too small: {len(vocabulary)}")
    token_to_index = {token: index for index, token in enumerate(vocabulary)}

    exact = F.normalize(exact_semantic.detach().cpu().float(), p=2, dim=-1)
    train_exact = exact.index_select(0, train_indices.detach().cpu())
    sums = torch.zeros((len(vocabulary), exact.shape[1]), dtype=torch.float32)
    counts = torch.zeros((len(vocabulary),), dtype=torch.float32)
    targets = torch.zeros((len(sentences), len(vocabulary)), dtype=torch.bool)
    for local_row, global_row in enumerate(train_rows):
        for token in content_words(sentences[global_row]):
            index = token_to_index.get(token)
            if index is None:
                continue
            sums[index] += train_exact[local_row]
            counts[index] += 1.0
            targets[global_row, index] = True
    total = train_exact.sum(dim=0, keepdim=True)
    positive = sums / counts[:, None].clamp_min(1.0)
    negative = (total - sums) / (len(train_rows) - counts)[:, None].clamp_min(1.0)
    directions = F.normalize(positive - negative, p=2, dim=-1)
    return vocabulary, directions, targets


def semantic_objective_train_step(
    *,
    adapter: nn.Module,
    semantic_inputs: torch.Tensor,
    exact_semantic: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alignment_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    context_weight: float,
    content_weight: float,
    content_targets: torch.Tensor | None,
    content_directions: torch.Tensor | None,
    content_temperature: float,
    content_negative_topk: int,
    raw_anchor_weight: float,
    geometry_weight: float,
    rank_distill_weight: float,
    rank_distill_temperature: float,
) -> dict[str, float]:
    """Fast brain/adapter-only step for semantic and lexical supervision."""

    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = semantic_inputs.to(device=device, dtype=torch.float32)
    targets = exact_semantic.to(device=device, dtype=torch.float32)
    predicted = adapter.project_semantic(inputs).float()
    if tuple(predicted.shape) != tuple(targets.shape):
        raise ValueError(
            f"Semantic-only shape mismatch: predicted={predicted.shape} targets={targets.shape}"
        )

    cosine = F.cosine_similarity(predicted, targets, dim=-1)
    alignment_loss = 1.0 - cosine.mean()
    scaled_alignment = alignment_weight * alignment_loss

    contrastive_loss = predicted.sum() * 0.0
    if contrastive_weight > 0.0:
        if contrastive_temperature <= 0.0:
            raise ValueError("semantic contrastive temperature must be positive")
        predicted_norm = F.normalize(predicted, p=2, dim=-1)
        target_norm = F.normalize(targets, p=2, dim=-1)
        logits = predicted_norm @ target_norm.T / contrastive_temperature
        labels = torch.arange(logits.shape[0], device=device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
    scaled_contrastive = contrastive_weight * contrastive_loss

    context_loss = predicted.sum() * 0.0
    if context_weight > 0.0:
        context_from_projected = getattr(adapter, "context_from_projected_semantic", None)
        if context_from_projected is None:
            raise ValueError("Semantic-only context loss requires context_from_projected_semantic().")
        predicted_context, _ = context_from_projected(predicted)
        with torch.no_grad():
            target_context, _ = context_from_projected(targets)
        context_loss = 1.0 - F.cosine_similarity(
            predicted_context.float(), target_context.float(), dim=-1
        ).mean()
    scaled_context = context_weight * context_loss

    content_loss = predicted.sum() * 0.0
    if content_weight > 0.0:
        if content_targets is None or content_directions is None:
            raise ValueError("Semantic-only content loss requires target masks and directions.")
        content_loss = semantic_content_ranking_loss(
            predicted,
            content_targets.to(device=device),
            content_directions.to(device=device),
            temperature=content_temperature,
            negative_topk=content_negative_topk,
        )
    scaled_content = content_weight * content_loss
    preservation = semantic_preservation_losses(
        predicted,
        inputs,
        exact_targets=(targets if rank_distill_weight > 0.0 else None),
        rank_temperature=rank_distill_temperature,
    )
    scaled_raw_anchor = raw_anchor_weight * preservation["anchor"]
    scaled_geometry = geometry_weight * preservation["geometry"]
    scaled_rank_distill = rank_distill_weight * preservation["rank_distill"]
    total = (
        scaled_alignment
        + scaled_contrastive
        + scaled_context
        + scaled_content
        + scaled_raw_anchor
        + scaled_geometry
        + scaled_rank_distill
    )
    total.backward()
    optimizer.step()

    return {
        "loss": float(total.detach().cpu()),
        "denoiser_loss": 0.0,
        "decoder_loss": 0.0,
        "content_bow_loss": 0.0,
        "content_bow_loss_scaled": 0.0,
        "content_precision_loss": 0.0,
        "content_precision_loss_scaled": 0.0,
        "semantic_alignment_loss": float(alignment_loss.detach().cpu()),
        "semantic_alignment_loss_scaled": float(scaled_alignment.detach().cpu()),
        "semantic_alignment_cosine": float(cosine.mean().detach().cpu()),
        "semantic_contrastive_loss": float(contrastive_loss.detach().cpu()),
        "semantic_contrastive_loss_scaled": float(scaled_contrastive.detach().cpu()),
        "semantic_context_loss": float(context_loss.detach().cpu()),
        "semantic_context_loss_scaled": float(scaled_context.detach().cpu()),
        "semantic_content_loss": float(content_loss.detach().cpu()),
        "semantic_content_loss_scaled": float(scaled_content.detach().cpu()),
        "semantic_raw_anchor_loss": float(preservation["anchor"].detach().cpu()),
        "semantic_raw_anchor_loss_scaled": float(scaled_raw_anchor.detach().cpu()),
        "semantic_raw_anchor_cosine": float(preservation["anchor_cosine"].detach().cpu()),
        "semantic_geometry_loss": float(preservation["geometry"].detach().cpu()),
        "semantic_geometry_loss_scaled": float(scaled_geometry.detach().cpu()),
        "semantic_rank_distill_loss": float(preservation["rank_distill"].detach().cpu()),
        "semantic_rank_distill_loss_scaled": float(scaled_rank_distill.detach().cpu()),
    }


def validation_checkpoint_score(metrics: dict, metric: str) -> float:
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    if metric == "generation_t5_top1":
        return float(retrieval.get("top1", metrics.get("exact_match", 0.0)))
    if metric == "word_overlap":
        return float(quality.get("words_overlap", 0.0))
    if metric == "content_word_overlap":
        return float(quality.get("content_words_overlap", 0.0))
    if metric == "word_content_mean":
        word_overlap = float(quality.get("words_overlap", 0.0))
        content_overlap = float(quality.get("content_words_overlap", 0.0))
        return 0.5 * (word_overlap + content_overlap)
    if metric == "negative_wer":
        word_error_rate = quality.get("word_error_rate")
        if word_error_rate is None:
            raise ValueError("word_error_rate is missing from generation metrics")
        return -float(word_error_rate)
    if metric == "semantic_top5":
        semantic_interface = metrics.get("semantic_interface") or {}
        if "top5" not in semantic_interface:
            raise ValueError("semantic_interface.top5 is missing from generation metrics")
        return float(semantic_interface["top5"])
    raise ValueError(f"Unsupported checkpoint metric: {metric}")


def learning_rate_for_step(
    *,
    base_lr: float,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if base_lr <= 0.0:
        raise ValueError("base_lr must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between zero and one")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * max(1, step) / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported LR schedule: {schedule}")
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * multiplier


def curriculum_loss_weights(
    *,
    epoch: float,
    curriculum_epochs: float,
    final_denoiser_weight: float,
    final_decoder_weight: float,
    curriculum_denoiser_weight: float,
    curriculum_decoder_weight: float,
) -> tuple[float, float]:
    if curriculum_epochs > 0.0 and epoch <= curriculum_epochs:
        return curriculum_denoiser_weight, curriculum_decoder_weight
    return final_denoiser_weight, final_decoder_weight


@torch.no_grad()
def project_semantic_vectors(
    adapter: nn.Module,
    semantic_inputs: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    project_semantic = getattr(adapter, "project_semantic", None)
    if project_semantic is None:
        raise ValueError("Semantic projection requires adapter.project_semantic().")
    adapter.eval()
    projected = []
    for start in range(0, semantic_inputs.shape[0], batch_size):
        batch = semantic_inputs[start : start + batch_size].to(device=device, dtype=torch.float32)
        projected.append(project_semantic(batch).detach().cpu().float())
    return F.normalize(torch.cat(projected, dim=0), p=2, dim=-1)


@torch.no_grad()
def semantic_interface_metrics(
    adapter: nn.Module,
    semantic_inputs: torch.Tensor,
    semantic_targets: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predicted = project_semantic_vectors(
        adapter,
        semantic_inputs,
        device=device,
        batch_size=batch_size,
    )
    target = F.normalize(semantic_targets.detach().cpu().float(), p=2, dim=-1)
    if predicted.shape != target.shape:
        raise ValueError(
            f"Semantic interface shape mismatch: predicted={tuple(predicted.shape)} "
            f"target={tuple(target.shape)}"
        )
    similarities = predicted @ target.T
    matched = similarities.diag()
    ranks = 1 + (similarities > matched.unsqueeze(1)).sum(dim=1)
    return {
        "matched_cosine_mean": float(matched.mean()),
        "matched_cosine_median": float(matched.median()),
        "top1": float((ranks == 1).float().mean()),
        "top5": float((ranks <= min(5, len(ranks))).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def build_semantic_adapter(
    *,
    args: argparse.Namespace,
    input_dim: int,
    context_dim: int,
) -> torch.nn.Module:
    if args.semantic_adapter_kind == "flat":
        return SemanticVectorContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            dropout=args.adapter_dropout,
            input_projection_dim=args.semantic_input_projection_dim or None,
            normalize_semantic_output=args.normalize_semantic_projection,
        )
    if args.semantic_adapter_kind == "delay_fusion":
        if args.semantic_input_projection_dim:
            raise ValueError(
                "--semantic-input-projection-dim cannot be combined with --semantic-adapter-kind delay_fusion."
            )
        return DelayFusionContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            dropout=args.adapter_dropout,
            num_delay_tokens=args.semantic_delay_tokens,
            normalize_semantic_output=args.normalize_semantic_projection,
        )
    if args.semantic_adapter_kind == "residual_mlp":
        if args.semantic_input_projection_dim:
            raise ValueError(
                "--semantic-input-projection-dim cannot be combined with --semantic-adapter-kind residual_mlp."
            )
        return ResidualMLPFusionContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            mapper_hidden_dim=args.semantic_mapper_hidden_dim,
            dropout=args.adapter_dropout,
            num_delay_tokens=args.semantic_delay_tokens,
            normalize_semantic_output=args.normalize_semantic_projection,
        )
    if args.semantic_adapter_kind == "residual_identity":
        if args.semantic_input_projection_dim:
            raise ValueError(
                "--semantic-input-projection-dim cannot be combined with "
                "--semantic-adapter-kind residual_identity."
            )
        return ResidualIdentityContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            mapper_hidden_dim=args.semantic_mapper_hidden_dim,
            dropout=args.adapter_dropout,
            normalize_semantic_output=args.normalize_semantic_projection,
        )
    if args.semantic_adapter_kind == "residual_delay_tokens":
        if args.semantic_input_projection_dim:
            raise ValueError(
                "--semantic-input-projection-dim cannot be combined with "
                "--semantic-adapter-kind residual_delay_tokens."
            )
        return ResidualDelayTokenFusionContextProjector(
            input_dim=input_dim,
            context_dim=context_dim,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            mapper_hidden_dim=args.semantic_mapper_hidden_dim,
            dropout=args.adapter_dropout,
            num_delay_tokens=args.semantic_delay_tokens,
            num_layers=args.semantic_delay_layers,
            num_heads=args.semantic_delay_heads,
            normalize_semantic_output=args.normalize_semantic_projection,
            preserve_delay_context=args.semantic_preserve_delay_context,
        )
    if args.semantic_input_projection_dim:
        raise ValueError(
            "--semantic-input-projection-dim is only supported by --semantic-adapter-kind flat; "
            "the delay-token adapter projects each delay block separately."
        )
    return DelayTokenContextProjector(
        input_dim=input_dim,
        context_dim=context_dim,
        context_length=args.context_length,
        hidden_dim=args.semantic_hidden_dim,
        dropout=args.adapter_dropout,
        num_delay_tokens=args.semantic_delay_tokens,
        num_layers=args.semantic_delay_layers,
        num_heads=args.semantic_delay_heads,
    )


def load_e2e_initialization(
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    adapter_mismatch: str = "error",
) -> dict[str, object]:
    # This is a trusted, locally produced ELF checkpoint. Explicitly disabling
    # weights-only loading keeps initialization compatible with PyTorch >=2.6,
    # whose default otherwise rejects the checkpoint's config metadata.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}: {type(ckpt).__name__}")

    loaded: list[str] = []
    parent_checkpoint = ckpt.get("parent_init_e2e_checkpoint")
    if parent_checkpoint:
        parent_checkpoint = str(parent_checkpoint)
        if Path(parent_checkpoint).resolve() == Path(checkpoint_path).resolve():
            raise ValueError("Compact checkpoint cannot refer to itself as its parent.")
        load_e2e_initialization(
            model,
            adapter,
            parent_checkpoint,
            device,
            adapter_mismatch=adapter_mismatch,
        )
        loaded.append("parent_init_e2e_checkpoint")
    if "model_state_dict" in ckpt:
        load_elf_state_dict(model, ckpt["model_state_dict"])
        loaded.append("model_state_dict")
    elif "params" in ckpt:
        load_elf_state_dict(model, ckpt["params"])
        loaded.append("params")

    trainable_model_state = ckpt.get("trainable_model_state_dict")
    if trainable_model_state is not None:
        missing, unexpected = model.load_state_dict(trainable_model_state, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected trainable ELF keys in {checkpoint_path}: {sorted(unexpected)}"
            )
        if not trainable_model_state:
            raise ValueError(f"Compact checkpoint {checkpoint_path} has no trainable ELF tensors.")
        logger.info(
            "Applied %d trainable ELF tensors from compact checkpoint; %d parent/base keys retained.",
            len(trainable_model_state),
            len(missing),
        )
        loaded.append("trainable_model_state_dict")

    trainable_adapter_state = ckpt.get("trainable_adapter_state_dict")
    if trainable_adapter_state is not None:
        if not trainable_adapter_state:
            raise ValueError(
                f"Compact checkpoint {checkpoint_path} has no trainable adapter tensors."
            )
        missing, unexpected = adapter.load_state_dict(
            trainable_adapter_state,
            strict=False,
        )
        if unexpected:
            raise ValueError(
                f"Unexpected trainable adapter keys in {checkpoint_path}: "
                f"{sorted(unexpected)}"
            )
        logger.info(
            "Applied %d trainable adapter tensors from compact checkpoint; "
            "%d parent/base keys retained.",
            len(trainable_adapter_state),
            len(missing),
        )
        loaded.append("trainable_adapter_state_dict")

    adapter_state = ckpt.get("adapter_state_dict")
    if adapter_state is None:
        # Accept bare adapter state dicts for quick local probes.
        adapter_keys = set(adapter.state_dict().keys())
        if adapter_keys and adapter_keys <= set(ckpt.keys()):
            adapter_state = ckpt

    if adapter_state is not None:
        adapter_state = dict(adapter_state)
        direct_keys = set(adapter.state_dict().keys())
        if not (direct_keys & set(adapter_state.keys())):
            prefix = "semantic_projector."
            stripped = {
                key[len(prefix):]: value
                for key, value in adapter_state.items()
                if key.startswith(prefix)
            }
            if stripped:
                adapter_state = stripped
        target_state = adapter.state_dict()
        if adapter_mismatch == "skip":
            incompatible_keys = sorted(
                key
                for key in set(target_state) & set(adapter_state)
                if tuple(target_state[key].shape) != tuple(adapter_state[key].shape)
            )
            if set(target_state) != set(adapter_state) or incompatible_keys:
                logger.warning(
                    "Skipping incompatible adapter initialization from %s: target_keys=%d source_keys=%d "
                    "shape_mismatches=%s",
                    checkpoint_path,
                    len(target_state),
                    len(adapter_state),
                    incompatible_keys,
                )
                adapter_state = None
        elif adapter_mismatch == "direct-identity":
            compatible_state = {
                key: value
                for key, value in adapter_state.items()
                if key in target_state
                and tuple(target_state[key].shape) == tuple(value.shape)
                and key.startswith("net.")
            }
            expected_net_keys = {key for key in target_state if key.startswith("net.")}
            if set(compatible_state) != expected_net_keys:
                raise ValueError(
                    "Could not recover the complete pretrained semantic context net for "
                    f"direct identity initialization: missing={sorted(expected_net_keys - set(compatible_state))}"
                )
            adapter_state = compatible_state
        if adapter_state is not None:
            missing, unexpected = adapter.load_state_dict(adapter_state, strict=False)
        if adapter_state is not None and (missing or unexpected):
            allowed_missing = {
                "mean-blocks": {"input_projection.weight"},
                "delay-fusion": {"fusion_logits"},
                "residual-mlp": {
                    key for key in target_state if key.startswith("semantic_mapper.")
                },
                "delay-context": {
                    "delay_context_projection.weight",
                    "delay_context_projection.bias",
                },
                "direct-identity": {
                    key for key in target_state if key.startswith("semantic_mapper.")
                },
            }.get(adapter_mismatch, set())
            if set(missing) != allowed_missing or unexpected:
                raise ValueError(
                    f"Could not strictly load semantic projector from {checkpoint_path}: "
                    f"missing={missing} unexpected={unexpected}"
                )
        if adapter_state is not None and (missing or unexpected):
            if adapter_mismatch == "mean-blocks":
                input_projection = getattr(adapter, "input_projection", None)
                if input_projection is None:
                    raise ValueError(
                        "--init-e2e-adapter-mismatch mean-blocks requires a nonzero "
                        "--semantic-input-projection-dim."
                    )
                logger.info(
                    "Retained mean-block initialized semantic input projection %d -> %d while loading pretrained adapter.",
                    input_projection.in_features,
                    input_projection.out_features,
                )
            elif adapter_mismatch == "delay-fusion":
                fusion_logits = getattr(adapter, "fusion_logits", None)
                if fusion_logits is None:
                    raise ValueError(
                        "--init-e2e-adapter-mismatch delay-fusion requires "
                        "--semantic-adapter-kind delay_fusion."
                    )
                logger.info(
                    "Retained uniform delay-fusion initialization with %d trainable logits.",
                    fusion_logits.numel(),
                )
            elif adapter_mismatch == "residual-mlp":
                semantic_mapper = getattr(adapter, "semantic_mapper", None)
                if semantic_mapper is None:
                    raise ValueError(
                        "--init-e2e-adapter-mismatch residual-mlp requires "
                        "--semantic-adapter-kind residual_mlp."
                    )
                logger.info(
                    "Retained zero-output residual semantic mapper with %d parameters.",
                    sum(param.numel() for param in semantic_mapper.parameters()),
                )
            elif adapter_mismatch == "delay-context":
                delay_context_projection = getattr(
                    adapter, "delay_context_projection", None
                )
                if delay_context_projection is None:
                    raise ValueError(
                        "--init-e2e-adapter-mismatch delay-context requires "
                        "--semantic-adapter-kind residual_delay_tokens and "
                        "--semantic-preserve-delay-context."
                    )
                logger.info(
                    "Retained zero-output explicit delay-context projection with %d parameters.",
                    sum(
                        param.numel()
                        for param in delay_context_projection.parameters()
                    ),
                )
            elif adapter_mismatch == "direct-identity":
                semantic_mapper = getattr(adapter, "semantic_mapper", None)
                if semantic_mapper is None:
                    raise ValueError(
                        "--init-e2e-adapter-mismatch direct-identity requires "
                        "--semantic-adapter-kind residual_identity."
                    )
                logger.info(
                    "Loaded the pretrained context net behind a zero-output direct semantic residual (%d parameters).",
                    sum(param.numel() for param in semantic_mapper.parameters()),
                )
        if adapter_state is not None:
            loaded.append("adapter_state_dict")

    if not loaded:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has neither loadable model nor adapter state. "
            f"Available keys: {sorted(ckpt.keys())}"
        )
    logger.info("Initialized semantic oracle state from %s: %s", checkpoint_path, ", ".join(loaded))
    return {"checkpoint": checkpoint_path, "loaded": loaded}


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

    # Array jobs may populate the same cache concurrently.  A fixed `.tmp`
    # filename lets one worker unlink or replace another worker's memmap.
    # Give every writer a private temporary path, then publish atomically.
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
        delete=False,
    )
    tmp_path = Path(handle.name)
    handle.close()
    logger.info(
        "Encoding target T5 latents to cache %s shape=%s dtype=%s",
        cache_path,
        expected_shape,
        np.dtype(storage_dtype).name,
    )
    try:
        mmap = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=storage_dtype, shape=expected_shape
        )
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
    finally:
        tmp_path.unlink(missing_ok=True)

    meta_payload = json.dumps(
        {
            "path": str(cache_path),
            "shape": expected_shape,
            "dtype": np.dtype(storage_dtype).name,
            "metadata": metadata,
        },
        indent=2,
    )
    meta_handle = tempfile.NamedTemporaryFile(
        prefix=f".{meta_path.name}.",
        suffix=".tmp",
        dir=meta_path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    meta_tmp_path = Path(meta_handle.name)
    try:
        meta_handle.write(meta_payload)
        meta_handle.flush()
        meta_handle.close()
        os.replace(meta_tmp_path, meta_path)
    finally:
        if not meta_handle.closed:
            meta_handle.close()
        meta_tmp_path.unlink(missing_ok=True)
    return np.load(cache_path, mmap_mode="r")


@torch.no_grad()
def frozen_flow_target_latent_memmap(
    *,
    cache_path: Path,
    model: nn.Module,
    adapter: nn.Module,
    semantic_vectors: torch.Tensor,
    target_length: int,
    context_length: int,
    latent_dim: int,
    config,
    sampling_steps: int,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.memmap:
    """Cache target latents produced by the frozen deployment flow from noise."""

    expected_shape = (
        int(semantic_vectors.shape[0]),
        int(target_length),
        int(latent_dim),
    )
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if tuple(cached.shape) != expected_shape:
            raise ValueError(
                f"Frozen-flow latent cache shape mismatch: {cached.shape} != {expected_shape}"
            )
        logger.info("Using frozen-flow target latent cache %s shape=%s", cache_path, cached.shape)
        return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npy")
    cached = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[int(sampling_steps)],
        cfgs=[float(cfg_scale)],
        self_cond_cfg_scales=[float(self_cond_cfg_scale)],
        time_schedule=config.time_schedule,
    )
    t_steps = get_sampling_steps(
        n_steps=int(sampling_steps),
        time_schedule=config.time_schedule,
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(
        device=device.type if device.type == "cuda" else "cpu"
    ).manual_seed(int(seed))
    model.eval()
    adapter.eval()
    use_batch_size = max(1, int(batch_size))
    try:
        for start in range(0, semantic_vectors.shape[0], use_batch_size):
            stop = min(start + use_batch_size, semantic_vectors.shape[0])
            context, context_mask = adapter(
                semantic_vectors[start:stop].to(device=device, dtype=torch.float32)
            )
            if context.shape[1] != context_length:
                raise ValueError(
                    f"Frozen-flow context length {context.shape[1]} != {context_length}"
                )
            zeros_target = torch.zeros(
                (context.shape[0], target_length, context.shape[-1]),
                dtype=context.dtype,
                device=device,
            )
            cond_seq = torch.cat([context, zeros_target], dim=1)
            cond_mask = torch.cat(
                [
                    context_mask.to(device=device, dtype=context.dtype),
                    torch.zeros(
                        (context.shape[0], target_length),
                        dtype=context.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            z = torch.randn(
                cond_seq.shape,
                generator=generator,
                device=device,
                dtype=context.dtype,
            ) * config.denoiser_noise_scale
            latent = _generate_samples_single_batch(
                model=model,
                generator=generator,
                z=z,
                t_steps=t_steps.to(dtype=context.dtype),
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
                config=config,
                sampling_config=sampling_config,
                cfg_scale=float(cfg_scale),
                self_cond_cfg_scale=float(self_cond_cfg_scale),
            )
            cached[start:stop] = (
                latent[:, context_length:].detach().cpu().to(torch.float16).numpy()
            )
            if start == 0 or stop == semantic_vectors.shape[0] or stop % (use_batch_size * 20) == 0:
                logger.info(
                    "cached frozen-flow target latents rows %d:%d / %d",
                    start,
                    stop,
                    semantic_vectors.shape[0],
                )
        cached.flush()
        os.replace(temporary_path, cache_path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    logger.info("Saved frozen-flow target latent cache %s", cache_path)
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
    text_semantic_vectors = semantic_vectors.clone()
    sentences = _strings(data[args.sentence_key])
    brain_input_summary = None
    adapter_input_dim = int(semantic_vectors.shape[-1])
    if args.brain_input_npz:
        if not args.brain_model_checkpoint:
            raise ValueError("--brain-model-checkpoint is required with --brain-input-npz.")
        brain_data = np.load(args.brain_input_npz, allow_pickle=True)
        for key in (args.brain_train_key, args.brain_val_key):
            if key not in brain_data:
                raise KeyError(
                    f"{args.brain_input_npz} is missing {key!r}; available={list(brain_data.files)}"
                )
        train_brain = np.asarray(brain_data[args.brain_train_key], dtype=np.float32)
        val_brain = np.asarray(brain_data[args.brain_val_key], dtype=np.float32)
        if train_brain.ndim != 2 or val_brain.ndim != 2:
            raise ValueError(
                f"Brain train/val arrays must be 2-D, got {train_brain.shape}/{val_brain.shape}."
            )
        if train_brain.shape[1] != val_brain.shape[1]:
            raise ValueError(
                f"Brain train/val dimensions disagree: {train_brain.shape}/{val_brain.shape}."
            )
        combined_brain = np.concatenate([train_brain, val_brain], axis=0)
        if combined_brain.shape[0] != len(sentences):
            raise ValueError(
                "Raw brain rows must exactly match the text NPZ rows: "
                f"brain={combined_brain.shape[0]} text={len(sentences)}."
            )
        metadata_pairs = (
            ("story", "train_story", "val_story", True),
            ("start_tr", "train_start_tr", "val_start_tr", False),
            ("stop_tr", "train_stop_tr", "val_stop_tr", False),
        )
        for text_key, brain_train_metadata_key, brain_val_metadata_key, is_string in metadata_pairs:
            if text_key not in data:
                continue
            if brain_train_metadata_key not in brain_data or brain_val_metadata_key not in brain_data:
                raise KeyError(
                    f"Cannot verify raw-brain alignment: missing {brain_train_metadata_key!r} "
                    f"or {brain_val_metadata_key!r} in {args.brain_input_npz}."
                )
            if is_string:
                expected_metadata = _strings(data[text_key])
                actual_metadata = _strings(
                    np.concatenate(
                        [brain_data[brain_train_metadata_key], brain_data[brain_val_metadata_key]]
                    )
                )
                matches = expected_metadata == actual_metadata
            else:
                expected_metadata = np.asarray(data[text_key])
                actual_metadata = np.concatenate(
                    [brain_data[brain_train_metadata_key], brain_data[brain_val_metadata_key]]
                )
                matches = bool(np.array_equal(expected_metadata, actual_metadata))
            if not matches:
                raise ValueError(
                    f"Raw-brain alignment check failed for metadata field {text_key!r}."
                )
        normalized_brain_val_roll = (
            int(args.brain_val_roll) % int(val_brain.shape[0])
            if val_brain.shape[0] > 0
            else 0
        )
        if normalized_brain_val_roll:
            val_brain = np.roll(
                val_brain,
                shift=normalized_brain_val_roll,
                axis=0,
            ).copy()
            combined_brain = np.concatenate([train_brain, val_brain], axis=0)
            logger.warning(
                "Applied held-out brain derangement roll=%d after raw-brain alignment checks.",
                normalized_brain_val_roll,
            )
        semantic_vectors = torch.as_tensor(combined_brain, dtype=torch.float32)
        adapter_input_dim = int(args.brain_semantic_output_dim)
        brain_input_summary = {
            "path": args.brain_input_npz,
            "train_shape": list(train_brain.shape),
            "val_shape": list(val_brain.shape),
            "combined_shape": list(combined_brain.shape),
            "semantic_output_dim": adapter_input_dim,
            "val_roll": normalized_brain_val_roll,
        }
        logger.info(
            "Using raw brain inputs from %s train=%s val=%s; semantic adapter input_dim=%d",
            args.brain_input_npz,
            train_brain.shape,
            val_brain.shape,
            adapter_input_dim,
        )
    total_n = semantic_vectors.shape[0] if args.num_examples <= 0 else min(args.num_examples, semantic_vectors.shape[0])
    if args.val_num_examples < 0:
        raise ValueError("--val-num-examples must be non-negative.")
    val_n = min(args.val_num_examples, max(0, total_n - 1))
    train_split_n = total_n - val_n
    if train_split_n <= 0:
        raise ValueError(f"Need at least one training example after holdout; got total_n={total_n}, val_n={val_n}.")
    semantic_vectors = semantic_vectors[:total_n]
    sentences = sentences[:total_n]
    normalized_semantic_val_roll = (
        int(args.semantic_val_roll) % int(val_n) if val_n > 0 else 0
    )
    if normalized_semantic_val_roll:
        semantic_vectors = semantic_vectors.clone()
        semantic_vectors[train_split_n:total_n] = torch.roll(
            semantic_vectors[train_split_n:total_n],
            shifts=normalized_semantic_val_roll,
            dims=0,
        )
        logger.warning(
            "Applied held-out semantic-condition derangement roll=%d; train rows and targets unchanged.",
            normalized_semantic_val_roll,
        )
    semantic_alignment_targets = None
    if args.semantic_alignment_target_npz:
        alignment_data = np.load(args.semantic_alignment_target_npz, allow_pickle=True)
        if args.semantic_alignment_target_key not in alignment_data:
            raise KeyError(
                f"{args.semantic_alignment_target_npz} is missing "
                f"{args.semantic_alignment_target_key!r}"
            )
        alignment_values = np.asarray(
            alignment_data[args.semantic_alignment_target_key],
            dtype=np.float32,
        )
        if alignment_values.ndim != 2 or alignment_values.shape[0] < total_n:
            raise ValueError(
                "Semantic alignment target shape must be [N, D] with at least total_n rows; "
                f"got {alignment_values.shape}, total_n={total_n}."
            )
        if args.semantic_alignment_target_sentence_key in alignment_data:
            alignment_sentences = _strings(
                alignment_data[args.semantic_alignment_target_sentence_key][:total_n]
            )
            if alignment_sentences != sentences:
                mismatch = next(
                    index
                    for index, (source, target) in enumerate(zip(sentences, alignment_sentences))
                    if source != target
                )
                mismatch_message = (
                    f"Semantic alignment target sentences differ at row {mismatch}: "
                    f"source={sentences[mismatch]!r} target={alignment_sentences[mismatch]!r}"
                )
                if not args.allow_semantic_alignment_sentence_mismatch:
                    raise ValueError(mismatch_message)
                logger.warning(
                    "%s. Continuing because the auxiliary temporal-target override was explicit.",
                    mismatch_message,
                )
        semantic_alignment_targets = torch.as_tensor(
            alignment_values[:total_n],
            dtype=torch.float32,
        )
        logger.info(
            "Loaded semantic alignment targets from %s key=%s shape=%s",
            args.semantic_alignment_target_npz,
            args.semantic_alignment_target_key,
            tuple(semantic_alignment_targets.shape),
        )
    semantic_objective_weight = (
        args.semantic_alignment_loss_weight
        + args.semantic_contrastive_loss_weight
        + args.semantic_context_loss_weight
        + args.semantic_content_loss_weight
        + args.semantic_raw_anchor_loss_weight
        + args.semantic_geometry_loss_weight
        + args.semantic_rank_distill_loss_weight
    )
    exact_target_objective_weight = (
        args.semantic_alignment_loss_weight
        + args.semantic_contrastive_loss_weight
        + args.semantic_context_loss_weight
        + args.semantic_content_loss_weight
        + args.semantic_rank_distill_loss_weight
    )
    if exact_target_objective_weight > 0.0 and semantic_alignment_targets is None:
        raise ValueError(
            "A semantic alignment/contrastive/context loss requires "
            "--semantic-alignment-target-npz."
        )
    condition_sources = None
    if args.train_condition_sources:
        if args.condition_source_key not in data:
            raise KeyError(
                f"{args.npz_path} is missing condition-source key {args.condition_source_key!r}; "
                f"available={list(data.files)}"
            )
        condition_sources = _strings(data[args.condition_source_key][:total_n])
    train_indices = select_train_indices(
        condition_sources,
        train_split_n=train_split_n,
        include_sources=args.train_condition_sources,
    )
    train_n = int(train_indices.numel())
    semantic_content_vocabulary: list[str] = []
    semantic_content_directions = None
    semantic_content_targets = None
    if args.semantic_content_loss_weight > 0.0:
        if semantic_alignment_targets is None:
            raise ValueError("Semantic content loss requires --semantic-alignment-target-npz.")
        (
            semantic_content_vocabulary,
            semantic_content_directions,
            semantic_content_targets,
        ) = build_semantic_content_prototypes(
            sentences,
            semantic_alignment_targets,
            train_indices,
            min_frequency=args.semantic_content_min_frequency,
            max_vocabulary=args.semantic_content_max_vocabulary,
        )
        logger.info(
            "Built train-only semantic content objective vocabulary=%d positive_density=%.6f",
            len(semantic_content_vocabulary),
            float(semantic_content_targets.index_select(0, train_indices).float().mean()),
        )
    balanced_train_pools = None
    if args.train_balance_key:
        if args.train_balance_key not in data:
            raise KeyError(
                f"{args.npz_path} is missing train-balance key {args.train_balance_key!r}; "
                f"available={list(data.files)}"
            )
        balance_labels = _strings(data[args.train_balance_key][:total_n])
        balanced_train_pools = build_balanced_index_pools(balance_labels, train_indices)
        logger.info(
            "Enabled uniform-%s training over %d groups (selected rows per group: min=%d max=%d)",
            args.train_balance_key,
            len(balanced_train_pools),
            min(len(pool) for pool in balanced_train_pools),
            max(len(pool) for pool in balanced_train_pools),
        )
    eval_pool_indices = (
        torch.arange(train_split_n, total_n, dtype=torch.long)
        if val_n > 0
        else train_indices
    )
    eval_split = "val" if val_n > 0 else "train"

    steps_per_epoch = int(np.ceil(train_n / args.batch_size))
    if args.epochs > 0:
        args.steps = int(np.ceil(steps_per_epoch * args.epochs))
    logger.info(
        "Loaded total_n=%d train_split_n=%d selected_train_n=%d train_sources=%s val_n=%d "
        "eval_split=%s semantic=%s batch_size=%d steps=%d epochs=%.2f",
        total_n,
        train_split_n,
        train_n,
        args.train_condition_sources,
        val_n,
        eval_split,
        tuple(semantic_vectors.shape),
        args.batch_size,
        args.steps,
        args.steps / max(1, steps_per_epoch),
    )

    dascoli_simulation: SimulatedDAscoliSentences | None = None
    dascoli_source_summary = None
    external_eval_source = False
    if args.sentence_source_eval_json and not args.dascoli_simulated_source:
        raise ValueError(
            "--sentence-source-eval-json currently requires --dascoli-simulated-source "
            "to provide leakage-safe corrupted training proposals."
        )
    if args.dascoli_simulated_source:
        if args.dascoli_expected_words <= 0:
            raise ValueError("--dascoli-expected-words must be positive.")
        if not 0.0 <= args.dascoli_global_trust <= 1.0:
            raise ValueError("--dascoli-global-trust must be in [0, 1].")
        if not 0.0 <= args.dascoli_confidence_floor <= 1.0:
            raise ValueError("--dascoli-confidence-floor must be in [0, 1].")
        if not 0.0 <= args.dascoli_eval_semantic_scale <= 1.0:
            raise ValueError("--dascoli-eval-semantic-scale must be in [0, 1].")
        if not 0.0 <= args.sentence_source_flow_end_time <= 1.0:
            raise ValueError("--sentence-source-flow-end-time must be in [0, 1].")
        if args.sentence_source_copy_confidence_threshold > 1.0:
            raise ValueError(
                "--sentence-source-copy-confidence-threshold must be negative or in [0, 1]."
            )
        if not 0.0 <= args.dascoli_source_dropout_prob <= 1.0:
            raise ValueError("--dascoli-source-dropout-prob must be in [0, 1].")
        if not 0.0 <= args.sentence_source_eval_confidence <= 1.0:
            raise ValueError("--sentence-source-eval-confidence must be in [0, 1].")
        selected_train_sentences = [sentences[index] for index in train_indices.tolist()]
        position_vocabulary, global_vocabulary = build_position_vocabulary(
            selected_train_sentences,
            expected_words=args.dascoli_expected_words,
        )
        train_simulation = simulate_dascoli_sentences(
            sentences[:train_split_n],
            position_vocabulary=position_vocabulary,
            global_vocabulary=global_vocabulary,
            accuracies=args.dascoli_train_accuracies,
            seed=args.seed + 70_001,
        )
        if val_n > 0:
            if args.sentence_source_eval_json:
                with Path(args.sentence_source_eval_json).open("r", encoding="utf-8") as handle:
                    external_payload = json.load(handle)
                validation_simulation = external_sentence_source(
                    external_payload,
                    expected_targets=sentences[train_split_n:total_n],
                    expected_words=args.dascoli_expected_words,
                    default_confidence=args.sentence_source_eval_confidence,
                    confidence_field=args.sentence_source_eval_confidence_field,
                )
                external_eval_source = True
            else:
                validation_simulation = simulate_dascoli_sentences(
                    sentences[train_split_n:total_n],
                    position_vocabulary=position_vocabulary,
                    global_vocabulary=global_vocabulary,
                    accuracies=[args.dascoli_eval_accuracy],
                    seed=args.seed + 70_003,
                )
            simulation_width = max(
                train_simulation.word_confidence.shape[1],
                validation_simulation.word_confidence.shape[1],
            )

            def pad_simulation_width(
                simulation: SimulatedDAscoliSentences,
            ) -> SimulatedDAscoliSentences:
                padding = simulation_width - simulation.word_confidence.shape[1]
                if padding <= 0:
                    return simulation
                return SimulatedDAscoliSentences(
                    sentences=simulation.sentences,
                    word_confidence=F.pad(simulation.word_confidence, (0, padding)),
                    correct_word_mask=F.pad(
                        simulation.correct_word_mask, (0, padding), value=False
                    ),
                    row_accuracy=simulation.row_accuracy,
                )

            train_simulation = pad_simulation_width(train_simulation)
            validation_simulation = pad_simulation_width(validation_simulation)
            dascoli_simulation = SimulatedDAscoliSentences(
                sentences=train_simulation.sentences + validation_simulation.sentences,
                word_confidence=torch.cat(
                    [train_simulation.word_confidence, validation_simulation.word_confidence], dim=0
                ),
                correct_word_mask=torch.cat(
                    [train_simulation.correct_word_mask, validation_simulation.correct_word_mask], dim=0
                ),
                row_accuracy=torch.cat(
                    [train_simulation.row_accuracy, validation_simulation.row_accuracy], dim=0
                ),
            )
        else:
            dascoli_simulation = train_simulation
        train_correct = dascoli_simulation.correct_word_mask[:train_split_n]
        eval_correct = dascoli_simulation.correct_word_mask[train_split_n:total_n]
        dascoli_source_summary = {
            "role": (
                "external_validation_sentence_proposal"
                if external_eval_source
                else "target_derived_validation_ceiling"
            ),
            "expected_words": args.dascoli_expected_words,
            "train_accuracies": list(args.dascoli_train_accuracies),
            "eval_accuracy_probability": args.dascoli_eval_accuracy,
            "eval_mode": args.dascoli_eval_mode,
            "global_trust": args.dascoli_global_trust,
            "confidence_floor": args.dascoli_confidence_floor,
            "eval_semantic_scale": args.dascoli_eval_semantic_scale,
            "source_flow_end_time": args.sentence_source_flow_end_time,
            "source_copy_confidence_threshold": (
                args.sentence_source_copy_confidence_threshold
                if args.sentence_source_copy_confidence_threshold >= 0.0
                else None
            ),
            "source_copy_function_words": args.sentence_source_copy_function_words,
            "source_dropout_probability": args.dascoli_source_dropout_prob,
            "external_eval_source_json": (
                str(Path(args.sentence_source_eval_json).resolve())
                if external_eval_source
                else None
            ),
            "external_eval_default_word_confidence": (
                args.sentence_source_eval_confidence if external_eval_source else None
            ),
            "external_eval_confidence_field": (
                args.sentence_source_eval_confidence_field if external_eval_source else None
            ),
            "validation_proposals_use_target_text": not external_eval_source,
            "train_realized_position_accuracy": float(train_correct.float().mean()),
            "eval_realized_position_accuracy": (
                float(eval_correct.float().mean()) if eval_correct.numel() else None
            ),
            "train_only_position_vocabulary_sizes": [len(words) for words in position_vocabulary],
            "train_only_global_vocabulary_size": len(global_vocabulary),
        }
        np.savez_compressed(
            output_dir / "dascoli_simulation.npz",
            sentence=np.asarray(dascoli_simulation.sentences, dtype=object),
            word_confidence=dascoli_simulation.word_confidence.numpy(),
            correct_word_mask=dascoli_simulation.correct_word_mask.numpy(),
            row_accuracy=dascoli_simulation.row_accuracy.numpy(),
            split=np.asarray(
                ["train"] * train_split_n + [eval_split] * val_n,
                dtype=object,
            ),
        )
        logger.warning(
            "Enabled sentence-source flow: train_position_accuracy=%.3f "
            "eval_position_accuracy=%s mode=%s role=%s.",
            dascoli_source_summary["train_realized_position_accuracy"],
            dascoli_source_summary["eval_realized_position_accuracy"],
            args.dascoli_eval_mode,
            dascoli_source_summary["role"],
        )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask = tokenize_sentences(tokenizer, sentences)
    target_length = input_ids.shape[1]
    source_input_ids = None
    source_attention_mask = None
    source_token_confidence = None
    source_copy_token_confidence = None
    if dascoli_simulation is not None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("D'Ascoli confidence alignment requires a fast tokenizer.")
        preliminary_source_ids, preliminary_source_mask = tokenize_sentences(
            tokenizer, dascoli_simulation.sentences
        )
        if preliminary_source_ids.shape[1] > target_length:
            expanded_target_length = int(preliminary_source_ids.shape[1])
            padding = expanded_target_length - target_length
            input_ids = F.pad(
                input_ids,
                (0, padding),
                value=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
            attention_mask = F.pad(attention_mask, (0, padding), value=0)
            target_length = expanded_target_length
            logger.warning(
                "Expanded target_length to %d to preserve every sentence-source token.",
                target_length,
            )
        source_token_lengths = preliminary_source_mask.sum(dim=1)
        truncated_source_rows = int((source_token_lengths > target_length).sum())
        dascoli_source_summary["source_rows_truncated_to_target_length"] = truncated_source_rows
        dascoli_source_summary["source_untruncated_max_tokens"] = int(
            preliminary_source_ids.shape[1]
        )
        if truncated_source_rows:
            logger.warning(
                "Truncating %d/%d simulated source rows to audited target_length=%d "
                "(source maximum=%d) to preserve the ELF-B validation contract.",
                truncated_source_rows,
                total_n,
                target_length,
                preliminary_source_ids.shape[1],
            )
        encoded_source = tokenizer(
            dascoli_simulation.sentences,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=target_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        source_input_ids = encoded_source["input_ids"].to(torch.long)
        source_attention_mask = encoded_source["attention_mask"].to(torch.long)
        source_raw_token_confidence = align_word_confidence_to_tokens(
            sentences=dascoli_simulation.sentences,
            word_confidence=dascoli_simulation.word_confidence,
            offset_mapping=encoded_source["offset_mapping"],
            attention_mask=source_attention_mask,
        )
        source_copy_token_confidence = source_raw_token_confidence
        if args.sentence_source_copy_function_words:
            source_copy_token_confidence = force_vocabulary_token_confidence(
                sentences=dascoli_simulation.sentences,
                token_confidence=source_raw_token_confidence,
                offset_mapping=encoded_source["offset_mapping"],
                attention_mask=source_attention_mask,
                preserved_words=set(_CONTENT_WORD_STOPWORDS),
            )
        source_token_confidence = transform_token_confidence(
            source_raw_token_confidence,
            source_attention_mask,
            global_trust=args.dascoli_global_trust,
            confidence_floor=args.dascoli_confidence_floor,
        )
    if args.eval_target_length < 0 or args.eval_target_length > target_length:
        raise ValueError(
            f"eval target length must be in [0, {target_length}], got "
            f"{args.eval_target_length}."
        )
    eval_generation_target_length = (
        int(args.eval_target_length)
        if args.eval_target_length > 0
        else int(target_length)
    )
    if eval_generation_target_length != target_length:
        logger.info(
            "Validation generation uses target canvas %d while the model/training "
            "canvas remains %d.",
            eval_generation_target_length,
            target_length,
        )
    decoder_position_logit_bias = build_decoder_position_logit_bias(
        input_ids,
        attention_mask,
        train_indices,
        vocabulary_size=len(tokenizer),
        strength=args.decoder_position_prior_strength,
        smoothing=args.decoder_position_prior_smoothing,
        clip=args.decoder_position_prior_clip,
    )
    if decoder_position_logit_bias is not None:
        logger.info(
            "Enabled train-only decoder position prior strength=%.4f smoothing=%.4f "
            "clip=%.3f shape=%s.",
            args.decoder_position_prior_strength,
            args.decoder_position_prior_smoothing,
            args.decoder_position_prior_clip,
            tuple(decoder_position_logit_bias.shape),
        )
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
    lora_layers: list[str] = []
    if args.elf_lora_rank > 0:
        lora_layers = inject_elf_lora(
            model,
            rank=args.elf_lora_rank,
            alpha=args.elf_lora_alpha,
            dropout=args.elf_lora_dropout,
            targets=args.elf_lora_targets.split(","),
            last_n_blocks=args.elf_lora_last_n_blocks,
        )
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        logger.info(
            "Injected ELF LoRA into %d linear layers; trainable LoRA parameters=%d",
            len(lora_layers),
            lora_parameter_count(model),
        )
    elif args.freeze_elf:
        for param in model.parameters():
            param.requires_grad_(False)
        trainable_names = []
    else:
        trainable_names = freeze_for_toy_tuning(model, args.last_n_blocks)
    brain_cross_attention_blocks: list[int] = []
    if args.elf_brain_cross_attention_last_n_blocks > 0:
        brain_cross_attention_blocks = model.enable_brain_cross_attention(
            condition_length=args.context_length,
            last_n_blocks=args.elf_brain_cross_attention_last_n_blocks,
            num_heads=(args.elf_brain_cross_attention_heads or None),
            attn_drop=args.elf_brain_cross_attention_dropout,
            proj_drop=args.elf_brain_cross_attention_dropout,
            decoder_only=args.elf_brain_cross_attention_decoder_only,
        )
        logger.info(
            "Enabled zero-init ELF brain cross-attention in blocks=%s parameters=%d decoder_only=%s",
            brain_cross_attention_blocks,
            sum(
                parameter.numel()
                for module in model.brain_cross_attention.values()
                for parameter in module.parameters()
            ),
            args.elf_brain_cross_attention_decoder_only,
        )
    if args.elf_brain_cross_attention_only:
        if not brain_cross_attention_blocks:
            raise ValueError(
                "--elf-brain-cross-attention-only requires a positive "
                "--elf-brain-cross-attention-last-n-blocks."
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in model.brain_cross_attention.values():
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        # train_step uses this marker to disable stochastic behavior in the
        # frozen ELF path while preserving gradients through the new adapters.
        model.brain_cross_attention_only = True
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    logger.info("Trainable ELF parameter groups: %d", len(trainable_names))
    logger.info("Trainable ELF parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    adapter = build_semantic_adapter(
        args=args,
        input_dim=adapter_input_dim,
        context_dim=encoder_config.d_model,
    ).to(device)
    init_summary = None
    if args.init_e2e_checkpoint:
        init_summary = load_e2e_initialization(
            model,
            adapter,
            args.init_e2e_checkpoint,
            device,
            adapter_mismatch=args.init_e2e_adapter_mismatch,
        )
    if args.train_semantic_input_only:
        for param in adapter.parameters():
            param.requires_grad_(False)
        input_projection = getattr(adapter, "input_projection", None)
        fusion_logits = getattr(adapter, "fusion_logits", None)
        semantic_mapper = getattr(adapter, "semantic_mapper", None)
        delay_context_projection = getattr(adapter, "delay_context_projection", None)
        if semantic_mapper is not None:
            for param in semantic_mapper.parameters():
                param.requires_grad_(True)
            if delay_context_projection is not None:
                for param in delay_context_projection.parameters():
                    param.requires_grad_(True)
            logger.info(
                "Semantic input-only training: unfroze residual mapper with %d parameters.",
                sum(param.numel() for param in semantic_mapper.parameters())
                + (
                    sum(param.numel() for param in delay_context_projection.parameters())
                    if delay_context_projection is not None
                    else 0
                ),
            )
        elif input_projection is not None:
            for param in input_projection.parameters():
                param.requires_grad_(True)
            logger.info(
                "Semantic input-only training: unfroze input projection with %d parameters.",
                sum(param.numel() for param in input_projection.parameters()),
            )
        elif fusion_logits is not None:
            fusion_logits.requires_grad_(True)
            logger.info(
                "Semantic input-only training: unfroze %d delay-fusion logits.",
                fusion_logits.numel(),
            )
        else:
            raise ValueError(
                "--train-semantic-input-only requires a flat adapter with an input projection "
                "or --semantic-adapter-kind delay_fusion/residual_mlp."
            )
    brain_trainable_names = None
    if args.semantic_ordered_head_checkpoint and not args.semantic_lexical_head_checkpoint:
        raise ValueError(
            "--semantic-ordered-head-checkpoint requires "
            "--semantic-lexical-head-checkpoint."
        )
    if args.brain_input_npz and (
        args.semantic_lexical_head_checkpoint or args.semantic_ordered_head_checkpoint
    ):
        raise ValueError(
            "Use either raw-brain lexical conditioning or direct-semantic lexical "
            "conditioning, not both."
        )
    if args.brain_input_npz:
        fmri2sem = load_mri2sem_model(
            args.brain_model_checkpoint,
            input_dim=int(semantic_vectors.shape[-1]),
            output_dim=args.brain_semantic_output_dim,
            hidden_dim=args.brain_hidden_dim,
            res_blocks=args.brain_res_blocks,
            dropout=args.brain_dropout,
            projector_mismatch=args.brain_projector_mismatch,
        ).to(device)
        brain_trainable_names = configure_mri2sem_trainable(fmri2sem, args.brain_unfreeze)
        if args.brain_lexical_head_checkpoint:
            lexical_checkpoint_payload = torch.load(
                args.brain_lexical_head_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            lexical_payload = lexical_checkpoint_payload.get(
                "packaged_lexical_head_bundle", lexical_checkpoint_payload
            )
            lexical_vocabulary = list(lexical_payload.get("vocabulary", []))
            if not lexical_vocabulary:
                raise ValueError("Lexical-head checkpoint has no vocabulary.")
            if lexical_payload.get("feature") != "semantic":
                raise ValueError(
                    "Brain lexical context requires a head trained on semantic features."
                )
            lexical_input_dim = int(lexical_payload.get("input_dim", 0))
            if lexical_input_dim != args.brain_semantic_output_dim:
                raise ValueError(
                    f"Lexical-head input_dim={lexical_input_dim} does not match "
                    f"brain semantic dim={args.brain_semantic_output_dim}."
                )
            content_head = ContentLogitHead(
                input_dim=lexical_input_dim,
                hidden_dim=int(lexical_payload.get("hidden_dim", 0)),
                output_dim=len(lexical_vocabulary),
            )
            content_head.load_state_dict(lexical_payload["head_state_dict"], strict=True)
            lexical_logit_prior = lexical_payload.get("logit_prior")
            if lexical_logit_prior is not None:
                lexical_logit_prior = torch.as_tensor(
                    lexical_logit_prior, dtype=torch.float32
                )

            exact_for_prototypes = F.normalize(
                text_semantic_vectors[:total_n].detach().cpu().float(), p=2, dim=-1
            )
            if args.brain_semantic_output_dim % exact_for_prototypes.shape[1]:
                raise ValueError(
                    "Brain semantic output must contain an integer number of lexical "
                    "prototype blocks: "
                    f"prototype={exact_for_prototypes.shape[1]} brain={args.brain_semantic_output_dim}."
                )
            lexical_repeat_factor = (
                args.brain_semantic_output_dim // exact_for_prototypes.shape[1]
            )
            token_to_index = {
                token: index for index, token in enumerate(lexical_vocabulary)
            }
            lexical_sums = torch.zeros_like(
                exact_for_prototypes.new_zeros(
                    (len(lexical_vocabulary), exact_for_prototypes.shape[1])
                )
            )
            lexical_counts = torch.zeros(len(lexical_vocabulary), dtype=torch.float32)
            for row in train_indices.detach().cpu().tolist():
                for token in set(match.group(0).lower() for match in _WORD_RE.finditer(sentences[row])):
                    index = token_to_index.get(token)
                    if index is None:
                        continue
                    lexical_sums[index] += exact_for_prototypes[row]
                    lexical_counts[index] += 1.0
            global_prototype = exact_for_prototypes.index_select(
                0, train_indices.detach().cpu()
            ).mean(dim=0)
            missing_prototypes = lexical_counts == 0
            lexical_prototypes = lexical_sums / lexical_counts[:, None].clamp_min(1.0)
            lexical_prototypes[missing_prototypes] = global_prototype
            lexical_prototypes = F.normalize(lexical_prototypes, p=2, dim=-1)
            if lexical_logit_prior is None:
                # Older probe checkpoints did not persist this field. Rebuild
                # it from exactly the same leakage-safe train rows used for
                # the prototypes; validation and test rows remain untouched.
                lexical_logit_prior = lexical_counts / float(train_indices.numel())
                logger.info(
                    "Reconstructed lexical logit prior from %d training rows.",
                    train_indices.numel(),
                )
            encoded_lexical_tokens = [
                tokenizer(token, add_special_tokens=False)["input_ids"][:8]
                for token in lexical_vocabulary
            ]
            lexical_token_width = max(
                1, max((len(ids) for ids in encoded_lexical_tokens), default=0)
            )
            lexical_token_ids = torch.full(
                (len(lexical_vocabulary), lexical_token_width),
                -1,
                dtype=torch.long,
            )
            for row, ids in enumerate(encoded_lexical_tokens):
                if ids:
                    lexical_token_ids[row, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)

            ordered_head = None
            ordered_log_prior = None
            ordered_token_ids = None
            ordered_prototypes = None
            if args.brain_ordered_head_checkpoint:
                ordered_payload = torch.load(
                    args.brain_ordered_head_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
                ordered_vocabulary = list(ordered_payload.get("vocabulary", []))
                if not ordered_vocabulary:
                    raise ValueError("Ordered-head checkpoint has no vocabulary.")
                ordered_positions = int(ordered_payload.get("positions", 0))
                ordered_input_dim = int(ordered_payload.get("input_dim", 0))
                ordered_hidden_dim = int(ordered_payload.get("hidden_dim", 0))
                if ordered_positions <= 0 or ordered_input_dim <= 0:
                    raise ValueError("Ordered-head checkpoint has invalid dimensions.")
                if args.brain_semantic_output_dim % ordered_input_dim:
                    raise ValueError(
                        "Ordered-head input must divide the delayed MRI2SEM output: "
                        f"ordered={ordered_input_dim} "
                        f"MRI2SEM={args.brain_semantic_output_dim}."
                    )
                ordered_head = OrderedWordLogitHead(
                    input_dim=ordered_input_dim,
                    hidden_dim=ordered_hidden_dim,
                    positions=ordered_positions,
                    vocabulary_size=len(ordered_vocabulary),
                )
                ordered_head.load_state_dict(
                    ordered_payload["head_state_dict"], strict=True
                )
                ordered_log_prior = torch.as_tensor(
                    ordered_payload["log_prior"], dtype=torch.float32
                )
                ordered_token_to_index = {
                    token: index for index, token in enumerate(ordered_vocabulary)
                }
                ordered_sums = torch.zeros(
                    (len(ordered_vocabulary), exact_for_prototypes.shape[1]),
                    dtype=exact_for_prototypes.dtype,
                )
                ordered_counts = torch.zeros(
                    len(ordered_vocabulary), dtype=torch.float32
                )
                for row in train_indices.detach().cpu().tolist():
                    for token in set(
                        match.group(0).lower()
                        for match in _WORD_RE.finditer(sentences[row])
                    ):
                        index = ordered_token_to_index.get(token)
                        if index is None:
                            continue
                        ordered_sums[index] += exact_for_prototypes[row]
                        ordered_counts[index] += 1.0
                ordered_prototypes = ordered_sums / ordered_counts[:, None].clamp_min(1.0)
                ordered_prototypes[ordered_counts == 0] = global_prototype
                ordered_prototypes = F.normalize(
                    ordered_prototypes, p=2, dim=-1
                )
                encoded_ordered_tokens = [
                    tokenizer(token, add_special_tokens=False)["input_ids"][:8]
                    if token not in {"<pad>", "<unk>"}
                    else []
                    for token in ordered_vocabulary
                ]
                ordered_token_width = max(
                    1, max((len(ids) for ids in encoded_ordered_tokens), default=0)
                )
                ordered_token_ids = torch.full(
                    (len(ordered_vocabulary), ordered_token_width),
                    -1,
                    dtype=torch.long,
                )
                for row, ids in enumerate(encoded_ordered_tokens):
                    if ids:
                        ordered_token_ids[row, : len(ids)] = torch.as_tensor(
                            ids, dtype=torch.long
                        )
            adapter = FMRI2SEMLexicalToELFContextAdapter(
                fmri2sem,
                adapter,
                content_head=content_head,
                lexical_prototypes=lexical_prototypes,
                lexical_vocabulary=lexical_vocabulary,
                lexical_logit_prior=lexical_logit_prior,
                lexical_token_ids=lexical_token_ids,
                lexical_topk=args.brain_lexical_topk,
                lexical_temperature=args.brain_lexical_temperature,
                lexical_prior_subtraction=args.brain_lexical_prior_subtraction,
                lexical_decode_bias_strength=args.brain_lexical_decode_bias_strength,
                lexical_max_prior_probability=args.brain_lexical_max_prior_probability,
                lexical_decode_bias_once=args.brain_lexical_decode_bias_once,
                lexical_decode_bias_mode=args.brain_lexical_decode_bias_mode,
                lexical_context_mode=args.brain_lexical_context_mode,
                lexical_decode_bias_max_positions=(
                    args.brain_lexical_decode_bias_max_positions
                ),
                ordered_head=ordered_head,
                ordered_log_prior=ordered_log_prior,
                ordered_token_ids=ordered_token_ids,
                ordered_prototypes=ordered_prototypes,
                ordered_prior_subtraction=args.brain_ordered_prior_subtraction,
                ordered_min_prior_probability=(
                    args.brain_ordered_min_prior_probability
                ),
                ordered_min_margin=args.brain_ordered_min_margin,
                ordered_decode_bias_strength=(
                    args.brain_ordered_decode_bias_strength
                ),
                ordered_decode_max_positions=(
                    args.brain_ordered_decode_max_positions
                ),
            ).to(device)
            if args.init_e2e_checkpoint:
                wrapper_payload = torch.load(
                    args.init_e2e_checkpoint, map_location="cpu", weights_only=False
                )
                wrapper_state = wrapper_payload.get("adapter_state_dict", {})
                gate_candidates = [
                    value
                    for key, value in wrapper_state.items()
                    if key.endswith("lexical_context_gate")
                ]
                if gate_candidates:
                    if len(gate_candidates) != 1:
                        raise ValueError(
                            "Lexical checkpoint contains multiple context-gate candidates."
                        )
                    gate_value = gate_candidates[0]
                    if tuple(gate_value.shape) != tuple(adapter.lexical_context_gate.shape):
                        raise ValueError(
                            "Lexical context-gate checkpoint shape mismatch: "
                            f"checkpoint={tuple(gate_value.shape)} "
                            f"current={tuple(adapter.lexical_context_gate.shape)}"
                        )
                    with torch.no_grad():
                        adapter.lexical_context_gate.copy_(
                            gate_value.to(
                                device=adapter.lexical_context_gate.device,
                                dtype=adapter.lexical_context_gate.dtype,
                            )
                        )
                    logger.info(
                        "Restored trained lexical context gate from %s.",
                        args.init_e2e_checkpoint,
                    )
                ordered_gate_candidates = [
                    value
                    for key, value in wrapper_state.items()
                    if key.endswith("ordered_context_gate")
                ]
                if ordered_gate_candidates:
                    if len(ordered_gate_candidates) != 1:
                        raise ValueError(
                            "Lexical checkpoint contains multiple ordered-gate candidates."
                        )
                    ordered_gate = getattr(adapter, "ordered_context_gate", None)
                    if ordered_gate is None:
                        raise ValueError(
                            "Checkpoint has an ordered context gate but no ordered head is configured."
                        )
                    ordered_gate_value = ordered_gate_candidates[0]
                    if tuple(ordered_gate_value.shape) != tuple(ordered_gate.shape):
                        raise ValueError(
                            "Ordered context-gate checkpoint shape mismatch: "
                            f"checkpoint={tuple(ordered_gate_value.shape)} "
                            f"current={tuple(ordered_gate.shape)}"
                        )
                    with torch.no_grad():
                        ordered_gate.copy_(
                            ordered_gate_value.to(
                                device=ordered_gate.device,
                                dtype=ordered_gate.dtype,
                            )
                        )
                    logger.info(
                        "Restored trained ordered context gate from %s.",
                        args.init_e2e_checkpoint,
                    )
            logger.info(
                "Enabled frozen lexical head vocabulary=%d topk=%d prior_subtraction=%.3f "
                "decode_bias=%.3f bias_once=%s bias_mode=%s max_prior=%.3f context_mode=%s "
                "content_max_positions=%d ordered_bias=%.3f ordered_prior_subtraction=%.3f "
                "ordered_min_prior=%.4f ordered_min_margin=%.3f "
                "ordered_max_positions=%d prototype_dim=%d delayed_repeat=%d "
                "missing_prototypes=%d; "
                "trainable context gate=%d",
                len(lexical_vocabulary),
                args.brain_lexical_topk,
                args.brain_lexical_prior_subtraction,
                args.brain_lexical_decode_bias_strength,
                args.brain_lexical_decode_bias_once,
                args.brain_lexical_decode_bias_mode,
                args.brain_lexical_max_prior_probability,
                args.brain_lexical_context_mode,
                args.brain_lexical_decode_bias_max_positions,
                args.brain_ordered_decode_bias_strength,
                args.brain_ordered_prior_subtraction,
                args.brain_ordered_min_prior_probability,
                args.brain_ordered_min_margin,
                args.brain_ordered_decode_max_positions,
                exact_for_prototypes.shape[1],
                lexical_repeat_factor,
                int(missing_prototypes.sum()),
                adapter.lexical_context_gate.numel(),
            )
        else:
            adapter = FMRI2SEMToELFContextAdapter(fmri2sem, adapter).to(device)
        logger.info(
            "Wrapped raw fMRI adapter with MRI2SEM mode=%s trainable=%d/%d parameters",
            args.brain_unfreeze,
            sum(parameter.numel() for parameter in fmri2sem.parameters() if parameter.requires_grad),
            sum(parameter.numel() for parameter in fmri2sem.parameters()),
        )
    elif args.semantic_lexical_head_checkpoint:
        lexical_payload = torch.load(
            args.semantic_lexical_head_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        lexical_payload = lexical_payload.get(
            "packaged_lexical_head_bundle", lexical_payload
        )
        lexical_vocabulary = [
            str(word) for word in lexical_payload.get("vocabulary", [])
        ]
        if not lexical_vocabulary:
            raise ValueError("Direct-semantic lexical-head checkpoint has no vocabulary.")
        if lexical_payload.get("feature") != "semantic":
            raise ValueError(
                "Direct-semantic lexical conditioning requires a head trained with "
                "feature='semantic'."
            )
        lexical_input_dim = int(lexical_payload.get("input_dim", 0))
        if lexical_input_dim != adapter_input_dim:
            raise ValueError(
                f"Lexical-head input_dim={lexical_input_dim} does not match "
                f"semantic NPZ dim={adapter_input_dim}."
            )
        content_head = ContentLogitHead(
            input_dim=lexical_input_dim,
            hidden_dim=int(lexical_payload.get("hidden_dim", 0)),
            output_dim=len(lexical_vocabulary),
        )
        content_head.load_state_dict(lexical_payload["head_state_dict"], strict=True)
        lexical_logit_prior = lexical_payload.get("logit_prior")
        if lexical_logit_prior is None:
            raise ValueError(
                "Direct-semantic lexical-head checkpoint lacks its train-only logit prior."
            )
        lexical_logit_prior = torch.as_tensor(
            lexical_logit_prior, dtype=torch.float32
        )

        # These prototypes are required by the shared lexical wrapper, but a
        # pure decoder-bias evaluation keeps the zero-initialized context gate
        # frozen. Build them only from the already-selected training rows so a
        # future gate ablation remains leakage-safe as well.
        prototype_semantic = F.normalize(
            text_semantic_vectors[:total_n].detach().cpu().float(), p=2, dim=-1
        )
        token_to_index = {
            token: index for index, token in enumerate(lexical_vocabulary)
        }
        lexical_sums = torch.zeros(
            (len(lexical_vocabulary), prototype_semantic.shape[1]),
            dtype=torch.float32,
        )
        lexical_counts = torch.zeros(
            len(lexical_vocabulary), dtype=torch.float32
        )
        for row in train_indices.detach().cpu().tolist():
            row_words = {
                match.group(0).lower() for match in _WORD_RE.finditer(sentences[row])
            }
            for token in row_words:
                index = token_to_index.get(token)
                if index is None:
                    continue
                lexical_sums[index] += prototype_semantic[row]
                lexical_counts[index] += 1.0
        train_global = prototype_semantic.index_select(
            0, train_indices.detach().cpu()
        ).mean(dim=0)
        lexical_prototypes = lexical_sums / lexical_counts[:, None].clamp_min(1.0)
        lexical_prototypes[lexical_counts == 0] = train_global
        lexical_prototypes = F.normalize(lexical_prototypes, p=2, dim=-1)

        encoded_lexical_tokens = [
            tokenizer(token, add_special_tokens=False)["input_ids"][:8]
            for token in lexical_vocabulary
        ]
        lexical_token_width = max(
            1, max((len(ids) for ids in encoded_lexical_tokens), default=0)
        )
        lexical_token_ids = torch.full(
            (len(lexical_vocabulary), lexical_token_width),
            -1,
            dtype=torch.long,
        )
        for row, token_ids in enumerate(encoded_lexical_tokens):
            if token_ids:
                lexical_token_ids[row, : len(token_ids)] = torch.as_tensor(
                    token_ids, dtype=torch.long
                )

        ordered_head = None
        ordered_log_prior = None
        ordered_token_ids = None
        ordered_prototypes = None
        if args.semantic_ordered_head_checkpoint:
            ordered_payload = torch.load(
                args.semantic_ordered_head_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            ordered_vocabulary = [
                str(word) for word in ordered_payload.get("vocabulary", [])
            ]
            ordered_positions = int(ordered_payload.get("positions", 0))
            ordered_input_dim = int(ordered_payload.get("input_dim", 0))
            ordered_hidden_dim = int(ordered_payload.get("hidden_dim", 0))
            if not ordered_vocabulary or ordered_positions <= 0:
                raise ValueError("Direct-semantic ordered-head checkpoint is incomplete.")
            if ordered_input_dim != adapter_input_dim:
                raise ValueError(
                    f"Ordered-head input_dim={ordered_input_dim} does not match "
                    f"semantic NPZ dim={adapter_input_dim}."
                )
            ordered_head = OrderedWordLogitHead(
                input_dim=ordered_input_dim,
                hidden_dim=ordered_hidden_dim,
                positions=ordered_positions,
                vocabulary_size=len(ordered_vocabulary),
            )
            ordered_head.load_state_dict(
                ordered_payload["head_state_dict"], strict=True
            )
            ordered_log_prior = torch.as_tensor(
                ordered_payload["log_prior"], dtype=torch.float32
            )
            ordered_index = {
                token: index for index, token in enumerate(ordered_vocabulary)
            }
            ordered_sums = torch.zeros(
                (len(ordered_vocabulary), prototype_semantic.shape[1]),
                dtype=torch.float32,
            )
            ordered_counts = torch.zeros(
                len(ordered_vocabulary), dtype=torch.float32
            )
            for row in train_indices.detach().cpu().tolist():
                row_words = {
                    match.group(0).lower()
                    for match in _WORD_RE.finditer(sentences[row])
                }
                for token in row_words:
                    index = ordered_index.get(token)
                    if index is None:
                        continue
                    ordered_sums[index] += prototype_semantic[row]
                    ordered_counts[index] += 1.0
            ordered_prototypes = ordered_sums / ordered_counts[:, None].clamp_min(1.0)
            ordered_prototypes[ordered_counts == 0] = train_global
            ordered_prototypes = F.normalize(ordered_prototypes, p=2, dim=-1)
            encoded_ordered_tokens = [
                tokenizer(token, add_special_tokens=False)["input_ids"][:8]
                if token not in {"<pad>", "<unk>"}
                else []
                for token in ordered_vocabulary
            ]
            ordered_token_width = max(
                1, max((len(ids) for ids in encoded_ordered_tokens), default=0)
            )
            ordered_token_ids = torch.full(
                (len(ordered_vocabulary), ordered_token_width),
                -1,
                dtype=torch.long,
            )
            for row, token_ids in enumerate(encoded_ordered_tokens):
                if token_ids:
                    ordered_token_ids[row, : len(token_ids)] = torch.as_tensor(
                        token_ids, dtype=torch.long
                    )

        adapter = FMRI2SEMLexicalToELFContextAdapter(
            nn.Identity(),
            adapter,
            content_head=content_head,
            lexical_prototypes=lexical_prototypes,
            lexical_vocabulary=lexical_vocabulary,
            lexical_logit_prior=lexical_logit_prior,
            lexical_token_ids=lexical_token_ids,
            lexical_topk=args.brain_lexical_topk,
            lexical_temperature=args.brain_lexical_temperature,
            lexical_prior_subtraction=args.brain_lexical_prior_subtraction,
            lexical_decode_bias_strength=args.brain_lexical_decode_bias_strength,
            lexical_max_prior_probability=args.brain_lexical_max_prior_probability,
            lexical_decode_bias_once=args.brain_lexical_decode_bias_once,
            lexical_decode_bias_mode=args.brain_lexical_decode_bias_mode,
            lexical_context_mode=args.brain_lexical_context_mode,
            lexical_decode_bias_max_positions=(
                args.brain_lexical_decode_bias_max_positions
            ),
            ordered_head=ordered_head,
            ordered_log_prior=ordered_log_prior,
            ordered_token_ids=ordered_token_ids,
            ordered_prototypes=ordered_prototypes,
            ordered_prior_subtraction=args.brain_ordered_prior_subtraction,
            ordered_min_prior_probability=(
                args.brain_ordered_min_prior_probability
            ),
            ordered_min_margin=args.brain_ordered_min_margin,
            ordered_decode_bias_strength=(
                args.brain_ordered_decode_bias_strength
            ),
            ordered_decode_max_positions=(
                args.brain_ordered_decode_max_positions
            ),
        ).to(device)
        logger.info(
            "Enabled direct-semantic frozen lexical head vocabulary=%d topk=%d "
            "prior_subtraction=%.3f decode_bias=%.3f bias_once=%s bias_mode=%s "
            "max_prior=%.3f ordered=%s ordered_bias=%.3f; inference still receives "
            "only the NPZ semantic vector.",
            len(lexical_vocabulary),
            args.brain_lexical_topk,
            args.brain_lexical_prior_subtraction,
            args.brain_lexical_decode_bias_strength,
            args.brain_lexical_decode_bias_once,
            args.brain_lexical_decode_bias_mode,
            args.brain_lexical_max_prior_probability,
            bool(ordered_head is not None),
            args.brain_ordered_decode_bias_strength,
        )
    if args.freeze_semantic_adapter:
        for param in adapter.parameters():
            param.requires_grad_(False)
        logger.info("Froze the complete semantic/raw-brain adapter.")
    if args.train_lexical_context_gate_only:
        lexical_context_gate = getattr(adapter, "lexical_context_gate", None)
        if lexical_context_gate is None:
            raise ValueError(
                "--train-lexical-context-gate-only requires a lexical brain adapter; "
                "set --brain-lexical-head-checkpoint."
            )
        for param in adapter.parameters():
            param.requires_grad_(False)
        lexical_context_gate.requires_grad_(True)
        ordered_context_gate = getattr(adapter, "ordered_context_gate", None)
        if ordered_context_gate is not None:
            ordered_context_gate.requires_grad_(True)
        logger.info(
            "Lexical-context-gate-only training: unfroze %d values.",
            lexical_context_gate.numel()
            + (0 if ordered_context_gate is None else ordered_context_gate.numel()),
        )
    if args.eval_only:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in adapter.parameters():
            param.requires_grad_(False)
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

    source_latents = None
    need_source_latents = bool(
        dascoli_simulation is not None
        and (not args.eval_only or args.dascoli_eval_mode != "semantic_only")
    )
    if need_source_latents:
        logger.info("Encoding sentence-source T5 latents in memory")
        source_latents = encode_text_batched(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            device=device,
            batch_size=args.target_latent_encode_batch_size,
        ).to(torch.float16)

    if args.decoder_flow_latent_direct and not args.decoder_flow_latent_cache:
        raise ValueError(
            "--decoder-flow-latent-direct requires --decoder-flow-latent-cache."
        )
    if args.decoder_flow_latent_cache and dascoli_simulation is not None:
        raise ValueError(
            "Frozen-flow decoder latents cannot be combined with a D'Ascoli sentence source."
        )
    decoder_training_source_latents: torch.Tensor | np.memmap | None = source_latents
    if args.decoder_flow_latent_cache:
        decoder_training_source_latents = frozen_flow_target_latent_memmap(
            cache_path=Path(args.decoder_flow_latent_cache),
            model=model,
            adapter=adapter,
            # Cache only the physical training prefix. Validation ADA rows are
            # not needed to construct a decoder-training input.
            semantic_vectors=semantic_vectors[:train_split_n],
            target_length=target_length,
            context_length=args.context_length,
            latent_dim=encoder_config.d_model,
            config=config,
            sampling_steps=args.num_sampling_steps,
            cfg_scale=args.cfg_scale,
            self_cond_cfg_scale=args.self_cond_cfg_scale,
            device=device,
            batch_size=args.decoder_flow_latent_batch_size,
            seed=args.decoder_flow_latent_seed,
        )

    def get_decoder_training_source_latents(
        indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if isinstance(decoder_training_source_latents, torch.Tensor):
            return decoder_training_source_latents.index_select(0, indices)
        if isinstance(decoder_training_source_latents, np.memmap):
            return select_memmap_rows(decoder_training_source_latents, indices)
        return None

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
    decoder_token_weights = build_content_token_weights(
        tokenizer,
        sentences,
        target_ids,
        target_mask,
        content_weight=args.decoder_content_token_weight,
    )
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

    noise_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(args.seed + 17)
    order_generator = torch.Generator().manual_seed(args.seed + 29)

    def make_eval_generator() -> torch.Generator:
        # Reuse identical diffusion noise at every validation checkpoint so a
        # WER improvement reflects model changes rather than a luckier sample.
        return torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(
            args.seed + 100_003
        )
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
                "eval_generation_target_length": eval_generation_target_length,
                "num_total_examples": total_n,
                "num_train_examples": train_n,
                "train_split_boundary": train_split_n,
                "num_val_examples": val_n,
                "eval_split": eval_split,
                "train_target_mask_density": train_target_mask_density,
                "init_e2e_summary": init_summary,
                "brain_input_summary": brain_input_summary,
                "brain_trainable_names": brain_trainable_names,
                "elf_lora_layers": lora_layers,
                "dascoli_source_summary": dascoli_source_summary,
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
                "eval_generation_target_length": eval_generation_target_length,
                "num_total_examples": total_n,
                "num_train_examples": train_n,
                "train_split_boundary": train_split_n,
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
        wandb.define_metric("semantic_interface/*", step_metric="eval/epoch")

    eval_available = int(eval_pool_indices.numel())
    eval_n = eval_available if args.eval_num_examples <= 0 else min(args.eval_num_examples, eval_available)
    eval_indices = eval_pool_indices[:eval_n]
    retrieval_n = (
        0
        if args.retrieval_num_examples < 0
        else (
            eval_n
            if args.retrieval_num_examples == 0
            else min(args.retrieval_num_examples, eval_available)
        )
    )
    retrieval_indices = eval_pool_indices[:retrieval_n]
    eval_target_latents = get_target_latents(eval_indices)
    retrieval_target_latents = get_target_latents(retrieval_indices) if retrieval_n > 0 else eval_target_latents
    eval_generation_target_latents = eval_target_latents[
        :, :eval_generation_target_length
    ]
    eval_generation_target_mask = target_mask.index_select(0, eval_indices)[
        :, :eval_generation_target_length
    ]
    best_score = None

    def eval_source_arguments() -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        bool,
    ]:
        if dascoli_simulation is None or args.dascoli_eval_mode == "semantic_only":
            return None, None, None, None, None, False
        if source_latents is None or source_token_confidence is None or source_attention_mask is None:
            raise RuntimeError("D'Ascoli source tensors were not encoded.")
        latents = source_latents.index_select(0, eval_indices)[
            :, :eval_generation_target_length
        ]
        if args.dascoli_eval_mode == "joint_unweighted":
            confidence = source_attention_mask.index_select(0, eval_indices)[
                :, :eval_generation_target_length
            ].to(torch.float32)
        else:
            confidence = source_token_confidence.index_select(0, eval_indices)[
                :, :eval_generation_target_length
            ]
        copy_ids = None
        copy_mask = None
        copy_confidence = None
        if args.sentence_source_copy_confidence_threshold >= 0.0:
            if source_copy_token_confidence is None:
                raise RuntimeError("Raw sentence-source confidence was not encoded.")
            copy_ids = source_input_ids.index_select(0, eval_indices)[
                :, :eval_generation_target_length
            ]
            copy_mask = source_attention_mask.index_select(0, eval_indices)[
                :, :eval_generation_target_length
            ]
            copy_confidence = source_copy_token_confidence.index_select(0, eval_indices)[
                :, :eval_generation_target_length
            ]
        disable_semantic = args.dascoli_eval_mode == "sentence_only"
        if args.dascoli_eval_mode == "joint_shuffled":
            latents = latents.roll(1, dims=0)
            confidence = confidence.roll(1, dims=0)
            if copy_ids is not None:
                copy_ids = copy_ids.roll(1, dims=0)
                copy_mask = copy_mask.roll(1, dims=0)
                copy_confidence = copy_confidence.roll(1, dims=0)
        return latents, confidence, copy_ids, copy_mask, copy_confidence, disable_semantic

    def attach_dascoli_eval_metadata(metrics: dict) -> None:
        if dascoli_simulation is None:
            return
        eval_rows = eval_indices.tolist()
        paired_rows = eval_rows
        if args.dascoli_eval_mode == "joint_shuffled" and eval_rows:
            paired_rows = eval_rows[-1:] + eval_rows[:-1]
        paired_source_sentences = [
            dascoli_simulation.sentences[index] for index in paired_rows
        ]
        eval_target_sentences = select_strings(sentences, eval_indices)
        paired_position_scores = []
        paired_bag_f1_scores = []
        for source_sentence, target_sentence in zip(
            paired_source_sentences, eval_target_sentences
        ):
            source_words = normalized_words(source_sentence)
            target_words = normalized_words(target_sentence)
            paired_position_scores.append(
                sum(source == target for source, target in zip(source_words, target_words))
                / max(1, len(target_words))
            )
            source_counts = Counter(source_words)
            target_counts = Counter(target_words)
            overlap = sum((source_counts & target_counts).values())
            precision = overlap / max(1, len(source_words))
            recall = overlap / max(1, len(target_words))
            paired_bag_f1_scores.append(
                0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
            )
        eval_word_confidence = dascoli_simulation.word_confidence.index_select(
            0, eval_indices
        )
        paired_word_confidence = (
            eval_word_confidence.roll(1, dims=0)
            if args.dascoli_eval_mode == "joint_shuffled"
            else eval_word_confidence
        )
        sentence_source_metadata = {
            **dascoli_source_summary,
            "realized_eval_position_accuracy": float(
                dascoli_simulation.correct_word_mask.index_select(0, eval_indices).float().mean()
            ),
            "mean_eval_word_confidence": float(
                dascoli_simulation.word_confidence.index_select(0, eval_indices).mean()
            ),
            "paired_source_position_accuracy": float(np.mean(paired_position_scores)),
            "paired_source_word_f1": float(np.mean(paired_bag_f1_scores)),
            "source_sentences": paired_source_sentences,
            "unshuffled_source_sentences": [
                dascoli_simulation.sentences[index] for index in eval_rows
            ],
            "word_confidence": paired_word_confidence.tolist(),
            "unshuffled_word_confidence": eval_word_confidence.tolist(),
            "correct_word_mask": dascoli_simulation.correct_word_mask.index_select(
                0, eval_indices
            ).tolist(),
        }
        metrics["sentence_source"] = sentence_source_metadata
        if not external_eval_source:
            metrics["dascoli_simulation"] = sentence_source_metadata

    (
        eval_source_latents,
        eval_source_confidence,
        eval_source_token_ids,
        eval_source_attention_mask,
        eval_source_copy_confidence,
        eval_disable_semantic,
    ) = eval_source_arguments()

    def current_semantic_interface_metrics() -> dict[str, float] | None:
        if semantic_alignment_targets is None:
            return None
        return semantic_interface_metrics(
            adapter,
            semantic_vectors.index_select(0, eval_indices),
            semantic_alignment_targets.index_select(0, eval_indices),
            device=device,
            batch_size=args.retrieval_batch_size,
        )

    if args.eval_only:
        epoch = 0.0
        eval_semantic_predictions = project_semantic_vectors(
            adapter,
            semantic_vectors.index_select(0, eval_indices),
            device=device,
            batch_size=args.retrieval_batch_size,
        )
        np.savez_compressed(
            output_dir / "eval_semantic_predictions.npz",
            input_embeddings=eval_semantic_predictions.numpy().astype(np.float32),
            sentence=np.asarray(select_strings(sentences, eval_indices), dtype=object),
            split=np.asarray([eval_split] * eval_n, dtype=object),
            schema_json=np.asarray(
                json.dumps(
                    {
                        "role": "leakage_safe_model_predicted_semantic_vectors",
                        "source": "adapter.project_semantic(eval_inputs)",
                        "reference_text_used_for_prediction": False,
                        "num_examples": eval_n,
                        "semantic_dim": int(eval_semantic_predictions.shape[-1]),
                    }
                ),
                dtype=object,
            ),
        )
        if retrieval_n > 0:
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
            retrieval_metrics["step"] = 0
            retrieval_metrics["epoch"] = epoch
            retrieval_metrics["split"] = eval_split
            retrieval_metrics["num_train_examples"] = train_n
            retrieval_metrics["num_eval_examples"] = retrieval_n
            with (output_dir / "retrieval_step_000000.json").open("w", encoding="utf-8") as f:
                json.dump(retrieval_metrics, f, ensure_ascii=False, indent=2)
            logger.info(
                "eval-only retrieval combined_top1=%.3f combined_top5=%.3f mean_rank=%.2f",
                retrieval_metrics["combined"]["top1"],
                retrieval_metrics["combined"]["top5"],
                retrieval_metrics["combined"]["mean_rank"],
            )

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
            target_latents=(
                eval_generation_target_latents
                if args.generation_t5_retrieval
                else None
            ),
            target_mask=(
                eval_generation_target_mask
                if args.generation_t5_retrieval
                else None
            ),
            target_length=eval_generation_target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=make_eval_generator(),
            condition_source="semantic",
            condition_batch_size=args.batch_size,
            source_latents=eval_source_latents,
            source_confidence=eval_source_confidence,
            source_token_ids=eval_source_token_ids,
            source_attention_mask=eval_source_attention_mask,
            source_copy_confidence=eval_source_copy_confidence,
            source_copy_confidence_threshold=(
                args.sentence_source_copy_confidence_threshold
                if args.sentence_source_copy_confidence_threshold >= 0.0
                else None
            ),
            source_edit_latent_preservation=args.sentence_source_edit_latent_preservation,
            disable_semantic_condition=eval_disable_semantic,
            semantic_condition_scale=args.dascoli_eval_semantic_scale,
            source_flow_end_time=args.sentence_source_flow_end_time,
            decoder_position_logit_bias=decoder_position_logit_bias,
        )
        eval_metrics["step"] = 0
        eval_metrics["epoch"] = epoch
        eval_metrics["split"] = eval_split
        eval_metrics["num_train_examples"] = train_n
        eval_metrics["num_eval_examples"] = eval_n
        eval_metrics["eval_num_examples"] = eval_n
        interface_metrics = current_semantic_interface_metrics()
        if interface_metrics is not None:
            eval_metrics["semantic_interface"] = interface_metrics
        attach_dascoli_eval_metadata(eval_metrics)
        eval_metrics["eval_checkpoint_scores"] = eval_checkpoint_scores(eval_metrics)
        with (output_dir / "eval_step_000000.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
        with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

        quality = eval_metrics.get("generation_quality", {})
        retrieval = eval_metrics.get("generation_t5_retrieval", {})
        logger.info(
            "eval-only exact=%.3f well_structured=%.3f words_overlap=%.3f "
            "content_words_overlap=%.3f gen_t5_top5=%s",
            eval_metrics["exact_match"],
            quality.get("well_structured_sentence", float("nan")),
            quality.get("words_overlap", float("nan")),
            quality.get("content_words_overlap", float("nan")),
            retrieval.get("top5"),
        )
        for idx in range(min(5, eval_n)):
            logger.info("[%d] target=%r generated=%r", idx, eval_metrics["targets"][idx], eval_metrics["generated"][idx])
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
            if interface_metrics is not None:
                payload.update(
                    {f"semantic_interface/{key}": value for key, value in interface_metrics.items()}
                )
            safe_wandb_log(payload, step=0)
            run.summary["best_score"] = retrieval.get("top1", eval_metrics["exact_match"]) if retrieval else eval_metrics["exact_match"]
            run.finish()
        logger.info("Finished eval-only pass.")
        return

    model_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adapter_params = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not model_params and not adapter_params:
        raise ValueError("No trainable parameters. Use --eval-only for frozen evaluation.")
    if args.save_trainable_only_checkpoint:
        if not args.init_e2e_checkpoint:
            raise ValueError(
                "--save-trainable-only-checkpoint requires --init-e2e-checkpoint."
            )
    effective_elf_lr = args.elf_lr if args.elf_lr > 0.0 else args.lr
    if isinstance(adapter, FMRI2SEMToELFContextAdapter):
        brain_params = [
            parameter for parameter in adapter.fmri2sem.parameters() if parameter.requires_grad
        ]
        brain_param_ids = {id(parameter) for parameter in brain_params}
        nonbrain_adapter_params = [
            parameter for parameter in adapter_params if id(parameter) not in brain_param_ids
        ]
        parameter_groups = []
        if nonbrain_adapter_params:
            parameter_groups.append({"params": nonbrain_adapter_params, "lr_scale": 1.0})
        if model_params:
            parameter_groups.append(
                {
                    "params": model_params,
                    "lr_scale": effective_elf_lr / args.lr,
                }
            )
        if brain_params:
            parameter_groups.append(
                {
                    "params": brain_params,
                    "lr_scale": args.brain_lr / args.lr,
                }
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        parameter_groups = []
        if adapter_params:
            parameter_groups.append({"params": adapter_params, "lr_scale": 1.0})
        if model_params:
            parameter_groups.append(
                {
                    "params": model_params,
                    "lr_scale": effective_elf_lr / args.lr,
                }
            )
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    saved_eval_checkpoints: list[dict] = []
    eval_index = 0

    def training_checkpoint_payload(*, checkpoint_step: int, checkpoint_epoch: float, score):
        payload = {
            "step": checkpoint_step,
            "epoch": checkpoint_epoch,
            "score": score,
            "args": vars(args),
            "config": {
                key: value
                for key, value in vars(config).items()
                if isinstance(value, (str, int, float, bool, type(None), list, tuple))
            },
        }
        if args.save_trainable_only_checkpoint:
            trainable_model_names = {
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            trainable_adapter_names = {
                name for name, parameter in adapter.named_parameters() if parameter.requires_grad
            }
            model_state = model.state_dict()
            adapter_state = adapter.state_dict()
            payload["checkpoint_format"] = (
                "trainable_e2e_delta_v2"
                if trainable_adapter_names
                else "trainable_elf_delta_v1"
            )
            payload["parent_init_e2e_checkpoint"] = args.init_e2e_checkpoint
            if trainable_model_names:
                payload["trainable_model_state_dict"] = {
                    name: model_state[name].detach().cpu()
                    for name in sorted(trainable_model_names)
                }
            if trainable_adapter_names:
                payload["trainable_adapter_state_dict"] = {
                    name: adapter_state[name].detach().cpu()
                    for name in sorted(trainable_adapter_names)
                }
        else:
            payload["checkpoint_format"] = "full_v1"
            payload["model_state_dict"] = model.state_dict()
            payload["adapter_state_dict"] = adapter.state_dict()
        return payload

    for step in range(1, args.steps + 1):
        current_lr = learning_rate_for_step(
            base_lr=args.lr,
            step=step,
            total_steps=args.steps,
            schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr * parameter_group.get("lr_scale", 1.0)
        epoch = step * args.batch_size / max(1, train_n)
        if balanced_train_pools is not None:
            indices = sample_balanced_indices(
                balanced_train_pools,
                batch_size=args.batch_size,
                generator=order_generator,
            )
        else:
            batches = make_batches(train_n, args.batch_size, order_generator)
            indices = train_indices.index_select(0, batches[(step - 1) % len(batches)])
        batch_source_confidence = (
            source_token_confidence.index_select(0, indices)
            if source_token_confidence is not None
            else None
        )
        if (
            batch_source_confidence is not None
            and args.dascoli_source_dropout_prob > 0.0
        ):
            source_drop = torch.rand(
                (batch_source_confidence.shape[0], 1), generator=order_generator
            ) < args.dascoli_source_dropout_prob
            batch_source_confidence = torch.where(
                source_drop,
                torch.zeros_like(batch_source_confidence),
                batch_source_confidence,
            )
        batch = OverfitBatch(
            indices=indices,
            meg=meg.index_select(0, indices),
            meg_lengths=meg_lengths.index_select(0, indices),
            semantic_vectors=semantic_vectors.index_select(0, indices),
            subject_ids=subject_ids.index_select(0, indices),
            target_latents=get_target_latents(indices),
            target_ids=target_ids.index_select(0, indices),
            target_mask=train_target_mask.index_select(0, indices),
            target_weights=decoder_token_weights.index_select(0, indices),
            source_latents=get_decoder_training_source_latents(indices),
            source_confidence=batch_source_confidence,
        )
        active_denoiser_weight, active_decoder_weight = curriculum_loss_weights(
            epoch=epoch,
            curriculum_epochs=args.curriculum_semantic_only_epochs,
            final_denoiser_weight=args.denoiser_loss_weight,
            final_decoder_weight=args.decoder_loss_weight,
            curriculum_denoiser_weight=args.curriculum_denoiser_loss_weight,
            curriculum_decoder_weight=args.curriculum_decoder_loss_weight,
        )
        config.denoiser_loss_weight = active_denoiser_weight
        config.decoder_loss_weight = active_decoder_weight
        batch_alignment_targets = (
            semantic_alignment_targets.index_select(0, indices)
            if semantic_alignment_targets is not None
            else None
        )
        batch_content_targets = (
            semantic_content_targets.index_select(0, indices)
            if semantic_content_targets is not None
            else None
        )
        if args.semantic_objective_only:
            if batch_alignment_targets is None:
                raise ValueError("--semantic-objective-only requires semantic alignment targets.")
            metrics = semantic_objective_train_step(
                adapter=adapter,
                semantic_inputs=batch.semantic_vectors,
                exact_semantic=batch_alignment_targets,
                optimizer=optimizer,
                device=device,
                alignment_weight=args.semantic_alignment_loss_weight,
                contrastive_weight=args.semantic_contrastive_loss_weight,
                contrastive_temperature=args.semantic_contrastive_temperature,
                context_weight=args.semantic_context_loss_weight,
                content_weight=args.semantic_content_loss_weight,
                content_targets=batch_content_targets,
                content_directions=semantic_content_directions,
                content_temperature=args.semantic_content_temperature,
                content_negative_topk=args.semantic_content_negative_topk,
                raw_anchor_weight=args.semantic_raw_anchor_loss_weight,
                geometry_weight=args.semantic_geometry_loss_weight,
                rank_distill_weight=args.semantic_rank_distill_loss_weight,
                rank_distill_temperature=args.semantic_rank_distill_temperature,
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
                condition_source="semantic",
                cond_dropout_prob=args.cond_dropout_prob,
                train_timestep_mode=args.train_timestep_mode,
                train_timestep_steps=args.train_timestep_steps,
                semantic_alignment_loss_weight=args.semantic_alignment_loss_weight,
                semantic_alignment_loss_type=args.semantic_alignment_loss_type,
                semantic_alignment_targets=batch_alignment_targets,
                semantic_contrastive_loss_weight=args.semantic_contrastive_loss_weight,
                semantic_contrastive_temperature=args.semantic_contrastive_temperature,
                semantic_context_loss_weight=args.semantic_context_loss_weight,
                semantic_content_loss_weight=args.semantic_content_loss_weight,
                semantic_content_targets=batch_content_targets,
                semantic_content_directions=semantic_content_directions,
                semantic_content_temperature=args.semantic_content_temperature,
                semantic_content_negative_topk=args.semantic_content_negative_topk,
                semantic_raw_anchor_loss_weight=args.semantic_raw_anchor_loss_weight,
                semantic_geometry_loss_weight=args.semantic_geometry_loss_weight,
                semantic_rank_distill_loss_weight=args.semantic_rank_distill_loss_weight,
                semantic_rank_distill_temperature=args.semantic_rank_distill_temperature,
                decoder_content_bow_loss_weight=args.decoder_content_bow_loss_weight,
                decoder_content_precision_loss_weight=args.decoder_content_precision_loss_weight,
                decoder_content_precision_topk=args.decoder_content_precision_topk,
                decoder_condition_pairing_margin_weight=(
                    args.decoder_condition_pairing_margin_weight
                ),
                decoder_condition_pairing_margin=args.decoder_condition_pairing_margin,
            )
        config.denoiser_loss_weight = args.denoiser_loss_weight
        config.decoder_loss_weight = args.decoder_loss_weight
        if step == 1 or step % 10 == 0:
            logger.info(
                "step=%d epoch=%.2f loss=%.6f denoiser=%.6f decoder=%.6f "
                "content_bow=%.6f content_precision=%.6f pairing=%.6f sem_cos=%.4f sem_contrast=%.6f "
                "sem_context=%.6f sem_content=%.6f raw_anchor_cos=%.4f "
                "sem_geometry=%.6f rank_kd=%.6f",
                step,
                epoch,
                metrics["loss"],
                metrics["denoiser_loss"],
                metrics["decoder_loss"],
                metrics["content_bow_loss"],
                metrics["content_precision_loss"],
                metrics.get("condition_pairing_loss", 0.0),
                metrics["semantic_alignment_cosine"],
                metrics["semantic_contrastive_loss"],
                metrics["semantic_context_loss"],
                metrics["semantic_content_loss"],
                metrics["semantic_raw_anchor_cosine"],
                metrics["semantic_geometry_loss"],
                metrics["semantic_rank_distill_loss"],
            )
        if run is not None:
            safe_wandb_log(
                {
                    "train/loss": metrics["loss"],
                    "train/denoiser_loss": metrics["denoiser_loss"],
                    "train/decoder_loss": metrics["decoder_loss"],
                    "train/content_bow_loss": metrics["content_bow_loss"],
                    "train/content_bow_loss_scaled": metrics["content_bow_loss_scaled"],
                    "train/content_precision_loss": metrics["content_precision_loss"],
                    "train/content_precision_loss_scaled": metrics["content_precision_loss_scaled"],
                    "train/condition_pairing_loss": metrics.get("condition_pairing_loss", 0.0),
                    "train/condition_pairing_loss_scaled": metrics.get(
                        "condition_pairing_loss_scaled", 0.0
                    ),
                    "train/semantic_alignment_loss": metrics["semantic_alignment_loss"],
                    "train/semantic_alignment_cosine": metrics["semantic_alignment_cosine"],
                    "train/semantic_contrastive_loss": metrics["semantic_contrastive_loss"],
                    "train/semantic_context_loss": metrics["semantic_context_loss"],
                    "train/semantic_content_loss": metrics["semantic_content_loss"],
                    "train/semantic_content_loss_scaled": metrics["semantic_content_loss_scaled"],
                    "train/semantic_raw_anchor_loss": metrics["semantic_raw_anchor_loss"],
                    "train/semantic_raw_anchor_cosine": metrics["semantic_raw_anchor_cosine"],
                    "train/semantic_geometry_loss": metrics["semantic_geometry_loss"],
                    "train/semantic_rank_distill_loss": metrics["semantic_rank_distill_loss"],
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/lr": current_lr,
                    "train/denoiser_loss_weight": active_denoiser_weight,
                    "train/decoder_loss_weight": active_decoder_weight,
                },
                step=step,
            )

        # A negative --retrieval-num-examples intentionally disables this
        # expensive auxiliary evaluation.  Do not enter evaluate_retrieval
        # with an empty split: its context encoder requires a positive chunk
        # size.  Generation-time T5 retrieval in evaluate_generation remains
        # enabled independently.
        run_retrieval = retrieval_n > 0 and args.retrieval_eval_every > 0 and (
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
                safe_wandb_log(
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
            target_latents=(
                eval_generation_target_latents
                if args.generation_t5_retrieval
                else None
            ),
            target_mask=(
                eval_generation_target_mask
                if args.generation_t5_retrieval
                else None
            ),
            target_length=eval_generation_target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=make_eval_generator(),
            condition_source="semantic",
            condition_batch_size=args.batch_size,
            source_latents=eval_source_latents,
            source_confidence=eval_source_confidence,
            source_token_ids=eval_source_token_ids,
            source_attention_mask=eval_source_attention_mask,
            source_copy_confidence=eval_source_copy_confidence,
            source_copy_confidence_threshold=(
                args.sentence_source_copy_confidence_threshold
                if args.sentence_source_copy_confidence_threshold >= 0.0
                else None
            ),
            source_edit_latent_preservation=args.sentence_source_edit_latent_preservation,
            disable_semantic_condition=eval_disable_semantic,
            semantic_condition_scale=args.dascoli_eval_semantic_scale,
            source_flow_end_time=args.sentence_source_flow_end_time,
            decoder_position_logit_bias=decoder_position_logit_bias,
        )
        eval_metrics["step"] = step
        eval_metrics["epoch"] = epoch
        eval_metrics["split"] = eval_split
        eval_metrics["num_train_examples"] = train_n
        eval_metrics["num_eval_examples"] = eval_n
        eval_metrics["eval_num_examples"] = eval_n
        eval_metrics["learning_rate"] = current_lr
        interface_metrics = current_semantic_interface_metrics()
        if interface_metrics is not None:
            eval_metrics["semantic_interface"] = interface_metrics
        attach_dascoli_eval_metadata(eval_metrics)
        eval_metrics["eval_checkpoint_scores"] = eval_checkpoint_scores(eval_metrics)
        with (output_dir / f"eval_step_{step:06d}.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
        eval_index += 1
        save_eval_checkpoint(
            args=args,
            model=model,
            adapter=adapter,
            optimizer=optimizer,
            config=config,
            metrics=eval_metrics,
            eval_index=eval_index,
            saved_checkpoints=saved_eval_checkpoints,
        )
        retrieval = eval_metrics.get("generation_t5_retrieval", {})
        score = validation_checkpoint_score(eval_metrics, args.checkpoint_metric)
        if best_score is None or score > best_score:
            best_score = score
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
            if args.save_best_checkpoint:
                atomic_torch_save(
                    training_checkpoint_payload(
                        checkpoint_step=step,
                        checkpoint_epoch=epoch,
                        score=score,
                    ),
                    output_dir / "best.pt",
                )
        quality = eval_metrics.get("generation_quality", {})
        logger.info(
            "eval step=%d exact=%.3f WER=%.3f well_structured=%.3f words_overlap=%.3f content_words_overlap=%.3f gen_t5_top1=%s",
            step,
            eval_metrics["exact_match"],
            quality.get("word_error_rate", float("nan")),
            quality.get("well_structured_sentence", float("nan")),
            quality.get("words_overlap", float("nan")),
            quality.get("content_words_overlap", float("nan")),
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
            if interface_metrics is not None:
                payload.update(
                    {f"semantic_interface/{key}": value for key, value in interface_metrics.items()}
                )
            if args.wandb_sample_examples > 0:
                sample_count = min(args.wandb_sample_examples, len(eval_metrics["generated"]))
                overlap_per_sample = eval_metrics.get("word_overlap", {}).get("per_sample", [])
                content_overlap_per_sample = eval_metrics.get("word_overlap", {}).get("per_sample_content", [])
                content_overlap_counts = eval_metrics.get("word_overlap", {}).get(
                    "per_sample_content_overlap_counts",
                    [],
                )
                payload["eval/samples"] = wandb.Table(
                    columns=[
                        "index",
                        "target",
                        "generated",
                        "exact",
                        "words_overlap",
                        "content_words_overlap",
                        "content_overlap_words",
                    ],
                    data=[
                        [
                            idx,
                            eval_metrics["targets"][idx],
                            eval_metrics["generated"][idx],
                            int(eval_metrics["exact"][idx]),
                            float(overlap_per_sample[idx]) if idx < len(overlap_per_sample) else None,
                            (
                                float(content_overlap_per_sample[idx])
                                if idx < len(content_overlap_per_sample)
                                else None
                            ),
                            (
                                format_overlap_counts(content_overlap_counts[idx])
                                if idx < len(content_overlap_counts)
                                else ""
                            ),
                        ]
                        for idx in range(sample_count)
                    ],
                )
            safe_wandb_log(payload, step=step)
        for idx in range(min(5, eval_n)):
            logger.info("[%d] target=%r generated=%r", idx, eval_metrics["targets"][idx], eval_metrics["generated"][idx])

    if args.save_final_checkpoint:
        atomic_torch_save(
            training_checkpoint_payload(
                checkpoint_step=step,
                checkpoint_epoch=epoch,
                score=best_score,
            ),
            output_dir / "final.pt",
        )

    if run is not None:
        run.summary["best_score"] = best_score
        run.finish()
    logger.info("Finished. best_score=%s", best_score)


if __name__ == "__main__":
    main()
