#!/usr/bin/env python
"""Overfit ELF on a tiny LibriBrain MEG-to-sentence reconstruction task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
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
from nltk.stem import PorterStemmer
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import Config, SamplingConfig
from modules.meg_adapter import MEGContextAdapter
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
from utils.libribrain_utils import (
    build_libribrain_sentence_dataset,
    build_libribrain_sentence_dataset_per_book,
    build_sherlock1_session_run_keys,
    collate_libribrain_sentence_batch,
)
from utils.logging_utils import log_for_0
from utils.sampling_utils import (
    add_noise,
    get_sampling_steps,
    net_out_to_v_x,
    restore_cond,
    sample_timesteps,
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


@dataclass
class OverfitBatch:
    indices: torch.Tensor
    meg: torch.Tensor
    meg_lengths: torch.Tensor
    semantic_vectors: torch.Tensor
    subject_ids: torch.Tensor
    target_latents: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    target_weights: torch.Tensor | None = None


class SemanticVectorContextProjector(nn.Module):
    """Project one semantic embedding per sentence into ELF context tokens."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_length: int,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        input_projection_dim: int | None = None,
        normalize_semantic_output: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.context_dim = context_dim
        self.input_dim = input_dim
        self.normalize_semantic_output = bool(normalize_semantic_output)
        projected_dim = int(input_projection_dim or input_dim)
        if projected_dim <= 0 or projected_dim > input_dim:
            raise ValueError(
                f"input_projection_dim must be in [1, input_dim], got {projected_dim} for input_dim={input_dim}."
            )
        self.input_projection: nn.Linear | None = None
        if projected_dim != input_dim:
            if input_dim % projected_dim:
                raise ValueError(
                    f"input_dim={input_dim} must be divisible by input_projection_dim={projected_dim} "
                    "for mean-block initialization."
                )
            block_count = input_dim // projected_dim
            self.input_projection = nn.Linear(input_dim, projected_dim, bias=False)
            with torch.no_grad():
                self.input_projection.weight.zero_()
                identity = torch.eye(projected_dim, dtype=self.input_projection.weight.dtype)
                for block_idx in range(block_count):
                    start = block_idx * projected_dim
                    self.input_projection.weight[:, start : start + projected_dim].copy_(identity / block_count)
        self.net = nn.Sequential(
            nn.LayerNorm(projected_dim),
            nn.Linear(projected_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_length * context_dim),
        )

    def project_semantic(self, semantic_vectors: torch.Tensor) -> torch.Tensor:
        if self.input_projection is not None:
            semantic_vectors = self.input_projection(semantic_vectors)
        if self.normalize_semantic_output:
            semantic_vectors = F.normalize(semantic_vectors.float(), p=2, dim=-1)
        return semantic_vectors

    def context_from_projected_semantic(self, semantic_vectors: torch.Tensor):
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

    def forward(self, semantic_vectors: torch.Tensor):
        return self.context_from_projected_semantic(self.project_semantic(semantic_vectors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--semantic-data-path", default=None)
    parser.add_argument("--pnpl-root", default=None)
    parser.add_argument("--books", nargs="+", type=int, default=[1])
    parser.add_argument("--num-examples", type=int, default=8, help="Use 0 to load all non-empty examples.")
    parser.add_argument(
        "--eval-split",
        choices=["train_head", "sherlock1_sessions_11_12"],
        default="train_head",
        help=(
            "Validation source. train_head keeps the legacy first-N examples from training; "
            "sherlock1_sessions_11_12 loads Sherlock1 sessions 11/12 as held-out eval data."
        ),
    )
    parser.add_argument(
        "--eval-sherlock1-sessions",
        nargs="+",
        type=int,
        default=[11, 12],
        help="Sherlock1 sessions to load when --eval-split=sherlock1_sessions_11_12.",
    )
    parser.add_argument("--checkpoint_path", default="embedded-language-flows/ELF-B-owt-torch")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--output_dir", default="outputs/meg_context_overfit")
    parser.add_argument("--context_length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--epochs",
        type=float,
        default=0.0,
        help="If >0, override --steps with ceil(num_examples / batch_size * epochs).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--target-encode-batch-size",
        type=int,
        default=128,
        help="Batch size for precomputing T5 target latents; lower this to reduce startup GPU memory.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument(
        "--eval-num-examples",
        type=int,
        default=0,
        help="Evaluate at most this many examples each eval; 0 means all examples.",
    )
    parser.add_argument("--num_sampling_steps", type=int, default=32)
    parser.add_argument(
        "--train-timestep-mode",
        choices=["logit_normal", "sampling_schedule", "sampling_schedule_all"],
        default="logit_normal",
        help=(
            "Noise levels for denoiser training. sampling_schedule samples one sampler "
            "timestep per example; sampling_schedule_all trains on every sampler timestep."
        ),
    )
    parser.add_argument(
        "--train-timestep-steps",
        type=int,
        default=0,
        help="Number of sampler timesteps to use for schedule-based training; 0 uses num_sampling_steps.",
    )
    parser.add_argument(
        "--retrieval-eval-every",
        type=int,
        default=0,
        help="Run conditional-loss retrieval eval every N steps; 0 disables it.",
    )
    parser.add_argument(
        "--retrieval-num-examples",
        type=int,
        default=0,
        help="Number of examples/candidates for retrieval eval; 0 reuses eval-num-examples.",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--retrieval-t", type=float, default=0.5)
    parser.add_argument(
        "--generation-t5-retrieval",
        action="store_true",
        help="During generation eval, rank generated text against targets in mean-pooled T5 latent space.",
    )
    parser.add_argument(
        "--generation-gpt2-ppl",
        action="store_true",
        help="During generation eval, score generated text with a frozen causal LM perplexity.",
    )
    parser.add_argument("--gpt2-model-name", default="gpt2")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument("--last_n_blocks", type=int, default=-1)
    parser.add_argument(
        "--freeze-elf",
        action="store_true",
        help="Freeze all ELF weights and train only the condition adapter/projector.",
    )
    parser.add_argument(
        "--condition-source",
        choices=["meg", "semantic"],
        default="meg",
        help="Use MEG adapter output or semantic-vector projector as ELF condition.",
    )
    parser.add_argument("--denoiser_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="BrainDiffusion")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="meg-context-overfit")
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no-wandb-samples", action="store_true", help="Do not log generated text tables to WandB.")
    parser.add_argument(
        "--save-eval-checkpoints",
        action="store_true",
        help="Save top-scoring model/adapter checkpoints at generation evals.",
    )
    parser.add_argument(
        "--eval-checkpoint-top-k",
        type=int,
        default=3,
        help="Keep at most this many eval checkpoints, ranked by checkpoint score.",
    )
    parser.add_argument(
        "--eval-checkpoint-every",
        type=int,
        default=0,
        help="Also save every Nth generation eval checkpoint; 0 disables periodic saves.",
    )
    parser.add_argument(
        "--include-optimizer-in-checkpoints",
        action="store_true",
        help="Include optimizer state in eval checkpoints. This is much larger and only needed to resume training.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave unset for auto")
    parser.add_argument("--segment-ms", type=int, default=3000)
    parser.add_argument("--set-name", default="sentences")
    parser.add_argument("--embedding-type", default="SONAR")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-files", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--merger-channels", type=int, default=306)
    parser.add_argument("--conv-channels", type=int, default=320)
    parser.add_argument("--num-conv-layers", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilation-growth", type=int, default=2)
    parser.add_argument("--dilation-period", type=int, default=5)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--semantic-hidden-dim", type=int, default=2048)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--disable-subject-layers", action="store_true")
    parser.add_argument("--disable-temporal-attention", action="store_true")
    parser.add_argument("--norm-type", choices=["batch", "layer"], default="batch")
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
    config.decoder_prob = float(getattr(args, "decoder_prob", config.decoder_prob))
    config.decoder_noise_scale = args.decoder_noise_scale
    config.t_eps = 0.05
    config.time_schedule = "logit_normal"
    config.self_cond_prob = 0.5
    config.use_bf16 = True
    config.output_dir = args.output_dir
    config.denoiser_loss_weight = args.denoiser_loss_weight
    config.decoder_loss_weight = args.decoder_loss_weight
    config.attn_dropout = float(getattr(args, "elf_attn_dropout", 0.0))
    config.proj_dropout = float(getattr(args, "elf_proj_dropout", 0.0))
    return config


def tokenize_sentences(tokenizer, sentences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids_list = [tokenizer(sentence, add_special_tokens=True)["input_ids"] for sentence in sentences]
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


def rank_true_targets_by_similarity(similarity: torch.Tensor) -> dict:
    num_queries, num_candidates = similarity.shape
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)
    labels = torch.arange(num_queries, dtype=torch.long, device=similarity.device)
    ranks = (sorted_indices == labels[:, None]).nonzero()[:, 1] + 1
    return {
        "top1": float((ranks == 1).float().mean().cpu()),
        "top5": float((ranks <= min(5, num_candidates)).float().mean().cpu()),
        "mean_rank": float(ranks.float().mean().cpu()),
        "median_rank": float(ranks.float().median().cpu()),
        "ranks": ranks.detach().cpu().tolist(),
        "top_indices": sorted_indices[:, : min(5, num_candidates)].detach().cpu().tolist(),
    }


def mean_pool_latents(latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=latents.device, dtype=latents.dtype).unsqueeze(-1)
    pooled = (latents * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled.to(torch.float32), dim=-1)


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_QUALITY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "him", "his", "i", "in", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "she", "so", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "we",
    "were", "what", "when", "which", "who", "with", "you",
}
_CONTENT_WORD_STOPWORDS = _QUALITY_STOPWORDS | {
    "about", "after", "again", "against", "all", "am", "any", "because",
    "been", "before", "being", "below", "between", "both", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "further",
    "having", "here", "hers", "herself", "himself", "how", "if", "into",
    "itself", "just", "more", "most", "off", "once", "only", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "should", "some",
    "quite", "really", "such", "sure", "than", "then", "these", "those",
    "through", "too", "under", "until", "up", "us", "very", "where", "whom",
    "whose", "why", "will", "would", "your", "yours", "yourself",
    "yourselves", "well", "yes",
}
_PORTER_STEMMER = PorterStemmer()
_FUNCTION_CONTRACTION_SUFFIXES = {"s", "re", "ve", "d", "ll", "m"}
_FUNCTION_NEGATIVE_CONTRACTIONS = {
    "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't",
    "hadn't", "hasn't", "haven't", "isn't", "mustn't", "needn't", "shan't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
}


def _stem_tokens(tokens: Sequence[str]) -> list[str]:
    """Return stable morphology-normalized tokens while preserving token count.

    Porter stemming is intentionally used instead of a corpus-backed lemmatizer:
    it needs no downloaded language model on ARC and captures the common plural
    and tense variants relevant to the overlap audit.
    """

    return [
        _PORTER_STEMMER.stem(token) if len(token) > 3 else token
        for token in tokens
    ]


def _is_function_word(token: str) -> bool:
    if token in _CONTENT_WORD_STOPWORDS or token in _FUNCTION_NEGATIVE_CONTRACTIONS:
        return True
    if "'" not in token:
        return False
    base, suffix = token.rsplit("'", 1)
    return base in _CONTENT_WORD_STOPWORDS and suffix in _FUNCTION_CONTRACTION_SUFFIXES


def _function_word_tokens(text: str) -> list[str]:
    """Return stopword-defined function tokens, including common contractions."""

    return [token for token in _word_tokens(text) if _is_function_word(token)]


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _content_word_tokens(text: str) -> list[str]:
    return [
        token
        for token in _word_tokens(text)
        if token not in _CONTENT_WORD_STOPWORDS and any(char.isalpha() for char in token)
    ]


def _counted_overlap(
    generated_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> tuple[float, float, float, float, dict[str, int]]:
    if not generated_tokens or not target_tokens:
        return 0.0, 0.0, 0.0, 0.0, {}

    generated_counts = {}
    for token in generated_tokens:
        generated_counts[token] = generated_counts.get(token, 0) + 1
    target_counts = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap_counts = {
        token: min(count, target_counts.get(token, 0))
        for token, count in generated_counts.items()
        if target_counts.get(token, 0) > 0
    }
    overlap_count = sum(overlap_counts.values())
    precision = overlap_count / max(1, len(generated_tokens))
    recall = overlap_count / max(1, len(target_tokens))
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
    generated_set = set(generated_tokens)
    target_set = set(target_tokens)
    jaccard = len(generated_set & target_set) / max(1, len(generated_set | target_set))
    return float(precision), float(recall), float(f1), float(jaccard), dict(sorted(overlap_counts.items()))


def format_overlap_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return ""
    return ", ".join(f"{token}x{count}" if count > 1 else token for token, count in counts.items())


def _word_error_counts(
    reference_tokens: Sequence[str],
    hypothesis_tokens: Sequence[str],
) -> tuple[int, int, int]:
    """Return insertion, deletion, and substitution counts for one token pair."""
    previous = [(index, index, 0, 0) for index in range(len(hypothesis_tokens) + 1)]
    for reference_index, reference_token in enumerate(reference_tokens, start=1):
        current = [(reference_index, 0, reference_index, 0)]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            error, insertions, deletions, substitutions = previous[hypothesis_index]
            candidates = [(error + 1, insertions, deletions + 1, substitutions)]

            error, insertions, deletions, substitutions = current[hypothesis_index - 1]
            candidates.append((error + 1, insertions + 1, deletions, substitutions))

            error, insertions, deletions, substitutions = previous[hypothesis_index - 1]
            if reference_token == hypothesis_token:
                candidates.append((error, insertions, deletions, substitutions))
            else:
                candidates.append((error + 1, insertions, deletions, substitutions + 1))
            current.append(min(candidates, key=lambda item: (item[0], item[1] + item[2], item[3])))
        previous = current
    _errors, insertions, deletions, substitutions = previous[-1]
    return insertions, deletions, substitutions


def word_overlap_metrics(generated: Sequence[str], targets: Sequence[str]) -> dict:
    precisions = []
    recalls = []
    f1s = []
    jaccards = []
    overlap_counts = []
    content_precisions = []
    content_recalls = []
    content_f1s = []
    content_jaccards = []
    content_overlap_counts = []
    stemmed_precisions = []
    stemmed_recalls = []
    stemmed_f1s = []
    stemmed_jaccards = []
    stemmed_overlap_counts = []
    stemmed_content_precisions = []
    stemmed_content_recalls = []
    stemmed_content_f1s = []
    stemmed_content_jaccards = []
    stemmed_content_overlap_counts = []
    generated_function_word_counts = []
    generated_function_word_fractions = []
    generated_function_word_tokens = []
    target_function_word_counts = []
    target_function_word_fractions = []
    target_function_word_tokens = []
    total_generated_words = 0
    total_generated_function_words = 0
    total_target_function_words = 0
    per_sample_wer = []
    total_reference_words = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0

    for generated_text, target_text in zip(generated, targets):
        generated_tokens = _word_tokens(generated_text)
        target_tokens = _word_tokens(target_text)
        precision, recall, f1, jaccard, counts = _counted_overlap(generated_tokens, target_tokens)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        jaccards.append(jaccard)
        overlap_counts.append(counts)

        insertions, deletions, substitutions = _word_error_counts(target_tokens, generated_tokens)
        errors = insertions + deletions + substitutions
        per_sample_wer.append(errors / max(1, len(target_tokens)))
        total_reference_words += len(target_tokens)
        total_insertions += insertions
        total_deletions += deletions
        total_substitutions += substitutions

        content_precision, content_recall, content_f1, content_jaccard, content_counts = _counted_overlap(
            _content_word_tokens(generated_text),
            _content_word_tokens(target_text),
        )
        content_precisions.append(content_precision)
        content_recalls.append(content_recall)
        content_f1s.append(content_f1)
        content_jaccards.append(content_jaccard)
        content_overlap_counts.append(content_counts)

        stemmed_precision, stemmed_recall, stemmed_f1, stemmed_jaccard, stemmed_counts = _counted_overlap(
            _stem_tokens(generated_tokens),
            _stem_tokens(target_tokens),
        )
        stemmed_precisions.append(stemmed_precision)
        stemmed_recalls.append(stemmed_recall)
        stemmed_f1s.append(stemmed_f1)
        stemmed_jaccards.append(stemmed_jaccard)
        stemmed_overlap_counts.append(stemmed_counts)

        generated_content_tokens = _content_word_tokens(generated_text)
        target_content_tokens = _content_word_tokens(target_text)
        (
            stemmed_content_precision,
            stemmed_content_recall,
            stemmed_content_f1,
            stemmed_content_jaccard,
            stemmed_content_counts,
        ) = _counted_overlap(
            _stem_tokens(generated_content_tokens),
            _stem_tokens(target_content_tokens),
        )
        stemmed_content_precisions.append(stemmed_content_precision)
        stemmed_content_recalls.append(stemmed_content_recall)
        stemmed_content_f1s.append(stemmed_content_f1)
        stemmed_content_jaccards.append(stemmed_content_jaccard)
        stemmed_content_overlap_counts.append(stemmed_content_counts)

        generated_function_tokens = _function_word_tokens(generated_text)
        target_function_tokens = _function_word_tokens(target_text)
        generated_function_word_counts.append(len(generated_function_tokens))
        generated_function_word_fractions.append(len(generated_function_tokens) / max(1, len(generated_tokens)))
        generated_function_word_tokens.append(generated_function_tokens)
        target_function_word_counts.append(len(target_function_tokens))
        target_function_word_fractions.append(len(target_function_tokens) / max(1, len(target_tokens)))
        target_function_word_tokens.append(target_function_tokens)
        total_generated_words += len(generated_tokens)
        total_generated_function_words += len(generated_function_tokens)
        total_target_function_words += len(target_function_tokens)

    summary = {
        "words_overlap": float(np.mean(f1s)) if f1s else 0.0,
        "words_overlap_precision": float(np.mean(precisions)) if precisions else 0.0,
        "words_overlap_recall": float(np.mean(recalls)) if recalls else 0.0,
        "words_overlap_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "content_words_overlap": float(np.mean(content_f1s)) if content_f1s else 0.0,
        "content_words_overlap_precision": float(np.mean(content_precisions)) if content_precisions else 0.0,
        "content_words_overlap_recall": float(np.mean(content_recalls)) if content_recalls else 0.0,
        "content_words_overlap_jaccard": float(np.mean(content_jaccards)) if content_jaccards else 0.0,
        # Explicit aliases preserve the historical scores while making their
        # exact-token contract unambiguous in new reports.
        "exact_words_overlap": float(np.mean(f1s)) if f1s else 0.0,
        "exact_words_overlap_precision": float(np.mean(precisions)) if precisions else 0.0,
        "exact_words_overlap_recall": float(np.mean(recalls)) if recalls else 0.0,
        "exact_words_overlap_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "exact_content_words_overlap": float(np.mean(content_f1s)) if content_f1s else 0.0,
        "exact_content_words_overlap_precision": float(np.mean(content_precisions)) if content_precisions else 0.0,
        "exact_content_words_overlap_recall": float(np.mean(content_recalls)) if content_recalls else 0.0,
        "exact_content_words_overlap_jaccard": float(np.mean(content_jaccards)) if content_jaccards else 0.0,
        "stemmed_words_overlap": float(np.mean(stemmed_f1s)) if stemmed_f1s else 0.0,
        "stemmed_words_overlap_precision": float(np.mean(stemmed_precisions)) if stemmed_precisions else 0.0,
        "stemmed_words_overlap_recall": float(np.mean(stemmed_recalls)) if stemmed_recalls else 0.0,
        "stemmed_words_overlap_jaccard": float(np.mean(stemmed_jaccards)) if stemmed_jaccards else 0.0,
        "stemmed_content_words_overlap": float(np.mean(stemmed_content_f1s)) if stemmed_content_f1s else 0.0,
        "stemmed_content_words_overlap_precision": float(np.mean(stemmed_content_precisions)) if stemmed_content_precisions else 0.0,
        "stemmed_content_words_overlap_recall": float(np.mean(stemmed_content_recalls)) if stemmed_content_recalls else 0.0,
        "stemmed_content_words_overlap_jaccard": float(np.mean(stemmed_content_jaccards)) if stemmed_content_jaccards else 0.0,
        "decoded_word_count": int(total_generated_words),
        "decoded_function_word_count": int(total_generated_function_words),
        "decoded_function_word_fraction": float(total_generated_function_words / max(1, total_generated_words)),
        "target_function_word_count": int(total_target_function_words),
        "target_function_word_fraction": float(total_target_function_words / max(1, total_reference_words)),
        "word_error_rate": float(
            (total_insertions + total_deletions + total_substitutions) / max(1, total_reference_words)
        ),
        "word_error_reference_words": int(total_reference_words),
        "word_error_insertions": int(total_insertions),
        "word_error_deletions": int(total_deletions),
        "word_error_substitutions": int(total_substitutions),
    }
    return {
        "summary": summary,
        "per_sample": f1s,
        "per_sample_precision": precisions,
        "per_sample_recall": recalls,
        "per_sample_jaccard": jaccards,
        "per_sample_overlap_counts": overlap_counts,
        "per_sample_content": content_f1s,
        "per_sample_content_precision": content_precisions,
        "per_sample_content_recall": content_recalls,
        "per_sample_content_jaccard": content_jaccards,
        "per_sample_content_overlap_counts": content_overlap_counts,
        "per_sample_stemmed": stemmed_f1s,
        "per_sample_stemmed_precision": stemmed_precisions,
        "per_sample_stemmed_recall": stemmed_recalls,
        "per_sample_stemmed_jaccard": stemmed_jaccards,
        "per_sample_stemmed_overlap_counts": stemmed_overlap_counts,
        "per_sample_stemmed_content": stemmed_content_f1s,
        "per_sample_stemmed_content_precision": stemmed_content_precisions,
        "per_sample_stemmed_content_recall": stemmed_content_recalls,
        "per_sample_stemmed_content_jaccard": stemmed_content_jaccards,
        "per_sample_stemmed_content_overlap_counts": stemmed_content_overlap_counts,
        "per_sample_decoded_function_word_count": generated_function_word_counts,
        "per_sample_decoded_function_word_fraction": generated_function_word_fractions,
        "per_sample_decoded_function_words": generated_function_word_tokens,
        "per_sample_target_function_word_count": target_function_word_counts,
        "per_sample_target_function_word_fraction": target_function_word_fractions,
        "per_sample_target_function_words": target_function_word_tokens,
        "per_sample_wer": per_sample_wer,
    }


def generation_repetition_metrics(texts: Sequence[str]) -> dict[str, float]:
    unique_ratios = []
    repeated_token_fractions = []
    max_token_runs = []
    repeated_bigram_fractions = []
    repeated_trigram_fractions = []
    avg_word_lengths = []
    long_word_fractions = []
    vowel_missing_fractions = []
    alpha_token_fractions = []
    stopword_fractions = []
    length_scores = []
    well_structured_scores = []
    degen_flags = []

    for text in texts:
        tokens = text.strip().split()
        if not tokens:
            unique_ratios.append(0.0)
            repeated_token_fractions.append(1.0)
            max_token_runs.append(0.0)
            repeated_bigram_fractions.append(1.0)
            repeated_trigram_fractions.append(1.0)
            avg_word_lengths.append(0.0)
            long_word_fractions.append(1.0)
            vowel_missing_fractions.append(1.0)
            alpha_token_fractions.append(0.0)
            stopword_fractions.append(0.0)
            length_scores.append(0.0)
            well_structured_scores.append(0.0)
            degen_flags.append(1.0)
            continue

        clean_tokens = [token.strip(".,;:!?\"'()[]{}").lower() for token in tokens]
        clean_tokens = [token for token in clean_tokens if token]
        alpha_tokens = [token for token in clean_tokens if token.isalpha()]
        alpha_fraction = len(alpha_tokens) / max(1, len(clean_tokens))
        avg_word_length = float(np.mean([len(token) for token in alpha_tokens])) if alpha_tokens else 0.0
        long_word_fraction = sum(len(token) > 14 for token in alpha_tokens) / max(1, len(alpha_tokens))
        vowel_missing_fraction = (
            sum(len(token) > 3 and not any(char in "aeiou" for char in token) for token in alpha_tokens)
            / max(1, len(alpha_tokens))
        )
        stopword_fraction = sum(token in _QUALITY_STOPWORDS for token in alpha_tokens) / max(1, len(alpha_tokens))
        length_score = min(len(tokens), 40) / 40.0 if len(tokens) < 40 else max(0.0, 1.0 - (len(tokens) - 40) / 80.0)

        unique_ratio = len(set(tokens)) / len(tokens)
        repeated_token_fraction = 1.0 - unique_ratio
        max_run = 1
        current_run = 1
        for prev, cur in zip(tokens, tokens[1:]):
            current_run = current_run + 1 if cur == prev else 1
            max_run = max(max_run, current_run)

        def repeated_ngram_fraction(n: int) -> float:
            if len(tokens) < n:
                return 0.0
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            return 1.0 - (len(set(ngrams)) / len(ngrams))

        bigram_repeat = repeated_ngram_fraction(2)
        trigram_repeat = repeated_ngram_fraction(3)
        is_degenerate = max_run >= 5 or unique_ratio < 0.4 or bigram_repeat > 0.4
        stopword_score = max(0.0, 1.0 - abs(stopword_fraction - 0.45) / 0.45)
        word_shape_score = 1.0 - min(1.0, long_word_fraction + vowel_missing_fraction)
        repetition_score = max(0.0, 1.0 - max(repeated_token_fraction, bigram_repeat, min(max_run / 10.0, 1.0)))
        well_structured_score = float(np.mean([
            unique_ratio,
            repetition_score,
            word_shape_score,
            alpha_fraction,
            stopword_score,
            length_score,
        ]))

        unique_ratios.append(unique_ratio)
        repeated_token_fractions.append(repeated_token_fraction)
        max_token_runs.append(float(max_run))
        repeated_bigram_fractions.append(bigram_repeat)
        repeated_trigram_fractions.append(trigram_repeat)
        avg_word_lengths.append(avg_word_length)
        long_word_fractions.append(long_word_fraction)
        vowel_missing_fractions.append(vowel_missing_fraction)
        alpha_token_fractions.append(alpha_fraction)
        stopword_fractions.append(stopword_fraction)
        length_scores.append(length_score)
        well_structured_scores.append(well_structured_score)
        degen_flags.append(float(is_degenerate))

    return {
        "unique_token_ratio": float(np.mean(unique_ratios)),
        "repeated_token_fraction": float(np.mean(repeated_token_fractions)),
        "max_token_run": float(np.mean(max_token_runs)),
        "repeated_bigram_fraction": float(np.mean(repeated_bigram_fractions)),
        "repeated_trigram_fraction": float(np.mean(repeated_trigram_fractions)),
        "avg_word_length": float(np.mean(avg_word_lengths)),
        "long_word_fraction": float(np.mean(long_word_fractions)),
        "vowel_missing_fraction": float(np.mean(vowel_missing_fractions)),
        "alpha_token_fraction": float(np.mean(alpha_token_fractions)),
        "stopword_fraction": float(np.mean(stopword_fractions)),
        "length_score": float(np.mean(length_scores)),
        "well_structured_sentence": float(np.mean(well_structured_scores)),
        "degen_fraction": float(np.mean(degen_flags)),
    }


@torch.no_grad()
def encode_text_batched(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder: nn.Module,
    latent_mean: float,
    latent_std: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    latents = []
    batch_size = max(1, batch_size)
    for start in range(0, input_ids.shape[0], batch_size):
        end = min(start + batch_size, input_ids.shape[0])
        batch_latents = encode_text(
            input_ids=input_ids[start:end].to(device),
            attention_mask=attention_mask[start:end].to(device),
            encoder=encoder,
            latent_mean=latent_mean,
            latent_std=latent_std,
            use_bf16=False,
        )
        latents.append(batch_latents.detach().cpu())
    return torch.cat(latents, dim=0)


@torch.no_grad()
def causal_lm_perplexity(
    texts: Sequence[str],
    *,
    tokenizer,
    model: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    safe_texts = [text if text.strip() else "." for text in texts]
    encoded = tokenizer(
        safe_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = (shift_labels != -100).to(torch.float32)
    token_loss = F.cross_entropy(
        shift_logits.transpose(1, 2).to(torch.float32),
        shift_labels,
        reduction="none",
        ignore_index=-100,
    )
    per_text_nll = (token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
    per_text_ppl = torch.exp(per_text_nll.clamp(max=20.0))
    return {
        "gpt2_nll": float(per_text_nll.mean().cpu()),
        "gpt2_perplexity": float(per_text_ppl.mean().cpu()),
    }


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
        attn_drop=float(getattr(config, "attn_dropout", 0.0)),
        proj_drop=float(getattr(config, "proj_dropout", 0.0)),
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        vocab_size=vocab_size,
        bottleneck_dim=128,
        gradient_checkpointing=False,
    ).to(device)

    ckpt_root = _download_hf_checkpoint(args.checkpoint_path) or args.checkpoint_path
    ckpt = _restore_checkpoint(ckpt_root)
    if ckpt is None:
        raise ValueError(f"Could not restore checkpoint from {args.checkpoint_path}")
    if "params" in ckpt:
        params = ckpt["params"]
    elif "model_state_dict" in ckpt:
        params = ckpt["model_state_dict"]
    else:
        raise ValueError(
            f"Checkpoint {args.checkpoint_path} has neither 'params' nor 'model_state_dict'. "
            f"Available keys: {sorted(ckpt.keys())}"
        )
    model.load_state_dict(params)
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


def extract_subject_label(info: dict) -> str:
    for key in ("subject", "participant", "sub", "subject_id"):
        value = info.get(key)
        if value not in (None, ""):
            return str(value)
    return "0"


def examples_from_dataset(
    *,
    dataset,
    condition_source: str,
    max_examples: int,
) -> dict[str, torch.Tensor | list[str] | list[dict] | dict[str, int]]:
    if condition_source == "semantic":
        datasets = list(getattr(dataset, "datasets", [dataset]))
        semantic_vectors = []
        sentences = []
        info_list = []
        for ds in datasets:
            for sample in getattr(ds, "samples", []):
                subject, session, task, run, onset, offset, semantic_vector, sentence = sample
                sentence = str(sentence).strip()
                if not sentence:
                    continue
                semantic_vectors.append(torch.as_tensor(semantic_vector, dtype=torch.float32))
                sentences.append(sentence)
                info_list.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "run": run,
                        "onset": float(onset),
                        "offset": float(offset),
                        "sentence": sentence,
                    }
                )
                if max_examples > 0 and len(sentences) >= max_examples:
                    break
            if max_examples > 0 and len(sentences) >= max_examples:
                break

        if max_examples > 0 and len(sentences) < max_examples:
            raise ValueError(
                f"Requested {max_examples} examples but found only {len(sentences)} non-empty sentences."
            )
        if not sentences:
            raise ValueError("No non-empty semantic-vector sentence examples found.")

        subject_labels = [extract_subject_label(info) for info in info_list]
        num_examples = len(sentences)
        return {
            "meg": torch.zeros((num_examples, 1, 1), dtype=torch.float32),
            "meg_time_mask": torch.ones((num_examples, 1), dtype=torch.bool),
            "meg_lengths": torch.ones((num_examples,), dtype=torch.long),
            "semantic_vectors": torch.stack(semantic_vectors),
            "sentences": sentences,
            "info": info_list,
            "subject_labels": subject_labels,
        }

    samples = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if len(item) == 3:
            _, _, info = item
        elif len(item) == 2:
            info = {}
        else:
            raise ValueError(f"Unexpected LibriBrain sample format of length {len(item)}")

        sentence = str(info.get("sentence", "")).strip()
        if not sentence:
            continue

        samples.append(item)
        if max_examples > 0 and len(samples) >= max_examples:
            break

    if max_examples > 0 and len(samples) < max_examples:
        raise ValueError(
            f"Requested {max_examples} examples but found only {len(samples)} non-empty sentences."
        )

    batch = collate_libribrain_sentence_batch(samples)
    subject_labels = [extract_subject_label(info) for info in batch["info"]]

    batch["subject_labels"] = subject_labels
    return batch


def assign_subject_ids(
    batch: dict[str, torch.Tensor | list[str] | list[dict] | dict[str, int]],
    subject_to_id: dict[str, int],
) -> None:
    subject_labels = batch["subject_labels"]
    batch["subject_to_id"] = subject_to_id
    batch["subject_ids"] = torch.tensor([subject_to_id[subject] for subject in subject_labels], dtype=torch.long)


def load_overfit_examples(args: argparse.Namespace) -> dict[str, torch.Tensor | list[str] | list[dict] | dict[str, int]]:
    dataset = build_libribrain_sentence_dataset_per_book(
        data_path=args.data_path,
        books=args.books,
        semantic_data_path=args.semantic_data_path,
        pnpl_root=args.pnpl_root,
        set_name=args.set_name,
        embedding_type=args.embedding_type,
        segment_ms=None if args.segment_ms <= 0 else args.segment_ms,
        standardize=not args.no_standardize,
        preload_files=args.preload_files,
    )
    batch = examples_from_dataset(
        dataset=dataset,
        condition_source=args.condition_source,
        max_examples=args.num_examples,
    )
    unique_subjects = sorted(set(batch["subject_labels"]))
    subject_to_id = {subject: idx for idx, subject in enumerate(unique_subjects)}
    assign_subject_ids(batch, subject_to_id)
    return batch


def load_eval_examples(
    args: argparse.Namespace,
    train_batch: dict[str, torch.Tensor | list[str] | list[dict] | dict[str, int]],
) -> dict[str, torch.Tensor | list[str] | list[dict] | dict[str, int]]:
    if args.eval_split == "train_head":
        return train_batch
    if args.eval_split != "sherlock1_sessions_11_12":
        raise ValueError(f"Unsupported eval split: {args.eval_split}")

    dataset = build_libribrain_sentence_dataset(
        data_path=args.data_path,
        books=[1],
        run_keys=build_sherlock1_session_run_keys(args.eval_sherlock1_sessions),
        semantic_data_path=args.semantic_data_path,
        pnpl_root=args.pnpl_root,
        set_name=args.set_name,
        embedding_type=args.embedding_type,
        segment_ms=None if args.segment_ms <= 0 else args.segment_ms,
        standardize=not args.no_standardize,
        preload_files=args.preload_files,
    )
    eval_batch = examples_from_dataset(
        dataset=dataset,
        condition_source=args.condition_source,
        max_examples=args.eval_num_examples,
    )
    subject_labels = sorted(set(train_batch["subject_labels"]) | set(eval_batch["subject_labels"]))
    subject_to_id = {subject: idx for idx, subject in enumerate(subject_labels)}
    assign_subject_ids(train_batch, subject_to_id)
    assign_subject_ids(eval_batch, subject_to_id)
    return eval_batch


def build_batches(
    meg: torch.Tensor,
    meg_lengths: torch.Tensor,
    semantic_vectors: torch.Tensor,
    subject_ids: torch.Tensor,
    target_latents: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> list[OverfitBatch]:
    num_examples = target_latents.shape[0]
    perm = torch.randperm(num_examples, generator=generator)
    batches = []
    for start in range(0, num_examples, batch_size):
        indices = perm[start:start + batch_size]
        batches.append(
            OverfitBatch(
                indices=indices,
                meg=meg.index_select(0, indices),
                meg_lengths=meg_lengths.index_select(0, indices),
                semantic_vectors=semantic_vectors.index_select(0, indices),
                subject_ids=subject_ids.index_select(0, indices),
                target_latents=target_latents.index_select(0, indices),
                target_ids=target_ids.index_select(0, indices),
                target_mask=target_mask.index_select(0, indices),
            )
        )
    return batches


def optimizer_grad_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    global_sq = 0.0
    global_max = 0.0
    global_tensors = 0

    for group_idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group{group_idx}")).replace("/", "_")
        group_sq = 0.0
        group_max = 0.0
        group_tensors = 0
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            grad = grad.detach().float()
            group_sq += float(torch.sum(grad * grad).cpu())
            group_max = max(group_max, float(grad.abs().max().cpu()))
            group_tensors += 1
        metrics[f"grad_{group_name}_l2"] = group_sq ** 0.5
        metrics[f"grad_{group_name}_max_abs"] = group_max
        metrics[f"grad_{group_name}_tensor_count"] = float(group_tensors)
        global_sq += group_sq
        global_max = max(global_max, group_max)
        global_tensors += group_tensors

    metrics["grad_global_l2"] = global_sq ** 0.5
    metrics["grad_global_max_abs"] = global_max
    metrics["grad_global_tensor_count"] = float(global_tensors)
    return metrics


def build_decoder_training_latent(
    x0: torch.Tensor,
    decoder_noise: torch.Tensor,
    decoder_lambda: torch.Tensor,
    cond_seq_mask: torch.Tensor,
    *,
    preserve_condition_prefix: bool,
) -> torch.Tensor:
    """Construct decoder corruption, optionally matching inference conditioning.

    ELF sampling always restores condition tokens to ``x0``.  New explicit
    decoder cross-attention therefore opts into the same clean-prefix contract;
    historical training modes retain their original full-sequence corruption.
    """

    decoder_z = decoder_lambda * x0 + (1.0 - decoder_lambda) * decoder_noise
    if preserve_condition_prefix:
        decoder_z = restore_cond(decoder_z, x0, cond_seq_mask)
    return decoder_z


def content_bow_coverage_loss(
    target_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize target content tokens that are unlikely to occur anywhere in the output.

    Content-token positions are identified by weights greater than one. For each such
    target token, the loss computes the probability that at least one valid output
    position emits that vocabulary item. This complements position-wise decoder CE
    with an order-tolerant content-coverage objective.
    """

    if target_logits.ndim != 3:
        raise ValueError(f"target_logits must have shape [batch, length, vocab], got {target_logits.shape}")
    expected_shape = target_logits.shape[:2]
    for name, tensor in (
        ("target_ids", target_ids),
        ("target_mask", target_mask),
        ("target_weights", target_weights),
    ):
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")

    content_mask = (target_weights > 1.0) & (target_mask > 0.0)
    if not bool(content_mask.any()):
        return target_logits.sum() * 0.0

    token_probabilities = F.softmax(target_logits.float(), dim=-1)
    requested_ids = target_ids[:, None, :].expand(-1, target_logits.shape[1], -1)
    requested_probabilities = token_probabilities.gather(dim=2, index=requested_ids)

    output_mask = (target_mask > 0.0)[:, :, None]
    requested_probabilities = torch.where(
        output_mask,
        requested_probabilities,
        torch.zeros_like(requested_probabilities),
    )
    requested_probabilities = requested_probabilities.clamp(min=0.0, max=1.0 - 1e-6)
    log_absence_probability = torch.log1p(-requested_probabilities).sum(dim=1)
    presence_probability = -torch.expm1(log_absence_probability)
    per_token_loss = -torch.log(presence_probability.clamp_min(1e-8))
    return per_token_loss.masked_select(content_mask).mean()


def content_bow_precision_loss(
    target_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    negative_topk: int = 32,
) -> torch.Tensor:
    """Penalize likely content tokens that do not belong to each target.

    Candidate content-token ids are collected from the current batch's marked
    target spans. For each sample, candidates used by another target but absent
    from its own target are negatives. The loss penalizes the most likely
    negative tokens, complementing :func:`content_bow_coverage_loss`, which only
    rewards recall. Restricting negatives to marked content tokens avoids
    treating punctuation, special tokens, and ordinary function words as
    hallucinations.
    """

    if target_logits.ndim != 3:
        raise ValueError(f"target_logits must have shape [batch, length, vocab], got {target_logits.shape}")
    expected_shape = target_logits.shape[:2]
    for name, tensor in (
        ("target_ids", target_ids),
        ("target_mask", target_mask),
        ("target_weights", target_weights),
    ):
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    if negative_topk < 0:
        raise ValueError("negative_topk must be non-negative")

    content_mask = (target_weights > 1.0) & (target_mask > 0.0)
    if not bool(content_mask.any()):
        return target_logits.sum() * 0.0

    candidate_ids = torch.unique(target_ids.masked_select(content_mask))
    if candidate_ids.numel() < 2:
        return target_logits.sum() * 0.0

    token_probabilities = F.softmax(target_logits.float(), dim=-1)
    candidate_probabilities = token_probabilities.index_select(dim=2, index=candidate_ids)
    output_mask = (target_mask > 0.0)[:, :, None]
    candidate_probabilities = torch.where(
        output_mask,
        candidate_probabilities,
        torch.zeros_like(candidate_probabilities),
    ).clamp(min=0.0, max=1.0 - 1e-6)
    log_absence_probability = torch.log1p(-candidate_probabilities).sum(dim=1)
    presence_probability = -torch.expm1(log_absence_probability)

    sample_has_candidate = (
        (target_ids[:, :, None] == candidate_ids[None, None, :]) & content_mask[:, :, None]
    ).any(dim=1)
    negative_mask = ~sample_has_candidate
    if not bool(negative_mask.any()):
        return target_logits.sum() * 0.0

    per_candidate_loss = -torch.log1p(-presence_probability.clamp(max=1.0 - 1e-6))
    if negative_topk > 0:
        masked_loss = per_candidate_loss.masked_fill(~negative_mask, float("-inf"))
        k = min(negative_topk, candidate_ids.numel())
        top_loss = masked_loss.topk(k=k, dim=1).values
        finite = torch.isfinite(top_loss)
        if not bool(finite.any()):
            return target_logits.sum() * 0.0
        return top_loss.masked_select(finite).mean()
    return per_candidate_loss.masked_select(negative_mask).mean()


def semantic_content_ranking_loss(
    predicted_semantic: torch.Tensor,
    target_content_mask: torch.Tensor,
    content_directions: torch.Tensor,
    *,
    temperature: float = 0.1,
    negative_topk: int = 32,
) -> torch.Tensor:
    """Rank each row's target content words above hard lexical negatives.

    ``content_directions`` are fixed train-only semantic prototype directions,
    and ``target_content_mask`` marks the words belonging to each row. The
    objective updates the predicted semantic vector directly, before the ELF
    context projector, so lexical supervision does not have to traverse the
    entire diffusion/decoder path.
    """

    if predicted_semantic.ndim != 2:
        raise ValueError(
            f"predicted_semantic must have shape [batch, dim], got {predicted_semantic.shape}"
        )
    if content_directions.ndim != 2 or content_directions.shape[1] != predicted_semantic.shape[1]:
        raise ValueError(
            "content_directions must have shape [vocabulary, semantic_dim], got "
            f"{content_directions.shape} for semantic_dim={predicted_semantic.shape[1]}"
        )
    expected_mask_shape = (predicted_semantic.shape[0], content_directions.shape[0])
    if tuple(target_content_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"target_content_mask must have shape {expected_mask_shape}, got {target_content_mask.shape}"
        )
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if negative_topk <= 0:
        raise ValueError("negative_topk must be positive")

    positive_mask = target_content_mask.to(dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1) & (~positive_mask).any(dim=1)
    if not bool(valid_rows.any()):
        return predicted_semantic.sum() * 0.0

    predicted = F.normalize(predicted_semantic.float(), p=2, dim=-1)
    directions = F.normalize(content_directions.float(), p=2, dim=-1)
    scores = predicted @ directions.T
    negative_scores = scores.masked_fill(positive_mask, float("-inf"))
    k = min(negative_topk, max(1, content_directions.shape[0] - 1))
    hard_negative_scores = negative_scores.topk(k=k, dim=1).values
    finite_negatives = torch.isfinite(hard_negative_scores)
    pairwise_margin = (
        hard_negative_scores[:, None, :] - scores[:, :, None]
    ) / temperature
    pair_mask = positive_mask[:, :, None] & finite_negatives[:, None, :] & valid_rows[:, None, None]
    if not bool(pair_mask.any()):
        return predicted_semantic.sum() * 0.0
    return F.softplus(pairwise_margin.masked_select(pair_mask)).mean()


def raw_semantic_teacher(
    semantic_inputs: torch.Tensor,
    *,
    output_dim: int,
) -> torch.Tensor:
    """Recover the deterministic raw semantic baseline used before adaptation.

    A same-width input is already in the teacher space.  A wider input is
    interpreted as contiguous, equal-width delay blocks and averaged exactly,
    matching the mean-block initialization used by the fMRI MiniLM bridge.
    """

    if semantic_inputs.ndim != 2:
        raise ValueError(
            f"Raw semantic teacher expects [B, D], got {tuple(semantic_inputs.shape)}."
        )
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
    input_dim = int(semantic_inputs.shape[-1])
    if input_dim == output_dim:
        return semantic_inputs
    if input_dim % output_dim:
        raise ValueError(
            f"Input dimension {input_dim} is not divisible by teacher output dimension {output_dim}."
        )
    return semantic_inputs.reshape(
        semantic_inputs.shape[0], input_dim // output_dim, output_dim
    ).mean(dim=1)


def semantic_preservation_losses(
    predicted_semantic: torch.Tensor,
    semantic_inputs: torch.Tensor,
    *,
    exact_targets: torch.Tensor | None = None,
    rank_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Protect raw semantic identity while a downstream text objective trains.

    ``anchor`` limits per-row movement from the deterministic raw bridge,
    ``geometry`` preserves off-diagonal pairwise cosine structure, and
    ``rank_distill`` preserves the raw bridge's in-batch distribution over
    exact semantic candidates.  Only the student/predicted path receives
    gradients.
    """

    if predicted_semantic.ndim != 2:
        raise ValueError(
            f"Predicted semantic must have shape [B, D], got {tuple(predicted_semantic.shape)}."
        )
    teacher = raw_semantic_teacher(
        semantic_inputs.float(), output_dim=int(predicted_semantic.shape[-1])
    ).detach()
    student = predicted_semantic.float()
    teacher_normalized = F.normalize(teacher, p=2, dim=-1)
    student_normalized = F.normalize(student, p=2, dim=-1)

    anchor_cosine = F.cosine_similarity(student, teacher, dim=-1).mean()
    anchor_loss = 1.0 - anchor_cosine

    if student.shape[0] > 1:
        teacher_geometry = teacher_normalized @ teacher_normalized.T
        student_geometry = student_normalized @ student_normalized.T
        off_diagonal = ~torch.eye(
            student.shape[0], dtype=torch.bool, device=student.device
        )
        geometry_loss = F.mse_loss(
            student_geometry.masked_select(off_diagonal),
            teacher_geometry.masked_select(off_diagonal),
        )
    else:
        geometry_loss = student.sum() * 0.0

    rank_distill_loss = student.sum() * 0.0
    if exact_targets is not None and student.shape[0] > 1:
        if rank_temperature <= 0.0:
            raise ValueError("semantic rank-distillation temperature must be positive")
        targets = exact_targets.detach().float()
        if tuple(targets.shape) != tuple(student.shape):
            raise ValueError(
                "Semantic rank-distillation shape mismatch: "
                f"student={tuple(student.shape)} exact_targets={tuple(targets.shape)}"
            )
        target_normalized = F.normalize(targets, p=2, dim=-1)
        teacher_logits = teacher_normalized @ target_normalized.T / rank_temperature
        student_logits = student_normalized @ target_normalized.T / rank_temperature
        rank_distill_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

    return {
        "anchor": anchor_loss,
        "anchor_cosine": anchor_cosine,
        "geometry": geometry_loss,
        "rank_distill": rank_distill_loss,
    }


def train_step(
    *,
    model: nn.Module,
    adapter: nn.Module,
    batch: OverfitBatch,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    noise_generator: torch.Generator,
    condition_source: str,
    cond_dropout_prob: float,
    train_timestep_mode: str,
    train_timestep_steps: int,
    semantic_alignment_loss_weight: float = 0.0,
    semantic_alignment_loss_type: str = "cosine",
    semantic_alignment_targets: torch.Tensor | None = None,
    semantic_contrastive_loss_weight: float = 0.0,
    semantic_contrastive_temperature: float = 0.07,
    semantic_context_loss_weight: float = 0.0,
    semantic_context_teacher: nn.Module | None = None,
    semantic_content_loss_weight: float = 0.0,
    semantic_content_targets: torch.Tensor | None = None,
    semantic_content_directions: torch.Tensor | None = None,
    semantic_content_temperature: float = 0.1,
    semantic_content_negative_topk: int = 32,
    semantic_raw_anchor_loss_weight: float = 0.0,
    semantic_geometry_loss_weight: float = 0.0,
    semantic_rank_distill_loss_weight: float = 0.0,
    semantic_rank_distill_temperature: float = 0.07,
    semantic_rank_teacher_adapter: nn.Module | None = None,
    decoder_content_bow_loss_weight: float = 0.0,
    decoder_content_precision_loss_weight: float = 0.0,
    decoder_content_precision_topk: int = 32,
    decoder_condition_pairing_margin_weight: float = 0.0,
    decoder_condition_pairing_margin: float = 0.1,
) -> dict[str, float]:
    model.train()
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        adapter.train()
    else:
        # A frozen brain adapter must also be behaviorally frozen.  Leaving it
        # in train mode would keep dropout active and turn the supposedly fixed
        # condition into a moving target for a cross-attention-only stage.
        adapter.eval()
    model_deterministic = bool(
        getattr(model, "brain_cross_attention_only", False)
    )

    meg = batch.meg.to(device=device, dtype=torch.float32)
    meg_lengths = batch.meg_lengths.to(device=device, dtype=torch.long)
    semantic_vectors = batch.semantic_vectors.to(device=device, dtype=torch.float32)
    subject_ids = batch.subject_ids.to(device=device, dtype=torch.long)
    target_latents = batch.target_latents.to(device=device, dtype=torch.float32)
    target_ids = batch.target_ids.to(device)
    target_token_mask = batch.target_mask.to(device=device, dtype=torch.float32)
    target_token_weights = (
        batch.target_weights.to(device=device, dtype=torch.float32)
        if batch.target_weights is not None
        else torch.ones_like(target_token_mask)
    )
    alignment_targets = (
        semantic_alignment_targets.to(device=device, dtype=torch.float32)
        if semantic_alignment_targets is not None
        else None
    )

    def build_conditioned_x0():
        predicted_semantic = None
        if condition_source == "meg":
            adapter_output = adapter(meg, meg_lengths=meg_lengths, subjects=subject_ids)
            context = adapter_output.context
            cond_seq_mask = adapter_output.context_mask.to(dtype=context.dtype)
            encoded_sequence = getattr(adapter_output, "encoded_sequence", None)
            if encoded_sequence is not None and encoded_sequence.ndim == 3 and encoded_sequence.shape[1] == 1:
                predicted_semantic = encoded_sequence[:, 0, :]
        elif condition_source == "semantic":
            condition_context = getattr(adapter, "condition_context", None)
            project_semantic = getattr(adapter, "project_semantic", None)
            context_from_projected = getattr(adapter, "context_from_projected_semantic", None)
            if condition_context is not None:
                predicted_semantic, context, cond_seq_mask = condition_context(semantic_vectors)
            elif project_semantic is not None and context_from_projected is not None:
                predicted_semantic = project_semantic(semantic_vectors)
                context, cond_seq_mask = context_from_projected(predicted_semantic)
            else:
                context, cond_seq_mask = adapter(semantic_vectors)
        else:
            raise ValueError(f"Unsupported condition_source: {condition_source}")

        if cond_dropout_prob > 0.0:
            drop = torch.rand(
                (context.shape[0], 1, 1),
                generator=noise_generator,
                device=device,
            ) < cond_dropout_prob
            context = torch.where(drop, torch.zeros_like(context), context)

        x0 = torch.cat([context, target_latents], dim=1)
        target_valid_mask = target_token_mask.to(dtype=cond_seq_mask.dtype)
        target_prefix = torch.zeros(
            (x0.shape[0], target_latents.shape[1]),
            dtype=cond_seq_mask.dtype,
            device=device,
        )
        cond_seq_mask = torch.cat([cond_seq_mask, target_prefix], dim=1)
        attention_mask = torch.cat([cond_seq_mask[:, : context.shape[1]], target_valid_mask], dim=1)
        target_loss_mask = torch.cat([torch.zeros_like(cond_seq_mask[:, : context.shape[1]]), target_valid_mask], dim=1)
        sc_scale = torch.ones((x0.shape[0],), dtype=x0.dtype, device=device)
        return context, x0, cond_seq_mask, attention_mask, sc_scale, target_loss_mask, predicted_semantic

    optimizer.zero_grad(set_to_none=True)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"

    def denoiser_loss_for_t(t: torch.Tensor) -> torch.Tensor:
        _, x0, cond_seq_mask, attention_mask, sc_scale, target_loss_mask, _ = build_conditioned_x0()
        noise = torch.randn(x0.shape, generator=noise_generator, device=device, dtype=x0.dtype)
        z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask.unsqueeze(-1))
        pred, _ = model(
            z,
            t,
            attention_mask=attention_mask,
            deterministic=model_deterministic,
            self_cond_cfg_scale=sc_scale,
        )
        v_pred, _ = net_out_to_v_x(pred, z, t, config.t_eps)
        t_expanded = t.reshape(-1, 1, 1)
        v_target = (x0 - z) / torch.clamp(1.0 - t_expanded, min=config.t_eps)
        latent_target_mask = target_loss_mask.unsqueeze(-1)
        denoiser_loss = (
            ((v_pred - v_target) ** 2 * latent_target_mask).sum()
            / latent_target_mask.sum().clamp_min(1.0)
            / x0.shape[-1]
        )
        return denoiser_loss

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        if train_timestep_mode == "logit_normal":
            t_values = [
                sample_timesteps(
                    batch_size=target_latents.shape[0],
                    P_mean=config.denoiser_p_mean,
                    P_std=config.denoiser_p_std,
                    time_schedule=config.time_schedule,
                    device=device,
                    dtype=target_latents.dtype,
                )
            ]
        elif train_timestep_mode == "sampling_schedule":
            timestep_count = train_timestep_steps if train_timestep_steps > 0 else 32
            schedule = get_sampling_steps(
                n_steps=timestep_count,
                time_schedule=config.time_schedule,
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
                device=device,
                dtype=target_latents.dtype,
            )
            indices = torch.randint(
                low=0,
                high=schedule.shape[0],
                size=(target_latents.shape[0],),
                generator=noise_generator,
                device=device,
            )
            t_values = [schedule.index_select(0, indices)]
        elif train_timestep_mode == "sampling_schedule_all":
            timestep_count = train_timestep_steps if train_timestep_steps > 0 else 32
            schedule = get_sampling_steps(
                n_steps=timestep_count,
                time_schedule=config.time_schedule,
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
                device=device,
                dtype=target_latents.dtype,
            )
            t_values = [
                torch.full(
                    (target_latents.shape[0],),
                    float(timestep.detach().cpu()),
                    dtype=target_latents.dtype,
                    device=device,
                )
                for timestep in schedule
            ]
        else:
            raise ValueError(f"Unsupported train_timestep_mode: {train_timestep_mode}")

    denoiser_loss_value = 0.0
    if config.denoiser_loss_weight > 0.0:
        for t in t_values:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                denoiser_loss = denoiser_loss_for_t(t)
                scaled_denoiser_loss = config.denoiser_loss_weight * denoiser_loss / len(t_values)
            if scaled_denoiser_loss.requires_grad:
                scaled_denoiser_loss.backward()
            denoiser_loss_value += float(denoiser_loss.detach().cpu()) / len(t_values)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        context, x0, cond_seq_mask, attention_mask, sc_scale, _target_loss_mask, predicted_semantic = (
            build_conditioned_x0()
        )
        decoder_lambda = torch.sigmoid(
            torch.randn(x0.shape[:2], generator=noise_generator, device=device, dtype=x0.dtype)
            * config.decoder_p_std + config.decoder_p_mean
        ).unsqueeze(-1)
        decoder_noise = (
            torch.randn(x0.shape, generator=noise_generator, device=device, dtype=x0.dtype)
            * config.decoder_noise_scale
        )
        # Decoder-only brain attention sees the same clean condition memory at
        # training and inference; legacy/shared-flow modes keep the historical
        # full-sequence corruption contract.
        decoder_z = build_decoder_training_latent(
            x0,
            decoder_noise,
            decoder_lambda,
            cond_seq_mask,
            preserve_condition_prefix=bool(
                getattr(model, "brain_cross_attention_decoder_only", False)
            ),
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
            deterministic=model_deterministic,
            self_cond_cfg_scale=sc_scale,
            decoder_step_active=True,
        )
        target_logits = decoder_logits[:, context.shape[1]:]
        ce_per_token = F.cross_entropy(
            target_logits.transpose(1, 2).to(torch.float32),
            target_ids,
            reduction="none",
        )
        weighted_target_mask = target_token_mask * target_token_weights
        decoder_loss = (
            (ce_per_token * weighted_target_mask).sum()
            / weighted_target_mask.sum().clamp_min(1.0)
        )
        scaled_decoder_loss = config.decoder_loss_weight * decoder_loss
        if decoder_content_bow_loss_weight < 0.0:
            raise ValueError("decoder_content_bow_loss_weight must be non-negative")
        content_bow_loss = (
            content_bow_coverage_loss(
                target_logits,
                target_ids,
                target_token_mask,
                target_token_weights,
            )
            if decoder_content_bow_loss_weight > 0.0
            else target_logits.sum() * 0.0
        )
        scaled_content_bow_loss = decoder_content_bow_loss_weight * content_bow_loss
        if decoder_content_precision_loss_weight < 0.0:
            raise ValueError("decoder_content_precision_loss_weight must be non-negative")
        content_precision_loss = (
            content_bow_precision_loss(
                target_logits,
                target_ids,
                target_token_mask,
                target_token_weights,
                negative_topk=decoder_content_precision_topk,
            )
            if decoder_content_precision_loss_weight > 0.0
            else target_logits.sum() * 0.0
        )
        scaled_content_precision_loss = decoder_content_precision_loss_weight * content_precision_loss
        if decoder_condition_pairing_margin_weight < 0.0:
            raise ValueError(
                "decoder_condition_pairing_margin_weight must be non-negative"
            )
        condition_pairing_loss = target_logits.sum() * 0.0
        if decoder_condition_pairing_margin_weight > 0.0 and x0.shape[0] > 1:
            condition_length = context.shape[1]
            rolled_decoder_input = torch.cat(
                [
                    decoder_input[:, :condition_length].roll(1, dims=0),
                    decoder_input[:, condition_length:],
                ],
                dim=1,
            )
            rolled_attention_mask = torch.cat(
                [
                    attention_mask[:, :condition_length].roll(1, dims=0),
                    attention_mask[:, condition_length:],
                ],
                dim=1,
            )
            _, rolled_decoder_logits = model(
                rolled_decoder_input,
                torch.ones((x0.shape[0],), dtype=x0.dtype, device=device),
                attention_mask=rolled_attention_mask,
                deterministic=model_deterministic,
                self_cond_cfg_scale=sc_scale,
                decoder_step_active=True,
            )
            rolled_target_logits = rolled_decoder_logits[:, condition_length:]
            rolled_ce_per_token = F.cross_entropy(
                rolled_target_logits.transpose(1, 2).to(torch.float32),
                target_ids,
                reduction="none",
            )
            pairing_mask = target_token_mask
            matched_nll = (
                (ce_per_token * pairing_mask).sum(dim=1)
                / pairing_mask.sum(dim=1).clamp_min(1.0)
            )
            rolled_nll = (
                (rolled_ce_per_token * pairing_mask).sum(dim=1)
                / pairing_mask.sum(dim=1).clamp_min(1.0)
            )
            condition_pairing_loss = F.softplus(
                matched_nll - rolled_nll + decoder_condition_pairing_margin
            ).mean()
        scaled_condition_pairing_loss = (
            decoder_condition_pairing_margin_weight * condition_pairing_loss
        )
        semantic_alignment_loss = None
        semantic_alignment_cosine = None
        scaled_semantic_alignment_loss = x0.new_tensor(0.0)
        semantic_contrastive_loss = None
        semantic_context_loss = None
        semantic_content_loss = None
        scaled_semantic_contrastive_loss = x0.new_tensor(0.0)
        scaled_semantic_context_loss = x0.new_tensor(0.0)
        scaled_semantic_content_loss = x0.new_tensor(0.0)
        semantic_raw_anchor_loss = None
        semantic_raw_anchor_cosine = None
        semantic_geometry_loss = None
        semantic_rank_distill_loss = None
        scaled_semantic_raw_anchor_loss = x0.new_tensor(0.0)
        scaled_semantic_geometry_loss = x0.new_tensor(0.0)
        scaled_semantic_rank_distill_loss = x0.new_tensor(0.0)
        any_semantic_objective = (
            semantic_alignment_loss_weight > 0.0
            or semantic_contrastive_loss_weight > 0.0
            or semantic_context_loss_weight > 0.0
            or semantic_content_loss_weight > 0.0
            or semantic_raw_anchor_loss_weight > 0.0
            or semantic_geometry_loss_weight > 0.0
            or semantic_rank_distill_loss_weight > 0.0
        )
        if any_semantic_objective:
            if predicted_semantic is None:
                raise ValueError("Semantic alignment loss requires an adapter with one semantic encoded token.")
            target_semantic = alignment_targets if alignment_targets is not None else semantic_vectors
            if bool(getattr(adapter, "normalize_semantic_output", False)):
                target_semantic = F.normalize(target_semantic.float(), p=2, dim=-1)
            if tuple(predicted_semantic.shape) != tuple(target_semantic.shape):
                raise ValueError(
                    "Semantic alignment shape mismatch: "
                    f"predicted={tuple(predicted_semantic.shape)} target={tuple(target_semantic.shape)}"
                )
            pred_semantic = predicted_semantic.float()
            target_semantic = target_semantic.float()
            semantic_alignment_cosine = F.cosine_similarity(pred_semantic, target_semantic, dim=-1)
            if semantic_alignment_loss_weight > 0.0:
                if semantic_alignment_loss_type == "cosine":
                    semantic_alignment_loss = 1.0 - semantic_alignment_cosine.mean()
                elif semantic_alignment_loss_type == "mse":
                    semantic_alignment_loss = F.mse_loss(pred_semantic, target_semantic)
                else:
                    raise ValueError(f"Unsupported semantic_alignment_loss_type={semantic_alignment_loss_type!r}")
                scaled_semantic_alignment_loss = semantic_alignment_loss_weight * semantic_alignment_loss
            if semantic_contrastive_loss_weight > 0.0:
                if semantic_contrastive_temperature <= 0.0:
                    raise ValueError("semantic_contrastive_temperature must be positive")
                pred_normalized = F.normalize(pred_semantic, p=2, dim=-1)
                target_normalized = F.normalize(target_semantic, p=2, dim=-1)
                logits = pred_normalized @ target_normalized.T / semantic_contrastive_temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                semantic_contrastive_loss = 0.5 * (
                    F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
                )
                scaled_semantic_contrastive_loss = (
                    semantic_contrastive_loss_weight * semantic_contrastive_loss
                )
            if semantic_context_loss_weight > 0.0:
                with torch.no_grad():
                    if semantic_context_teacher is not None:
                        semantic_context_teacher.eval()
                        target_context, _ = semantic_context_teacher(target_semantic)
                    else:
                        context_from_projected = getattr(
                            adapter, "context_from_projected_semantic", None
                        )
                        if context_from_projected is None:
                            raise ValueError(
                                "Semantic context consistency requires either a frozen "
                                "semantic_context_teacher or context_from_projected_semantic()."
                            )
                        target_context, _ = context_from_projected(target_semantic)
                semantic_context_loss = 1.0 - F.cosine_similarity(
                    context.float(), target_context.float(), dim=-1
                ).mean()
                scaled_semantic_context_loss = semantic_context_loss_weight * semantic_context_loss
            if semantic_content_loss_weight > 0.0:
                if semantic_content_targets is None or semantic_content_directions is None:
                    raise ValueError(
                        "Semantic content loss requires target masks and train-only prototype directions."
                    )
                semantic_content_loss = semantic_content_ranking_loss(
                    pred_semantic,
                    semantic_content_targets.to(device=device),
                    semantic_content_directions.to(device=device),
                    temperature=semantic_content_temperature,
                    negative_topk=semantic_content_negative_topk,
                )
                scaled_semantic_content_loss = semantic_content_loss_weight * semantic_content_loss
            if (
                semantic_raw_anchor_loss_weight > 0.0
                or semantic_geometry_loss_weight > 0.0
                or semantic_rank_distill_loss_weight > 0.0
            ):
                preservation = semantic_preservation_losses(
                    pred_semantic,
                    semantic_vectors,
                    exact_targets=(
                        target_semantic if semantic_rank_distill_loss_weight > 0.0 else None
                    ),
                    rank_temperature=semantic_rank_distill_temperature,
                )
                semantic_raw_anchor_loss = preservation["anchor"]
                semantic_raw_anchor_cosine = preservation["anchor_cosine"]
                semantic_geometry_loss = preservation["geometry"]
                semantic_rank_distill_loss = preservation["rank_distill"]
                if (
                    semantic_rank_distill_loss_weight > 0.0
                    and semantic_rank_teacher_adapter is not None
                ):
                    with torch.no_grad():
                        semantic_rank_teacher_adapter.eval()
                        teacher_output = semantic_rank_teacher_adapter(
                            meg,
                            meg_lengths=meg_lengths,
                            subjects=subject_ids,
                        )
                        teacher_semantic = teacher_output.encoded_sequence.squeeze(1)
                    semantic_rank_distill_loss = semantic_preservation_losses(
                        pred_semantic,
                        teacher_semantic,
                        exact_targets=target_semantic,
                        rank_temperature=semantic_rank_distill_temperature,
                    )["rank_distill"]
                scaled_semantic_raw_anchor_loss = (
                    semantic_raw_anchor_loss_weight * semantic_raw_anchor_loss
                )
                scaled_semantic_geometry_loss = (
                    semantic_geometry_loss_weight * semantic_geometry_loss
                )
                scaled_semantic_rank_distill_loss = (
                    semantic_rank_distill_loss_weight * semantic_rank_distill_loss
                )
        scaled_decoder_objective = (
            scaled_decoder_loss
            + scaled_content_bow_loss
            + scaled_content_precision_loss
            + scaled_condition_pairing_loss
            + scaled_semantic_alignment_loss
            + scaled_semantic_contrastive_loss
            + scaled_semantic_context_loss
            + scaled_semantic_content_loss
            + scaled_semantic_raw_anchor_loss
            + scaled_semantic_geometry_loss
            + scaled_semantic_rank_distill_loss
        )
    scaled_decoder_objective.backward()

    grad_metrics = optimizer_grad_metrics(optimizer)
    optimizer.step()
    loss_value = (
        config.denoiser_loss_weight * denoiser_loss_value
        + float(scaled_decoder_loss.detach().cpu())
        + float(scaled_content_bow_loss.detach().cpu())
        + float(scaled_content_precision_loss.detach().cpu())
        + float(scaled_condition_pairing_loss.detach().cpu())
        + float(scaled_semantic_alignment_loss.detach().cpu())
        + float(scaled_semantic_contrastive_loss.detach().cpu())
        + float(scaled_semantic_context_loss.detach().cpu())
        + float(scaled_semantic_content_loss.detach().cpu())
        + float(scaled_semantic_raw_anchor_loss.detach().cpu())
        + float(scaled_semantic_geometry_loss.detach().cpu())
        + float(scaled_semantic_rank_distill_loss.detach().cpu())
    )
    metrics = {
        "loss": loss_value,
        "denoiser_loss": denoiser_loss_value,
        "decoder_loss": float(decoder_loss.detach().cpu()),
        "content_bow_loss": float(content_bow_loss.detach().cpu()),
        "content_bow_loss_scaled": float(scaled_content_bow_loss.detach().cpu()),
        "content_precision_loss": float(content_precision_loss.detach().cpu()),
        "content_precision_loss_scaled": float(scaled_content_precision_loss.detach().cpu()),
        "condition_pairing_loss": float(condition_pairing_loss.detach().cpu()),
        "condition_pairing_loss_scaled": float(
            scaled_condition_pairing_loss.detach().cpu()
        ),
        "semantic_alignment_loss": (
            float(semantic_alignment_loss.detach().cpu()) if semantic_alignment_loss is not None else 0.0
        ),
        "semantic_alignment_loss_scaled": float(scaled_semantic_alignment_loss.detach().cpu()),
        "semantic_alignment_cosine": (
            float(semantic_alignment_cosine.mean().detach().cpu()) if semantic_alignment_cosine is not None else 0.0
        ),
        "semantic_contrastive_loss": (
            float(semantic_contrastive_loss.detach().cpu()) if semantic_contrastive_loss is not None else 0.0
        ),
        "semantic_contrastive_loss_scaled": float(scaled_semantic_contrastive_loss.detach().cpu()),
        "semantic_context_loss": (
            float(semantic_context_loss.detach().cpu()) if semantic_context_loss is not None else 0.0
        ),
        "semantic_context_loss_scaled": float(scaled_semantic_context_loss.detach().cpu()),
        "semantic_content_loss": (
            float(semantic_content_loss.detach().cpu()) if semantic_content_loss is not None else 0.0
        ),
        "semantic_content_loss_scaled": float(scaled_semantic_content_loss.detach().cpu()),
        "semantic_raw_anchor_loss": (
            float(semantic_raw_anchor_loss.detach().cpu())
            if semantic_raw_anchor_loss is not None
            else 0.0
        ),
        "semantic_raw_anchor_loss_scaled": float(
            scaled_semantic_raw_anchor_loss.detach().cpu()
        ),
        "semantic_raw_anchor_cosine": (
            float(semantic_raw_anchor_cosine.detach().cpu())
            if semantic_raw_anchor_cosine is not None
            else 0.0
        ),
        "semantic_geometry_loss": (
            float(semantic_geometry_loss.detach().cpu())
            if semantic_geometry_loss is not None
            else 0.0
        ),
        "semantic_geometry_loss_scaled": float(scaled_semantic_geometry_loss.detach().cpu()),
        "semantic_rank_distill_loss": (
            float(semantic_rank_distill_loss.detach().cpu())
            if semantic_rank_distill_loss is not None
            else 0.0
        ),
        "semantic_rank_distill_loss_scaled": float(
            scaled_semantic_rank_distill_loss.detach().cpu()
        ),
    }
    metrics.update(grad_metrics)
    return metrics


@torch.no_grad()
def encode_condition_contexts(
    *,
    adapter: nn.Module,
    meg: torch.Tensor,
    meg_lengths: torch.Tensor,
    semantic_vectors: torch.Tensor,
    subject_ids: torch.Tensor,
    device: torch.device,
    condition_source: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode long MEG windows without materializing attention for the full split."""
    total = int(meg.shape[0] if condition_source == "meg" else semantic_vectors.shape[0])
    chunk_size = total if batch_size <= 0 else min(total, int(batch_size))
    contexts: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        if condition_source == "meg":
            output = adapter(
                meg[start:stop].to(device=device, dtype=torch.float32),
                meg_lengths=meg_lengths[start:stop].to(device=device, dtype=torch.long),
                subjects=subject_ids[start:stop].to(device=device, dtype=torch.long),
            )
            context = output.context
            mask = output.context_mask.to(device=device, dtype=context.dtype)
        elif condition_source == "semantic":
            context, mask = adapter(semantic_vectors[start:stop].to(device=device, dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported condition_source: {condition_source}")
        contexts.append(context)
        masks.append(mask.to(device=device, dtype=context.dtype))
    return torch.cat(contexts, dim=0), torch.cat(masks, dim=0)


@torch.no_grad()
def evaluate_generation(
    *,
    model: nn.Module,
    adapter: nn.Module,
    meg: torch.Tensor,
    meg_lengths: torch.Tensor,
    semantic_vectors: torch.Tensor,
    subject_ids: torch.Tensor,
    tokenizer,
    encoder: nn.Module | None = None,
    gpt2_tokenizer=None,
    gpt2_model: nn.Module | None = None,
    target_sentences: Sequence[str],
    target_latents: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    target_length: int,
    context_length: int,
    config: Config,
    sampling_config: SamplingConfig,
    device: torch.device,
    generator: torch.Generator,
    condition_source: str,
    condition_batch_size: int = 0,
) -> dict:
    model.eval()
    adapter.eval()

    context, context_mask = encode_condition_contexts(
        adapter=adapter,
        meg=meg,
        meg_lengths=meg_lengths,
        semantic_vectors=semantic_vectors,
        subject_ids=subject_ids,
        device=device,
        condition_source=condition_source,
        batch_size=condition_batch_size,
    )

    generation_batch_size = context.shape[0] if condition_batch_size <= 0 else int(condition_batch_size)
    generated: list[str] = []
    lexical_predictions: list[str] = []
    for start in range(0, context.shape[0], generation_batch_size):
        context_chunk = context[start : start + generation_batch_size]
        mask_chunk = context_mask[start : start + generation_batch_size]
        zeros_target = torch.zeros(
            (context_chunk.shape[0], target_length, context_chunk.shape[-1]),
            dtype=context_chunk.dtype,
            device=device,
        )
        cond_seq = torch.cat([context_chunk, zeros_target], dim=1)
        cond_mask = torch.cat(
            [
                mask_chunk,
                torch.zeros((context_chunk.shape[0], target_length), dtype=context_chunk.dtype, device=device),
            ],
            dim=1,
        )
        t_steps = get_sampling_steps(
            n_steps=sampling_config.num_sampling_steps[0],
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean,
            P_std=config.denoiser_p_std,
            device=device,
            dtype=context_chunk.dtype,
        )
        z = torch.randn(cond_seq.shape, generator=generator, device=device, dtype=context_chunk.dtype)
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
        lexical_token_bias = None
        lexical_sequence_ids = None
        lexical_sequence_bias = None
        ordered_token_ids = None
        ordered_token_bias = None
        lexical_bias_method = getattr(adapter, "lexical_token_logit_bias", None)
        if lexical_bias_method is not None and condition_source == "semantic":
            semantic_chunk = semantic_vectors[start : start + generation_batch_size].to(
                device=device, dtype=torch.float32
            )
            lexical_token_bias = lexical_bias_method(
                semantic_chunk,
                vocabulary_size=len(tokenizer),
            )
            if bool(getattr(adapter, "lexical_decode_bias_once", False)):
                lexical_sequence_method = getattr(
                    adapter, "lexical_token_sequence_bias", None
                )
                if (
                    getattr(adapter, "lexical_decode_bias_mode", "sequence")
                    == "sequence"
                    and lexical_sequence_method is not None
                ):
                    sequence_payload = lexical_sequence_method(
                        semantic_chunk, vocabulary_size=len(tokenizer)
                    )
                    if sequence_payload is not None:
                        lexical_sequence_ids, lexical_sequence_bias = sequence_payload
                        lexical_token_bias = None
            ordered_bias_method = getattr(
                adapter, "ordered_token_position_bias", None
            )
            if ordered_bias_method is not None:
                ordered_payload = ordered_bias_method(
                    semantic_chunk, vocabulary_size=len(tokenizer)
                )
                if ordered_payload is not None:
                    ordered_token_ids, ordered_token_bias = ordered_payload
            lexical_vocabulary = getattr(adapter, "lexical_vocabulary", None)
            lexical_selection = getattr(adapter, "lexical_selection", None)
            if lexical_vocabulary is not None and lexical_selection is not None:
                raw_semantic = adapter.fmri2sem(semantic_chunk)
                lexical_indices, _ = lexical_selection(raw_semantic)
                lexical_predictions.extend(
                    " ".join(lexical_vocabulary[index] for index in row)
                    for row in lexical_indices.detach().cpu().tolist()
                )
        predicted_ids = _dlm_decode_batch(
            z=latent,
            model=model,
            t_final_val=t_steps[-1].item(),
            config=config,
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
            token_logit_bias=lexical_token_bias,
            token_logit_bias_once=bool(
                getattr(adapter, "lexical_decode_bias_once", False)
            ),
            token_logit_bias_target_start=context_chunk.shape[1],
            token_logit_bias_max_positions=int(
                getattr(adapter, "lexical_decode_bias_max_positions", 0)
            ),
            token_sequence_ids=lexical_sequence_ids,
            token_sequence_bias=lexical_sequence_bias,
            ordered_token_ids=ordered_token_ids,
            ordered_token_bias=ordered_token_bias,
            ordered_token_max_positions=int(
                getattr(adapter, "ordered_decode_max_positions", 0)
            ),
        )
        shift = torch.full((context_chunk.shape[0],), context_length, dtype=torch.long, device=device)
        predicted_ids = shift_left(predicted_ids, shift, 0)[:, :target_length]
        predicted_ids = mask_after_eos(
            predicted_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        generated.extend(
            tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True).strip()
            for row in predicted_ids
        )
    normalized_targets = [sentence.strip() for sentence in target_sentences]
    exact = [int(gen == tgt) for gen, tgt in zip(generated, normalized_targets)]
    overlap = word_overlap_metrics(generated, normalized_targets)
    generation_quality = generation_repetition_metrics(generated)
    generation_quality.update(overlap["summary"])
    metrics = {
        "generated": generated,
        "targets": normalized_targets,
        "exact": exact,
        "exact_match": float(sum(exact) / len(exact)),
        "generation_quality": generation_quality,
        "word_overlap": overlap,
    }
    if len(lexical_predictions) == len(normalized_targets):
        metrics["lexical_predictions"] = lexical_predictions
        metrics["lexical_probe_quality"] = word_overlap_metrics(
            lexical_predictions, normalized_targets
        )["summary"]
    if encoder is not None and target_latents is not None and target_mask is not None:
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
        target_pooled = mean_pool_latents(target_latents.to(device), target_mask.to(device))
        similarity = generated_pooled @ target_pooled.T
        metrics["generation_t5_retrieval"] = rank_true_targets_by_similarity(similarity)
    if gpt2_tokenizer is not None and gpt2_model is not None:
        metrics["generation_quality"].update(causal_lm_perplexity(
            generated,
            tokenizer=gpt2_tokenizer,
            model=gpt2_model,
            device=device,
        ))
    return metrics


@torch.no_grad()
def evaluate_retrieval(
    *,
    model: nn.Module,
    adapter: nn.Module,
    meg: torch.Tensor,
    meg_lengths: torch.Tensor,
    semantic_vectors: torch.Tensor,
    subject_ids: torch.Tensor,
    target_latents: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_sentences: Sequence[str],
    config: Config,
    device: torch.device,
    condition_source: str,
    retrieval_batch_size: int,
    retrieval_t: float,
    condition_batch_size: int = 0,
) -> dict:
    model.eval()
    adapter.eval()

    contexts, context_masks = encode_condition_contexts(
        adapter=adapter,
        meg=meg,
        meg_lengths=meg_lengths,
        semantic_vectors=semantic_vectors,
        subject_ids=subject_ids,
        device=device,
        condition_source=condition_source,
        batch_size=condition_batch_size,
    )

    target_latents = target_latents.to(device)
    target_ids = target_ids.to(device)
    target_mask = target_mask.to(device=device, dtype=torch.float32)

    num_queries = contexts.shape[0]
    num_candidates = target_latents.shape[0]
    denoiser_scores = torch.empty((num_queries, num_candidates), dtype=torch.float32, device=device)
    decoder_scores = torch.empty((num_queries, num_candidates), dtype=torch.float32, device=device)
    combined_scores = torch.empty((num_queries, num_candidates), dtype=torch.float32, device=device)

    pairs = [(query_idx, cand_idx) for query_idx in range(num_queries) for cand_idx in range(num_candidates)]
    for start in range(0, len(pairs), retrieval_batch_size):
        chunk = pairs[start : start + retrieval_batch_size]
        query_idx = torch.tensor([pair[0] for pair in chunk], dtype=torch.long, device=device)
        cand_idx = torch.tensor([pair[1] for pair in chunk], dtype=torch.long, device=device)

        context = contexts.index_select(0, query_idx)
        context_mask = context_masks.index_select(0, query_idx)
        candidate_latents = target_latents.index_select(0, cand_idx)
        candidate_ids = target_ids.index_select(0, cand_idx)
        candidate_mask = target_mask.index_select(0, cand_idx)

        x0 = torch.cat([context, candidate_latents], dim=1)
        target_prefix = torch.zeros(
            (x0.shape[0], candidate_latents.shape[1]),
            dtype=context_mask.dtype,
            device=device,
        )
        target_valid_mask = candidate_mask.to(dtype=context_mask.dtype)
        cond_seq_mask = torch.cat([context_mask, target_prefix], dim=1)
        attention_mask = torch.cat([context_mask, target_valid_mask], dim=1)
        latent_target_mask = torch.cat([torch.zeros_like(context_mask), target_valid_mask], dim=1).unsqueeze(-1)

        t = torch.full((x0.shape[0],), retrieval_t, dtype=x0.dtype, device=device)
        # Use a deterministic sinusoidal perturbation so each pair is scored stably.
        noise = torch.sin(torch.arange(x0.numel(), device=device, dtype=x0.dtype)).reshape_as(x0)
        z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask.unsqueeze(-1))

        use_bf16 = bool(config.use_bf16) and device.type == "cuda"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred, _ = model(
                z,
                t,
                attention_mask=attention_mask,
                deterministic=True,
                self_cond_cfg_scale=torch.ones((x0.shape[0],), dtype=x0.dtype, device=device),
            )
            v_pred, _ = net_out_to_v_x(pred, z, t, config.t_eps)
            t_expanded = t.reshape(-1, 1, 1)
            v_target = (x0 - z) / torch.clamp(1.0 - t_expanded, min=config.t_eps)
            denoiser_per_pair = (
                ((v_pred - v_target) ** 2 * latent_target_mask).sum(dim=(1, 2))
                / latent_target_mask.sum(dim=(1, 2)).clamp_min(1.0)
                / x0.shape[-1]
            )

            decoder_input = (
                torch.cat([z, torch.zeros_like(z)], dim=-1)
                if config.self_cond_prob > 0
                else z
            )
            _, decoder_logits = model(
                decoder_input,
                torch.ones((x0.shape[0],), dtype=x0.dtype, device=device),
                attention_mask=attention_mask,
                deterministic=True,
                self_cond_cfg_scale=torch.ones((x0.shape[0],), dtype=x0.dtype, device=device),
                decoder_step_active=True,
            )
            target_logits = decoder_logits[:, context.shape[1]:]
            ce_per_token = F.cross_entropy(
                target_logits.transpose(1, 2).to(torch.float32),
                candidate_ids,
                reduction="none",
            )
            decoder_per_pair = (
                (ce_per_token * candidate_mask).sum(dim=1)
                / candidate_mask.sum(dim=1).clamp_min(1.0)
            )
            combined_per_pair = (
                config.denoiser_loss_weight * denoiser_per_pair
                + config.decoder_loss_weight * decoder_per_pair
            )

        denoiser_scores[query_idx, cand_idx] = denoiser_per_pair.to(torch.float32)
        decoder_scores[query_idx, cand_idx] = decoder_per_pair.to(torch.float32)
        combined_scores[query_idx, cand_idx] = combined_per_pair.to(torch.float32)

    def _rank_metrics(scores: torch.Tensor) -> dict:
        sorted_indices = torch.argsort(scores, dim=1)
        labels = torch.arange(num_queries, dtype=torch.long, device=device)
        ranks = (sorted_indices == labels[:, None]).nonzero()[:, 1] + 1
        true_scores = scores[labels, labels]
        masked = scores.clone()
        masked[labels, labels] = float("inf")
        best_distractor_scores = masked.min(dim=1).values
        random_indices = (labels + 1) % num_candidates
        return {
            "top1": float((ranks == 1).float().mean().cpu()),
            "top5": float((ranks <= min(5, num_candidates)).float().mean().cpu()),
            "mean_rank": float(ranks.float().mean().cpu()),
            "median_rank": float(ranks.float().median().cpu()),
            "true_loss_mean": float(true_scores.mean().cpu()),
            "best_distractor_loss_mean": float(best_distractor_scores.mean().cpu()),
            "next_candidate_loss_mean": float(scores[labels, random_indices].mean().cpu()),
            "true_minus_best_distractor_mean": float(
                (true_scores - best_distractor_scores).mean().cpu()
            ),
            "true_minus_next_candidate_mean": float(
                (true_scores - scores[labels, random_indices]).mean().cpu()
            ),
            "ranks": ranks.detach().cpu().tolist(),
            "top_indices": sorted_indices[:, : min(5, num_candidates)].detach().cpu().tolist(),
        }

    metrics = {
        "targets": [sentence.strip() for sentence in target_sentences],
        "combined": _rank_metrics(combined_scores),
        "denoiser": _rank_metrics(denoiser_scores),
        "decoder": _rank_metrics(decoder_scores),
    }
    return metrics


def maybe_init_wandb(args: argparse.Namespace, config: Config):
    if not args.use_wandb or wandb is None:
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        notes=args.wandb_notes,
        config={
            "checkpoint_path": args.checkpoint_path,
            "books": args.books,
            "num_examples": args.num_examples,
            "context_length": args.context_length,
            "condition_source": args.condition_source,
            "embedding_type": args.embedding_type,
            "steps": args.steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_encode_batch_size": args.target_encode_batch_size,
            "eval_num_examples": args.eval_num_examples,
            "retrieval_eval_every": args.retrieval_eval_every,
            "retrieval_num_examples": args.retrieval_num_examples,
            "generation_t5_retrieval": args.generation_t5_retrieval,
            "generation_gpt2_ppl": args.generation_gpt2_ppl,
            "gpt2_model_name": args.gpt2_model_name,
            "train_timestep_mode": args.train_timestep_mode,
            "train_timestep_steps": args.train_timestep_steps,
            "log_wandb_samples": not args.no_wandb_samples,
            "lr": args.lr,
            "freeze_elf": args.freeze_elf,
            "last_n_blocks": args.last_n_blocks,
            "decoder_loss_weight": args.decoder_loss_weight,
            "denoiser_loss_weight": args.denoiser_loss_weight,
            "cond_dropout_prob": args.cond_dropout_prob,
            "max_length": config.max_length,
        },
    )


def save_results(output_dir: str, step: int, metrics: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"eval_step_{step:04d}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_best_results(output_dir: str, metrics: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "best_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def eval_checkpoint_scores(metrics: dict) -> dict[str, float]:
    """Score eval checkpoints by combined quality plus individual eval criteria."""
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    well_structured = quality.get("well_structured_sentence")
    content_words_overlap = quality.get("content_words_overlap")
    words_overlap = quality.get("words_overlap")
    word_error_rate = quality.get("word_error_rate")
    mean_rank = retrieval.get("mean_rank")
    eval_num_examples = max(1, int(metrics.get("eval_num_examples") or 1))
    scores = {}
    if well_structured is not None and mean_rank is not None:
        scores["structured_rank"] = float(well_structured) - (float(mean_rank) / eval_num_examples)
    elif mean_rank is not None:
        scores["structured_rank"] = -float(mean_rank) / eval_num_examples
    else:
        scores["structured_rank"] = float(metrics.get("exact_match") or 0.0)
    if well_structured is not None:
        scores["well_structured_sentence"] = float(well_structured)
    if content_words_overlap is not None:
        scores["content_words_overlap"] = float(content_words_overlap)
    if words_overlap is not None:
        scores["words_overlap"] = float(words_overlap)
    if word_error_rate is not None:
        scores["negative_word_error_rate"] = -float(word_error_rate)
    if mean_rank is not None:
        scores["mean_rank"] = -float(mean_rank)
    if retrieval.get("top5") is not None:
        scores["generation_t5_top5"] = float(retrieval["top5"])
    return scores


def save_eval_checkpoint(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    metrics: dict,
    eval_index: int,
    saved_checkpoints: list[dict],
) -> None:
    if not args.save_eval_checkpoints:
        return

    top_k = max(0, int(args.eval_checkpoint_top_k))
    periodic = int(args.eval_checkpoint_every)
    scores = eval_checkpoint_scores(metrics)
    step = int(metrics["step"])
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    should_save = False
    if top_k > 0:
        for criterion, score in scores.items():
            criterion_items = [item for item in saved_checkpoints if criterion in item["scores"]]
            if len(criterion_items) < top_k or score > min(item["scores"][criterion] for item in criterion_items):
                should_save = True
                break
    if periodic > 0 and eval_index % periodic == 0:
        should_save = True
    if not should_save:
        return

    payload = {
        "step": step,
        "epoch": metrics.get("epoch"),
        "score": scores["structured_rank"],
        "scores": scores,
        "metrics": metrics,
        "args": vars(args),
        "config": {
            key: value
            for key, value in vars(config).items()
            if isinstance(value, (str, int, float, bool, type(None), list, tuple))
        },
        "model_state_dict": model.state_dict(),
        "adapter_state_dict": adapter.state_dict(),
    }
    if args.include_optimizer_in_checkpoints:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    path = checkpoint_dir / f"eval_step_{step:08d}_score_{scores['structured_rank']:.6f}.pt"
    torch.save(payload, path)
    saved_checkpoints.append({"path": path, "score": scores["structured_rank"], "scores": scores, "step": step})

    periodic_paths = set()
    if periodic > 0:
        periodic_paths = {
            item["path"]
            for item in saved_checkpoints
            if item["step"] % (periodic * max(1, int(args.eval_every))) == 0
        }

    keep_paths = set(periodic_paths)
    criteria = sorted({criterion for item in saved_checkpoints for criterion in item["scores"]})
    criterion_rankings = {}
    for criterion in criteria:
        ranked_for_criterion = sorted(
            [item for item in saved_checkpoints if criterion in item["scores"]],
            key=lambda item: item["scores"][criterion],
            reverse=True,
        )
        criterion_rankings[criterion] = ranked_for_criterion[:top_k]
        keep_paths.update(item["path"] for item in criterion_rankings[criterion])
    for item in list(saved_checkpoints):
        if item["path"] in keep_paths:
            continue
        try:
            item["path"].unlink()
        except FileNotFoundError:
            pass
        saved_checkpoints.remove(item)

    manifest = sorted(saved_checkpoints, key=lambda item: item["score"], reverse=True)
    criterion_rankings = {
        criterion: [
            {"path": str(item["path"]), "score": item["scores"][criterion], "step": item["step"]}
            for item in sorted(
                [checkpoint for checkpoint in saved_checkpoints if criterion in checkpoint["scores"]],
                key=lambda checkpoint: checkpoint["scores"][criterion],
                reverse=True,
            )[:top_k]
        ]
        for criterion in sorted({criterion for item in saved_checkpoints for criterion in item["scores"]})
    }
    manifest_path = checkpoint_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "criteria": criterion_rankings,
                "checkpoints": [
                    {"path": str(item["path"]), "score": item["score"], "scores": item["scores"], "step": item["step"]}
                    for item in manifest
                ],
            },
            f,
            indent=2,
        )
    log_for_0(f"Saved eval checkpoint {path} scores={scores}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    log_for_0(f"Using device: {device}")

    dataset_batch = load_overfit_examples(args)
    eval_batch = load_eval_examples(args, dataset_batch)
    target_sentences = dataset_batch["sentences"]
    eval_sentences = eval_batch["sentences"]
    meg = dataset_batch["meg"]
    meg_lengths = dataset_batch["meg_lengths"]
    semantic_vectors = dataset_batch["semantic_vectors"]
    subject_ids = dataset_batch["subject_ids"]
    subject_to_id = dataset_batch["subject_to_id"]
    eval_meg = eval_batch["meg"]
    eval_meg_lengths = eval_batch["meg_lengths"]
    eval_semantic_vectors = eval_batch["semantic_vectors"]
    eval_subject_ids = eval_batch["subject_ids"]

    log_for_0(
        f"Loaded {len(target_sentences)} examples across {len(subject_to_id)} subjects; "
        f"meg_shape={tuple(meg.shape)} semantic_shape={tuple(semantic_vectors.shape)}"
    )
    if eval_batch is dataset_batch:
        log_for_0("Eval split: train_head; generation/retrieval eval uses the first eval examples from training data.")
    else:
        log_for_0(
            f"Eval split: {args.eval_split}; loaded {len(eval_sentences)} held-out examples "
            f"from Sherlock1 sessions={args.eval_sherlock1_sessions}."
        )
    num_train_examples = len(target_sentences)
    steps_per_epoch = int(np.ceil(num_train_examples / args.batch_size))
    if args.epochs > 0:
        args.steps = int(np.ceil(steps_per_epoch * args.epochs))
    log_for_0(
        f"Training schedule: num_examples={num_train_examples} "
        f"batch_size={args.batch_size} steps_per_epoch={steps_per_epoch} "
        f"epochs={args.steps / steps_per_epoch:.3f} steps={args.steps}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = tokenize_sentences(tokenizer, target_sentences)
    eval_input_ids, eval_attention_mask = (
        (input_ids, attention_mask)
        if eval_batch is dataset_batch
        else tokenize_sentences(tokenizer, eval_sentences)
    )
    target_length = input_ids.shape[1]
    eval_target_length = eval_input_ids.shape[1]
    max_length = args.context_length + max(target_length, eval_target_length)
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

    if args.condition_source == "meg":
        adapter = MEGContextAdapter(
            in_channels=int(meg.shape[1]),
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
    elif args.condition_source == "semantic":
        adapter = SemanticVectorContextProjector(
            input_dim=int(semantic_vectors.shape[-1]),
            context_dim=encoder_config.d_model,
            context_length=args.context_length,
            hidden_dim=args.semantic_hidden_dim,
            dropout=args.adapter_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported condition_source: {args.condition_source}")
    log_for_0(f"Trainable adapter parameters: {sum(p.numel() for p in adapter.parameters() if p.requires_grad):,}")

    log_for_0(f"Encoding target T5 latents in batches of {args.target_encode_batch_size}")
    target_latents = encode_text_batched(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        device=device,
        batch_size=args.target_encode_batch_size,
    )
    target_ids = input_ids.detach().cpu()
    target_mask = attention_mask.detach().cpu().to(torch.float32)
    if eval_batch is dataset_batch:
        eval_target_latents = target_latents
        eval_target_ids = target_ids
        eval_target_mask = target_mask
    else:
        eval_target_latents = encode_text_batched(
            input_ids=eval_input_ids,
            attention_mask=eval_attention_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            device=device,
            batch_size=args.target_encode_batch_size,
        )
        eval_target_ids = eval_input_ids.detach().cpu()
        eval_target_mask = eval_attention_mask.detach().cpu().to(torch.float32)

    gpt2_tokenizer = None
    gpt2_model = None
    if args.generation_gpt2_ppl:
        log_for_0(f"Loading generation quality LM: {args.gpt2_model_name}")
        gpt2_tokenizer = AutoTokenizer.from_pretrained(args.gpt2_model_name)
        if gpt2_tokenizer.pad_token_id is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.gpt2_model_name).to(device)
        gpt2_model.eval()
        for param in gpt2_model.parameters():
            param.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad] + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )

    run = maybe_init_wandb(args, config)
    if run is not None:
        wandb.define_metric("train/epoch")
        wandb.define_metric("eval/epoch")
        wandb.define_metric("retrieval/epoch")
        wandb.define_metric("train/*", step_metric="train/epoch")
        wandb.define_metric("eval/*", step_metric="eval/epoch")
        wandb.define_metric("retrieval/*", step_metric="retrieval/epoch")
        wandb.define_metric("generation_t5_retrieval/*", step_metric="eval/epoch")
        wandb.define_metric("generation_quality/*", step_metric="eval/epoch")
    noise_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    noise_generator.manual_seed(args.seed + 17)
    batch_generator = torch.Generator().manual_seed(args.seed + 29)
    eval_count = len(eval_sentences) if args.eval_num_examples <= 0 else min(args.eval_num_examples, len(eval_sentences))
    eval_slice = slice(0, eval_count)
    retrieval_count = (
        eval_count
        if args.retrieval_num_examples <= 0
        else min(args.retrieval_num_examples, len(eval_sentences))
    )
    retrieval_slice = slice(0, retrieval_count)

    best_metrics = None
    best_eval_score = None
    eval_index = 0
    saved_checkpoints: list[dict] = []
    for step in range(1, args.steps + 1):
        epoch = min(step * args.batch_size / num_train_examples, args.steps / steps_per_epoch)
        batches = build_batches(
            meg=meg,
            meg_lengths=meg_lengths,
            semantic_vectors=semantic_vectors,
            subject_ids=subject_ids,
            target_latents=target_latents,
            target_ids=target_ids,
            target_mask=target_mask,
            batch_size=args.batch_size,
            generator=batch_generator,
        )
        batch = batches[(step - 1) % len(batches)]
        loss_metrics = train_step(
            model=model,
            adapter=adapter,
            batch=batch,
            optimizer=optimizer,
            config=config,
            device=device,
            noise_generator=noise_generator,
            condition_source=args.condition_source,
            cond_dropout_prob=args.cond_dropout_prob,
            train_timestep_mode=args.train_timestep_mode,
            train_timestep_steps=args.train_timestep_steps
            if args.train_timestep_steps > 0
            else args.num_sampling_steps,
        )

        if step % 10 == 0 or step == 1:
            log_for_0(
                f"step={step} epoch={epoch:.4f} loss={loss_metrics['loss']:.6f} "
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
                    "train/epoch": epoch,
                    "train/examples_seen": step * args.batch_size,
                },
                step=step,
            )

        if step % args.eval_every != 0 and step != args.steps:
            if args.retrieval_eval_every <= 0 or step % args.retrieval_eval_every != 0:
                continue
        run_generation_eval = step % args.eval_every == 0 or step == args.steps
        run_retrieval_eval = args.retrieval_eval_every > 0 and (
            step % args.retrieval_eval_every == 0 or step == args.steps
        )

        if run_retrieval_eval:
            retrieval_metrics = evaluate_retrieval(
                model=model,
                adapter=adapter,
                meg=eval_meg[retrieval_slice],
                meg_lengths=eval_meg_lengths[retrieval_slice],
                semantic_vectors=eval_semantic_vectors[retrieval_slice],
                subject_ids=eval_subject_ids[retrieval_slice],
                target_latents=eval_target_latents[retrieval_slice],
                target_ids=eval_target_ids[retrieval_slice],
                target_mask=eval_target_mask[retrieval_slice],
                target_sentences=eval_sentences[retrieval_slice],
                config=config,
                device=device,
                condition_source=args.condition_source,
                retrieval_batch_size=args.retrieval_batch_size,
                retrieval_t=args.retrieval_t,
            )
            retrieval_metrics["step"] = step
            retrieval_metrics["epoch"] = epoch
            retrieval_metrics["eval_num_examples"] = retrieval_count
            os.makedirs(args.output_dir, exist_ok=True)
            retrieval_path = Path(args.output_dir) / f"retrieval_step_{step:04d}.json"
            with retrieval_path.open("w", encoding="utf-8") as f:
                json.dump(retrieval_metrics, f, ensure_ascii=False, indent=2)

            log_for_0(
                "retrieval "
                f"step={step} epoch={epoch:.4f} "
                f"combined_top1={retrieval_metrics['combined']['top1']:.3f} "
                f"combined_top5={retrieval_metrics['combined']['top5']:.3f} "
                f"combined_mean_rank={retrieval_metrics['combined']['mean_rank']:.2f} "
                f"denoiser_top1={retrieval_metrics['denoiser']['top1']:.3f} "
                f"decoder_top1={retrieval_metrics['decoder']['top1']:.3f}"
            )
            if run is not None:
                wandb.log(
                    {
                        "retrieval/combined_top1": retrieval_metrics["combined"]["top1"],
                        "retrieval/combined_top5": retrieval_metrics["combined"]["top5"],
                        "retrieval/combined_mean_rank": retrieval_metrics["combined"]["mean_rank"],
                        "retrieval/combined_true_minus_best": retrieval_metrics["combined"][
                            "true_minus_best_distractor_mean"
                        ],
                        "retrieval/denoiser_top1": retrieval_metrics["denoiser"]["top1"],
                        "retrieval/decoder_top1": retrieval_metrics["decoder"]["top1"],
                        "retrieval/epoch": epoch,
                    },
                    step=step,
                )

        if not run_generation_eval:
            continue

        metrics = evaluate_generation(
            model=model,
            adapter=adapter,
            meg=eval_meg[eval_slice],
            meg_lengths=eval_meg_lengths[eval_slice],
            semantic_vectors=eval_semantic_vectors[eval_slice],
            subject_ids=eval_subject_ids[eval_slice],
            tokenizer=tokenizer,
            encoder=encoder if args.generation_t5_retrieval else None,
            gpt2_tokenizer=gpt2_tokenizer,
            gpt2_model=gpt2_model,
            target_sentences=eval_sentences[eval_slice],
            target_latents=eval_target_latents[eval_slice] if args.generation_t5_retrieval else None,
            target_mask=eval_target_mask[eval_slice] if args.generation_t5_retrieval else None,
            target_length=eval_target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=noise_generator,
            condition_source=args.condition_source,
        )
        metrics["step"] = step
        metrics["epoch"] = epoch
        metrics["eval_num_examples"] = eval_count
        metrics["eval_split"] = args.eval_split
        metrics["subjects"] = eval_batch["subject_labels"][eval_slice]
        checkpoint_scores = eval_checkpoint_scores(metrics)
        metrics["eval_checkpoint_scores"] = checkpoint_scores
        save_results(args.output_dir, step, metrics)
        eval_index += 1
        save_eval_checkpoint(
            args=args,
            model=model,
            adapter=adapter,
            optimizer=optimizer,
            config=config,
            metrics=metrics,
            eval_index=eval_index,
            saved_checkpoints=saved_checkpoints,
        )
        eval_score = checkpoint_scores["structured_rank"]
        if best_eval_score is None or eval_score > best_eval_score:
            best_eval_score = eval_score
            best_metrics = metrics
            save_best_results(args.output_dir, best_metrics)
            log_for_0(f"New best eval structured_rank={eval_score:.6f}; saved best_metrics.json")

        quality = metrics.get("generation_quality", {})
        log_for_0(
            f"eval step={step} epoch={epoch:.4f} exact_match={metrics['exact_match']:.3f} "
            f"words_overlap={quality.get('words_overlap', float('nan')):.3f} "
            f"content_words_overlap={quality.get('content_words_overlap', float('nan')):.3f}"
        )
        for idx, (target, generated, exact) in enumerate(
            zip(metrics["targets"], metrics["generated"], metrics["exact"])
        ):
            log_for_0(f"[{idx}] exact={exact} target={target!r} generated={generated!r}")

        if run is not None:
            payload = {
                "eval/exact_match": metrics["exact_match"],
                "eval/epoch": epoch,
            }
            if "generation_t5_retrieval" in metrics:
                gen_retrieval = metrics["generation_t5_retrieval"]
                payload.update(
                    {
                        "generation_t5_retrieval/top1": gen_retrieval["top1"],
                        "generation_t5_retrieval/top5": gen_retrieval["top5"],
                        "generation_t5_retrieval/mean_rank": gen_retrieval["mean_rank"],
                        "generation_t5_retrieval/median_rank": gen_retrieval["median_rank"],
                    }
                )
            if "generation_quality" in metrics:
                generation_quality = metrics["generation_quality"]
                payload.update({
                    f"generation_quality/{key}": value
                    for key, value in generation_quality.items()
                    if isinstance(value, (int, float))
                })
            if args.no_wandb_samples:
                wandb.log(payload, step=step)
                continue
            overlap_per_sample = metrics.get("word_overlap", {}).get("per_sample", [])
            content_overlap_per_sample = metrics.get("word_overlap", {}).get("per_sample_content", [])
            content_overlap_counts = metrics.get("word_overlap", {}).get("per_sample_content_overlap_counts", [])
            table = wandb.Table(
                columns=[
                    "epoch",
                    "step",
                    "id",
                    "target",
                    "generated",
                    "exact",
                    "words_overlap",
                    "content_words_overlap",
                    "content_overlap_words",
                ]
            )
            for idx, (target, generated, exact) in enumerate(
                zip(metrics["targets"], metrics["generated"], metrics["exact"])
            ):
                overlap = float(overlap_per_sample[idx]) if idx < len(overlap_per_sample) else None
                content_overlap = (
                    float(content_overlap_per_sample[idx])
                    if idx < len(content_overlap_per_sample)
                    else None
                )
                content_words = (
                    format_overlap_counts(content_overlap_counts[idx])
                    if idx < len(content_overlap_counts)
                    else ""
                )
                table.add_data(epoch, step, idx, target, generated, exact, overlap, content_overlap, content_words)
            payload["eval/samples"] = table
            wandb.log(payload, step=step)

        if metrics["exact_match"] == 1.0:
            log_for_0("Reached perfect exact match; stopping early.")
            break

    if run is not None and best_metrics is not None:
        run.summary["best_exact_match"] = best_metrics["exact_match"]
        run.summary["best_structured_rank"] = best_eval_score
        run.finish()

    if best_metrics is not None:
        log_for_0(f"Best exact match: {best_metrics['exact_match']:.3f}")
        log_for_0(f"Best structured rank: {best_eval_score:.6f}")
        log_for_0(f"Saved best metrics to {Path(args.output_dir) / 'best_metrics.json'}")


if __name__ == "__main__":
    main()
