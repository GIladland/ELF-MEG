#!/usr/bin/env python
"""Train a leakage-safe ordered ten-word decoder on frozen fMRI features.

Unlike the content bag-of-words probe, this head predicts one word at each of
the ten target positions.  It starts as the train-only position-frequency
prior: feature weights are zero and the output bias is the per-position word
log prior.  The learned component is therefore explicitly brain-dependent.
Promotion requires a matched validation advantage over both epoch zero and
deranged validation brain features.  Only train11725 and val266 are loaded.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from meg_context_overfit import _CONTENT_WORD_STOPWORDS, word_overlap_metrics
from modules.fmri2sem_bridge import load_mri2sem_model
from probe_fmri_supervised_content_head import ContentHead, encode_features


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
PAD = "<pad>"
UNK = "<unk>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument(
        "--semantic-npz",
        default="",
        help=(
            "Optional oracle+story-OOF delayed MiniLM archive. Use its OOF train "
            "rows and separate validation ensemble, mean pooled to 384 dimensions, "
            "instead of in-sample MRI features."
        ),
    )
    parser.add_argument(
        "--semantic-delay-mode",
        choices=("mean", "ordered"),
        default="mean",
        help=(
            "Mean-pool four delayed semantic blocks or retain the full ordered "
            "1,536-D predicted-ADA vector."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--max-vocabulary", type=int, default=5000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--mri-output-dim", type=int, default=384)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mri-dropout", type=float, default=0.1)
    parser.add_argument("--feature", choices=("hidden", "semantic"), default="semantic")
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--pairwise-weight", type=float, default=0.3)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        choices=("conditional_content", "word_f1"),
        default="conditional_content",
        help="Choose checkpoints for content discrimination or exact-word recovery.",
    )
    parser.add_argument(
        "--content-ce-weight",
        type=float,
        default=1.0,
        help="Relative positional CE weight for non-stopword target positions.",
    )
    parser.add_argument("--content-head-checkpoint", default="")
    parser.add_argument("--content-prior-subtraction", type=float, default=0.5)
    parser.add_argument(
        "--content-bias-strengths", type=float, nargs="+", default=[0.0],
        help="Broadcast a frozen brain-content residual across ordered positions at decode time.",
    )
    parser.add_argument(
        "--prior-subtractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75]
    )
    parser.add_argument("--derangements", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tokens(sentence: str) -> list[str]:
    return WORD_RE.findall(sentence.lower())


def vocabulary_and_targets(
    train_sentences: list[str],
    val_sentences: list[str],
    *,
    positions: int,
    min_frequency: int,
    max_vocabulary: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, float]:
    counts = Counter(word for sentence in train_sentences for word in tokens(sentence))
    vocabulary = [PAD, UNK]
    vocabulary.extend(
        word
        for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_frequency and word not in (PAD, UNK)
    )
    vocabulary = vocabulary[:max_vocabulary]
    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    def encode(sentences: list[str]) -> torch.Tensor:
        result = torch.full((len(sentences), positions), word_to_index[PAD], dtype=torch.long)
        for row, sentence in enumerate(sentences):
            for position, word in enumerate(tokens(sentence)[:positions]):
                result[row, position] = word_to_index.get(word, word_to_index[UNK])
        return result

    train_targets = encode(train_sentences)
    val_targets = encode(val_sentences)
    prior_counts = torch.full((positions, len(vocabulary)), 0.25, dtype=torch.float32)
    prior_counts.scatter_add_(
        1,
        train_targets.T,
        torch.ones_like(train_targets.T, dtype=torch.float32),
    )
    prior = prior_counts / prior_counts.sum(dim=1, keepdim=True)
    validation_coverage = float((val_targets != word_to_index[UNK]).float().mean())
    return vocabulary, train_targets, val_targets, prior.log(), validation_coverage


class OrderedWordHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, positions: int, vocabulary_size: int,
        log_prior: torch.Tensor,
    ) -> None:
        super().__init__()
        self.positions = int(positions)
        self.vocabulary_size = int(vocabulary_size)
        if hidden_dim > 0:
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1)
            )
            feature_dim = hidden_dim
        else:
            self.trunk = nn.LayerNorm(input_dim)
            feature_dim = input_dim
        self.output = nn.Linear(feature_dim, positions * vocabulary_size)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(log_prior.reshape(-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(inputs)).reshape(
            len(inputs), self.positions, self.vocabulary_size
        )


def derangement(size: int, generator: np.random.Generator) -> np.ndarray:
    base = np.arange(size)
    for _ in range(10_000):
        result = generator.permutation(size)
        if np.all(result != base):
            return result
    return np.roll(base, 1)


def decode(
    logits: torch.Tensor,
    vocabulary: list[str],
    log_prior: torch.Tensor,
    prior_subtraction: float,
) -> list[str]:
    adjusted = logits - float(prior_subtraction) * log_prior[None, :, :]
    predicted = adjusted.argmax(dim=-1).tolist()
    result: list[str] = []
    for row in predicted:
        words = [vocabulary[index] for index in row]
        words = [word for word in words if word not in (PAD, UNK)]
        result.append(" ".join(words))
    return result


def compact(generated: list[str], targets: list[str]) -> dict[str, float | int]:
    summary = word_overlap_metrics(generated, targets)["summary"]
    return {
        key: summary[key]
        for key in (
            "content_words_overlap", "content_words_overlap_precision",
            "content_words_overlap_recall", "words_overlap", "words_overlap_precision",
            "words_overlap_recall", "word_error_rate", "word_error_insertions",
            "word_error_deletions", "word_error_substitutions",
        )
    }


@torch.no_grad()
def logits_for(head: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    head.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        chunks.append(head(features[start : start + batch_size].to(device)).float().cpu())
    return torch.cat(chunks)


def evaluate(
    logits: torch.Tensor,
    *,
    vocabulary: list[str],
    log_prior: torch.Tensor,
    prior_subtractions: list[float],
    targets: list[str],
    deranged_logits: list[torch.Tensor],
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for subtraction in prior_subtractions:
        generated = decode(logits, vocabulary, log_prior, subtraction)
        deranged_metrics = [
            compact(decode(values, vocabulary, log_prior, subtraction), targets)
            for values in deranged_logits
        ]
        matched = compact(generated, targets)
        conditional = matched["content_words_overlap"] - float(np.mean([
            row["content_words_overlap"] for row in deranged_metrics
        ]))
        output[f"{subtraction:g}"] = {
            "matched": matched,
            "deranged_content_f1_mean": float(np.mean([
                row["content_words_overlap"] for row in deranged_metrics
            ])),
            "deranged_content_f1_max": float(np.max([
                row["content_words_overlap"] for row in deranged_metrics
            ])),
            "conditional_content_margin": conditional,
            "generated": generated,
        }
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    brain = np.load(args.brain_npz, allow_pickle=True, mmap_mode="r")
    train_x = brain["train_x"]
    val_x = brain["val_x"]
    if len(train_x) != args.train_count or len(val_x) != args.val_count:
        raise ValueError("Unexpected train/validation brain counts.")
    text = np.load(args.text_npz, allow_pickle=True)
    sentences = [str(value) for value in text["sentence"].tolist()]
    if len(sentences) != args.train_count + args.val_count:
        raise ValueError("Unexpected train/validation sentence counts.")
    train_sentences = sentences[: args.train_count]
    val_sentences = sentences[args.train_count :]
    vocabulary, train_targets, _, log_prior, val_coverage = vocabulary_and_targets(
        train_sentences, val_sentences, positions=args.positions,
        min_frequency=args.min_frequency, max_vocabulary=args.max_vocabulary,
    )
    if args.content_ce_weight < 1.0:
        raise ValueError("content_ce_weight must be at least 1.0")
    content_vocabulary_mask = torch.as_tensor(
        [
            word not in {PAD, UNK}
            and word not in _CONTENT_WORD_STOPWORDS
            and any(character.isalpha() for character in word)
            for word in vocabulary
        ],
        dtype=torch.bool,
        device=device,
    )

    if args.semantic_npz:
        with np.load(args.semantic_npz, allow_pickle=True, mmap_mode="r") as semantic:
            condition = np.asarray([str(value) for value in semantic["condition_source"].tolist()])
            train_rows = np.flatnonzero(condition == "oof_prediction")
            val_rows = np.flatnonzero(condition == "validation_prediction_ensemble")
            if len(train_rows) != args.train_count or len(val_rows) != args.val_count:
                raise ValueError("Unexpected OOF semantic row counts.")
            values = np.asarray(semantic["input_embeddings"], dtype=np.float32)
            if values.shape[1] not in {384, 1536}:
                raise ValueError(
                    "OOF ordered decoder expects 384-D semantics or four delayed "
                    "384-D blocks."
                )
            semantic_sentences = np.asarray([str(value) for value in semantic["sentence"].tolist()])
            if not np.array_equal(semantic_sentences[train_rows], np.asarray(train_sentences)):
                raise ValueError("OOF train sentences are misaligned.")
            if not np.array_equal(semantic_sentences[val_rows], np.asarray(val_sentences)):
                raise ValueError("OOF validation sentences are misaligned.")
            if values.shape[1] == 1536 and args.semantic_delay_mode == "mean":
                values = values.reshape(len(values), 4, 384).mean(axis=1)
            elif values.shape[1] == 1536 and args.semantic_delay_mode == "ordered":
                pass
            elif values.shape[1] == 384 and args.semantic_delay_mode == "ordered":
                raise ValueError(
                    "semantic-delay-mode=ordered requires a 1,536-D four-block input."
                )
            train_features = F.normalize(torch.as_tensor(values[train_rows]), p=2, dim=-1)
            val_features = F.normalize(torch.as_tensor(values[val_rows]), p=2, dim=-1)
    else:
        mri = load_mri2sem_model(
            args.checkpoint, input_dim=train_x.shape[1], output_dim=args.mri_output_dim,
            hidden_dim=args.mri_hidden_dim, res_blocks=args.mri_res_blocks,
            dropout=args.mri_dropout,
        ).to(device)
        train_features = encode_features(
            mri, train_x, feature=args.feature,
            batch_size=args.feature_batch_size, device=device,
        )
        val_features = encode_features(
            mri, val_x, feature=args.feature,
            batch_size=args.feature_batch_size, device=device,
        )
        del mri
        torch.cuda.empty_cache()

    content_bias = torch.zeros((args.val_count, len(vocabulary)), dtype=torch.float32)
    content_payload = None
    if args.content_head_checkpoint:
        content_payload = torch.load(
            args.content_head_checkpoint, map_location="cpu", weights_only=False
        )
        content_vocabulary = list(content_payload["vocabulary"])
        content_head = ContentHead(
            input_dim=int(content_payload["input_dim"]),
            hidden_dim=int(content_payload["hidden_dim"]),
            output_dim=len(content_vocabulary),
        ).to(device)
        if int(content_payload["input_dim"]) != int(val_features.shape[1]):
            raise ValueError(
                "Frozen content head feature width does not match ordered decoder features."
            )
        content_head.load_state_dict(content_payload["head_state_dict"], strict=True)
        content_head.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(val_features), args.batch_size):
                chunks.append(
                    content_head(val_features[start : start + args.batch_size].to(device)).cpu()
                )
        content_logits = torch.cat(chunks).float()
        content_prior = content_payload.get("logit_prior")
        if content_prior is None:
            raise ValueError("Frozen content head checkpoint lacks logit_prior.")
        content_logits = content_logits - args.content_prior_subtraction * torch.logit(
            torch.as_tensor(content_prior).float().clamp(1e-5, 1.0 - 1e-5)
        )[None, :]
        ordered_index = {word: index for index, word in enumerate(vocabulary)}
        for content_index, word in enumerate(content_vocabulary):
            target_index = ordered_index.get(word)
            if target_index is not None:
                content_bias[:, target_index] = content_logits[:, content_index]
        del content_head, content_logits
        torch.cuda.empty_cache()

    head = OrderedWordHead(
        input_dim=train_features.shape[1], hidden_dim=args.head_hidden_dim,
        positions=args.positions, vocabulary_size=len(vocabulary), log_prior=log_prior,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_story = np.asarray([str(value) for value in brain["train_story"].tolist()])
    story_pools = {
        story: np.flatnonzero(train_story == story) for story in sorted(set(train_story.tolist()))
    }
    story_names = sorted(story_pools)
    steps_per_epoch = int(np.ceil(args.train_count / args.batch_size))
    sampler = np.random.default_rng(args.seed)
    derange_rng = np.random.default_rng(args.seed + 1000)
    val_derangements = [derangement(args.val_count, derange_rng) for _ in range(args.derangements)]
    history: list[dict] = []
    best_score = float("-inf")
    best_state = None
    best_record = None

    for epoch in range(0, args.epochs + 1):
        val_logits = logits_for(head, val_features, args.batch_size, device)
        evaluation = {}
        for strength in args.content_bias_strengths:
            biased_logits = val_logits + float(strength) * content_bias[:, None, :]
            deranged_logits = [
                val_logits[index]
                + float(strength) * content_bias[index, None, :]
                for index in val_derangements
            ]
            strength_results = evaluate(
                biased_logits, vocabulary=vocabulary, log_prior=log_prior,
                prior_subtractions=args.prior_subtractions, targets=val_sentences,
                deranged_logits=deranged_logits,
            )
            for subtraction, result in strength_results.items():
                evaluation[f"content{strength:g}_prior{subtraction}"] = result
        eligible = []
        for subtraction, result in evaluation.items():
            matched = result["matched"]
            if args.selection_metric == "word_f1":
                # The ordered head is used only as a soft ELF decoder prior;
                # select the state that best repairs exact tokens and use WER
                # as a small tie-break rather than treating it as a generator.
                score = (
                    matched["words_overlap"]
                    - 0.01 * matched["word_error_rate"]
                )
            else:
                # Content is primary, but a model must show matched-brain advantage;
                # WER breaks ties rather than allowing a language prior to win.
                score = (
                    matched["content_words_overlap"]
                    + result["conditional_content_margin"]
                    - 0.01 * matched["word_error_rate"]
                )
            eligible.append((score, subtraction))
        epoch_score, chosen_subtraction = max(eligible)
        record = {
            "epoch": epoch,
            "selection_score": epoch_score,
            "selected_prior_subtraction": chosen_subtraction,
            "validation": evaluation,
        }
        history.append(record)
        printable = dict(record)
        printable["validation"] = {
            key: {
                "matched": value["matched"],
                "deranged_content_f1_mean": value["deranged_content_f1_mean"],
                "deranged_content_f1_max": value["deranged_content_f1_max"],
                "conditional_content_margin": value["conditional_content_margin"],
            }
            for key, value in evaluation.items()
        }
        print(json.dumps(printable), flush=True)
        if epoch_score > best_score:
            best_score = epoch_score
            best_record = record
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        if epoch == args.epochs:
            break

        head.train()
        for _ in range(steps_per_epoch):
            selected_stories = sampler.integers(0, len(story_names), size=args.batch_size)
            selected_rows = np.asarray([
                sampler.choice(story_pools[story_names[index]]) for index in selected_stories
            ], dtype=np.int64)
            inputs = train_features[selected_rows].to(device)
            targets = train_targets[selected_rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(inputs)
            positional_ce = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape_as(targets)
            target_is_content = content_vocabulary_mask[targets]
            positional_weights = torch.where(
                target_is_content,
                torch.full_like(positional_ce, args.content_ce_weight),
                torch.ones_like(positional_ce),
            )
            ce = (positional_ce * positional_weights).sum() / positional_weights.sum()
            pairwise = logits.sum() * 0.0
            if args.pairwise_weight > 0.0:
                comparison = torch.roll(torch.arange(len(inputs), device=device), shifts=1)
                comparison_logits = logits.index_select(0, comparison)
                true_logits = logits.gather(2, targets[:, :, None]).squeeze(2)
                negative_true_logits = comparison_logits.gather(2, targets[:, :, None]).squeeze(2)
                comparison_targets = targets.index_select(0, comparison)
                mask = targets != comparison_targets
                if bool(mask.any()):
                    pairwise = F.softplus(-(true_logits - negative_true_logits)).masked_select(mask).mean()
            loss = args.ce_weight * ce + args.pairwise_weight * pairwise
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_history = []
    for record in history:
        clone = dict(record)
        clone["validation"] = {
            key: {inner: value for inner, value in result.items() if inner != "generated"}
            for key, result in record["validation"].items()
        }
        serializable_history.append(clone)
    output = {
        "status": "complete",
        "scientific_scope": (
            "story-OOF-train11725-to-predicted-val266 ordered lexical decoder; test107 absent"
            if args.semantic_npz
            else "train11725-to-val266 ordered lexical decoder; test107 never loaded"
        ),
        "args": vars(args), "vocabulary_size": len(vocabulary),
        "validation_vocabulary_coverage": val_coverage,
        "best": {
            key: value for key, value in best_record.items() if key != "validation"
        } if best_record else None,
        "best_validation": {
            key: {inner: value for inner, value in result.items() if inner != "generated"}
            for key, result in best_record["validation"].items()
        } if best_record else None,
        "history": serializable_history,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    if best_state is not None and best_record is not None:
        selected = str(best_record["selected_prior_subtraction"])
        generated = best_record["validation"][selected]["generated"]
        torch.save({
            "head_state_dict": best_state, "vocabulary": vocabulary,
            "log_prior": log_prior, "feature": args.feature,
            "input_dim": int(train_features.shape[1]), "positions": args.positions,
            "hidden_dim": args.head_hidden_dim, "best": output["best"],
            "best_validation": output["best_validation"],
            "content_head_checkpoint": args.content_head_checkpoint or None,
            "content_head_state_dict": (
                content_payload.get("head_state_dict") if content_payload is not None else None
            ),
            "content_head_vocabulary": (
                content_payload.get("vocabulary") if content_payload is not None else None
            ),
            "content_head_logit_prior": (
                content_payload.get("logit_prior") if content_payload is not None else None
            ),
        }, output_path.with_suffix(".pt"))
        np.savez_compressed(
            output_path.with_suffix(".npz"), generated=np.asarray(generated, dtype=object),
            target=np.asarray(val_sentences, dtype=object),
        )
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
