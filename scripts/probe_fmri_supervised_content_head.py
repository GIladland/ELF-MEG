#!/usr/bin/env python
"""Probe train-only content-word signal in a frozen MRI2SEM encoder.

This deliberately bypasses MiniLM, the semantic adapter, and ELF.  A small
multilabel classifier is fitted on the 11,725 training brain rows and scored
only on the 266 validation rows.  The 107-row test split is never loaded.
The result answers a narrow architectural question: does the frozen MRI
hidden state retain lexical information that is lost by the semantic bridge?
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

from modules.fmri2sem_bridge import load_mri2sem_model


NON_CONTENT_WORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself just let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves out
    over own same shan't she she'd she'll she's should shouldn't so some such
    than that that's the their theirs them themselves then there there's these
    they they'd they'll they're they've this those through to too under until
    up very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    also anyway basically er erm huh like mhm mm hmm oh okay ok right uh um
    well yeah yep yes gonna gotta kinda sort sorta wanna y'know
    """.split()
)
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument(
        "--semantic-npz",
        default="",
        help=(
            "Optional oracle+story-OOF archive. If provided, use its OOF train "
            "predictions and validation prediction ensemble as features instead "
            "of an in-sample MRI forward pass. Four 384-D delay blocks are either "
            "kept ordered or mean pooled, then normalized."
        ),
    )
    parser.add_argument(
        "--semantic-delay-mode",
        choices=("mean", "ordered"),
        default="mean",
        help=(
            "How to expose a 4x384 delayed semantic prediction to the lexical head. "
            "'mean' reproduces the original 384-D probe; 'ordered' retains all four "
            "delay blocks as a 1536-D feature."
        ),
    )
    parser.add_argument(
        "--validation-text-npz",
        default="",
        help=(
            "Optional archive supplying only the validation labels. This permits training on a "
            "broad temporal window while selecting checkpoints on the exact held-out target."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-key", default="train_x")
    parser.add_argument("--val-key", default="val_x")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--mri-output-dim", type=int, default=1536)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mri-dropout", type=float, default=0.1)
    parser.add_argument("--feature", choices=["hidden", "semantic"], default="hidden")
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--prior-initialization",
        action="store_true",
        help="Zero-initialize feature weights and initialize logits to train content-word log priors.",
    )
    parser.add_argument(
        "--shuffle-train-pairing-seed",
        type=int,
        default=-1,
        help=(
            "If non-negative, permute training targets relative to brain features with this seed. "
            "Column frequencies are unchanged, so this is a matched language-prior null."
        ),
    )
    parser.add_argument("--min-frequency", type=int, default=5)
    parser.add_argument("--max-vocabulary", type=int, default=3000)
    parser.add_argument(
        "--temporal-context-offset-trs",
        default="0",
        help=(
            "Comma-separated same-story start-TR offsets to concatenate for OOF semantic "
            "features (for example -10,0,10). Missing boundary neighbours are zero-filled."
        ),
    )
    parser.add_argument("--negative-topk", type=int, default=64)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument(
        "--positive-idf-power", type=float, default=0.0,
        help="Raise train-only inverse-document-frequency weights to this power for positive losses.",
    )
    parser.add_argument(
        "--selection-metric", choices=("f1", "idf_f1"), default="f1",
    )
    parser.add_argument(
        "--pairwise-positive-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for matched-vs-deranged brain ranking on target words. The comparison row "
            "is required not to contain that word, so this term cannot be solved by word priors."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def words(sentence: str) -> list[str]:
    return [word for word in WORD_RE.findall(sentence.lower()) if word not in NON_CONTENT_WORDS]


def build_vocabulary(sentences: list[str], min_frequency: int, max_vocabulary: int) -> list[str]:
    counts = Counter(word for sentence in sentences for word in set(words(sentence)))
    ordered = sorted(
        (word for word, count in counts.items() if count >= min_frequency),
        key=lambda word: (-counts[word], word),
    )
    return ordered[:max_vocabulary]


def target_matrix(sentences: list[str], vocabulary: list[str]) -> torch.Tensor:
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    result = torch.zeros((len(sentences), len(vocabulary)), dtype=torch.bool)
    for row, sentence in enumerate(sentences):
        indices = [word_to_idx[word] for word in set(words(sentence)) if word in word_to_idx]
        if indices:
            result[row, indices] = True
    return result


def temporal_context_features(
    values: np.ndarray,
    rows: np.ndarray,
    stories: np.ndarray,
    starts: np.ndarray,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    """Concatenate normalized same-story features at exact start-TR offsets."""
    subset = np.asarray(values[rows], dtype=np.float32)
    subset = subset / np.maximum(np.linalg.norm(subset, axis=1, keepdims=True), 1e-8)
    row_stories = np.asarray(stories[rows]).astype(str)
    row_starts = np.asarray(starts[rows], dtype=np.int64)
    lookup = {
        (story, int(start)): index
        for index, (story, start) in enumerate(zip(row_stories.tolist(), row_starts.tolist()))
    }
    zero = np.zeros(subset.shape[1], dtype=np.float32)
    stacked = []
    for story, start in zip(row_stories.tolist(), row_starts.tolist()):
        stacked.append(np.concatenate([
            subset[lookup[(story, int(start) + offset)]]
            if (story, int(start) + offset) in lookup else zero
            for offset in offsets
        ]))
    return F.normalize(torch.as_tensor(np.stack(stacked)), p=2, dim=-1)


@torch.no_grad()
def encode_features(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    feature: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
        hidden = model.encoder(batch)
        output = hidden if feature == "hidden" else F.normalize(model.projector(hidden), p=2, dim=-1)
        chunks.append(output.float().cpu())
    return torch.cat(chunks, dim=0)


class ContentHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def hard_negative_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    negative_topk: int,
    negative_weight: float,
    positive_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_terms = F.softplus(-logits)
    if positive_weights is None:
        positives = positive_terms.masked_select(targets)
        positive_loss = positives.mean() if positives.numel() else logits.sum() * 0.0
    else:
        weights = positive_weights.to(positive_terms)[None, :].expand_as(positive_terms)
        selected_weights = weights.masked_select(targets)
        positive_loss = (
            (positive_terms * weights).masked_select(targets).sum()
            / selected_weights.sum().clamp_min(1e-8)
            if selected_weights.numel() else logits.sum() * 0.0
        )
    negative_losses = F.softplus(logits).masked_fill(targets, float("-inf"))
    k = min(negative_topk, max(1, logits.shape[1] - 1))
    hard_negatives = negative_losses.topk(k=k, dim=1).values
    finite = torch.isfinite(hard_negatives)
    negative_loss = (
        hard_negatives.masked_select(finite).mean() if bool(finite.any()) else logits.sum() * 0.0
    )
    return positive_loss + negative_weight * negative_loss, positive_loss, negative_loss


@torch.no_grad()
def topk_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks=(1, 3, 5, 8, 10, 20),
    idf_weights: torch.Tensor | None = None,
) -> dict:
    result: dict[str, dict[str, float]] = {}
    target_counts = targets.sum(dim=1).clamp_min(1)
    for k in ks:
        use_k = min(k, logits.shape[1])
        top = logits.topk(k=use_k, dim=1).indices
        hits = targets.gather(1, top).sum(dim=1).float()
        precision = hits / float(use_k)
        recall = hits / target_counts.float()
        f1 = torch.where(
            precision + recall > 0,
            2.0 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )
        result[str(k)] = {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "nonzero_fraction": float((hits > 0).float().mean()),
            "mean_hits": float(hits.mean()),
        }
        if idf_weights is not None:
            weights = idf_weights.float()
            selected_weights = weights[top]
            weighted_hits = (targets.gather(1, top).float() * selected_weights).sum(dim=1)
            weighted_precision = weighted_hits / selected_weights.sum(dim=1).clamp_min(1e-8)
            weighted_recall = weighted_hits / (
                targets.float() * weights[None, :]
            ).sum(dim=1).clamp_min(1e-8)
            weighted_f1 = torch.where(
                weighted_precision + weighted_recall > 0,
                2.0 * weighted_precision * weighted_recall
                / (weighted_precision + weighted_recall),
                torch.zeros_like(weighted_precision),
            )
            result[str(k)].update({
                "idf_precision": float(weighted_precision.mean()),
                "idf_recall": float(weighted_recall.mean()),
                "idf_f1": float(weighted_f1.mean()),
            })
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    brain = np.load(args.brain_npz, allow_pickle=True, mmap_mode="r")
    train_x = brain[args.train_key]
    val_x = brain[args.val_key]
    if len(train_x) != args.train_count or len(val_x) != args.val_count:
        raise ValueError(
            f"Unexpected brain split sizes train={len(train_x)} val={len(val_x)}; "
            f"expected {args.train_count}/{args.val_count}."
        )

    text = np.load(args.text_npz, allow_pickle=True)
    all_sentences = [str(value) for value in text[args.sentence_key].tolist()]
    expected_total = args.train_count + args.val_count
    if len(all_sentences) != expected_total:
        raise ValueError(f"Expected {expected_total} text rows, got {len(all_sentences)}.")
    train_sentences = all_sentences[: args.train_count]
    val_sentences = all_sentences[args.train_count :]
    if args.validation_text_npz:
        validation_text = np.load(args.validation_text_npz, allow_pickle=True)
        validation_sentences = [
            str(value) for value in validation_text[args.sentence_key].tolist()
        ]
        if len(validation_sentences) != expected_total:
            raise ValueError(
                f"Validation text archive must contain {expected_total} rows, "
                f"got {len(validation_sentences)}."
            )
        val_sentences = validation_sentences[args.train_count :]
    vocabulary = build_vocabulary(train_sentences, args.min_frequency, args.max_vocabulary)
    train_targets = target_matrix(train_sentences, vocabulary)
    val_targets = target_matrix(val_sentences, vocabulary)
    document_frequency = train_targets.sum(dim=0).float()
    idf_weights = torch.log(
        torch.tensor(float(len(train_targets) + 1)) / (document_frequency + 1.0)
    ) + 1.0
    positive_weights = idf_weights.pow(args.positive_idf_power)
    if args.shuffle_train_pairing_seed >= 0:
        pairing_generator = torch.Generator(device="cpu").manual_seed(args.shuffle_train_pairing_seed)
        train_targets = train_targets.index_select(
            0,
            torch.randperm(len(train_targets), generator=pairing_generator),
        )

    if args.semantic_npz:
        with np.load(args.semantic_npz, allow_pickle=True, mmap_mode="r") as semantic:
            condition = np.asarray([str(value) for value in semantic["condition_source"].tolist()])
            train_rows = np.flatnonzero(condition == "oof_prediction")
            val_rows = np.flatnonzero(condition == "validation_prediction_ensemble")
            if len(train_rows) != args.train_count or len(val_rows) != args.val_count:
                raise ValueError("Unexpected OOF semantic row counts.")
            values = np.asarray(semantic["input_embeddings"], dtype=np.float32)
            if values.shape[1] == 1536 and args.semantic_delay_mode == "mean":
                values = values.reshape(len(values), 4, 384).mean(axis=1)
            elif values.shape[1] == 1536 and args.semantic_delay_mode == "ordered":
                pass
            elif values.shape[1] != 384:
                raise ValueError(f"Unsupported semantic feature dimension {values.shape[1]}.")
            offsets = tuple(
                int(value.strip())
                for value in args.temporal_context_offset_trs.replace(":", ",").split(",")
                if value.strip()
            )
            if not offsets:
                raise ValueError("At least one temporal context offset is required.")
            semantic_sentences = np.asarray([str(value) for value in semantic[args.sentence_key].tolist()])
            if not np.array_equal(semantic_sentences[train_rows], np.asarray(train_sentences)):
                # Temporally faithful auxiliary targets intentionally have different text
                # from the exact ten-word archive.  Permit that only when the segment identity
                # contract proves row-for-row alignment.
                for key in ("story", "start_tr", "stop_tr"):
                    if key not in semantic.files or key not in text.files:
                        raise ValueError(
                            "OOF semantic train sentences differ and segment metadata is missing."
                        )
                    if not np.array_equal(
                        np.asarray(semantic[key])[train_rows], np.asarray(text[key])[: args.train_count]
                    ):
                        raise ValueError(
                            f"OOF semantic train rows are misaligned on segment key {key!r}."
                        )
            if not np.array_equal(semantic_sentences[val_rows], np.asarray(val_sentences)):
                raise ValueError("OOF semantic validation sentences are misaligned.")
            if offsets == (0,):
                train_features = F.normalize(torch.as_tensor(values[train_rows]), p=2, dim=-1)
                val_features = F.normalize(torch.as_tensor(values[val_rows]), p=2, dim=-1)
            else:
                for key in ("story", "start_tr"):
                    if key not in semantic.files:
                        raise ValueError(f"Temporal context requires semantic metadata {key!r}.")
                stories = np.asarray(semantic["story"])
                starts = np.asarray(semantic["start_tr"])
                train_features = temporal_context_features(
                    values, train_rows, stories, starts, offsets,
                )
                val_features = temporal_context_features(
                    values, val_rows, stories, starts, offsets,
                )
    else:
        model = load_mri2sem_model(
            args.checkpoint,
            input_dim=int(train_x.shape[1]),
            output_dim=args.mri_output_dim,
            hidden_dim=args.mri_hidden_dim,
            res_blocks=args.mri_res_blocks,
            dropout=args.mri_dropout,
        ).to(device)
        train_features = encode_features(
            model,
            train_x,
            feature=args.feature,
            batch_size=args.feature_batch_size,
            device=device,
        )
        val_features = encode_features(
            model,
            val_x,
            feature=args.feature,
            batch_size=args.feature_batch_size,
            device=device,
        )
        del model
        torch.cuda.empty_cache()

    head = ContentHead(
        input_dim=int(train_features.shape[1]),
        hidden_dim=args.head_hidden_dim,
        output_dim=len(vocabulary),
    ).to(device)
    if args.prior_initialization:
        final_layer = head.net[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("ContentHead final layer must be linear for prior initialization.")
        prior = train_targets.float().mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
        with torch.no_grad():
            nn.init.zeros_(final_layer.weight)
            final_layer.bias.copy_(torch.logit(prior).to(final_layer.bias.device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history: list[dict] = []
    head.eval()
    initial_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(val_features), args.batch_size):
            initial_logits.append(head(val_features[start : start + args.batch_size].to(device)).cpu())
    initial_record = {
        "epoch": 0,
        "train_loss": None,
        "train_positive_loss": None,
        "train_hard_negative_loss": None,
        "train_pairwise_positive_loss": None,
        "validation": topk_metrics(
            torch.cat(initial_logits), val_targets, idf_weights=idf_weights,
        ),
    }
    history.append(initial_record)
    print(json.dumps(initial_record), flush=True)
    best: dict | None = initial_record
    best_state: dict[str, torch.Tensor] | None = {
        key: value.detach().cpu() for key, value in head.state_dict().items()
    }

    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        losses: list[float] = []
        positive_losses: list[float] = []
        negative_losses: list[float] = []
        pairwise_losses: list[float] = []
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            inputs = train_features.index_select(0, indices).to(device)
            targets = train_targets.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(inputs)
            loss, positive_loss, negative_loss = hard_negative_multilabel_loss(
                logits,
                targets,
                negative_topk=args.negative_topk,
                negative_weight=args.negative_weight,
                positive_weights=positive_weights,
            )
            pairwise_loss = logits.sum() * 0.0
            if args.pairwise_positive_weight > 0.0 and len(indices) > 1:
                comparison_rows = torch.roll(
                    torch.arange(len(indices), device=device), shifts=1
                )
                comparison_targets = targets.index_select(0, comparison_rows)
                rank_mask = targets & ~comparison_targets
                if bool(rank_mask.any()):
                    comparison_logits = logits.index_select(0, comparison_rows)
                    pairwise_terms = F.softplus(-(logits - comparison_logits))
                    pairwise_weights = positive_weights.to(device)[None, :].expand_as(logits)
                    selected_weights = pairwise_weights.masked_select(rank_mask)
                    pairwise_loss = (
                        (pairwise_terms * pairwise_weights).masked_select(rank_mask).sum()
                        / selected_weights.sum().clamp_min(1e-8)
                    )
                    loss = loss + args.pairwise_positive_weight * pairwise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            positive_losses.append(float(positive_loss.detach().cpu()))
            negative_losses.append(float(negative_loss.detach().cpu()))
            pairwise_losses.append(float(pairwise_loss.detach().cpu()))

        head.eval()
        val_logits_chunks: list[torch.Tensor] = []
        for start in range(0, len(val_features), args.batch_size):
            val_logits_chunks.append(head(val_features[start : start + args.batch_size].to(device)).cpu())
        metrics = topk_metrics(
            torch.cat(val_logits_chunks), val_targets, idf_weights=idf_weights,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_positive_loss": float(np.mean(positive_losses)),
            "train_hard_negative_loss": float(np.mean(negative_losses)),
            "train_pairwise_positive_loss": float(np.mean(pairwise_losses)),
            "validation": metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        selection_key = "idf_f1" if args.selection_metric == "idf_f1" else "f1"
        score = metrics["5"][selection_key]
        if best is None or score > best["validation"]["5"][selection_key]:
            best = record
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}

    output = {
        "status": "complete",
        "scientific_scope": (
            "story-OOF-train11725-to-predicted-val266 supervised lexical probe; test107 absent"
            if args.semantic_npz
            else "train11725-to-val266 supervised lexical probe; test107 never loaded"
        ),
        "args": vars(args),
        "brain_train_shape": list(train_x.shape),
        "brain_val_shape": list(val_x.shape),
        "feature_shape": list(train_features.shape),
        "vocabulary_size": len(vocabulary),
        "train_positive_fraction": float(train_targets.float().mean()),
        "val_positive_fraction": float(val_targets.float().mean()),
        "val_target_words_in_train_vocabulary_mean": float(val_targets.sum(dim=1).float().mean()),
        "best": best,
        "history": history,
        "vocabulary": vocabulary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    if best_state is not None:
        torch.save(
            {
                "head_state_dict": best_state,
                "vocabulary": vocabulary,
                "feature": args.feature,
                "input_dim": int(train_features.shape[1]),
                "hidden_dim": args.head_hidden_dim,
                "logit_prior": train_targets.float().mean(dim=0),
                "idf_weights": idf_weights,
                "best": best,
            },
            output_path.with_suffix(".pt"),
        )
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
