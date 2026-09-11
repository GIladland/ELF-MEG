#!/usr/bin/env python3
"""Audit train-only lexical signal in each MRI2SEM delayed MiniLM block.

This diagnostic never loads the brain test split. It builds word directions
from train rows only, evaluates them on the held-out validation rows, and
compares individual delay blocks with feature- and score-level fusion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.fmri2sem_bridge import load_mri2sem_model  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "about", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "quite", "really", "same", "she",
    "should", "so", "some", "such", "sure", "than", "that", "the", "their",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "we", "well", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "will", "with", "would", "yes", "you", "your", "yours",
    "yourself", "yourselves",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-x-key", default="train_x")
    parser.add_argument("--val-x-key", default="val_x")
    parser.add_argument("--train-y-key", default="train_y")
    parser.add_argument("--val-y-key", default="val_y")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--num-delays", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-vocabulary", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
        for value in values.tolist()
    ]


def content_words(sentence: str) -> set[str]:
    return {
        token
        for token in WORD_RE.findall(sentence.lower())
        if token not in STOPWORDS and any(character.isalpha() for character in token)
    }


def build_vocabulary(train_sentences: list[str], min_frequency: int, maximum: int):
    frequencies = Counter(
        token for sentence in train_sentences for token in content_words(sentence)
    )
    vocabulary = [
        token
        for token, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_frequency
    ][:maximum]
    if len(vocabulary) < 2:
        raise ValueError(f"Content vocabulary too small: {len(vocabulary)}")
    return vocabulary


def target_matrix(sentences: list[str], vocabulary: list[str]) -> torch.Tensor:
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    targets = torch.zeros((len(sentences), len(vocabulary)), dtype=torch.bool)
    for row, sentence in enumerate(sentences):
        for token in content_words(sentence):
            index = token_to_index.get(token)
            if index is not None:
                targets[row, index] = True
    return targets


def prototype_directions(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features.float(), p=2, dim=-1)
    membership = targets.float()
    counts = membership.sum(dim=0)
    positive = membership.T @ features / counts[:, None].clamp_min(1.0)
    negative_counts = features.shape[0] - counts
    negative = (
        features.sum(dim=0, keepdim=True) - membership.T @ features
    ) / negative_counts[:, None].clamp_min(1.0)
    return F.normalize(positive - negative, p=2, dim=-1)


@torch.inference_mode()
def predict(model, values: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs = []
    model.eval()
    for start in range(0, values.shape[0], batch_size):
        batch = torch.as_tensor(
            np.asarray(values[start : start + batch_size]),
            dtype=torch.float32,
            device=device,
        )
        outputs.append(model(batch).detach().cpu())
    return torch.cat(outputs, dim=0)


def topk_metrics(scores: torch.Tensor, targets: torch.Tensor, ks=(1, 3, 5, 10, 25, 50, 100)):
    result = {}
    positives = int(targets.sum())
    rows = targets.shape[0]
    for requested_k in ks:
        k = min(requested_k, scores.shape[1])
        indices = scores.topk(k=k, dim=1).indices
        hits = targets.gather(1, indices)
        hit_count = int(hits.sum())
        rows_with_hit = float(hits.any(dim=1).float().mean())
        result[str(requested_k)] = {
            "micro_recall": hit_count / max(1, positives),
            "micro_precision": hit_count / max(1, rows * k),
            "rows_with_hit": rows_with_hit,
            "hits": hit_count,
        }
    return result


def normalized_scores(features: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2, dim=-1) @ F.normalize(
        directions.float(), p=2, dim=-1
    ).T


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    brain = np.load(args.brain_npz, allow_pickle=True)
    text = np.load(args.text_npz, allow_pickle=True)
    train_x = brain[args.train_x_key]
    val_x = brain[args.val_x_key]
    train_n = train_x.shape[0]
    val_n = val_x.shape[0]
    sentences = strings(text[args.sentence_key])
    if len(sentences) != train_n + val_n:
        raise ValueError(
            f"Text/brain row mismatch: sentences={len(sentences)} train={train_n} val={val_n}"
        )
    train_sentences = sentences[:train_n]
    val_sentences = sentences[train_n:]
    vocabulary = build_vocabulary(
        train_sentences, args.min_frequency, args.max_vocabulary
    )
    train_targets = target_matrix(train_sentences, vocabulary)
    val_targets = target_matrix(val_sentences, vocabulary)

    model = load_mri2sem_model(
        args.checkpoint,
        input_dim=train_x.shape[1],
        output_dim=args.num_delays * 384,
        hidden_dim=args.hidden_dim,
        res_blocks=args.res_blocks,
        dropout=0.0,
    ).to(device)
    predicted_train = predict(model, train_x, device, args.batch_size).reshape(
        train_n, args.num_delays, -1
    )
    predicted_val = predict(model, val_x, device, args.batch_size).reshape(
        val_n, args.num_delays, -1
    )

    report = {
        "contract": {
            "brain_npz": args.brain_npz,
            "text_npz": args.text_npz,
            "checkpoint": args.checkpoint,
            "train_rows": train_n,
            "val_rows": val_n,
            "test_rows_loaded": 0,
            "vocabulary_size": len(vocabulary),
            "validation_target_instances": int(val_targets.sum()),
        },
        "representations": {},
    }

    exact = torch.as_tensor(np.asarray(text["input_embeddings"]), dtype=torch.float32)
    exact_train, exact_val = exact[:train_n], exact[train_n:]
    exact_directions = prototype_directions(exact_train, train_targets)
    report["representations"]["oracle_exact384"] = topk_metrics(
        normalized_scores(exact_val, exact_directions), val_targets
    )
    report["representations"]["predicted_delay_mean_to_exact384"] = topk_metrics(
        normalized_scores(predicted_val.mean(dim=1), exact_directions), val_targets
    )

    predicted_block_scores = []
    for delay in range(args.num_delays):
        directions = prototype_directions(predicted_train[:, delay], train_targets)
        scores = normalized_scores(predicted_val[:, delay], directions)
        predicted_block_scores.append(scores)
        report["representations"][f"predicted_delay{delay + 1}_train_dictionary"] = (
            topk_metrics(scores, val_targets)
        )
    stacked_predicted_scores = torch.stack(predicted_block_scores, dim=0)
    report["representations"]["predicted_delay_score_mean"] = topk_metrics(
        stacked_predicted_scores.mean(dim=0), val_targets
    )
    report["representations"]["predicted_delay_score_max"] = topk_metrics(
        stacked_predicted_scores.max(dim=0).values, val_targets
    )

    if args.train_y_key in brain and args.val_y_key in brain:
        oracle_train = torch.as_tensor(
            np.asarray(brain[args.train_y_key]), dtype=torch.float32
        ).reshape(train_n, args.num_delays, -1)
        oracle_val = torch.as_tensor(
            np.asarray(brain[args.val_y_key]), dtype=torch.float32
        ).reshape(val_n, args.num_delays, -1)
        cross_scores = []
        for delay in range(args.num_delays):
            directions = prototype_directions(oracle_train[:, delay], train_targets)
            oracle_scores = normalized_scores(oracle_val[:, delay], directions)
            predicted_scores = normalized_scores(predicted_val[:, delay], directions)
            cross_scores.append(predicted_scores)
            report["representations"][f"oracle_delay{delay + 1}"] = topk_metrics(
                oracle_scores, val_targets
            )
            report["representations"][f"predicted_delay{delay + 1}_to_oracle_dictionary"] = (
                topk_metrics(predicted_scores, val_targets)
            )
        stacked_cross_scores = torch.stack(cross_scores, dim=0)
        report["representations"]["predicted_to_oracle_delay_score_mean"] = topk_metrics(
            stacked_cross_scores.mean(dim=0), val_targets
        )
        report["representations"]["predicted_to_oracle_delay_score_max"] = topk_metrics(
            stacked_cross_scores.max(dim=0).values, val_targets
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
