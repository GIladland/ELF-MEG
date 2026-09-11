#!/usr/bin/env python
"""Train and evaluate a row-aligned semantic-space mapper between ELF NPZs."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ALIGNMENT_KEYS = ("story", "start_tr", "stop_tr", "split", "sentence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-val", required=True)
    parser.add_argument("--target-train-val", required=True)
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--target-test", required=True)
    parser.add_argument("--source-predicted-test", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-sentence-mismatch",
        action="store_true",
        help=(
            "Align rows by story/start_tr/stop_tr/split while allowing source and target "
            "text contracts to differ. Use only when the target-space change is intentional."
        ),
    )
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Learn a zero-initialized residual correction; requires equal input/output dimensions.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BrainDiffusion")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="semantic_mapper")
    parser.add_argument("--wandb-run-name", default="tang3072_to_minilm384")
    parser.add_argument("--wandb-notes", default=None)
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in array.tolist()]


def parse_schema(value: np.ndarray | None) -> dict[str, Any]:
    if value is None:
        return {}
    raw = str(value.tolist())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"source_schema_raw": raw}
    return parsed if isinstance(parsed, dict) else {"source_schema": parsed}


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        if "input_embeddings" not in data.files:
            raise KeyError(f"{path} has no input_embeddings")
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "schema_json"}
        schema = parse_schema(data["schema_json"] if "schema_json" in data.files else None)
    vectors = np.asarray(arrays["input_embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError(f"Expected a finite 2-D embedding matrix in {path}, got {vectors.shape}")
    arrays["input_embeddings"] = vectors
    return arrays, schema


def assert_aligned(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    label: str,
    *,
    alignment_keys: tuple[str, ...] = ALIGNMENT_KEYS,
) -> None:
    source_n = len(source["input_embeddings"])
    target_n = len(target["input_embeddings"])
    if source_n != target_n:
        raise ValueError(f"{label} row mismatch: source={source_n}, target={target_n}")
    for key in alignment_keys:
        if key not in source or key not in target:
            raise KeyError(f"{label} requires aligned key {key!r}")
        source_value = strings(source[key]) if source[key].dtype.kind in {"O", "S", "U"} else source[key]
        target_value = strings(target[key]) if target[key].dtype.kind in {"O", "S", "U"} else target[key]
        if not np.array_equal(np.asarray(source_value), np.asarray(target_value)):
            raise ValueError(f"{label} source/target rows differ at key {key!r}")


class SemanticMapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if residual and input_dim != output_dim:
            raise ValueError("Residual semantic mapping requires equal input/output dimensions")
        self.residual = bool(residual)
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )

        if self.residual:
            final = self.net[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Semantic mapper final module must be Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        output = self.net(vectors)
        if self.residual:
            output = vectors + output
        return F.normalize(output, dim=-1)


def batch_indices(n: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    order = torch.randperm(n, generator=generator)
    return [order[start : start + batch_size] for start in range(0, n, batch_size)]


def mapper_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    contrastive_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = F.normalize(target, dim=-1)
    cosine_loss = (1.0 - (predicted * target).sum(dim=-1)).mean()
    contrastive = torch.zeros((), device=predicted.device)
    if contrastive_weight:
        logits = predicted @ target.T / temperature
        labels = torch.arange(len(predicted), device=predicted.device)
        contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    total = cosine_loss + contrastive_weight * contrastive
    return total, {
        "loss": float(total.detach().cpu()),
        "cosine_loss": float(cosine_loss.detach().cpu()),
        "contrastive_loss": float(contrastive.detach().cpu()),
    }


@torch.no_grad()
def predict(model: nn.Module, vectors: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, len(vectors), batch_size):
        batch = torch.as_tensor(vectors[start : start + batch_size], dtype=torch.float32, device=device)
        rows.append(model(batch).float().cpu().numpy())
    return np.vstack(rows).astype(np.float32)


def retrieval_metrics(predicted: np.ndarray, target: np.ndarray, sentences: list[str]) -> dict[str, Any]:
    predicted = predicted / np.linalg.norm(predicted, axis=1, keepdims=True).clip(min=1e-8)
    target = target / np.linalg.norm(target, axis=1, keepdims=True).clip(min=1e-8)
    scores = predicted @ target.T
    order = np.argsort(-scores, axis=1)
    n = len(predicted)
    ranks = np.empty(n, dtype=np.int64)
    equivalent_ranks = np.empty(n, dtype=np.int64)
    sentence_array = np.asarray(sentences, dtype=object)
    for row in range(n):
        ranks[row] = int(np.flatnonzero(order[row] == row)[0]) + 1
        equivalent = sentence_array[order[row]] == sentence_array[row]
        equivalent_ranks[row] = int(np.flatnonzero(equivalent)[0]) + 1
    matched = scores[np.arange(n), np.arange(n)]
    if n > 1:
        mismatch = scores[~np.eye(n, dtype=bool)]
        mismatch_mean = float(mismatch.mean())
    else:
        mismatch_mean = float("nan")

    def rank_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "top1": float((values <= 1).mean()),
            "top5": float((values <= min(5, n)).mean()),
            "top10": float((values <= min(10, n)).mean()),
            "mean_rank": float(values.mean()),
            "median_rank": float(np.median(values)),
            "mean_percentile": float((1.0 - (values - 1) / max(1, n - 1)).mean()),
        }

    return {
        "n": n,
        "chance_top1": 1.0 / n,
        "index_retrieval": rank_summary(ranks),
        "same_sentence_retrieval": rank_summary(equivalent_ranks),
        "matched_cosine_mean": float(matched.mean()),
        "matched_cosine_median": float(np.median(matched)),
        "mismatch_cosine_mean": mismatch_mean,
    }


def save_mapped_npz(
    path: Path,
    template: dict[str, np.ndarray],
    mapped: np.ndarray,
    *,
    source_path: str,
    target_path: str,
    checkpoint_path: str,
    role: str,
) -> None:
    if len(mapped) != len(template["input_embeddings"]):
        raise ValueError(f"Mapped/template row mismatch for {role}")
    output = dict(template)
    output["input_embeddings"] = np.asarray(mapped, dtype=np.float32)
    schema = {
        "semantic_mapper": {
            "role": role,
            "source_npz": source_path,
            "target_space_template_npz": target_path,
            "checkpoint": checkpoint_path,
            "output_dim": int(mapped.shape[1]),
            "l2_normalized": True,
            "row_alignment_preserved": True,
        }
    }
    output["schema_json"] = np.asarray(json.dumps(schema, indent=2), dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **output)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_train_val, _ = load_npz(args.source_train_val)
    target_train_val, _ = load_npz(args.target_train_val)
    source_test, _ = load_npz(args.source_test)
    target_test, _ = load_npz(args.target_test)
    alignment_keys = (
        tuple(key for key in ALIGNMENT_KEYS if key != "sentence")
        if args.allow_sentence_mismatch
        else ALIGNMENT_KEYS
    )
    assert_aligned(
        source_train_val,
        target_train_val,
        "train_val",
        alignment_keys=alignment_keys,
    )
    assert_aligned(source_test, target_test, "test", alignment_keys=alignment_keys)
    if not 0 < args.val_count < len(source_train_val["input_embeddings"]):
        raise ValueError(f"Invalid val-count={args.val_count}")

    source_dim = int(source_train_val["input_embeddings"].shape[1])
    target_dim = int(target_train_val["input_embeddings"].shape[1])
    if source_test["input_embeddings"].shape[1] != source_dim:
        raise ValueError("Source train/test dimensions differ")
    if target_test["input_embeddings"].shape[1] != target_dim:
        raise ValueError("Target train/test dimensions differ")

    split = len(source_train_val["input_embeddings"]) - args.val_count
    train_source = source_train_val["input_embeddings"][:split]
    train_target = target_train_val["input_embeddings"][:split]
    val_source = source_train_val["input_embeddings"][split:]
    val_target = target_train_val["input_embeddings"][split:]
    val_sentences = strings(target_train_val["sentence"])[split:]

    model = SemanticMapper(
        source_dim,
        target_dim,
        args.hidden_dim,
        args.dropout,
        residual=args.residual,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 29)
    run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            config={**vars(args), "source_dim": source_dim, "target_dim": target_dim, "train_count": split},
        )

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_totals = {"loss": 0.0, "cosine_loss": 0.0, "contrastive_loss": 0.0}
        batches = batch_indices(len(train_source), args.batch_size, generator)
        for indices in batches:
            source_batch = torch.as_tensor(train_source[indices.numpy()], dtype=torch.float32, device=device)
            target_batch = torch.as_tensor(train_target[indices.numpy()], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(source_batch)
            loss, parts = mapper_loss(
                predicted,
                target_batch,
                contrastive_weight=args.contrastive_weight,
                temperature=args.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in parts.items():
                epoch_totals[key] += value
        for key in epoch_totals:
            epoch_totals[key] /= max(1, len(batches))

        if epoch % args.eval_every:
            continue
        val_predicted = predict(model, val_source, args.batch_size, device)
        val_metrics = retrieval_metrics(val_predicted, val_target, val_sentences)
        score = val_metrics["index_retrieval"]["mean_percentile"]
        row = {"epoch": epoch, "train": epoch_totals, "val": val_metrics}
        history.append(row)
        logger.info(
            "epoch=%d loss=%.4f val_top1=%.4f val_top5=%.4f val_percentile=%.4f matched=%.4f",
            epoch,
            epoch_totals["loss"],
            val_metrics["index_retrieval"]["top1"],
            val_metrics["index_retrieval"]["top5"],
            score,
            val_metrics["matched_cosine_mean"],
        )
        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": epoch_totals["loss"],
                    "train/cosine_loss": epoch_totals["cosine_loss"],
                    "train/contrastive_loss": epoch_totals["contrastive_loss"],
                    "val/top1": val_metrics["index_retrieval"]["top1"],
                    "val/top5": val_metrics["index_retrieval"]["top5"],
                    "val/top10": val_metrics["index_retrieval"]["top10"],
                    "val/mean_percentile": score,
                    "val/matched_cosine": val_metrics["matched_cosine_mean"],
                }
            )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "source_dim": source_dim,
                    "target_dim": target_dim,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += args.eval_every
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch=%d; best_epoch=%d", epoch, best_epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    val_predicted = predict(model, val_source, args.batch_size, device)
    test_predicted = predict(model, source_test["input_embeddings"], args.batch_size, device)
    final_metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_validation": retrieval_metrics(val_predicted, val_target, val_sentences),
        "oracle_test": retrieval_metrics(
            test_predicted,
            target_test["input_embeddings"],
            strings(target_test["sentence"]),
        ),
        "history": history,
    }

    train_val_predicted = predict(model, source_train_val["input_embeddings"], args.batch_size, device)
    output_space_name = "minilm384" if target_dim == 384 else f"semantic{target_dim}"
    save_mapped_npz(
        output_dir / f"mapped_train_val_{output_space_name}.npz",
        target_train_val,
        train_val_predicted,
        source_path=args.source_train_val,
        target_path=args.target_train_val,
        checkpoint_path=str(checkpoint_path),
        role="oracle_train_val",
    )
    save_mapped_npz(
        output_dir / f"mapped_oracle_test_{output_space_name}.npz",
        target_test,
        test_predicted,
        source_path=args.source_test,
        target_path=args.target_test,
        checkpoint_path=str(checkpoint_path),
        role="oracle_test",
    )

    if args.source_predicted_test:
        source_predicted_test, _ = load_npz(args.source_predicted_test)
        assert_aligned(
            source_predicted_test,
            target_test,
            "predicted_test",
            alignment_keys=alignment_keys,
        )
        mapped_predicted_test = predict(model, source_predicted_test["input_embeddings"], args.batch_size, device)
        final_metrics["mri2sem_predicted_test"] = retrieval_metrics(
            mapped_predicted_test,
            target_test["input_embeddings"],
            strings(target_test["sentence"]),
        )
        save_mapped_npz(
            output_dir / f"mapped_mri2sem_test_{output_space_name}.npz",
            target_test,
            mapped_predicted_test,
            source_path=args.source_predicted_test,
            target_path=args.target_test,
            checkpoint_path=str(checkpoint_path),
            role="mri2sem_predicted_test",
        )

    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    logger.info("Final metrics:\n%s", json.dumps({k: v for k, v in final_metrics.items() if k != "history"}, indent=2))
    if run is not None:
        for prefix in ("oracle_test", "mri2sem_predicted_test"):
            if prefix not in final_metrics:
                continue
            metrics = final_metrics[prefix]
            wandb.log(
                {
                    f"{prefix}/top1": metrics["index_retrieval"]["top1"],
                    f"{prefix}/top5": metrics["index_retrieval"]["top5"],
                    f"{prefix}/top10": metrics["index_retrieval"]["top10"],
                    f"{prefix}/mean_percentile": metrics["index_retrieval"]["mean_percentile"],
                    f"{prefix}/matched_cosine": metrics["matched_cosine_mean"],
                }
            )
        run.summary["best_epoch"] = best_epoch
        run.finish()


if __name__ == "__main__":
    main()
