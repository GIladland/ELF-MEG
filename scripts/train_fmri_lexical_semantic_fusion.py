#!/usr/bin/env python
"""Fuse a frozen direct-MiniLM brain vector with its predicted content words.

The fMRI model, semantic residual mapper, and lexical head are frozen.  Only a
small zero-output fusion MLP is trained on train11725 and selected on val266.
The 107-row test split is never loaded.  The exported vectors live *before*
the existing semantic residual mapper, so the proven MiniLM-to-ELF checkpoint
can consume them without changing its interface.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modules.fmri2sem_bridge import load_mri2sem_model
from probe_fmri_supervised_content_head import ContentHead, encode_features, target_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-npz", required=True)
    parser.add_argument("--text-npz", required=True)
    parser.add_argument("--mri-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--export-npz", default="")
    parser.add_argument("--train-count", type=int, default=11725)
    parser.add_argument("--val-count", type=int, default=266)
    parser.add_argument("--mri-output-dim", type=int, default=384)
    parser.add_argument("--mri-hidden-dim", type=int, default=2048)
    parser.add_argument("--mri-res-blocks", type=int, default=4)
    parser.add_argument("--mri-dropout", type=float, default=0.1)
    parser.add_argument("--semantic-mapper-hidden-dim", type=int, default=1024)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--lexical-topk", type=int, default=5)
    parser.add_argument("--lexical-temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--alignment-weight", type=float, default=4.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.4)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def strings(values: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in values.tolist()]


def load_mapper(checkpoint_path: str, input_dim: int, hidden_dim: int) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("adapter_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("MRI checkpoint has no adapter_state_dict for semantic mapper recovery.")
    mapper = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, input_dim),
    )
    for prefix in ("semantic_projector.semantic_mapper.", "semantic_mapper."):
        mapper_state = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if mapper_state:
            mapper.load_state_dict(mapper_state, strict=True)
            return mapper
    raise ValueError("Checkpoint has no recoverable residual semantic mapper.")


class LexicalFusion(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, raw_semantic: torch.Tensor, lexical_semantic: torch.Tensor) -> torch.Tensor:
        return raw_semantic + self.net(torch.cat([raw_semantic, lexical_semantic], dim=-1))


def retrieval(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = F.normalize(predicted.float(), p=2, dim=-1)
    target = F.normalize(target.float(), p=2, dim=-1)
    similarities = predicted @ target.T
    matched = similarities.diag()
    ranks = 1 + (similarities > matched[:, None]).sum(dim=1)
    return {
        "matched_cosine_mean": float(matched.mean()),
        "top1": float((ranks == 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }


def lexical_vectors(
    logits: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    values, indices = logits.topk(k=min(topk, logits.shape[1]), dim=1)
    weights = F.softmax(values / temperature, dim=1)
    selected = prototypes[indices]
    return F.normalize((selected * weights[:, :, None]).sum(dim=1), p=2, dim=-1)


def apply_mapper(mapper: nn.Module, raw: torch.Tensor) -> torch.Tensor:
    return raw + mapper(raw)


def main() -> None:
    args = parse_args()
    if args.lexical_temperature <= 0.0:
        raise ValueError("lexical temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    brain = np.load(args.brain_npz, allow_pickle=True, mmap_mode="r")
    train_x, val_x = brain["train_x"], brain["val_x"]
    if len(train_x) != args.train_count or len(val_x) != args.val_count:
        raise ValueError(f"Unexpected brain rows {len(train_x)}/{len(val_x)}")
    text = np.load(args.text_npz, allow_pickle=True)
    sentences = strings(text["sentence"])
    exact = torch.as_tensor(np.asarray(text["input_embeddings"], dtype=np.float32))
    expected = args.train_count + args.val_count
    if len(sentences) != expected or len(exact) != expected:
        raise ValueError(f"Expected {expected} target rows, got {len(sentences)}/{len(exact)}")
    exact = F.normalize(exact, p=2, dim=-1)
    train_exact, val_exact = exact[: args.train_count], exact[args.train_count :]

    mri = load_mri2sem_model(
        args.mri_checkpoint,
        input_dim=int(train_x.shape[1]),
        output_dim=args.mri_output_dim,
        hidden_dim=args.mri_hidden_dim,
        res_blocks=args.mri_res_blocks,
        dropout=args.mri_dropout,
    ).to(device)
    train_raw = encode_features(
        mri, train_x, feature="semantic", batch_size=args.feature_batch_size, device=device
    )
    val_raw = encode_features(
        mri, val_x, feature="semantic", batch_size=args.feature_batch_size, device=device
    )
    del mri

    head_payload = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    vocabulary = list(head_payload["vocabulary"])
    head = ContentHead(
        input_dim=int(head_payload["input_dim"]),
        hidden_dim=int(head_payload["hidden_dim"]),
        output_dim=len(vocabulary),
    ).to(device)
    head.load_state_dict(head_payload["head_state_dict"], strict=True)
    head.eval()
    with torch.no_grad():
        train_logits = torch.cat(
            [head(train_raw[start : start + args.feature_batch_size].to(device)).cpu() for start in range(0, len(train_raw), args.feature_batch_size)]
        )
        val_logits = torch.cat(
            [head(val_raw[start : start + args.feature_batch_size].to(device)).cpu() for start in range(0, len(val_raw), args.feature_batch_size)]
        )
    del head

    train_word_targets = target_matrix(sentences[: args.train_count], vocabulary).float()
    counts = train_word_targets.sum(dim=0).clamp_min(1.0)
    prototypes = F.normalize((train_word_targets.T @ train_exact) / counts[:, None], p=2, dim=-1)
    train_lexical = lexical_vectors(
        train_logits, prototypes, topk=args.lexical_topk, temperature=args.lexical_temperature
    )
    val_lexical = lexical_vectors(
        val_logits, prototypes, topk=args.lexical_topk, temperature=args.lexical_temperature
    )

    mapper = load_mapper(
        args.mri_checkpoint, args.mri_output_dim, args.semantic_mapper_hidden_dim
    ).to(device).eval()
    for parameter in mapper.parameters():
        parameter.requires_grad_(False)
    fusion = LexicalFusion(args.mri_output_dim, args.fusion_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    @torch.no_grad()
    def evaluate() -> tuple[dict[str, float], torch.Tensor]:
        fusion.eval()
        chunks = []
        for start in range(0, len(val_raw), args.batch_size):
            raw = val_raw[start : start + args.batch_size].to(device)
            lexical = val_lexical[start : start + args.batch_size].to(device)
            fused_raw = fusion(raw, lexical)
            chunks.append(apply_mapper(mapper, fused_raw).float().cpu())
        mapped = torch.cat(chunks)
        return retrieval(mapped, val_exact), mapped

    with torch.no_grad():
        base_post = torch.cat(
            [apply_mapper(mapper, val_raw[start : start + args.batch_size].to(device)).cpu() for start in range(0, len(val_raw), args.batch_size)]
        )
    base_metrics = retrieval(base_post, val_exact)
    lexical_metrics = retrieval(val_lexical, val_exact)
    initial_metrics, _ = evaluate()
    history = [{"epoch": 0, "loss": None, "validation": initial_metrics}]
    best = history[0]
    best_state = {key: value.detach().cpu() for key, value in fusion.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        fusion.train()
        order = torch.randperm(args.train_count, generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            raw = train_raw.index_select(0, indices).to(device)
            lexical = train_lexical.index_select(0, indices).to(device)
            target = train_exact.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            fused_raw = fusion(raw, lexical)
            predicted = apply_mapper(mapper, fused_raw)
            alignment = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
            predicted_norm = F.normalize(predicted.float(), p=2, dim=-1)
            target_norm = F.normalize(target.float(), p=2, dim=-1)
            logits = predicted_norm @ target_norm.T / args.contrastive_temperature
            labels = torch.arange(len(indices), device=device)
            contrastive = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            loss = args.alignment_weight * alignment + args.contrastive_weight * contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _ = evaluate()
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["top5"] > best["validation"]["top5"] or (
            metrics["top5"] == best["validation"]["top5"]
            and metrics["mean_rank"] < best["validation"]["mean_rank"]
        ):
            best = record
            best_state = {key: value.detach().cpu() for key, value in fusion.state_dict().items()}

    fusion.load_state_dict(best_state, strict=True)
    best_metrics, _ = evaluate()
    with torch.no_grad():
        all_fused_raw = []
        for raw_cpu, lexical_cpu in ((train_raw, train_lexical), (val_raw, val_lexical)):
            chunks = []
            for start in range(0, len(raw_cpu), args.batch_size):
                chunks.append(
                    fusion(
                        raw_cpu[start : start + args.batch_size].to(device),
                        lexical_cpu[start : start + args.batch_size].to(device),
                    ).float().cpu()
                )
            all_fused_raw.append(torch.cat(chunks))

    output = {
        "status": "complete",
        "scientific_scope": "train11725 to val266; test107 never loaded",
        "args": vars(args),
        "base_post_mapper": base_metrics,
        "lexical_prototype_only": lexical_metrics,
        "best": best,
        "best_recomputed": best_metrics,
        "history": history,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    torch.save(
        {
            "fusion_state_dict": best_state,
            "head_checkpoint": args.head_checkpoint,
            "mri_checkpoint": args.mri_checkpoint,
            "vocabulary": vocabulary,
            "prototypes": prototypes,
            "best": best,
            "args": vars(args),
        },
        output_path.with_suffix(".pt"),
    )
    if args.export_npz:
        export_path = Path(args.export_npz)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            export_path,
            input_embeddings=torch.cat(all_fused_raw).numpy().astype(np.float32),
            sentence=np.asarray(sentences),
            schema_json=np.asarray(json.dumps({
                "source": str(output_path),
                "space": "direct384 lexical-fused raw semantic before frozen residual mapper",
                "train_count": args.train_count,
                "val_count": args.val_count,
                "test_loaded": False,
            })),
        )
    print(json.dumps({"base": base_metrics, "best": best}, indent=2))


if __name__ == "__main__":
    main()
