#!/usr/bin/env python
"""Measure how much ELF diffusion uses the semantic conditioning vector."""

from __future__ import annotations

import argparse
import csv
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


def _write_tsv_light(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _early_finalize_if_requested() -> None:
    if "--finalize-json" not in sys.argv:
        return
    parser = argparse.ArgumentParser(description="Finalize semantic sensitivity JSON into TSVs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finalize-json", required=True)
    args, _ = parser.parse_known_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.finalize_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary_rows = payload["summary"]
    _write_tsv_light(output_dir / "semantic_condition_sensitivity_summary.tsv", summary_rows)
    results = payload["results"]
    names = [result["name"] for result in results]
    sample_rows = []
    if results:
        for idx, target in enumerate(results[0]["targets"]):
            row = {"index": idx, "target": target}
            for result in results:
                row[f"generated_{result['name']}"] = result["generated"][idx]
            sample_rows.append(row)
    _write_tsv_light(output_dir / "semantic_condition_sensitivity_samples.tsv", sample_rows)
    print(json.dumps({"summary": summary_rows, "names": names}, ensure_ascii=False, indent=2))
    print(f"summary_tsv={output_dir / 'semantic_condition_sensitivity_summary.tsv'}")
    print(f"samples_tsv={output_dir / 'semantic_condition_sensitivity_samples.tsv'}")
    raise SystemExit(0)


_early_finalize_if_requested()

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from configs.config import SamplingConfig
from modules.t5_encoder import get_encoder
from scripts.meg_context_overfit import (
    SemanticVectorContextProjector,
    build_config,
    encode_text_batched,
    evaluate_generation,
    load_pretrained_model,
    rank_true_targets_by_similarity,
    tokenize_sentences,
    word_overlap_metrics,
)
from scripts.train_npz_semantic_to_elf import load_e2e_initialization


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
    parser.add_argument("--reference-npz", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--input-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--model", default="ELF-B")
    parser.add_argument("--encoder_model_name", default="t5-small")
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--semantic-hidden-dim", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=75)
    parser.add_argument("--target-encode-batch-size", type=int, default=64)
    parser.add_argument("--num_sampling_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--self_cond_cfg_scale", type=float, default=1.0)
    parser.add_argument("--denoiser_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_noise_scale", type=float, default=2.5)
    parser.add_argument("--elf-attn-dropout", type=float, default=0.0)
    parser.add_argument("--elf-proj-dropout", type=float, default=0.0)
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--noise-scales", default="0.25,0.5,1.0")
    parser.add_argument("--mix-alphas", default="0.25,0.5,0.75")
    parser.add_argument("--same-noise", action="store_true", default=True)
    parser.add_argument("--different-noise", dest="same_noise", action="store_false")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--finalize-json",
        default="",
        help="Existing results JSON to convert into TSVs without rerunning diffusion.",
    )
    return parser.parse_args()


def strings(array: np.ndarray) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in array.tolist()]


def parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item) for item in value.split(",") if item.strip()]


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    for _ in range(10_000):
        perm = np.arange(n)
        rng.shuffle(perm)
        if np.all(perm != np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


def load_vectors(path: str, key: str, limit: int = 0) -> torch.Tensor:
    data = np.load(path, allow_pickle=True)
    vectors = np.asarray(data[key], dtype=np.float32)
    if limit > 0:
        vectors = vectors[:limit]
    return torch.as_tensor(vectors, dtype=torch.float32)


def vector_stats(vectors: torch.Tensor) -> dict[str, float]:
    vectors = vectors.float()
    return {
        "norm_mean": float(vectors.norm(dim=-1).mean()),
        "norm_std": float(vectors.norm(dim=-1).std(unbiased=False)),
        "dim_mean_abs": float(vectors.mean(dim=0).abs().mean()),
        "dim_std_mean": float(vectors.std(dim=0, unbiased=False).mean()),
        "global_std": float(vectors.std(unbiased=False)),
    }


def semantic_alignment(control: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    control = control.float()
    target = target.float()
    cosine = F.cosine_similarity(control, target, dim=-1)
    similarity = F.normalize(control, dim=-1) @ F.normalize(target, dim=-1).T
    ranks = rank_true_targets_by_similarity(similarity)
    return {
        "semantic_cosine_to_true_mean": float(cosine.mean()),
        "semantic_cosine_to_true_std": float(cosine.std(unbiased=False)),
        "semantic_mse_to_true": float(F.mse_loss(control, target)),
        "semantic_l2_to_true_mean": float((control - target).norm(dim=-1).mean()),
        "semantic_top1_vs_true": ranks["top1"],
        "semantic_top5_vs_true": ranks["top5"],
        "semantic_mean_rank_vs_true": ranks["mean_rank"],
        "semantic_median_rank_vs_true": ranks["median_rank"],
    }


def make_controls(
    semantic: torch.Tensor,
    reference: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> list[tuple[str, torch.Tensor, dict[str, object]]]:
    rng = np.random.default_rng(args.seed + 1000)
    semantic = semantic.float()
    reference = reference.float()
    mean = reference.mean(dim=0, keepdim=True)
    std = reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    global_std = reference.std(unbiased=False).clamp_min(1e-6)
    mean_norm = reference.norm(dim=-1).mean().clamp_min(1e-6)
    controls: list[tuple[str, torch.Tensor, dict[str, object]]] = [
        ("matched", semantic.clone(), {"control_type": "true_pairing"}),
        ("roll_1", semantic.roll(shifts=1, dims=0), {"control_type": "deterministic_wrong_pairing", "shift": 1}),
        ("mean", mean.expand_as(semantic).clone(), {"control_type": "reference_mean"}),
        ("zero", torch.zeros_like(semantic), {"control_type": "zero"}),
    ]

    for idx in range(max(0, args.shuffle_repeats)):
        perm = derangement(semantic.shape[0], rng)
        controls.append(
            (
                f"deranged_{idx}",
                semantic.index_select(0, torch.as_tensor(perm, dtype=torch.long)),
                {"control_type": "deranged_pairing", "fixed_points": int(np.sum(perm == np.arange(len(perm))))},
            )
        )

    normal = torch.Generator().manual_seed(args.seed + 2000)
    controls.append(
        (
            "gaussian_dim",
            mean + std * torch.randn(semantic.shape, generator=normal),
            {"control_type": "per_dimension_gaussian_reference"},
        )
    )
    random_unit = torch.randn(semantic.shape, generator=normal)
    random_unit = F.normalize(random_unit, dim=-1) * mean_norm
    controls.append(("random_unit_norm", random_unit, {"control_type": "random_direction_reference_norm"}))

    noise_base = torch.randn(semantic.shape, generator=normal)
    for scale in parse_float_list(args.noise_scales):
        controls.append(
            (
                f"matched_noise_{scale:g}",
                semantic + float(scale) * std * noise_base,
                {"control_type": "matched_plus_reference_scaled_noise", "noise_scale": float(scale)},
            )
        )

    perm = derangement(semantic.shape[0], rng)
    shuffled = semantic.index_select(0, torch.as_tensor(perm, dtype=torch.long))
    for alpha in parse_float_list(args.mix_alphas):
        alpha = float(alpha)
        controls.append(
            (
                f"mix_matched_{alpha:g}_deranged_{1.0 - alpha:g}",
                alpha * semantic + (1.0 - alpha) * shuffled,
                {"control_type": "matched_deranged_interpolation", "matched_alpha": alpha},
            )
        )
    return controls


def load_checkpointed_model_and_adapter(args: argparse.Namespace, semantic_dim: int, target_length: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
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
    adapter = SemanticVectorContextProjector(
        input_dim=semantic_dim,
        context_dim=encoder_config.d_model,
        context_length=args.context_length,
        hidden_dim=args.semantic_hidden_dim,
        dropout=0.0,
    ).to(device)
    load_e2e_initialization(model, adapter, args.checkpoint_path, device)
    model.eval()
    adapter.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(False)
    return tokenizer, encoder, model, adapter, config


def summarize_result(result: dict) -> dict[str, object]:
    quality = result.get("generation_quality", {})
    retrieval = result.get("generation_t5_retrieval", {})
    return {
        "name": result["name"],
        "exact_match": result.get("exact_match"),
        "words_overlap": quality.get("words_overlap"),
        "content_words_overlap": quality.get("content_words_overlap"),
        "well_structured_sentence": quality.get("well_structured_sentence"),
        "degen_fraction": quality.get("degen_fraction"),
        "unique_token_ratio": quality.get("unique_token_ratio"),
        "generation_t5_top1": retrieval.get("top1"),
        "generation_t5_top5": retrieval.get("top5"),
        "generation_t5_mean_rank": retrieval.get("mean_rank"),
        "generation_t5_median_rank": retrieval.get("median_rank"),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.finalize_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.finalize_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        summary_rows = payload["summary"]
        write_tsv(output_dir / "semantic_condition_sensitivity_summary.tsv", summary_rows)
        results = payload["results"]
        names = [result["name"] for result in results]
        sample_rows = []
        if results:
            for idx, target in enumerate(results[0]["targets"]):
                row = {"index": idx, "target": target}
                for result in results:
                    row[f"generated_{result['name']}"] = result["generated"][idx]
                sample_rows.append(row)
        write_tsv(output_dir / "semantic_condition_sensitivity_samples.tsv", sample_rows)
        print(json.dumps({"summary": summary_rows, "names": names}, ensure_ascii=False, indent=2))
        print(f"summary_tsv={output_dir / 'semantic_condition_sensitivity_summary.tsv'}")
        print(f"samples_tsv={output_dir / 'semantic_condition_sensitivity_samples.tsv'}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    semantic_np = np.asarray(data[args.input_key], dtype=np.float32)
    sentences_all = strings(data[args.sentence_key])
    n = semantic_np.shape[0] if args.num_examples <= 0 else min(args.num_examples, semantic_np.shape[0])
    semantic = torch.as_tensor(semantic_np[:n], dtype=torch.float32)
    sentences = sentences_all[:n]
    reference = load_vectors(args.reference_npz, args.input_key) if args.reference_npz else semantic

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    target_ids, target_mask = tokenize_sentences(tokenizer, sentences)
    target_length = int(target_ids.shape[1])
    tokenizer, encoder, model, adapter, config = load_checkpointed_model_and_adapter(
        args,
        semantic_dim=int(semantic.shape[-1]),
        target_length=target_length,
        device=device,
    )
    target_latents = encode_text_batched(
        input_ids=target_ids,
        attention_mask=target_mask,
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        device=device,
        batch_size=args.target_encode_batch_size,
    )
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale],
        self_cond_cfg_scales=[args.self_cond_cfg_scale],
        time_schedule=config.time_schedule,
    )

    controls = make_controls(semantic, reference, args=args)
    results: list[dict] = []
    summary_rows: list[dict] = []
    matched_generated: list[str] | None = None

    for control_idx, (name, control_vectors, control_meta) in enumerate(controls):
        logger.info("Evaluating control %s (%d/%d)", name, control_idx + 1, len(controls))
        generator_seed = args.seed + 17 if args.same_noise else args.seed + 17 + control_idx
        generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(generator_seed)
        metrics = evaluate_generation(
            model=model,
            adapter=adapter,
            meg=torch.zeros((n, 1, 1), dtype=torch.float32),
            meg_lengths=torch.ones((n,), dtype=torch.long),
            semantic_vectors=control_vectors,
            subject_ids=torch.zeros((n,), dtype=torch.long),
            tokenizer=tokenizer,
            encoder=encoder,
            target_sentences=sentences,
            target_latents=target_latents,
            target_mask=target_mask,
            target_length=target_length,
            context_length=args.context_length,
            config=config,
            sampling_config=sampling_config,
            device=device,
            generator=generator,
            condition_source="semantic",
        )
        metrics["name"] = name
        metrics["control_meta"] = control_meta
        metrics["semantic_alignment_to_true"] = semantic_alignment(control_vectors, semantic)
        metrics["semantic_vector_stats"] = vector_stats(control_vectors)
        if name == "matched":
            matched_generated = list(metrics["generated"])
        elif matched_generated is not None:
            metrics["generated_vs_matched_generation"] = {
                "exact_same_fraction": float(
                    np.mean([a == b for a, b in zip(metrics["generated"], matched_generated)])
                ),
                **{
                    f"matched_generation_{key}": value
                    for key, value in word_overlap_metrics(metrics["generated"], matched_generated)["summary"].items()
                },
            }
        result_summary = summarize_result(metrics)
        result_summary.update(metrics["semantic_alignment_to_true"])
        if "generated_vs_matched_generation" in metrics:
            result_summary.update(metrics["generated_vs_matched_generation"])
        summary_rows.append(result_summary)
        results.append(metrics)
        logger.info(
            "%s content=%.4f words=%.4f top5=%s sem_cos=%.4f same_as_matched=%s",
            name,
            result_summary["content_words_overlap"],
            result_summary["words_overlap"],
            result_summary["generation_t5_top5"],
            result_summary["semantic_cosine_to_true_mean"],
            result_summary.get("exact_same_fraction"),
        )

    payload = {
        "report_name": "Semantic Conditioning Sensitivity",
        "args": vars(args),
        "num_examples": n,
        "target_length": target_length,
        "same_noise": bool(args.same_noise),
        "reference_stats": vector_stats(reference),
        "true_semantic_stats": vector_stats(semantic),
        "summary": summary_rows,
        "results": results,
    }
    with (output_dir / "semantic_condition_sensitivity_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_tsv(output_dir / "semantic_condition_sensitivity_summary.tsv", summary_rows)

    sample_rows = []
    by_name = {result["name"]: result for result in results}
    names = [name for name, _, _ in controls]
    for idx, target in enumerate(sentences):
        row = {"index": idx, "target": target}
        for name in names:
            row[f"generated_{name}"] = by_name[name]["generated"][idx]
        sample_rows.append(row)
    write_tsv(output_dir / "semantic_condition_sensitivity_samples.tsv", sample_rows)

    print(json.dumps({"summary": summary_rows}, ensure_ascii=False, indent=2))
    print(f"summary_tsv={output_dir / 'semantic_condition_sensitivity_summary.tsv'}")
    print(f"samples_tsv={output_dir / 'semantic_condition_sensitivity_samples.tsv'}")


if __name__ == "__main__":
    main()
