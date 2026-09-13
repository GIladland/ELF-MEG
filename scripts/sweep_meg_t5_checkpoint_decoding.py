#!/usr/bin/env python3
"""Sweep global decoding controls for one frozen MEG-to-T5 checkpoint.

Every configuration is target-free at inference. Validation references choose
one global decoding contract; no row-wise oracle selection is performed and
the protected test split is never loaded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cap_generated_text_words import cap_text
from train_fmri_minilm_t5_prefix import (
    IdentityResidualCalibrator,
    MiniLMT5Prefix,
    compact,
    generate,
    inject_t5_cross_attention_lora,
    strings,
)


def comma_values(value: str, cast) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--embedding-key", default="input_embeddings")
    parser.add_argument("--sentence-key", default="sentence")
    parser.add_argument("--val-start", type=int, default=2652)
    parser.add_argument("--val-rows", type=int, default=110)
    parser.add_argument("--num-beams", default="1,2,4,8")
    parser.add_argument("--min-new-tokens", default="0,8,10")
    parser.add_argument("--max-new-tokens", default="16")
    parser.add_argument("--length-penalties", default="0.8,1.0,1.2")
    parser.add_argument("--repetition-penalties", default="1.0")
    parser.add_argument("--no-repeat-ngram-sizes", default="0")
    parser.add_argument("--max-output-words", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=55)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload["args"]
    if saved.get("projector_kind", "mlp") != "mlp":
        raise ValueError("This MEG sweep currently supports MLP prefix checkpoints only")
    model_name = saved.get("model_name", "t5-small")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    t5 = T5ForConditionalGeneration.from_pretrained(
        model_name, local_files_only=True
    ).to(device).eval()
    inject_t5_cross_attention_lora(
        t5,
        rank=int(saved.get("t5_lora_rank", 0)),
        alpha=float(saved.get("t5_lora_alpha", 8.0)),
        last_n_blocks=int(saved.get("t5_lora_last_n_blocks", 2)),
    )
    lora_state = payload.get("t5_lora_state_dict") or {}
    if lora_state:
        _, unexpected = t5.load_state_dict(lora_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected T5 LoRA keys: {unexpected}")
    semantic_dim = int(
        payload.get("semantic_dim")
        or payload["projector_state_dict"]["net.0.weight"].shape[0]
    )
    projector = MiniLMT5Prefix(
        input_dim=semantic_dim,
        hidden_dim=int(saved["projector_hidden_dim"]),
        prefix_length=int(saved["prefix_length"]),
        model_dim=t5.config.d_model,
        dropout=float(saved.get("dropout", 0.1)),
    ).to(device)
    projector.load_state_dict(payload["projector_state_dict"], strict=True)
    projector.eval()
    calibrator = None
    calibrator_state = payload.get("calibrator_state_dict")
    if calibrator_state is not None:
        if saved.get("brain_calibrator_kind") != "identity_residual":
            raise ValueError("Only same-space identity-residual calibration is supported")
        hidden_dim = int(calibrator_state["net.1.weight"].shape[0])
        calibrator = IdentityResidualCalibrator(
            semantic_dim, hidden_dim, float(saved.get("dropout", 0.1))
        ).to(device)
        calibrator.load_state_dict(calibrator_state, strict=True)
        calibrator.eval()
    for module in (t5, projector, calibrator):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return payload, tokenizer, t5, projector, calibrator, semantic_dim


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload, tokenizer, t5, projector, calibrator, semantic_dim = load_checkpoint(
        args.checkpoint, device
    )
    with np.load(args.semantic_npz, allow_pickle=True) as data:
        stop = args.val_start + args.val_rows
        semantic_np = np.asarray(data[args.embedding_key][args.val_start:stop], dtype=np.float32)
        targets = strings(data[args.sentence_key][args.val_start:stop])
    if semantic_np.shape != (args.val_rows, semantic_dim):
        raise ValueError(
            f"validation semantic shape {semantic_np.shape}, expected {(args.val_rows, semantic_dim)}"
        )
    semantic = F.normalize(torch.as_tensor(semantic_np), p=2, dim=-1)
    configurations = list(
        itertools.product(
            comma_values(args.num_beams, int),
            comma_values(args.min_new_tokens, int),
            comma_values(args.max_new_tokens, int),
            comma_values(args.length_penalties, float),
            comma_values(args.repetition_penalties, float),
            comma_values(args.no_repeat_ngram_sizes, int),
        )
    )
    # Length penalty is inoperative in greedy decoding; avoid duplicate rows.
    configurations = [
        row for row in configurations if row[0] > 1 or row[3] == 1.0
    ]
    records = []
    for beams, minimum, maximum, length_penalty, repetition, no_repeat in configurations:
        raw = generate(
            projector,
            t5,
            tokenizer,
            semantic,
            batch_size=args.batch_size,
            max_target_tokens=maximum,
            min_target_tokens=minimum,
            num_beams=beams,
            device=device,
            calibrator=calibrator,
            repetition_penalty=repetition,
            no_repeat_ngram_size=no_repeat,
            length_penalty=length_penalty,
            max_output_words=0,
        )
        generated = [cap_text(text, args.max_output_words) for text in raw]
        metrics = compact(generated, targets)
        record = {
            "config": {
                "num_beams": beams,
                "min_new_tokens": minimum,
                "max_new_tokens": maximum,
                "length_penalty": length_penalty,
                "repetition_penalty": repetition,
                "no_repeat_ngram_size": no_repeat,
                "max_output_words": args.max_output_words,
                "word_cap": "evaluator lexical-token prefix",
            },
            "matched": metrics,
            "generated": generated,
        }
        records.append(record)
        print(json.dumps({"config": record["config"], "matched": metrics}), flush=True)
    records.sort(
        key=lambda row: (
            -(row["matched"]["content_words_overlap"] + row["matched"]["words_overlap"]),
            row["matched"]["word_error_rate"],
        )
    )
    output = {
        "scientific_scope": (
            f"frozen-checkpoint MEG val{args.val_rows} global decoding sweep; "
            "protected test26 not loaded; validation targets choose one global contract"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_best": payload.get("best"),
        "sweep_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "targets": targets,
        "best": records[0],
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_json), "best": records[0]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
