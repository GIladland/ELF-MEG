"""Smoke-test the MEG-to-ELF adapter on LibriBrain batches."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from modules.meg_adapter import MEGContextAdapter
from utils.libribrain_utils import (
    build_libribrain_sentence_dataset_per_book,
    collate_libribrain_sentence_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--pnpl-root", default=None)
    parser.add_argument("--books", nargs="+", type=int, default=[1])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--context-dim", type=int, default=768)
    parser.add_argument("--n-subjects", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_libribrain_sentence_dataset_per_book(
        data_path=args.data_path,
        books=args.books,
        pnpl_root=args.pnpl_root,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_libribrain_sentence_batch,
    )

    adapter = MEGContextAdapter(
        in_channels=306,
        context_dim=args.context_dim,
        context_length=args.context_length,
        n_subjects=args.n_subjects,
    )
    adapter.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        with torch.no_grad():
            output = adapter(
                batch["meg"].float(),
                meg_lengths=batch["meg_lengths"],
            )

        print(
            f"batch={batch_idx} meg_shape={tuple(batch['meg'].shape)} "
            f"context_shape={tuple(output.context.shape)} "
            f"context_mask_shape={tuple(output.context_mask.shape)} "
            f"encoded_sequence_shape={tuple(output.encoded_sequence.shape)}"
        )
        print("context_mask_row0=", output.context_mask[0].tolist())
        print("sentence_row0=", batch["sentences"][0])


if __name__ == "__main__":
    main()
