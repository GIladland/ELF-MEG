from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.libribrain_utils import (  # noqa: E402
    build_libribrain_sentence_dataset_per_book,
    collate_libribrain_sentence_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PNPL LibriBrain sentence-aligned MEG loading for ELF."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="LibriBrain root containing Sherlock*/derivatives/{serialised,events}.",
    )
    parser.add_argument(
        "--semantic-data-path",
        type=str,
        default=None,
        help="Optional separate root for semantic-vector NPZ files.",
    )
    parser.add_argument(
        "--pnpl-root",
        type=str,
        default=None,
        help="Path to the PNPL checkout root that contains the `pnpl/` package folder.",
    )
    parser.add_argument(
        "--books",
        type=int,
        nargs="+",
        default=[1],
        help="Sherlock book indices to load, e.g. `--books 1 2 3`.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=2,
        help="How many batches to print before exiting.",
    )
    parser.add_argument(
        "--segment-ms",
        type=int,
        default=3000,
        help="Fixed window size in ms. Use `--segment-ms 0` to keep variable lengths.",
    )
    parser.add_argument(
        "--set-name",
        type=str,
        default="sentences",
        help="PNPL semantic NPZ `set` value to filter on.",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="SONAR",
        help="Which embedding bank PNPL should read from the NPZ.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable PNPL channel standardization.",
    )
    parser.add_argument(
        "--preload-files",
        action="store_true",
        help="Ask PNPL to prefetch referenced files before iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segment_ms = None if args.segment_ms <= 0 else args.segment_ms

    dataset = build_libribrain_sentence_dataset_per_book(
        data_path=args.data_path,
        books=args.books,
        semantic_data_path=args.semantic_data_path,
        pnpl_root=args.pnpl_root,
        set_name=args.set_name,
        embedding_type=args.embedding_type,
        segment_ms=segment_ms,
        standardize=not args.no_standardize,
        preload_files=args.preload_files,
    )

    print(f"dataset_type={type(dataset).__name__}")
    print(f"num_examples={len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_libribrain_sentence_batch,
    )

    for batch_idx, batch in enumerate(loader):
        print(
            f"batch={batch_idx} "
            f"meg_shape={tuple(batch['meg'].shape)} "
            f"mask_shape={tuple(batch['meg_time_mask'].shape)} "
            f"semantic_shape={tuple(batch['semantic_vectors'].shape)}"
        )
        print(f"lengths={batch['meg_lengths'].tolist()}")
        for example_idx, sentence in enumerate(batch["sentences"][:2]):
            info = batch["info"][example_idx]
            print(
                f"  sample[{example_idx}] "
                f"task={info.get('task')} session={info.get('session')} run={info.get('run')} "
                f"sentence={sentence!r}"
            )
        if batch_idx + 1 >= args.num_batches:
            break


if __name__ == "__main__":
    main()
