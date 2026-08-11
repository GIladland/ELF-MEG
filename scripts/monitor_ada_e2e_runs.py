#!/usr/bin/env python
"""Monitor ADA MEG2SEM->ELF runs and rank eval JSONs by content overlap."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_RUNS = [
    (
        "best_so_far_9kf5_freeze_adapter_train_elf",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_seg8000_9kf8gygo_epoch05_freeze_adapter_train_elf_normoracle_short3h",
    ),
    (
        "kbre17_full_e2e_stopped",
        "/data/engs-pnpl/glandau/elf-runs/sherlock_sva_meg2sem_ada002_seg8000_kbre5zbn_epoch17_full_e2e",
    ),
    (
        "projector_only_stopped",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_seg8000_9kf8gygo_epoch05_freeze_elf_train_projector_only_normoracle_l40s_3h",
    ),
    (
        "safegpu_align_full_adapter",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_seg8000_9kf8gygo_epoch05_freeze_elf_train_full_adapter_align05_normoracle_safegpu_3h",
    ),
    (
        "safegpu_align_full_e2e",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_seg8000_9kf8gygo_epoch05_full_e2e_align025_normoracle_safegpu_3h",
    ),
    (
        "safegpu_eval_gen64",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_9kf5_freeze_adapter_step4500_eval_gen64_oracle_safegpu",
    ),
    (
        "safegpu_align_m2s_only",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_seg8000_9kf8gygo_epoch05_freeze_elf_train_meg2sem_only_align05_normoracle_safegpu_3h",
    ),
    (
        "safegpu_eval_gen128",
        "/data/engs-pnpl/glandau/elf-runs/"
        "sherlock_sva_meg2sem_ada002_9kf5_freeze_adapter_step4500_eval_gen128_oracle_safegpu",
    ),
]


@dataclass
class EvalRecord:
    name: str
    path: str
    step: int
    epoch: float | None
    content: float
    top5: float
    words: float
    well_structured: float | None
    oracle_content: float | None
    oracle_top5: float | None
    semantic_cosine: float | None
    context_cosine: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", default="", help="Comma-separated SLURM job ids to show.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Additional run root to monitor.",
    )
    parser.add_argument("--loop", action="store_true", help="Repeat snapshots.")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between loop snapshots.")
    parser.add_argument("--iterations", type=int, default=96, help="Loop iterations; <=0 runs forever.")
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=ROOT run spec, got {spec!r}")
    name, root = spec.split("=", 1)
    return name.strip(), root.strip()


def load_eval_record(name: str, path: str) -> EvalRecord | None:
    try:
        with open(path, encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception as exc:
        print(f"{name}: skipped unreadable {path}: {exc}", flush=True)
        return None

    quality = metrics.get("generation_quality") or {}
    t5 = metrics.get("generation_t5_retrieval") or {}
    interface = metrics.get("meg2sem_interface") or {}
    oracle = metrics.get("oracle_ada") or {}
    oracle_quality = oracle.get("generation_quality") or {}
    oracle_t5 = oracle.get("generation_t5_retrieval") or {}
    return EvalRecord(
        name=name,
        path=path,
        step=int(metrics.get("step") or 0),
        epoch=metrics.get("epoch"),
        content=float(quality.get("content_words_overlap") or 0.0),
        top5=float(t5.get("top5") or 0.0),
        words=float(quality.get("words_overlap") or 0.0),
        well_structured=quality.get("well_structured_sentence"),
        oracle_content=oracle_quality.get("content_words_overlap"),
        oracle_top5=oracle_t5.get("top5"),
        semantic_cosine=interface.get("interface_semantic_cosine"),
        context_cosine=interface.get("interface_context_cosine"),
    )


def summarize_run(name: str, root: str) -> tuple[EvalRecord | None, EvalRecord | None, EvalRecord | None, int]:
    paths = glob.glob(os.path.join(root, "eval_step_*.json"))
    records = [record for path in paths if (record := load_eval_record(name, path)) is not None]
    if not records:
        return None, None, None, 0
    latest = max(records, key=lambda record: record.step)
    best_content = max(records, key=lambda record: (record.content, record.top5, record.words))
    best_top5 = max(records, key=lambda record: (record.top5, record.content, record.words))
    return latest, best_content, best_top5, len(records)


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=False)


def snapshot(runs: list[tuple[str, str]], jobs: str) -> None:
    print("=" * 88, flush=True)
    print(dt.datetime.now().isoformat(timespec="seconds"), flush=True)
    if jobs:
        print("===SQUEUE===", flush=True)
        run_command(
            [
                "bash",
                "-lc",
                "squeue --clusters=htc -j "
                + jobs
                + " -o '%.18i %.9P %.35j %.8T %.12M %.9l %.6D %.8Q %.12b %R'",
            ]
        )
        print("===START===", flush=True)
        run_command(
            [
                "bash",
                "-lc",
                "squeue --clusters=htc --start -j "
                + jobs
                + " -o '%.18i %.35j %.8T %.20S %.12b %R'",
            ]
        )

    all_best: list[EvalRecord] = []
    print("===RUNS===", flush=True)
    for name, root in runs:
        latest, best_content, best_top5, count = summarize_run(name, root)
        if best_content is None or best_top5 is None or latest is None:
            print(f"{name}: evals=0 root={root}", flush=True)
            continue
        all_best.append(best_content)
        print(
            "{}: evals={} latest_step={} latest_c={:.6g} latest_t5={:.6g} latest_words={:.6g} "
            "oracle_c={} oracle_t5={} sem_cos={} ctx_cos={}".format(
                name,
                count,
                latest.step,
                latest.content,
                latest.top5,
                latest.words,
                latest.oracle_content,
                latest.oracle_top5,
                latest.semantic_cosine,
                latest.context_cosine,
            ),
            flush=True,
        )
        print(
            "  best_content step={} epoch={} c={:.6g} t5={:.6g} words={:.6g} well={} file={}".format(
                best_content.step,
                best_content.epoch,
                best_content.content,
                best_content.top5,
                best_content.words,
                best_content.well_structured,
                os.path.basename(best_content.path),
            ),
            flush=True,
        )
        print(
            "  best_top5    step={} epoch={} c={:.6g} t5={:.6g} words={:.6g} well={} file={}".format(
                best_top5.step,
                best_top5.epoch,
                best_top5.content,
                best_top5.top5,
                best_top5.words,
                best_top5.well_structured,
                os.path.basename(best_top5.path),
            ),
            flush=True,
        )

    print("===GLOBAL_BY_CONTENT===", flush=True)
    for record in sorted(all_best, key=lambda item: (item.content, item.top5, item.words), reverse=True):
        print(
            "{} step={} epoch={} content={:.6g} top5={:.6g} words={:.6g} path={}".format(
                record.name,
                record.step,
                record.epoch,
                record.content,
                record.top5,
                record.words,
                record.path,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    runs = DEFAULT_RUNS + [parse_run_spec(spec) for spec in args.run]
    iteration = 0
    while True:
        snapshot(runs, args.jobs)
        iteration += 1
        if not args.loop:
            break
        if args.iterations > 0 and iteration >= args.iterations:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
