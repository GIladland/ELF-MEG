from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import ConcatDataset


def _candidate_pnpl_roots(explicit_root: str | os.PathLike[str] | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser().resolve())

    env_root = os.environ.get("PNPL_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    this_file = Path(__file__).resolve()
    candidates.extend(
        [
            this_file.parents[4] / "PNPL" / "pnpl",
            this_file.parents[5] / "PNPL" / "pnpl",
        ]
    )
    return candidates


def ensure_pnpl_importable(pnpl_root: str | os.PathLike[str] | None = None) -> Path:
    """Add the local PNPL checkout to sys.path and return its root."""
    for candidate in _candidate_pnpl_roots(pnpl_root):
        if (candidate / "pnpl").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    searched = ", ".join(str(path) for path in _candidate_pnpl_roots(pnpl_root))
    raise ImportError(
        "Could not find a PNPL checkout containing `pnpl/`. "
        f"Searched: {searched}"
    )


def _install_package_stub(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(path)]


def _load_module_from_file(module_name: str, file_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_libribrain_semantic_vectors(
    pnpl_root: str | os.PathLike[str] | None = None,
):
    root = ensure_pnpl_importable(pnpl_root)
    try:
        from pnpl.datasets.libribrain2025.semantic_vectors import LibriBrainSemanticVectors

        return LibriBrainSemanticVectors
    except ModuleNotFoundError:
        package_root = root / "pnpl"
        datasets_root = package_root / "datasets"
        libribrain_root = datasets_root / "libribrain2025"

        _install_package_stub("pnpl", package_root)
        _install_package_stub("pnpl.datasets", datasets_root)
        _install_package_stub("pnpl.datasets.libribrain2025", libribrain_root)

        _load_module_from_file("pnpl.datasets.utils", datasets_root / "utils.py")
        _load_module_from_file(
            "pnpl.datasets.libribrain2025.constants",
            libribrain_root / "constants.py",
        )
        _load_module_from_file(
            "pnpl.datasets.libribrain2025.base",
            libribrain_root / "base.py",
        )
        semantic_vectors_module = _load_module_from_file(
            "pnpl.datasets.libribrain2025.semantic_vectors",
            libribrain_root / "semantic_vectors.py",
        )
        return semantic_vectors_module.LibriBrainSemanticVectors


def build_sherlock_run_keys(book_idx: int) -> list[tuple[str, str, str, str]]:
    """Mirror the MEG2SEM Sherlock 1-9 session selection."""
    if book_idx == 9:
        return [("0", str(i), "Sherlock9", "1") for i in range(0, 13)]
    if book_idx == 8:
        return [("0", str(i), "Sherlock8", "1") for i in range(1, 11)]
    if book_idx == 7:
        return [("0", str(i), "Sherlock7", "1") for i in range(1, 15)]
    if book_idx == 6:
        return [("0", str(i), "Sherlock6", "1") for i in range(1, 14) if i not in (3, 10)]
    if book_idx == 5:
        return [("0", str(i), "Sherlock5", "1") for i in range(1, 16)]
    if book_idx == 4:
        return [("0", str(i), "Sherlock4", "1") for i in range(1, 13) if i not in (2, 9)]
    if book_idx == 3:
        return [("0", str(i), "Sherlock3", "1") for i in range(1, 13) if i != 2]
    if book_idx == 2:
        return [("0", str(i), "Sherlock2", "1") for i in range(1, 13) if i != 2]
    if book_idx == 1:
        return [("0", str(i), "Sherlock1", "1") for i in range(1, 11)]
    raise ValueError(f"Unsupported Sherlock book index: {book_idx} (expected 1..9)")


def build_sherlock_run_keys_for_books(
    books: Iterable[int | str],
) -> list[tuple[str, str, str, str]]:
    run_keys: list[tuple[str, str, str, str]] = []
    for book in books:
        run_keys.extend(build_sherlock_run_keys(int(book)))
    return run_keys


def _register_extra_libribrain_run_keys(
    run_keys: Sequence[tuple[str, str, str, str]],
) -> None:
    from pnpl.datasets.libribrain2025 import base as libribrain_base_module
    from pnpl.datasets.libribrain2025 import constants as libribrain_constants

    merged = list(libribrain_constants.RUN_KEYS)
    existing = set(merged)
    changed = False
    for run_key in run_keys:
        if run_key not in existing:
            merged.append(run_key)
            existing.add(run_key)
            changed = True
    if changed:
        libribrain_constants.RUN_KEYS = merged
        libribrain_base_module.RUN_KEYS = merged


def build_libribrain_sentence_dataset(
    data_path: str,
    books: Sequence[int | str],
    *,
    semantic_data_path: str | None = None,
    pnpl_root: str | os.PathLike[str] | None = None,
    preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
    set_name: str = "sentences",
    embedding_type: str = "SONAR",
    segment_ms: int | None = 3000,
    include_info: bool = True,
    standardize: bool = True,
    clipping_boundary: float | None = 10.0,
    preload_files: bool = False,
):
    """
    Load LibriBrain sentence-aligned MEG using semantic-vector NPZs for boundaries
    and sentence text. The embedding targets are still returned by PNPL, but the
    caller can ignore them and use only `info["sentence"]`.
    """
    LibriBrainSemanticVectors = import_libribrain_semantic_vectors(pnpl_root)

    class _FlexibleSemanticDataset(LibriBrainSemanticVectors):
        def __init__(self, *args, semantic_data_root: str | None = None, **kwargs):
            self.semantic_data_root = (
                os.path.abspath(os.path.expanduser(semantic_data_root))
                if semantic_data_root
                else None
            )
            super().__init__(*args, **kwargs)

        def _get_events_path(
            self,
            subject: str,
            session: str,
            task: str,
            run: str,
            event_type: str = "",
            semantic_vectors_variant: str | None = None,
        ) -> str:
            if event_type == "semantic_vectors" and self.semantic_data_root:
                fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_semantic_vectors"
                if semantic_vectors_variant:
                    fname += f"_{semantic_vectors_variant}"
                fname += ".npz"
                return os.path.join(
                    self.semantic_data_root,
                    task,
                    "derivatives",
                    "events",
                    fname,
                )
            return super()._get_events_path(
                subject=subject,
                session=session,
                task=task,
                run=run,
                event_type=event_type,
                semantic_vectors_variant=semantic_vectors_variant,
            )

    run_keys = build_sherlock_run_keys_for_books(books)
    _register_extra_libribrain_run_keys(run_keys)

    return _FlexibleSemanticDataset(
        data_path=data_path,
        semantic_data_root=semantic_data_path,
        preprocessing_str=preprocessing_str,
        include_run_keys=run_keys,
        include_info=include_info,
        set=set_name,
        embedding_type=embedding_type,
        segment_ms=segment_ms,
        standardize=standardize,
        clipping_boundary=clipping_boundary,
        preload_files=preload_files,
    )


def build_libribrain_sentence_dataset_per_book(
    data_path: str,
    books: Sequence[int | str],
    **kwargs,
):
    """Build per-book datasets and concatenate them, matching the MEG2SEM flow."""
    datasets = [
        build_libribrain_sentence_dataset(
            data_path=data_path,
            books=[int(book)],
            **kwargs,
        )
        for book in books
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def collate_libribrain_sentence_batch(batch):
    """Pad MEG time dimension and keep sentence strings / semantic vectors."""
    megs = []
    semantic_vectors = []
    info_list = []
    for item in batch:
        if len(item) == 3:
            meg, semantic_vector, info = item
        elif len(item) == 2:
            meg, semantic_vector = item
            info = {}
        else:
            raise ValueError(f"Unexpected LibriBrain sample format of length {len(item)}")
        megs.append(meg)
        semantic_vectors.append(torch.as_tensor(semantic_vector, dtype=torch.float32))
        info_list.append(info)

    channels = int(megs[0].shape[0])
    max_time = max(int(meg.shape[-1]) for meg in megs)
    batch_size = len(megs)

    meg_batch = torch.zeros((batch_size, channels, max_time), dtype=megs[0].dtype)
    meg_time_mask = torch.zeros((batch_size, max_time), dtype=torch.bool)
    lengths = torch.zeros((batch_size,), dtype=torch.long)

    for idx, meg in enumerate(megs):
        time_len = int(meg.shape[-1])
        meg_batch[idx, :, :time_len] = meg
        meg_time_mask[idx, :time_len] = True
        lengths[idx] = time_len

    sentences = [str(info.get("sentence", "")) for info in info_list]
    return {
        "meg": meg_batch,
        "meg_time_mask": meg_time_mask,
        "meg_lengths": lengths,
        "semantic_vectors": torch.stack(semantic_vectors),
        "sentences": sentences,
        "info": info_list,
    }
