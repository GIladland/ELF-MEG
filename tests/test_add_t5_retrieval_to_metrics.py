import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "add_t5_retrieval_to_metrics.py"
SPEC = importlib.util.spec_from_file_location("add_t5_retrieval_to_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_retrieval_metrics_exact_identity() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    metrics = MODULE.retrieval_metrics(embeddings, embeddings)
    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["mean_rank"] == 1.0
    assert metrics["median_rank"] == 1.0
    assert metrics["ranks"] == [1, 1, 1, 1]


def test_retrieval_metrics_rejects_shape_mismatch() -> None:
    try:
        MODULE.retrieval_metrics(
            np.zeros((2, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
        )
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("shape mismatch should fail")
