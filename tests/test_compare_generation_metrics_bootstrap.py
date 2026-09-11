import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "compare_generation_metrics_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("compare_generation_metrics_bootstrap", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reconstructs_per_sample_metrics_from_compact_generation_json() -> None:
    payload = {
        "generated": ["red cat", "one two three"],
        "targets": ["red dog", "one two"],
        "word_overlap": {"summary": {}},
    }
    metrics = MODULE.per_sample_text_metrics(payload)
    np.testing.assert_allclose(metrics["per_sample"], [0.5, 0.8])
    np.testing.assert_allclose(metrics["per_sample_wer"], [0.5, 0.5])
    assert metrics["per_sample_content"].shape == (2,)


def test_prefers_stored_per_sample_metrics() -> None:
    payload = {
        "word_overlap": {
            "per_sample_content": [0.1],
            "per_sample": [0.2],
            "per_sample_wer": [0.3],
        }
    }
    metrics = MODULE.per_sample_text_metrics(payload)
    np.testing.assert_allclose(metrics["per_sample_content"], [0.1])
    np.testing.assert_allclose(metrics["per_sample"], [0.2])
    np.testing.assert_allclose(metrics["per_sample_wer"], [0.3])
