import importlib.util
import csv
import json
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


def test_loads_decoding_sweep_best_as_standard_candidate(tmp_path: Path) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "targets": ["one target"],
                "best": {
                    "generated": ["one output"],
                    "matched": {"word_error_rate": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )
    payload = MODULE.load_json(str(path))
    assert payload["generated"] == ["one output"]
    assert payload["generation_quality"]["word_error_rate"] == 0.5


def test_loads_best_metrics_when_generated_is_also_top_level(tmp_path: Path) -> None:
    path = tmp_path / "sweep_top_level.json"
    path.write_text(
        json.dumps(
            {
                "targets": ["one target"],
                "generated": ["one output"],
                "best": {"matched": {"word_error_rate": 0.4}},
            }
        ),
        encoding="utf-8",
    )
    payload = MODULE.load_json(str(path))
    assert payload["generated"] == ["one output"]
    assert payload["generation_quality"]["word_error_rate"] == 0.4


def test_loads_bertscore_from_audit_csv_in_dataset_order(tmp_path: Path) -> None:
    path = tmp_path / "ranked.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "bertscore_raw_f1"]
        )
        writer.writeheader()
        writer.writerow({"index": 1, "bertscore_raw_f1": 0.2})
        writer.writerow({"index": 0, "bertscore_raw_f1": 0.1})
    np.testing.assert_allclose(MODULE.load_bertscore_f1(str(path)), [0.1, 0.2])
