import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "cap_generated_text_words.py"
SPEC = importlib.util.spec_from_file_location("cap_generated_text_words", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cap_uses_evaluator_tokens_and_preserves_prefix() -> None:
    text = "One-two, THREE! four five"
    assert MODULE.cap_text(text, 3) == "One-two, THREE"
    assert MODULE.words(MODULE.cap_text(text, 3)) == ["one", "two", "three"]


def test_cap_does_not_normalize_short_text() -> None:
    text = "Don't change THIS."
    assert MODULE.cap_text(text, 10) == text


def test_load_csv_restores_row_order(tmp_path: Path) -> None:
    path = tmp_path / "ranked.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "target", "generated"])
        writer.writeheader()
        writer.writerows(
            [
                {"index": 1, "target": "target one", "generated": "generation one"},
                {"index": 0, "target": "target zero", "generated": "generation zero"},
            ]
        )
    targets, generated, payload = MODULE.load_rows(metrics_json=None, input_csv=path)
    assert targets == ["target zero", "target one"]
    assert generated == ["generation zero", "generation one"]
    assert payload == {}


def test_load_csv_rejects_noncontiguous_indices(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("index,target,generated\n2,target,generation\n", encoding="utf-8")
    try:
        MODULE.load_rows(metrics_json=None, input_csv=path)
    except ValueError as error:
        assert "exactly 0..0" in str(error)
    else:
        raise AssertionError("noncontiguous indices should fail")
