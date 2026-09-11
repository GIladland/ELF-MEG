import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "rerank_candidates_by_roberta_mapper.py"
SPEC = importlib.util.spec_from_file_location("rerank_candidates_by_roberta_mapper", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ridge_recovers_linear_mapping() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    true_weights = rng.normal(size=(4, 3)).astype(np.float32)
    y = x @ true_weights + 0.2
    fit = MODULE.fit_ridge(x, y, 1e-6)
    predicted = MODULE.apply_ridge(x, *fit)
    expected = MODULE.normalize_rows(y)
    assert float(np.mean(np.sum(predicted * expected, axis=1))) > 0.999


def test_grouped_split_has_no_story_overlap() -> None:
    stories = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    train, val = MODULE.grouped_train_validation_split(stories, 49)
    assert set(stories[index] for index in train).isdisjoint(stories[index] for index in val)


def test_length_penalty_can_break_semantic_tie_without_references() -> None:
    scores = np.asarray([[0.8, 0.8], [0.9, 0.89]], dtype=np.float32)
    candidates = [
        ["one two three", "one two three four five six seven eight nine ten"],
        ["one two three four five six seven eight nine ten", "one two"],
    ]
    selected, adjusted = MODULE.select_with_length_penalty(
        scores, candidates, target_word_count=10, penalty=0.01
    )
    assert selected.tolist() == [1, 0]
    assert adjusted.shape == scores.shape


def test_negative_length_penalty_is_rejected() -> None:
    scores = np.asarray([[1.0]], dtype=np.float32)
    try:
        MODULE.select_with_length_penalty(
            scores, [["ten words are not needed here"]], target_word_count=10, penalty=-0.1
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative length penalty should fail")


def test_pool_delay_blocks_preserves_order_and_means() -> None:
    values = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 6.0, 8.0, 10.0]], dtype=np.float32
    )
    pooled = MODULE.pool_delay_blocks(values, 2)
    np.testing.assert_allclose(pooled, [[2.0, 3.0], [6.0, 8.0]])
    np.testing.assert_allclose(MODULE.pool_delay_blocks(values, 1), values)


def test_pool_delay_blocks_rejects_bad_partition() -> None:
    try:
        MODULE.pool_delay_blocks(np.zeros((2, 5), dtype=np.float32), 2)
    except ValueError as error:
        assert "cannot split" in str(error)
    else:
        raise AssertionError("non-divisible delay blocks should fail")
