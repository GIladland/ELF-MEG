from scripts.rank_generation_examples import content_tokens, edit_distance, overlap_f1, word_tokens


def test_rank_helpers_match_expected_word_metrics() -> None:
    assert word_tokens("Thirty years, old!") == ["thirty", "years", "old"]
    assert content_tokens("I got up to the train") == ["got", "train"]
    assert overlap_f1(["years"], ["thirty", "years"]) == 2.0 / 3.0
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_empty_overlap_and_deletion_distance() -> None:
    assert overlap_f1([], ["target"]) == 0.0
    assert edit_distance(["one", "two"], []) == 2
