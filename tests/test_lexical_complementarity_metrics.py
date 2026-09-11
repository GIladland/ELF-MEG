from scripts.meg_context_overfit import word_overlap_metrics


def test_exact_and_stemmed_overlap_are_reported_separately() -> None:
    metrics = word_overlap_metrics(
        ["my friends jumped with the dog"],
        ["my friend jumps with a dog"],
    )
    summary = metrics["summary"]

    assert summary["exact_words_overlap"] == summary["words_overlap"]
    assert summary["exact_content_words_overlap"] == summary["content_words_overlap"]
    assert summary["exact_words_overlap"] == 0.5
    assert summary["stemmed_words_overlap"] == 5.0 / 6.0
    assert summary["exact_content_words_overlap"] == 1.0 / 3.0
    assert summary["stemmed_content_words_overlap"] == 1.0


def test_function_word_counts_use_the_content_stopword_contract() -> None:
    metrics = word_overlap_metrics(
        ["my friends jumped with the dog"],
        ["my friend jumps with a dog"],
    )
    summary = metrics["summary"]

    assert summary["decoded_word_count"] == 6
    assert summary["decoded_function_word_count"] == 3
    assert summary["decoded_function_word_fraction"] == 0.5
    assert summary["target_function_word_count"] == 3
    assert summary["target_function_word_fraction"] == 0.5
    assert metrics["per_sample_decoded_function_words"] == [["my", "with", "the"]]


def test_function_word_counts_include_pronoun_and_auxiliary_contractions() -> None:
    metrics = word_overlap_metrics(["who's there and isn't ready"], ["who is there"])

    assert metrics["summary"]["decoded_function_word_count"] == 4
    assert metrics["per_sample_decoded_function_words"] == [["who's", "there", "and", "isn't"]]
