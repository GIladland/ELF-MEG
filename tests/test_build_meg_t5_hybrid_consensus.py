from scripts.build_meg_t5_hybrid_consensus import select_consensus_generation


def test_consensus_selects_supported_small_edit_without_reference_text():
    selected, support, minimum = select_consensus_generation(
        "the cat sat",
        ["the cat slept", "the cat slept", "the cat ran", "the cat sat"],
        minimum_support=0.5,
        max_source_edits=1,
    )
    assert selected == "the cat slept"
    assert support == minimum == 2


def test_consensus_returns_source_on_tie_or_large_edit():
    tied, _, _ = select_consensus_generation(
        "the cat sat",
        ["the cat slept", "the cat ran"],
        minimum_support=0.5,
        max_source_edits=1,
    )
    too_large, _, _ = select_consensus_generation(
        "the cat sat",
        ["a dog ran", "a dog ran"],
        minimum_support=0.5,
        max_source_edits=1,
    )
    assert tied == "the cat sat"
    assert too_large == "the cat sat"
