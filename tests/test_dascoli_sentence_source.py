import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.dascoli_sentence_source import (
    apply_source_token_copy,
    align_word_confidence_to_tokens,
    build_position_vocabulary,
    confidence_weighted_source_latents,
    preserve_editable_source_latents,
    external_sentence_source,
    force_vocabulary_token_confidence,
    simulate_dascoli_sentences,
    transform_token_confidence,
)


SENTENCES = [
    "one cat sat on the red rug all day long",
    "two dogs ran by a blue car one cold night",
    "three birds flew to the green tree at warm noon",
]


def test_simulated_extremes_are_exact_or_wrong_at_every_position():
    position_vocabulary, global_vocabulary = build_position_vocabulary(
        SENTENCES, expected_words=10
    )
    exact = simulate_dascoli_sentences(
        SENTENCES,
        position_vocabulary=position_vocabulary,
        global_vocabulary=global_vocabulary,
        accuracies=[1.0],
        seed=7,
    )
    wrong = simulate_dascoli_sentences(
        SENTENCES,
        position_vocabulary=position_vocabulary,
        global_vocabulary=global_vocabulary,
        accuracies=[0.0],
        seed=7,
    )
    assert exact.sentences == SENTENCES
    assert bool(exact.correct_word_mask.all())
    assert not bool(wrong.correct_word_mask.any())
    assert bool(((exact.word_confidence >= 0) & (exact.word_confidence <= 1)).all())


def test_simulation_is_deterministic_for_a_seed():
    position_vocabulary, global_vocabulary = build_position_vocabulary(
        SENTENCES, expected_words=10
    )
    kwargs = dict(
        position_vocabulary=position_vocabulary,
        global_vocabulary=global_vocabulary,
        accuracies=[0.25, 0.5, 0.75],
        seed=19,
    )
    first = simulate_dascoli_sentences(SENTENCES, **kwargs)
    second = simulate_dascoli_sentences(SENTENCES, **kwargs)
    assert first.sentences == second.sentences
    assert torch.equal(first.word_confidence, second.word_confidence)
    assert torch.equal(first.correct_word_mask, second.correct_word_mask)


def test_accuracy_curve_uses_nested_correct_word_masks():
    position_vocabulary, global_vocabulary = build_position_vocabulary(
        SENTENCES, expected_words=10
    )
    masks = []
    for accuracy in (0.25, 0.5, 0.75):
        simulation = simulate_dascoli_sentences(
            SENTENCES,
            position_vocabulary=position_vocabulary,
            global_vocabulary=global_vocabulary,
            accuracies=[accuracy],
            seed=31,
        )
        masks.append(simulation.correct_word_mask)
    assert not bool((masks[0] & ~masks[1]).any())
    assert not bool((masks[1] & ~masks[2]).any())


def test_word_confidence_maps_to_subwords_and_mean_to_eos():
    confidence = torch.tensor([[0.8, 0.2]])
    offsets = torch.tensor([[[0, 5], [6, 10], [0, 0], [0, 0]]])
    attention = torch.tensor([[1, 1, 1, 0]])
    aligned = align_word_confidence_to_tokens(
        sentences=["alpha beta"],
        word_confidence=confidence,
        offset_mapping=offsets,
        attention_mask=attention,
    )
    assert torch.allclose(aligned, torch.tensor([[0.8, 0.2, 0.5, 0.0]]))


def test_force_vocabulary_token_confidence_preserves_only_matching_words():
    confidence = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    offsets = torch.tensor([[[0, 3], [4, 9], [10, 13], [0, 0]]])
    attention = torch.tensor([[1, 1, 1, 1]])
    forced = force_vocabulary_token_confidence(
        sentences=["the river and"],
        token_confidence=confidence,
        offset_mapping=offsets,
        attention_mask=attention,
        preserved_words={"the", "and"},
    )
    assert torch.allclose(forced, torch.tensor([[1.0, 0.3, 1.0, 0.5]]))


def test_external_source_supports_short_proposals_and_default_confidence():
    source = external_sentence_source(
        {
            "generated": ["One cat sat", "two dogs ran by"],
            "targets": SENTENCES[:2],
        },
        expected_targets=SENTENCES[:2],
        expected_words=10,
        default_confidence=0.6,
    )
    assert source.sentences == ["One cat sat", "two dogs ran by"]
    assert torch.allclose(source.word_confidence[0, :3], torch.full((3,), 0.6))
    assert not bool(source.word_confidence[0, 3:].any())
    assert bool(source.correct_word_mask[0, :3].all())
    assert torch.isclose(source.row_accuracy[0], torch.tensor(0.3))


def test_external_source_preserves_punctuation_without_losing_tenth_lexical_word():
    sentence = "double double - i mean he said now we have to"
    confidence = [0.1 * index for index in range(11)]
    source = external_sentence_source(
        {
            "generated": [sentence],
            "targets": [SENTENCES[0]],
            "word_confidence": [confidence],
        },
        expected_targets=[SENTENCES[0]],
        expected_words=10,
        default_confidence=0.6,
    )
    assert source.sentences == [sentence]
    assert source.word_confidence.shape == (1, 11)
    assert torch.allclose(source.word_confidence[0], torch.tensor(confidence))


def test_external_source_supports_nested_confidence_field():
    source = external_sentence_source(
        {
            "generated": ["one two three"],
            "word_confidence_variants": {"margin": [[0.1, 0.4, 0.9]]},
        },
        expected_targets=["one target three"],
        expected_words=3,
        default_confidence=0.5,
        confidence_field="word_confidence_variants.margin",
    )
    assert torch.allclose(source.word_confidence[0], torch.tensor([0.1, 0.4, 0.9]))


def test_short_external_confidence_maps_mean_to_eos_without_padding_bias():
    confidence = torch.tensor([[0.8, 0.2, 0.0, 0.0]])
    offsets = torch.tensor([[[0, 5], [6, 10], [0, 0], [0, 0]]])
    attention = torch.tensor([[1, 1, 1, 0]])
    aligned = align_word_confidence_to_tokens(
        sentences=["alpha beta"],
        word_confidence=confidence,
        offset_mapping=offsets,
        attention_mask=attention,
    )
    assert torch.allclose(aligned, torch.tensor([[0.8, 0.2, 0.5, 0.0]]))


def test_confidence_weighted_source_has_correct_endpoints():
    source = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    confidence = torch.tensor([[1.0, 0.0, 0.25], [0.5, 0.75, 1.0]])
    mixed = confidence_weighted_source_latents(source, confidence, noise)
    assert torch.equal(mixed[0, 0], source[0, 0])
    assert torch.equal(mixed[0, 1], noise[0, 1])
    assert torch.allclose(mixed[0, 2], 0.25 * source[0, 2] + 0.75 * noise[0, 2])


def test_hard_copy_preserves_only_confident_nonpadding_tokens():
    generated = torch.tensor([[9, 9, 9, 9]])
    source = torch.tensor([[1, 2, 3, 0]])
    confidence = torch.tensor([[0.9, 0.4, 0.7, 1.0]])
    attention = torch.tensor([[1, 1, 1, 0]])
    copied, mask = apply_source_token_copy(
        generated, source, confidence, attention, threshold=0.7
    )
    assert torch.equal(copied, torch.tensor([[1, 9, 3, 9]]))
    assert torch.equal(mask, torch.tensor([[True, False, True, False]]))


def test_editable_latent_preservation_blends_only_uncopied_source_tokens():
    flow = torch.zeros(1, 4, 2)
    source = torch.full((1, 4, 2), 4.0)
    confidence = torch.tensor([[0.9, 0.2, 0.6, 0.1]])
    attention = torch.tensor([[1, 1, 1, 0]])
    blended, editable = preserve_editable_source_latents(
        flow,
        source,
        confidence,
        attention,
        copy_threshold=0.5,
        preservation=0.25,
    )
    assert torch.equal(editable, torch.tensor([[False, True, False, False]]))
    assert torch.equal(blended[0, 0], flow[0, 0])
    assert torch.equal(blended[0, 1], torch.ones(2))
    assert torch.equal(blended[0, 2], flow[0, 2])
    assert torch.equal(blended[0, 3], flow[0, 3])


def test_confidence_floor_preserves_padding_and_has_unweighted_endpoint():
    confidence = torch.tensor([[0.2, 0.8, 0.0]])
    attention = torch.tensor([[1, 1, 0]])
    floored = transform_token_confidence(
        confidence,
        attention,
        global_trust=0.5,
        confidence_floor=0.25,
    )
    assert torch.allclose(floored, torch.tensor([[0.325, 0.55, 0.0]]))
    unweighted = transform_token_confidence(
        confidence,
        attention,
        confidence_floor=1.0,
    )
    assert torch.equal(unweighted, attention.to(torch.float32))
