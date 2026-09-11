from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_fmri_minilm_t5_prefix import (  # noqa: E402
    MiniLMT5Prefix,
    IdentityResidualCalibrator,
    OrderedDelayT5Prefix,
    DelayWeightedCalibrator,
    ResidualBrainCalibrator,
    StaticBatchLogitBias,
    UnseenTokenBatchLogitBias,
    content_ranked_indices,
    content_token_bias,
    generate,
    format_keyword_prompts,
    inject_t5_cross_attention_lora,
    sequence_nll,
    train_content_idf_weights,
    tokenized_labels_and_weights,
)
from probe_fmri_supervised_content_head import ContentHead, temporal_context_features  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke check requires CUDA")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-small", local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projector = MiniLMT5Prefix(384, 32, 4, model.config.d_model, 0.0).to(device)
    semantic = torch.nn.functional.normalize(torch.randn(2, 384, device=device), dim=-1)
    context = projector(semantic)
    labels = tokenizer(
        ["a short sentence", "another small sentence"],
        padding=True,
        return_tensors="pt",
    )["input_ids"].to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    mask = torch.ones(context.shape[:2], dtype=torch.long, device=device)
    weighted_labels, label_weights = tokenized_labels_and_weights(
        tokenizer,
        ["the silver flowers", "a short sentence"],
        max_length=8,
        content_weight=3.0,
    )
    assert weighted_labels.shape == label_weights.shape == (2, 8)
    assert float(label_weights.max()) == 3.0
    assert bool((label_weights == 1.0).any())
    assert bool((label_weights == 0.0).any())
    idf_weights = train_content_idf_weights(
        ["common rare", "common ordinary", "common routine"],
        power=1.0,
        max_multiplier=3.0,
    )
    assert idf_weights["rare"] > idf_weights["common"]
    _, idf_label_weights = tokenized_labels_and_weights(
        tokenizer,
        ["the common rare"],
        max_length=8,
        content_weight=2.0,
        content_word_weights=idf_weights,
    )
    assert float(idf_label_weights.max()) > 2.0
    weighted_loss, per_row_nll = sequence_nll(
        model,
        context,
        mask,
        weighted_labels.to(device),
        label_weights.to(device),
    )
    assert weighted_loss.ndim == 0 and per_row_nll.shape == (2,)
    temporal = temporal_context_features(
        np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float32),
        np.arange(3), np.asarray(["a", "a", "b"]), np.asarray([0, 10, 0]),
        (-10, 0, 10),
    )
    assert temporal.shape == (3, 6)
    assert torch.allclose(temporal[0, :2], torch.zeros(2))
    assert temporal[0, 2] > 0 and temporal[0, 5] > 0
    assert torch.allclose(temporal[2, :2], torch.zeros(2))
    assert torch.allclose(temporal[2, 4:], torch.zeros(2))
    ranked_words = torch.tensor([[0, 1]])
    assert format_keyword_prompts(ranked_words, ["silver", "flowers"], "raw") == [
        "silver flowers"
    ]
    assert format_keyword_prompts(
        ranked_words, ["silver", "flowers"], "summarize"
    ) == ["summarize: silver flowers"]
    loss = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=context),
        attention_mask=mask,
        labels=labels,
    ).loss
    loss.backward()
    assert projector.net[-1].weight.grad is not None
    assert projector.net[-1].weight.grad.abs().sum() > 0

    calibrator = ResidualBrainCalibrator(384, 32, 0.0).to(device)
    calibrated = calibrator(semantic)
    # Zero-initialized residual means the new stage starts as an exact
    # normalized identity, preserving the pretrained brain representation.
    assert torch.allclose(calibrated, semantic, atol=1e-6)
    projector.zero_grad(set_to_none=True)
    calibrated_context = projector(calibrated)
    calibrated_loss = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=calibrated_context),
        attention_mask=mask,
        labels=labels,
    ).loss
    calibrated_loss.backward()
    assert calibrator.net[-1].weight.grad is not None
    assert calibrator.net[-1].weight.grad.abs().sum() > 0

    ada_semantic = torch.nn.functional.normalize(
        torch.randn(2, 1536, device=device), dim=-1
    )
    ada_calibrator = IdentityResidualCalibrator(1536, 32, 0.0).to(device)
    assert torch.allclose(ada_calibrator(ada_semantic), ada_semantic, atol=1e-6)
    ada_projector = MiniLMT5Prefix(
        1536, 32, 4, model.config.d_model, 0.0
    ).to(device)
    ada_loss = model(
        encoder_outputs=BaseModelOutput(
            last_hidden_state=ada_projector(ada_calibrator(ada_semantic))
        ),
        attention_mask=torch.ones(2, 4, dtype=torch.long, device=device),
        labels=labels,
    ).loss
    ada_loss.backward()
    assert ada_calibrator.net[-1].weight.grad is not None
    assert ada_calibrator.net[-1].weight.grad.abs().sum() > 0

    delayed = torch.nn.functional.normalize(torch.randn(2, 1536, device=device), dim=-1)
    delayed_calibrator = ResidualBrainCalibrator(1536, 32, 0.0).to(device)
    delayed_calibrated = delayed_calibrator(delayed)
    delayed_mean = torch.nn.functional.normalize(
        delayed.reshape(2, 4, 384).mean(dim=1), dim=-1
    )
    assert torch.allclose(delayed_calibrated, delayed_mean, atol=1e-6)
    delayed_context = projector(delayed_calibrated)
    delayed_loss = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=delayed_context),
        attention_mask=mask,
        labels=labels,
    ).loss
    delayed_loss.backward()
    assert delayed_calibrator.net[-1].weight.grad is not None
    assert delayed_calibrator.net[-1].weight.grad.abs().sum() > 0

    delay_weighted = DelayWeightedCalibrator().to(device)
    delay_weighted_output = delay_weighted(delayed)
    assert torch.allclose(delay_weighted_output, delayed_mean, atol=1e-6)
    delay_weighted_loss = model(
        encoder_outputs=BaseModelOutput(
            last_hidden_state=projector(delay_weighted_output)
        ),
        attention_mask=mask,
        labels=labels,
    ).loss
    delay_weighted_loss.backward()
    assert delay_weighted.delay_logits.grad is not None
    assert delay_weighted.log_scale.grad is not None
    assert delay_weighted.bias.grad is not None

    ordered_projector = OrderedDelayT5Prefix(
        prefix_length=4, model_dim=model.config.d_model, dropout=0.0, layers=1
    ).to(device)
    ordered_context = ordered_projector(delayed)
    assert ordered_context.shape == (2, 4, model.config.d_model)
    ordered_loss = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=ordered_context),
        attention_mask=torch.ones(2, 4, dtype=torch.long, device=device),
        labels=labels,
    ).loss
    ordered_loss.backward()
    assert ordered_projector.output[-1].weight.grad is not None
    assert ordered_projector.output[-1].weight.grad.abs().sum() > 0

    static_bias = torch.zeros(2, model.config.vocab_size, device=device)
    static_bias[0, 3] = 1.25
    static_bias[1, 4] = 2.5
    processed = StaticBatchLogitBias(static_bias)(
        torch.zeros(4, 1, dtype=torch.long, device=device),
        torch.zeros(4, model.config.vocab_size, device=device),
    )
    assert processed[0, 3] == processed[1, 3] == 1.25
    assert processed[2, 4] == processed[3, 4] == 2.5
    unseen_processed = UnseenTokenBatchLogitBias(static_bias)(
        torch.tensor([[0, 3], [0, 2], [0, 4], [0, 2]], device=device),
        torch.zeros(4, model.config.vocab_size, device=device),
    )
    assert unseen_processed[0, 3] == 0.0
    assert unseen_processed[1, 3] == 1.25
    assert unseen_processed[2, 4] == 0.0
    assert unseen_processed[3, 4] == 2.5

    content_head = ContentHead(384, 0, 2).to(device).eval()
    biased_generation = generate(
        projector, model, tokenizer, semantic.detach(),
        batch_size=2, max_target_tokens=4, num_beams=1, device=device,
        content_head=content_head,
        content_prior_logit=torch.zeros(2, device=device),
        content_vocabulary_token_ids=[
            tokenizer.encode("short", add_special_tokens=False),
            tokenizer.encode("another", add_special_tokens=False),
        ],
        content_bias_topk=1,
        content_bias_strength=0.25,
        content_vocabulary=["short", "another"],
        content_keyword_context=True,
    )
    assert len(biased_generation) == 2

    prior_filter_head = ContentHead(384, 0, 3).to(device).eval()
    with torch.no_grad():
        prior_filter_head.net[1].weight.zero_()
        prior_filter_head.net[1].bias.copy_(
            torch.tensor([3.0, 2.0, 1.0], device=device)
        )
    ranked_filtered = content_ranked_indices(
        prior_filter_head,
        semantic.detach(),
        torch.logit(torch.tensor([0.9, 0.01, 0.02], device=device)),
        topk=2,
        prior_subtraction=0.0,
        max_prior_probability=0.05,
    )
    assert ranked_filtered.tolist() == [[1, 2], [1, 2]]
    ranked_bias = content_token_bias(
        prior_filter_head,
        semantic.detach(),
        torch.logit(torch.tensor([0.9, 0.01, 0.02], device=device)),
        [[3], [4], [5]],
        topk=2,
        strength=1.0,
        prior_subtraction=0.0,
        model_vocab_size=model.config.vocab_size,
        max_prior_probability=0.05,
        weighting="ranked",
    )
    softmax_bias = content_token_bias(
        prior_filter_head,
        semantic.detach(),
        torch.logit(torch.tensor([0.9, 0.01, 0.02], device=device)),
        [[3], [4], [5]],
        topk=2,
        strength=1.0,
        prior_subtraction=0.0,
        model_vocab_size=model.config.vocab_size,
        max_prior_probability=0.05,
        weighting="softmax",
        temperature=0.5,
    )
    assert ranked_bias.shape == softmax_bias.shape
    assert not torch.allclose(ranked_bias[:, 4:6], softmax_bias[:, 4:6])

    injected = inject_t5_cross_attention_lora(
        model, rank=2, alpha=4.0, last_n_blocks=1
    )
    assert len(injected) == 4
    lora_context = projector(semantic.detach())
    lora_loss = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=lora_context),
        attention_mask=mask,
        labels=labels,
    ).loss
    lora_loss.backward()
    lora_b_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if ".lora_B" in name
    ]
    assert lora_b_gradients and all(gradient is not None for gradient in lora_b_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in lora_b_gradients) > 0.0
    with torch.no_grad():
        generated = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=context.detach()),
            attention_mask=mask,
            max_new_tokens=4,
            num_beams=1,
        )
    assert generated.shape[0] == 2
    print(
        "frozen T5 prefix forward/generation plus projector and identity-calibrator "
        "gradient checks passed"
    )


if __name__ == "__main__":
    main()
