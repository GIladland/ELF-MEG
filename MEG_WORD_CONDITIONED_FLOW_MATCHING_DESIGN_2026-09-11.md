# MEG semantic + D'Ascoli word-conditioned ELF flow matching

Date: 2026-09-11

## Tested baseline: sentence-source conditional transport

The 2026-09-12 val110 experiment validates this as a useful baseline but not
as the final fusion. At 49.82% target-derived word accuracy, full-strength
semantic plus confidence-weighted source reaches word+content sum `0.12450`,
whereas sentence-only full trust reaches `0.64683` and a small semantic scale
of `0.035` reaches `0.65497`. Directly emitting the simulated D'Ascoli sentence
reaches `0.95226`. The confidence/Gaussian blend and full semantic prefix both
discard useful lexical structure. See
`MEG_DASCOLI_SENTENCE_FLOW_RESULTS_2026-09-12.md`.

The next implementation should always start on the complete source-sentence
manifold, pass confidence as a separate feature, and gate semantic
cross-attention residuals inside ELF. The posterior-word-memory design below
is now recommended rather than deferred.

D'Ascoli is expected to produce a complete ordered ten-word sentence plus one
calibrated confidence per word.  The primary experiment therefore treats that
sentence as the **source distribution of the flow**, while the existing
MEG-predicted ADA embedding remains a clean semantic condition:

```text
D'Ascoli sentence -> frozen T5 encoder -> h [B,L,512]
word confidence -> T5 offset alignment -> c [B,L,1]
Gaussian prior -> epsilon [B,L,512]

s = c*h + (1-c)*(sigma*epsilon)
z_t = t*x_target + (1-t)*s
v* = (x_target-z_t)/(1-t) = x_target-s

MEG qc4wyals ADA -> frozen audited adapter -> C_sem [B,64,512]
v_theta(z_t,t | C_sem) -> target flow
```

At inference, sampling starts from `s` at `t=0` and integrates to `t=1`.
Thus a confident word starts near D'Ascoli's proposed T5 state, an uncertain
word starts near ordinary diffusion noise, and ELF can use ADA semantics to
repair the complete proposal.  There is one global flow time; confidence
changes the per-token source state rather than inventing incompatible local
timesteps.

This path is implemented in:

- `src/modules/dascoli_sentence_source.py`: controlled full-sentence simulator,
  word-to-subword confidence alignment, and source/noise blending;
- `scripts/meg_context_overfit.py`: source-to-target flow and decoder training,
  plus source-start sampling; and
- `scripts/train_npz_semantic_to_elf.py`: target-derived D'Ascoli simulation,
  modality dropout, source controls, and audit metadata.

Training mixes assumed D'Ascoli word accuracies `0.25/0.50/0.75/0.90`.  Ten
percent of source rows and ten percent of semantic conditions are independently
dropped so the trained checkpoint retains semantic-only and sentence-only
branches instead of learning compulsory copying.

The fixed validation matrix is:

| Arm | Purpose |
|---|---|
| semantic-only | Preserve/compare the audited qc4wyals ELF-B path |
| sentence-only | Measure how much the D'Ascoli proposal carries by itself |
| joint unweighted | Test naive “start from the sentence” |
| joint confidence-weighted | Primary proposed system |
| joint shuffled | Verify gains require the matched D'Ascoli row |
| weighted accuracy 0/25/50/75/90/100% | Response curve and achievable ceiling |

All simulations are target-derived validation ceilings and must be labelled as
such.  They estimate the value of a future D'Ascoli classifier at a given
accuracy/calibration level; they are not brain-conditioned lexical results.
Protected test26 remains absent.

## Recommended next architecture: posterior word memory

Condition ELF on two separate pieces of evidence:

1. the existing frozen `qc4wyals` 1,536-D ADA prediction, which produces the
   established 64-token semantic context; and
2. an ordered, calibrated D'Ascoli posterior over words, represented as ten
   soft word-memory tokens in the same T5-small latent space used by ELF.

Do **not** concatenate word logits to the ADA vector, average predicted words
into one sentence vector, or hard-clamp generated tokens. Those choices erase
word order, make a false high-confidence word dominate the sentence, and make
it difficult to measure whether ELF used semantic versus lexical evidence.

The first implementation should add the word memory as a gated residual within
the existing 64 context positions:

```text
qc4wyals ADA [B,1536] -> frozen/current semantic adapter -> C_sem [B,64,512]

D'Ascoli top-k posterior [B,10,K]
    -> train-only T5 word prototype table [V,512]
    -> posterior-weighted ordered word tokens [B,10,512]
    -> small word resampler/slot projector -> C_word [B,64,512]

C = C_sem + tanh(g_word) * C_word
x0 = concat(C, target T5 states)
v_theta(z_t, t | C) -> target flow
```

`g_word` is one scalar per context slot and is initialized to zero. Step zero
is therefore exactly the imported semantic-only checkpoint. Keeping 64 total
context positions also preserves target rotary positions, existing adapter
shapes, checkpoint loading, and the current decoding shift.

If ordinary self-attention ignores the lexical residual, the second
implementation should expose the ten word tokens as a separate masked memory
through zero-output-initialized target-to-word cross-attention in the final two
to four ELF-M blocks. Keep separate cross-attention parameters for the flow and
discrete decoder branches. The earlier fMRI screen showed that sharing newly
trained cross-attention between the denoising and decoder objectives could
damage an already useful flow trajectory; the decoder-only correction was the
safer route. The new word classifier may be stronger, but that failure mode
should still determine the staging.

## Why soft ordered word memory is the right interface

The D'Ascoli model should export a distribution rather than a single word at
each of the ten positions. For position `p`, retain the top `K` prior-corrected
logits and compute

```text
l'_p,w = l_p,w - alpha * log pi_p,w
q_p = softmax(topk(l'_p) / tau)
e_p = sum_w q_p,w * E_word[w]
```

where `pi` is estimated on training rows only and `E_word` is a train-only
word-prototype table in T5-small encoder space. Start with `K=5` and sweep
`K in {3,5,8}`, `alpha in {0,0.25,0.5}`, and temperature calibrated from OOF
training predictions. Preserve the top-k probabilities, entropy, and logit
margin as inputs; do not use a hard confidence threshold in the first screen.
In the previous fMRI lexical work, confidence and top-1/top-2 margin were not
monotonic with held-out hits, while weak signal spread across the top-five set.

Use ordered positions whenever D'Ascoli predicts them. If only bag-of-words
evidence is available, keep each selected word as a separate rank-labelled
memory token and mark the layout `bag`; do not invent position labels.

### Building `E_word`

Use the exact frozen `t5-small` encoder associated with the ELF checkpoint.
For every training sentence:

1. tokenize with offsets;
2. encode the full sentence;
3. mean the encoder states of subword pieces overlapping each whitespace word;
4. average those contextual states by normalized vocabulary item; and
5. L2-normalize the resulting prototype.

This places word evidence on the target latent manifold more directly than the
T5 input embedding table or ADA sentence prototypes. Build prototypes from the
2,652 training rows only. Validation text must not contribute to prototypes,
priors, vocabulary, calibration, or model fitting.

For a word with no usable prototype, use a learned `<unk>` vector and mark that
position invalid for lexical scoring. D'Ascoli should normally prevent this by
exporting only training-vocabulary IDs.

## Word context encoder

A compact initial module is sufficient:

```python
class WordPosteriorContext(nn.Module):
    def __init__(self, prototypes, positions=10, context_length=64):
        super().__init__()
        self.register_buffer("prototypes", prototypes)  # [V, 512]
        self.position = nn.Embedding(positions, 512)
        self.confidence = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 512))
        self.token_norm = nn.LayerNorm(512)
        self.resampler = nn.TransformerDecoder(..., num_layers=1)
        self.queries = nn.Parameter(torch.randn(1, context_length, 512) * 0.02)

    def forward(self, topk_ids, topk_probs, valid_mask):
        selected = self.prototypes[topk_ids]                 # [B,10,K,512]
        words = (topk_probs[..., None] * selected).sum(2)   # [B,10,512]
        entropy = -(topk_probs.clamp_min(1e-8).log() * topk_probs).sum(2)
        margin = topk_probs[..., 0] - topk_probs[..., 1]
        features = torch.stack([topk_probs[..., 0], entropy, margin], dim=-1)
        words = self.token_norm(words + self.position.weight[None])
        words = words + self.confidence(features)
        memory_padding_mask = ~valid_mask.bool()
        queries = self.queries.expand(len(words), -1, -1)
        return self.resampler(
            tgt=queries,
            memory=words,
            memory_key_padding_mask=memory_padding_mask,
        )
```

The exact implementation can use one cross-attention/resampler layer with four
or eight heads and a 1,024- or 2,048-wide feed-forward block. The experiment is
data-limited; a large word adapter is more likely to memorize the 2,652 rows.

Mask invalid/OOV positions in attention. Keep confidence as a feature rather
than multiplying the entire token by top-1 probability: even low-confidence
posteriors may carry useful relative lexical information.

## Flow-matching objective

The current ELF objective remains valid. For target latent sequence `x` and
noise `epsilon`, the existing interpolation is

```text
z_t = t*x + (1-t)*epsilon*sigma
v*  = (x-z_t)/(1-t)
L_FM = masked_mean(||v_theta(z_t,t | C_sem,C_word)-v*||^2)
```

Condition tokens remain clean and receive no flow loss. Only target positions
are noised and scored. The important change is that the conditional vector
field receives `C_sem + gated(C_word)`.

Train with both the existing flow loss and ELF decoder cross-entropy:

```text
L = lambda_FM * L_FM + lambda_CE * L_CE
```

Do not initially add a loss that rewards outputting the classifier's top word.
That would treat classifier errors as ground truth. Ground-truth target-token CE
already teaches ELF when lexical evidence is useful.

If the learned word gate stays near zero despite a useful classifier, add a
small matched-versus-deranged condition-ranking term, not a direct predicted-
word target:

```text
L_pair = relu(m + L_target(matched word memory)
                    - L_target(row-permuted word memory))
```

Use a fixed train-row derangement and report the unscaled losses separately.

## Independent condition dropout and guidance

The model must see missing and mismatched modalities during training. Use a
categorical condition-drop schedule rather than one dropout flag:

| Training condition | Initial probability |
|---|---:|
| semantic + word | 0.70 |
| semantic only | 0.10 |
| word only | 0.10 |
| unconditional | 0.10 |

This enables separate semantic and lexical guidance at sampling time. A
three-pass compositional rule is:

```text
v0   = v(z,t | empty)
vsem = v(z,t | semantic)
vfull= v(z,t | semantic,word)

v = v0 + s_sem*(vsem-v0) + s_word*(vfull-vsem)
```

At `s_sem=s_word=1`, this reduces to the full conditional vector field. Start
with both scales at one. Only sweep `s_word` after a fixed checkpoint is
selected, because guidance tuning and checkpoint tuning on the same 110 rows
will otherwise multiply validation selection.

For the first residual-in-context implementation, construct the three context
tensors outside the model. A later separate-memory implementation can thread
`word_memory` and `word_memory_mask` through the sampler explicitly.

## Training stages

### Stage 0: interface and oracle ceilings

Before brain-predicted words:

- verify word-posterior packing and row alignment;
- run exact target words as word memory to measure the architectural ceiling;
- run the training-position prior as word memory;
- run fixed row-permuted exact words as a negative control; and
- assert that zero word gate reproduces the semantic-only output bit-for-bit
  in deterministic mode.

The exact-word arm is an oracle only and must never be reported as
brain-conditioned performance.

### Stage 1: decoder-side lexical use

Freeze `qc4wyals`, the semantic adapter, the ELF-M flow path, and D'Ascoli.
Train the word context module, its zero-initialized gate, and (if needed) a
separate decoder-only target-to-word cross-attention in the final two blocks.
Set denoiser loss weight to zero in this stage. This cheaply tests whether the
word posterior is useful to ELF without perturbing the established trajectory.

### Stage 2: word-conditioned flow

Initialize from the best Stage-1 checkpoint. Add a distinct zero-initialized
flow word adapter in the final two blocks, or enable the residual word context
for the denoiser branch. Train with a low learning rate (initial screen:
`3e-5` and `1e-4`) and the normal flow loss. Keep the base ELF and semantic
path frozen; optionally allow rank-4 LoRA only in the final two to four blocks.

Promote to this stage only if Stage 1 beats semantic-only and fixed derangements
on lexical metrics. A classifier that fails this gate should be improved before
more diffusion capacity is exposed.

### Stage 3: tightly controlled joint adaptation

Only after Stage 2 succeeds, consider unfreezing the semantic input projector
or the final ELF LoRA blocks. D'Ascoli remains frozen. Use OOF D'Ascoli train
predictions, never in-sample predictions or exact train words, as the primary
training condition.

## Required D'Ascoli inputs

ELF needs two separate prediction packs with the same vocabulary contract:

- OOF predictions for all 2,652 Apple training rows, made by story-grouped
  folds; and
- predictions for all 110 validation rows, made without fitting on those rows.

The detailed producer contract is in
`DASCOLLI_WORD_CLASSIFICATION_HANDOVER_2026-09-11.md`.

The ELF loader must check all of the following before training:

- row count and row-ID equality against the semantic pack;
- exact sentence equality for alignment only;
- `reference_text_used_for_prediction == false`;
- OOF fold IDs are present for every training row;
- vocabulary and normalization hashes match train and validation packs;
- top-k probabilities are finite, non-negative, and sum to one over valid
  candidates;
- word IDs lie in `[0,V)`; and
- the protected 26-row test path is not present in arguments or metadata.

## Code change map in ELF

The recommended implementation touches these locations:

1. `src/modules/word_conditioning.py` (new): posterior-to-T5 prototypes,
   positional/confidence features, 64-slot resampler, and gated fusion wrapper.
2. `scripts/train_npz_semantic_to_elf.py`: word-prediction NPZ arguments,
   strict row alignment, OOF checks, independent condition dropout, checkpoint
   metadata, and lexical diagnostics.
3. `scripts/meg_context_overfit.py`: extend `OverfitBatch` with top-k word
   evidence and pass the fused condition through train/evaluation paths.
4. `src/modules/model.py` only for the second implementation: optional,
   zero-initialized target-to-word cross-attention with separate flow and
   decoder module dictionaries.
5. `src/utils/sampling_utils.py` only for separate-memory/compositional CFG:
   thread word memory and form unconditional/semantic/full forward passes.
6. tests: zero-gate checkpoint preservation, no gradient into D'Ascoli or the
   prototype table, word-mask behavior, train/validation row mismatch failure,
   modality dropout, compositional-CFG identity at scales one, and derangement.

The existing `FMRI2SEMLexicalToELFContextAdapter` is a useful pattern for
zero-initialized gates, prior correction, partitioned lexical memory, and
frozen classifier behavior. It should not be reused unchanged: it owns an
fMRI-to-semantic module and constructs sentence-semantic prototypes, whereas
this experiment receives independent D'Ascoli word posteriors and should use
direct T5 word prototypes.

## Validation and promotion contract

Keep the existing 110 rows, candidate bank, ten-word cap, and seed policy. For
every reported system include:

- the raw D'Ascoli top-1 sentence as a no-ELF baseline;
- word F1, content-word F1, their sum/mean, WER, raw BERTScore, BLEU-1,
  ROUGE-1, generated Top-1/Top-5, and semantic-interface retrieval;
- correct-source-position retention, wrong-source-position correction, and
  false changes to high-confidence correct positions;
- D'Ascoli position Top-1/Top-5 accuracy, sentence word/content set F1,
  train-vocabulary coverage, and calibration error;
- semantic-only, word-only, full, zero-word, frequency-prior, and at least 200
  fixed row-deranged word conditions; and
- a 100,000-permutation one-to-one row test for the final selected system.

Select checkpoints on a declared validation metric, preferably the existing
word/content mean, with WER and semantic retrieval as guardrails. A candidate
is not a lexical advance if its gain disappears under matched-versus-deranged
comparison or consists only of train-common words. Also report train-IDF and
noncommon content F1.

The 26 `birthofanation` rows remain unopened during classifier development,
word-adapter fitting, hyperparameter selection, guidance selection, and
qualitative inspection.
