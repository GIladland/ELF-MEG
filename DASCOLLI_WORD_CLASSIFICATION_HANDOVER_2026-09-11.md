# D'Ascoli handover: ordered word classification for Apple MEG -> ELF

Date: 2026-09-11

## Objective

Extend the D'Ascoli codebase to predict lexical evidence for each ten-word
Apple MEG fragment. The production output is a calibrated top-k word posterior
for each of ten ordered positions, exported for direct conditioning of ELF
alongside the frozen `qc4wyals` ADA semantic prediction.

This is a classifier handover, not an instruction to decode the protected test
story. Development and selection stop at the 110-row validation split.

The consumer-side design is documented in
`MEG_WORD_CONDITIONED_FLOW_MATCHING_DESIGN_2026-09-11.md`.

## Consumer experiment result and what it means for D'Ascoli

The simulated val110 experiment is complete; see
`MEG_DASCOLI_SENTENCE_FLOW_RESULTS_2026-09-12.md`. A target-derived sentence
with 49.82% correct ordered words has direct word/content F1
`0.49909/0.45317` and T5 retrieval Top-1/Top-5 `0.59091/0.77273`. This proves
that a successful ordered D'Ascoli decoder could add far more lexical
information than the current ADA-only generator.

It also makes the export contract stricter:

- the raw reconstructed sentence is a mandatory baseline and deliverable;
- calibrated top-k logits, entropy, and margin are needed separately from the
  top-1 sentence;
- do not bake confidence into an embedding magnitude or replace uncertain
  word states with noise;
- report ordered correct-position accuracy as well as bag-of-words F1; and
- produce story-grouped OOF training predictions, because ELF must learn from
  realistic correlated classifier errors rather than target-derived random
  replacements.

In the first ELF experiment, roughly 50% simulated position accuracy was the
point where full-strength joint flow began to improve clearly over
semantic-only generation. This is a planning threshold, not a required or
guaranteed real-model threshold: the simulator inherits true syntax and uses
optimistically separated confidence distributions.

A second, leakage-safe consumer experiment now validates the proposed
sentence-plus-confidence interface with a real brain-conditioned source. A
frozen T5 direct projector supplies a full sentence and target-free beam
probabilities; ELF-B hard-copies tokens above a global confidence threshold
and edits the rest while also receiving frozen `qc4wyals` predicted ADA. On
val110, threshold `0.20`, flow end `0.25`, and semantic scale `0.10` improve
the stronger T5 source from content/word F1 `0.02669/0.10398` to
`0.02834/0.10508`, while WER improves from `0.97909` to `0.97727`. See
`MEG_T5_ELF_GAP_CLOSURE_2026-09-12.md`.

This result is small and validation-selected, but it gives a concrete reason
to export calibrated position confidence. One low-confidence source token
(`bed`, confidence `0.124`) is changed to the target-relevant `seat`; a second
low-confidence trailing token is removed only when predicted-ADA semantic
transport is enabled. Do not copy the numerical threshold `0.20` into the
D'Ascoli model: its probability calibration will differ and must be selected
under the downstream validation protocol.

## Critical difference from the current D'Ascoli task

The current D'Ascoli pipeline in `/Users/gilad/Desktop/Projects/DAscolli`:

- loads `ParkerJones2025` Sherlock word events through `neuralset`;
- creates one MEG segment per known word onset;
- predicts a T5-large target embedding with `SimpleConvTimeAgg`;
- optionally applies a sentence transformer to grouped word predictions; and
- converts embedding similarities into a word distribution during retrieval.

The Apple decoder comparison uses a different contract:

- LibriBrain2/TheMoth, not the Sherlock validation/test runs;
- one `(306,2500)` MEG array per ten-second fragment;
- exactly ten whitespace-delimited target words per row;
- 2,652 train rows and 110 held-out validation rows; and
- a separate protected 26-row `birthofanation` pack.

Changing `SigLip` to cross-entropy in the existing entry point is therefore
not sufficient. Add a packed-Apple dataset path and an ordered ten-position
head. The existing Sherlock task can be used for backbone pretraining, but it
cannot substitute for Apple training or validation.

## Non-negotiable data contract

Canonical ARC root:

```text
ROOT=/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1
PREFIX=tang_themoth_exact384_ada002_textmid_seg10000

TRAIN=$ROOT/packed/${PREFIX}_train_dedup_fp16.npz
VAL=$ROOT/packed/${PREFIX}_val_fp16.npz
PROTECTED_TEST=$ROOT/packed/${PREFIX}_test_fp16.npz
```

Expected arrays used by this task:

```text
meg            float16/float32  [N,306,2500]
sentence       string           [N]
meg_lengths    integer          [N]          (optional; default 2500)
story          string           [N]          (preferred grouping key, if present)
session        string/integer   [N]          (story/group metadata)
run            string/integer   [N]
start_samples  integer          [N]
```

Ignore the ADA target array when training the word classifier. It belongs to
the independent semantic path.

Hard assertions before the first model forward:

- train has 2,652 rows and validation has 110 rows;
- every sentence produces exactly ten words under the normalization below;
- `meg.shape[1:] == (306,2500)`;
- every `meg_lengths` value is in `(0,2500]`;
- training and validation row IDs are unique and disjoint;
- grouped split metadata is present for OOF construction; and
- no opened path, config value, log line, cache key, callback, or automatic
  post-fit hook references `PROTECTED_TEST` or `birthofanation`.

The train file must be `train_dedup_fp16`, not the original 4,419-row pack,
which contained 1,767 exact duplicate MEG windows.

During development, any framework field named `test` must point to the
validation pack or be disabled. D'Ascoli's current callbacks automatically run
post-fit retrieval/decoding, so audit this explicitly.

## Stable row identity and normalization

Construct a stable row ID before batching. Do not include whether a prediction
is OOF or held-out in the ID; that is prediction metadata, not sample identity:

```text
sha256("tang_themoth_apples" + "\0" + session + "\0" + run + "\0" +
       start_samples + "\0" + exact_sentence)
```

Persist it in every prediction file. Row order alone is not an adequate join
key between repositories.

The segmentation contract is the ten whitespace-delimited stimulus tokens:

```python
surface_words = sentence.lower().split()
assert len(surface_words) == 10
```

Preserve these surface forms for targets and row alignment.  Two audited train
rows contain the source transcription `that''s`; the usual apostrophe regex
would split it into `that` and `s` and falsely produce eleven words.  Do not
silently normalize or split those rows.  A separate normalized form may be
stored for scoring, but it must not redefine classifier positions.

## Vocabulary

Build the vocabulary from training sentences only. Persist:

- `vocabulary`, ordered deterministically;
- `token_to_id` and `<unk>` ID;
- global frequency and per-position frequency;
- normalization code/version;
- SHA-256 of the ordered vocabulary; and
- train row-ID and sentence-list hashes.

Recommended first screen:

1. all training words with frequency at least two plus `<unk>`;
2. a frequency-at-least-five variant; and
3. the current top-250 setup only as a diagnostic baseline, not the production
   interface.

Map validation-only words to `<unk>` and report both overall accuracy and
accuracy conditional on train-vocabulary coverage. Do not add validation words
to the classifier after seeing them.

ELF benefits more from precise common/content words than from forced guesses at
rare OOVs; the semantic ADA stream remains available for global meaning.

## Recommended model

Predict all ten positions from the full ten-second segment. Preserve temporal
features instead of collapsing them before the ordered head:

```text
MEG [B,306,2500]
 -> existing D'Ascoli channel merger + subject layer + dilated ConvNet
 -> temporal features [B,H,T]
 -> masked adaptive pooling/downsampling to 64 steps
 -> 1-2 layer temporal Transformer/Conformer [B,64,H]
 -> ten learned position queries cross-attending to temporal features
 -> ordered hidden states [B,10,H]
 -> cosine/linear vocabulary classifier [B,10,V]

mean/max pool ordered hidden states
 -> auxiliary bag-of-words logits [B,V]
```

The ten-query cross-attention is only `10 x 64`, so it is inexpensive. It does
not require ground-truth word onset times and matches the downstream ten-slot
contract exactly.

Implementation guidance for the current code:

- reuse the internals of `neuraltrain.models.simpleconv.SimpleConv` through the
  convolutional `encoder`, but bypass `SimpleConvTimeAgg.time_agg_out`;
- set a compact backbone width (start with `H=320`) instead of producing 1,024
  channels at all 2,500 time steps;
- downsample before self-attention;
- retain current channel-position merger and subject handling; and
- allow partial initialization from the selected D'Ascoli/Sherlock backbone,
  logging every missing and unexpected key.

A cheap baseline may apply `Linear(H, 10*V)` to a globally pooled feature, but
it should not be the final design: every position would receive the same
collapsed temporal evidence.

### Classifier head

Start with a normalized cosine classifier:

```python
h = F.normalize(position_hidden, dim=-1)
w = F.normalize(classifier.weight, dim=-1)
logits = logit_scale.exp().clamp(max=100) * (h @ w.T) + bias
```

Initialize classifier rows from frozen T5 word embeddings or the existing
D'Ascoli target-embedding prototypes when dimensions permit, followed by a
trainable projection. This retains the useful geometry of the current
embedding-retrieval model while making probabilities and export IDs explicit.

Keep an auxiliary projected word embedding per position and retain the current
SigLIP/contrastive objective. The hybrid objective supports both closed-vocab
precision and open-vocabulary geometry:

```text
L = L_ordered_CE
  + 0.25 * L_bag_BCE
  + 0.25 * L_word_embedding
  + optional 0.05 * L_position_smoothness
```

Use these only as initial weights; log every component unscaled and scaled.

## Targets and losses

### Ordered cross-entropy

`target_ids` has shape `[B,10]`. Compute CE independently at each position and
mask `<unk>` from the primary in-vocabulary CE. Use label smoothing `0.05` in
the first screen.

Class imbalance must be handled carefully. Compare:

- plain CE plus prior correction at inference; and
- square-root inverse-frequency weighting capped at a modest value.

Do not make inverse-frequency weighting the sole production arm. Earlier ELF
lexical experiments showed that aggressive frequency manipulation can improve
nominal content scores while degrading fluent/common-word behavior.

### Bag-of-words auxiliary loss

Create a multi-hot vector from the ten target words, excluding `<unk>`. Use a
pooled head with `BCEWithLogitsLoss`. This teaches sentence-level presence
without discarding the ordered objective. The export remains ordered logits;
the bag head is diagnostic/auxiliary.

### Embedding auxiliary loss

Retain the current D'Ascoli embedding target or a fixed vocabulary prototype
for each word. Apply SigLIP/InfoNCE to position hidden states and target word
embeddings. Negatives must come only from training data. This gives sensible
geometry for words with few examples and permits an embedding-retrieval
baseline against the same validation rows.

## Train/validation and OOF predictions

ELF must be trained on classifier errors that resemble deployment errors. Do
not train the ELF word adapter using exact train words or in-sample D'Ascoli
predictions as its primary condition.

Produce five story-grouped OOF folds over the 2,652 training rows:

1. group by TheMoth story, never random row; use a verified one-story-per-session
   mapping only if the packed file has no explicit story field;
2. build vocabulary and normalization globally from training text only;
3. for each fold, fit on four groups of stories and predict the held-out group;
4. calibrate posterior temperature from the concatenated OOF predictions;
5. assemble exactly one OOF prediction per training row; and
6. fit final/ensemble models on all 2,652 train rows for the 110-row validation
   prediction, without validation fitting.

If vocabulary rows initialized from word embeddings are built globally, that
uses training labels only and is allowed. No validation text may enter learned
weights, priors, temperature calibration, or vocabulary construction.

Nearby windows from one story are correlated, so row-random OOF would be
optimistic even after exact-window deduplication.

Use at least three classifier seeds for the selected architecture. An ensemble
should average logits before prior correction and temperature scaling.

## Calibration and posterior selection

Estimate global and per-position priors on training rows. Fit a single
temperature on concatenated OOF logits first; per-position temperatures are an
ablation only if the calibration set is large enough.

Export top `K=8` candidates per position so ELF can sweep smaller `K` without
rerunning D'Ascoli. The exported posterior should be based on calibrated raw
logits. Also export priors so ELF can apply a declared prior subtraction:

```text
adjusted_logit[p,w] = calibrated_logit[p,w] - alpha * log_prior[p,w]
```

Do not choose `alpha` inside the D'Ascoli validation model. Export raw calibrated
evidence and let the downstream validation screen declare a small fixed sweep.

## Prediction NPZ contract

Write one train OOF pack and one validation pack. Required fields:

```text
schema_version          string scalar   "dascolli_apple_ordered_words_v1"
split                   string [N]      canonical source split: "train" or "val"
prediction_role         string scalar   "oof" or "heldout"
row_id                  string [N]
sentence                string [N]      alignment metadata only
story                   string [N]      explicit or reconstructed and verified
session                 string [N]
run                     string [N]
start_samples           int64  [N]

vocabulary              string [V]
vocabulary_sha256       string scalar
normalization           string scalar

topk_word_ids           int32  [N,10,8]
topk_logits             float32 [N,10,8] calibrated, before prior subtraction
topk_probabilities      float32 [N,10,8] sums to one over exported top-k
topk_probability_mass   float32 [N,10]   mass before top-k renormalization
position_valid_mask     bool   [N,10]
top1_probability        float32 [N,10]
posterior_entropy       float32 [N,10]   entropy of the full vocabulary posterior
top1_top2_margin        float32 [N,10]
predicted_word_ids      int32 [N,10]     calibrated top-1 IDs
predicted_sentence      string [N]       ten top-1 vocabulary items, in order
predicted_confidence    float32 [N,10]   calibrated probability of each top-1

global_log_prior        float32 [V]
position_log_prior      float32 [10,V]
fold_id                 int16 [N]       required for train OOF; -1 for val
model_seed_ids          int16 [M]
schema_json             JSON string scalar
```

`schema_json` must include:

```json
{
  "reference_text_used_for_prediction": false,
  "train_predictions_are_oof": true,
  "row_grouping": "TheMoth story",
  "word_positions": 10,
  "topk": 8,
  "logits_calibrated": true,
  "prior_subtraction_applied": false,
  "protected_test_loaded": false,
  "checkpoint_paths": [],
  "train_row_id_sha256": "...",
  "source_pack_sha256": "..."
}
```

For the validation pack set `train_predictions_are_oof` to false and record
that all producing checkpoints were trained only on the 2,652 training rows.

The probabilities over the exported top-k are a renormalized approximation;
`topk_probability_mass` records how much calibrated full-posterior probability
was retained. Keep `topk_logits` so ELF can change temperature/prior correction
correctly. If storage permits, an optional full-logit memmap may be exported
separately. Store additive-smoothed, finite log priors so unseen position/word
combinations do not create infinities.

Never put ground-truth word IDs into the production prediction NPZ. Store them
in a separate evaluation-only artifact so the ELF loader cannot accidentally
use them as condition evidence.

## Evaluation report before handoff to ELF

Report on all 110 validation rows:

- train-vocabulary coverage overall and by position;
- position Top-1/Top-5/Top-8 accuracy, micro and macro;
- mean reciprocal rank;
- top-1 reconstructed sequence word F1 and content-word F1;
- unordered sentence set precision/recall/F1 from top-1 and top-k unions;
- train-IDF-weighted content F1 and noncommon content F1;
- NLL, Brier score, ECE, and reliability bins;
- per-position metrics for positions 0 through 9; and
- three-seed mean, standard deviation, and ensemble result.

Required controls on the same rows:

1. global-frequency prior;
2. per-position-frequency prior;
3. current embedding-retrieval D'Ascoli baseline;
4. fixed Gaussian/noise input;
5. channel/time derangement if supported; and
6. at least 200 one-to-one row permutations of the selected predictions.

For the final selected classifier, run 100,000 row permutations for word F1,
content F1, and position accuracy. Lower-level token permutation is not the
primary null because it destroys the within-sentence output structure.

The classifier is ready for ELF only if matched MEG beats its declared
frequency baseline and row-permutation null on held-out lexical metrics, with
a positive result not confined entirely to the most common content words.
Training loss is not an acceptance criterion.

## File-by-file implementation plan in D'Ascoli

Suggested changes, preserving the existing Sherlock entry point:

1. `sentence_decoding/apple_word_data.py` (new)
   - load packed train/validation NPZs;
   - build/restore vocabulary;
   - compute row IDs and grouped fold IDs;
   - return MEG, length, ten target IDs, and metadata.
2. `sentence_decoding/ordered_word_model.py` (new)
   - reuse the SimpleConv temporal backbone;
   - downsample temporal states;
   - add ten query tokens, temporal attention, ordered classifier, bag head,
     and optional embedding projection.
3. `sentence_decoding/word_pl_module.py` (new)
   - calculate and log CE/BCE/embedding losses;
   - collect logits without reference-dependent postprocessing;
   - keep noise/derangement controls.
4. `sentence_decoding/word_metrics.py` (new)
   - vocabulary coverage, ordered top-k, set/content/IDF metrics, calibration,
     and row-permutation utilities.
5. `sentence_decoding/export_word_predictions.py` (new)
   - strict schema writer and round-trip validator;
   - separate OOF and validation outputs;
   - no automatic protected-test export.
6. `sentence_decoding/grids/apple_words.py` (new)
   - small declared model/loss/vocabulary screen and grouped OOF launcher.
7. tests
   - ten-word parsing and row-ID stability;
   - no train/validation overlap;
   - no protected-test access;
   - output/mask shapes;
   - fold coverage exactly once;
   - probability normalization and calibration round trip;
   - export alignment and vocabulary-hash mismatch failure.

Do not overload the existing `BrainModule._run_step`: it normalizes a single
embedding prediction and assumes `batch.data["feature"]` is a dense target.
An ordered `[B,10,V]` classifier has different loss, masking, metrics, and
export requirements. A separate Lightning module will be easier to audit and
will avoid breaking the established sentence decoder.

## Proposed CLI contract

After implementing the new entry point, the training interface should look
approximately like this (the command is a target interface, not currently an
existing executable):

```bash
python -m sentence_decoding.grids.apple_words \
  --train-npz "$TRAIN" \
  --val-npz "$VAL" \
  --disable-test \
  --vocab-min-frequency 2 \
  --positions 10 \
  --export-topk 8 \
  --oof-folds 5 \
  --group-key story \
  --seeds 49 63 82 \
  --output-dir /data/engs-pnpl/glandau/dascolli-apple-words/v1
```

Expected final files:

```text
v1/vocabulary.json
v1/oof_train_word_predictions.npz
v1/val110_word_predictions.npz
v1/val110_word_classifier_metrics.json
v1/val110_word_classifier_permutation.json
v1/checkpoints/...                 # classifier checkpoints only
```

## Handoff checklist

- [ ] Correct Apple packed data, row counts, and ten-word assertions.
- [ ] Protected test path absent and automatic post-fit test disabled.
- [ ] Train-only vocabulary, priors, IDF, and calibration.
- [ ] Story-grouped OOF prediction for every one of 2,652 train rows.
- [ ] Independent 110-row validation prediction with identical vocabulary hash.
- [ ] Ordered top-eight logits/probabilities and masks satisfy the NPZ schema.
- [ ] No reference-derived word targets in the production prediction packs.
- [ ] Frequency, noise, embedding-retrieval, and row-permutation controls run.
- [ ] Three-seed and ensemble metrics reported on all 110 rows.
- [ ] ELF-side loader round-trip succeeds before any fusion model is trained.
