# Leakage-Safe fMRI MiniLM-to-Text Improvement Plan

## Scientific contract

- The 107 MRI2SEM test predictions are final-evaluation data only.
- They must not be used for fitting, checkpoint selection, hyperparameter selection, calibration, candidate construction from brain vectors, or qualitative model selection.
- The corresponding 107 sentences may be learned from perfect MiniLM embeddings in the explicitly labelled known-text/oracle stage.
- Architecture and hyperparameters are selected using story-level OOF training predictions and the 266 predicted validation rows.
- After validation selection is locked, evaluate exactly one selected open-ended model on the 107 test predictions.

## Current best open-ended baseline

Evaluation run:

```text
eval_fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829_unseenbrain_direct1536_adapteronly_test107_20260829
```

W&B run: `ie1khz0f` in `giladland-university-of-oxford/BrainDiffusion`.

Training checkpoint:

```text
fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829
```

The held-out test metrics are:

- Raw WER: `0.9897196262` (`1059 / 1070` edit errors)
- Fixed ten-word-cap WER: `0.9813084112` (`1050 / 1070` edit errors)
- Exact match: `0 / 107`
- Word overlap F1 / precision / recall / Jaccard: `0.04629 / 0.06324 / 0.03832 / 0.02734`
- Content overlap F1 / precision / recall / Jaccard: `0.00634 / 0.01090 / 0.00460 / 0.00374`
- Generated-text retrieval Top-1 / Top-5: `0.00935 / 0.06542`
- Generated-text retrieval mean / median rank: `50.34 / 49`
- Mean generated length: `6.0` words for a fixed ten-word target
- Insertions / deletions / substitutions: `9 / 437 / 613`
- Unique-token ratio / repeated-token fraction: `0.73047 / 0.26953`
- Mean maximum token run: `1.57944`
- Repeated-bigram / repeated-trigram fraction: `0.13744 / 0.10338`
- Mean word length / long-word fraction: `3.40238 / 0.08411`
- Vowel-missing / alphabetic-token fraction: `0.08910 / 0.87448`
- Stopword fraction: `0.45684`
- Length score / well-structured-sentence score: `0.15000 / 0.64260`
- Degenerate-generation fraction: `0.12150`
- RoBERTa-large BERTScore precision / recall / F1, baseline-rescaled: `-0.52904 / -0.50818 / -0.51746`

The negative rescaled BERTScore confirms that semantic similarity is below the ordinary English-pair baseline. The example `thirty years old ...` -> `eight eight and eighty seven and seven years` has BERTScore F1 `-0.22294`: better than the skewed run average because it retains the age/years topic, but only rank `76 / 107`; the age is wrong and the event content is absent.

The five highest BERTScore-F1 test examples are listed separately from lexical overlap:

1. `team was healthy and then the radio came alive avalanche` -> `i decided to  the`: `0.09800`
2. `one moment when i knew peace at my school when` -> `i know i am very very confident and`: `0.04879`
3. `got out as i walked forward another friend was telling` -> `i got aa`: `0.03456`
4. `unison we stepped forward i push down i hit something` -> `i first i my his hair`: `0.02146`
5. `year at my new school it finally began to resemble` -> `wife i loved i loved respected my life`: `0.01463`

BERTScore can reward a loose topical association even when the decoded event is wrong, so it is always reported beside WER, content overlap, and retrieval rather than treated as sufficient evidence by itself.

The checkpoint's best 266-row predicted-validation result was selected at step `4000`:

- WER: `0.9763157895`
- Word overlap: `0.05774`
- Content overlap: `0.00672`
- Generated retrieval Top-1 / Top-5: `0.03008 / 0.05263`
- Generated retrieval mean / median rank: `115.98 / 102`
- Word overlap precision / recall / Jaccard: `0.08065 / 0.04887 / 0.03561`
- Content overlap precision / recall / Jaccard: `0.00872 / 0.00601 / 0.00407`
- Insertions / deletions / substitutions: `13 / 1075 / 1509`

Only three of the 107 test generations shared any content word with their target. The five highest word-overlap examples are:

1. Target: `got out as i walked forward another friend was telling`
   Generated: `i got aa`
   WER `0.9`, word overlap `0.3077`, content overlap `0.25`.
2. Target: `unison we stepped forward i push down i hit something`
   Generated: `i first i my his hair`
   WER `0.9`, word overlap `0.25`, content overlap `0.0`.
3. Target: `street and across the street at a coffee shop called`
   Generated: `a seas in daham's mouth and know`
   WER `1.0`, word overlap `0.2353`, content overlap `0.0`.
4. Target: `start relating to each other differently across the racial divide`
   Generated: `the and and`
   WER `0.9`, word overlap `0.1538`, content overlap `0.0`.
5. Target: `history will know that in the late fifties and early`
   Generated: `if we know a's`
   WER `0.9`, word overlap `0.1429`, content overlap `0.2857`.

## Current architecture and failure mode

```text
40,000-D lagged/pooled fMRI
  -> frozen MindEye-style residual MRI2SEM MLP
  -> 1,536-D concatenated delayed MiniLM prediction
  -> trainable Linear(1,536, 384), mean-block initialized
  -> trainable LayerNorm + MLP(384, 4096, 64*512)
  -> 64 x 512 ELF conditioning context
  -> frozen known-text ELF-B
  -> ten-word text
```

MRI2SEM is trained separately on 11,725 brain segments and predicts four ordered 384-D delay blocks. During bridge training its story-level OOF predictions are precomputed, so diffusion gradients do not reach MRI2SEM.

The bridge training set contains 11,725 oracle rows plus 11,725 story-level OOF prediction rows. The 266-row validation tail contains only the five-fold prediction ensemble. The test set is excluded.

Despite the name `adapter`, the trainable context projector has about 136.4M parameters, zero dropout, and only about 11.7K distinct training sentences. Its final `4096 -> 32768` layer dominates the parameter count. Updating this full projector can destroy the useful known-text MiniLM-to-ELF mapping while trying to absorb MRI2SEM noise. The initial `1536 -> 384` projection also collapses delay structure before the context mapping.

## Phase 1: preserve the known-text decoder and learn only delayed-input fusion

Implemented on 2026-08-29:

- Added `DelayFusionContextProjector`.
- It reshapes `1536` into ordered `4 x 384` blocks.
- It learns one feature-wise softmax weight per delay: only `4 x 384 = 1,536` trainable parameters.
- It initializes to the exact arithmetic mean of all four blocks.
- It reuses and freezes the already trained 384-D MiniLM-to-ELF context MLP.
- ELF remains frozen.
- Added a projection-only control that trains only the existing `1536 -> 384` linear layer (`589,824` parameters) while freezing the context MLP and ELF.

Validation sweep job: `8680024` (`0-3`), with no test evaluation:

| Array task | Input module | Learning rate | Trainable downstream parameters |
|---|---|---:|---:|
| 0 | feature-wise delay fusion | `1e-3` | `1,536` |
| 1 | feature-wise delay fusion | `5e-3` | `1,536` |
| 2 | full linear projection | `1e-4` | `589,824` |
| 3 | full linear projection | `5e-4` | `589,824` |

Each model trains for 20 epochs on the same oracle+OOF mixture and selects checkpoints only on the 266 predicted validation rows. The first gate is validation WER below `0.9763`; secondary criteria are content overlap, generated retrieval, length, and stability.

Early Phase-1 results did not clear the WER gate. Feature-wise fusion raised content overlap to roughly `0.0174-0.0178` from `0.0067`, but its best WER remained `0.9962-0.9977`. The low-rate full linear projection reached WER `0.9868`. These are useful content-sensitivity signals, but they are not candidates for test evaluation.

## Phase 2: explicitly align noisy predictions to the known-text semantic interface

Implemented and submitted on 2026-08-29 as validation-only array job `8680152` after the first Phase-1 results improved content overlap but worsened WER:

1. Attach the exact 384-D MiniLM embedding of each ten-word target to every oracle/OOF row.
2. Train the small fusion with a weighted objective:

   ```text
   L = L_diffusion
       + lambda_cos * (1 - cosine(fused_pred, exact_minilm))
       + lambda_contrastive * symmetric_in_batch_contrastive_loss
       + lambda_consistency * distance(context_pred, context_oracle)
   ```

3. Use story-balanced batches so overlapping stride-2 segments do not dominate.
4. Compare oracle-only, OOF-only, and mixed batches on the same predicted validation set.
5. Tune loss weights and learning rate on validation only.

The implementation reports fused-vector matched cosine, Top-1/Top-5 retrieval, and rank on the 266 predicted validation rows in addition to text metrics. Four settings compare cosine-only alignment, cosine+contrastive+context consistency, a stronger alignment setting, and a full linear-projection control. ELF and the known-text context MLP remain frozen; only the delayed-input fusion/projection is trainable.

A validation-only L2-normalized mean-fusion diagnostic (job `8680186`) produced WER `1.0068`, word/content F1 `0.05195 / 0.01460`, generated retrieval Top-1/Top-5 `0.0150 / 0.0789`, and exact-MiniLM interface cosine `0.0838`. Normalization helps retrieval slightly but does not solve the semantic mismatch.

The initial alignment array `8680152` also failed the WER gate. Feature-wise fusion remained near interface cosine `0.0838`; the best full-linear setting reached only WER `0.9955` and interface cosine `0.0756`. This ruled out simply increasing the auxiliary alignment weight on a linear/factorized fusion.

## Phase 2b: compact nonlinear residual mapper

Implemented and submitted as validation-only job array `8680277`:

```text
4 x 384 delayed MiniLM blocks
  -> exact block mean
  + zero-output initialized MLP(1536, hidden, 384) residual
  -> optional L2 normalization
  -> frozen known-text MiniLM-to-ELF context MLP
  -> frozen ELF
```

The sweep compares 512- and 1,024-wide residual mappers, normalization on/off, and learning rates `1e-4`/`5e-4`. The trainable front end has roughly one to two million parameters, starts at the existing mean-fusion solution, and uses exact-MiniLM cosine, symmetric contrastive, context-consistency, and diffusion objectives. Selection remains restricted to predicted validation rows; test107 is absent.

This phase directly optimizes the distribution gap instead of asking text-generation loss alone to discover it through a frozen diffusion model.

Completed validation results from array `8680277`:

| Mapper | Norm | LR | WER | Word F1 | Content F1 | Interface cosine |
|---|---:|---:|---:|---:|---:|---:|
| residual MLP, hidden 512 | no | `1e-4` | `0.97820` | `0.06973` | `0.01121` | `0.16940` |
| residual MLP, hidden 512 | yes | `1e-4` | `0.97857` | `0.06431` | `0.01285` | `0.16091` |
| residual MLP, hidden 1024 | yes | `1e-4` | `0.97820` | `0.06230` | `0.00570` | `0.17349` |
| residual MLP, hidden 512 | yes | `5e-4` | **`0.97594`** | `0.06120` | `0.00654` | **`0.17757`** |

The last row is the first nominal WER improvement over the `0.97632` baseline, but it is only one fewer edit (`2596` versus `2597`) and its retrieval/content metrics are not stronger. It is therefore a lead, not a test-ready winner. Job `8681242` is decoding the WER and content leads under three additional fixed diffusion seeds to test robustness.

## Phase 2c: distribution-matched OOF-only training

Implemented after inspecting the packed-condition labels:

- Rows `0:11725`: oracle delayed MiniLM.
- Rows `11725:23450`: story-level OOF MRI2SEM predictions.
- Rows `23450:23716`: 266 held-out validation prediction ensembles.

The trainer can now filter training rows by `condition_source` while retaining their original indices for the row-aligned T5 latent and exact-MiniLM caches. A split-safety unit test verifies that no row at or beyond the validation boundary can enter training.

Validation-only array job `8681291` trains compact residual mappers using only the 11,725 `oof_prediction` rows. It compares normalization, hidden width, learning rate, stronger cosine alignment, and stronger contrastive alignment. Oracle inputs and validation vectors are excluded; test107 remains absent.

The replacement clean array is job `8681705` (the earlier scheduler-mutated request was cancelled before training). Task 0 completed successfully:

- Run: `fmri_minilm_knowntext_oofonly_h512_lr1e4_inputonly_ep20_b32_seed49_20260829`
- W&B: `ro31zkye`
- Best checkpoint: step `1000`, epoch `2.729`
- Validation WER: **`0.973684`** (`2590 / 2660` errors), versus baseline `0.976316` (`2597 / 2660`)
- Word/content F1: **`0.06620 / 0.00949`**, versus baseline `0.05774 / 0.00672`
- Generated retrieval Top-1/Top-5: `0.00752 / 0.03008`, worse than baseline `0.03008 / 0.05263`
- Exact-interface cosine/Top-1/Top-5: `0.16905 / 0.01880 / 0.07895`
- Post-hoc validation BERTScore P/R/F1: `-0.09330 / -0.09506 / -0.09327`, versus the matched baseline validation F1 `-0.34629`

This is a real seven-error WER improvement with simultaneous lexical/content gains, but the generated-text retrieval regression means it is a strong lead rather than a locked winner. The remaining OOF-only variants, 32-step seed robustness, and matched baseline validation BERTScore are still required.

The matched baseline validation BERTScore is now complete: F1 `-0.34629`. Thus task 0 improves BERTScore by `+0.25302` absolute on the identical 266 references.

Task 0's five highest validation BERTScore examples are:

1. `dangerous edgy pictures of women and he wants to shoot` -> `the wants to get to hear the the`: `0.18491`
2. `finished writing my first book i returned to big sur` -> `high grade i began to want to the theme dog`: `0.17575`
3. `becoming an old lady and i have this huge crush` -> `want that i was really keen about the substance`: `0.17389`
4. `i got a phone call about our boy's baby sister` -> `the purpose was to feel the about the thing`: `0.16659`
5. `i wanted i asked to go on vacation to see` -> `that become i really wanted to want want`: `0.15705`

These show why BERTScore is complementary rather than decisive: several preserve a relation or topic while remaining factually and lexically poor.

Completed OOF-only candidate comparison so far:

| Candidate | WER | Word F1 | Content F1 | Gen Top-1/5 | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline full adapter | `0.97632` | `0.05774` | `0.00672` | `0.0301 / 0.0526` | `-0.34629` |
| task 0: h512, no norm, LR `1e-4` | `0.97368` | **`0.06620`** | **`0.00949`** | `0.0075 / 0.0301` | **`-0.09327`** |
| task 1: h512, norm, LR `1e-4` | `0.97519` | `0.06259` | `0.00563` | `0.0038 / 0.0376` | pending |
| task 2: h512, norm, LR `5e-4` | **`0.97331`** | `0.06512` | `0.00528` | `0.0113 / 0.0414` | `-0.09883` |
| task 3: h512, strong alignment | `1.02519` | `0.05831` | `0.01151` | `0.0000 / 0.0226` | pending |

Tasks 4/5 were scheduler-cancelled after early checkpoints, so they are not final candidates. Their partial step-1000 diagnostics did not beat the balanced task-0 profile.

Diffusion seed is a material confound. Across seeds 7/23/101, the original baseline WERs are `0.97594 / 0.97444 / 0.97669` (mean `0.97569`), while the earlier mixed-residual mapper gives `0.97331 / 0.97444 / 0.97030` (mean `0.97268`). The residual improvement averages `0.00301` WER, about eight errors, but varies by seed. A clean multi-seed run for the new OOF/story-balanced leader will be required before test evaluation. Test107 remains absent.

The OOF stories contribute 73 to 255 segments (median 167), so a uniform-story sampler was also implemented and verified by the ARC test suite. It samples a story uniformly and then a selected OOF row within that story; validation-tail rows cannot enter its pools. Clean replacement job `8682202` runs the top three story-balanced settings.

The first story-balanced checkpoint (task 0, step 1000) is the strongest result so far:

- Run: `fmri_minilm_knowntext_oofonly_h512_lr1e4_balanced_story_inputonly_ep20_b32_seed49_20260829`
- W&B: `r66sbj0n`
- WER `0.97030` (`2581 / 2660`, 16 fewer errors than seed-49 baseline)
- Word/content F1 `0.07227 / 0.00699`
- Generated retrieval Top-1/Top-5 `0.01880 / 0.04887`
- Exact-interface cosine `0.16576`
- Post-hoc BERTScore F1 `-0.10133` (unbalanced task 0: `-0.09327`; baseline: `-0.34629`)

It improves WER and lexical F1 while recovering most of the retrieval lost by unbalanced OOF-only training. The run is complete; it must still pass multi-seed decoding before it can be locked.

Its matched three-seed robustness result is strong:

| Metric | Baseline mean (seeds 7/23/101) | Balanced task-0 mean | Delta |
|---|---:|---:|---:|
| WER | `0.97569` | **`0.96642`** | **`-0.00927`** |
| Word F1 | `0.06496` | **`0.07160`** | **`+0.00664`** |
| Content F1 | `0.00836` | `0.00822` | `-0.00014` |
| Generated Top-1 | `0.00877` | **`0.01378`** | **`+0.00501`** |
| Generated Top-5 | `0.05013` | `0.04637` | `-0.00376` |

Per-seed balanced WER is `0.96842 / 0.96353 / 0.96729`, improving over the matched baseline by `20 / 29 / 25` word errors. This establishes that the gain is not a lucky diffusion seed.

Balanced task-0 BERTScore F1 is also stable across those seeds: `-0.10031 / -0.12763 / -0.10300` (mean `-0.11031`, SD `0.01505`). The matched baseline is `-0.14526 / -0.21253 / -0.28716` (mean `-0.21498`, SD `0.07098`), so the balanced mapper improves mean BERTScore by `+0.10467` and is substantially less seed-variable.

Story-balanced task 1 (normalized, LR `1e-4`, W&B `gg0dthtn`) has an even lower initial WER `0.96917` at seed 49 with word/content F1 `0.06847 / 0.01051` and BERTScore F1 `-0.10121`. ARC early-stopped the job after epoch 16.4, but its selected checkpoint is the valid epoch-2.73 checkpoint and five subsequent validation checks did not beat it. Its seed-7/23/101 WERs are `0.97256 / 0.97068 / 0.97180` (mean `0.97168`), with mean word F1 about `0.06675`. This is less robust than unnormalized task 0, whose matched mean WER is `0.96642` and word F1 is `0.07160`.

## Locked model selection and final unseen-brain test

Model selection was completed before opening test107. The locked winner is story-balanced task 0:

```text
fmri_minilm_knowntext_oofonly_h512_lr1e4_balanced_story_inputonly_ep20_b32_seed49_20260829
```

W&B run: `r66sbj0n`. Selected checkpoint: validation step `1000`.

The complete locked pipeline is:

```text
40,000-D fMRI segment
  -> separately trained, frozen MRI2SEM
  -> four ordered 384-D delayed MiniLM predictions
  -> arithmetic delay mean + zero-initialized residual MLP(1536, 512, 384)
  -> frozen known-text MiniLM-to-ELF context projector
  -> frozen ELF-B diffusion decoder
  -> ten-word text, 32 diffusion steps, seed 49
```

Only the residual mapper was trained in this improvement stage. It saw 11,725 story-level OOF MRI2SEM predictions with uniform story sampling and exact-MiniLM semantic auxiliary targets. It did not see oracle input rows, validation prediction rows, or any of the 107 test brain vectors. The perfect-MiniLM known-text decoder is allowed to know the sentence corpus, including the test sentences, but neither it nor the residual mapper was fitted to the test107 MRI2SEM outputs.

The one locked evaluation is:

```text
eval_fmri_minilm_knowntext_oofonly_h512_lr1e4_balanced_story_inputonly_ep20_b32_seed49_20260829_locked_steps32_seed49_unseenbrain_test107_20260830
```

ARC job `8682812` completed successfully. Test107 results:

- Raw WER: **`0.9757009346`** (`1044 / 1070` edit errors), versus `0.9897196262` (`1059 / 1070`) for the legitimate baseline.
- Fixed ten-word-cap WER: **`0.9728971963`** (`1041 / 1070`), versus `0.9813084112` (`1050 / 1070`).
- Exact match: `0 / 107`.
- Insertions / deletions / substitutions: `3 / 190 / 851`.
- Word-overlap F1 / precision / recall / Jaccard: **`0.05977 / 0.06782 / 0.05421 / 0.03779`**, versus baseline F1 `0.04629`.
- Content-overlap F1 / precision / recall / Jaccard: **`0.01018 / 0.01355 / 0.00834 / 0.00570`**, versus baseline F1 `0.00634`.
- Generated-text retrieval Top-1 / Top-5: `0.00000 / 0.05607`, versus baseline `0.00935 / 0.06542`.
- Generated-text retrieval mean / median rank: `51.02 / 48`, versus baseline `50.34 / 49`.
- Mean original / capped generated length: `8.252 / 8.224` words.
- Unique-token ratio / repeated-token fraction: `0.68195 / 0.31805`.
- Mean maximum token run: `2.28037`.
- Repeated-bigram / repeated-trigram fraction: `0.11805 / 0.05480`.
- Mean word length / long-word fraction: `3.71856 / 0.00000`.
- Vowel-missing / alphabetic-token fraction: `0.00000 / 0.99327`.
- Stopword fraction: `0.58786`.
- Length score / well-structured-sentence score: `0.20631 / 0.68759`.
- Degenerate-generation fraction: `0.14019`.
- RoBERTa-large BERTScore precision / recall / F1, baseline-rescaled: **`-0.12376 / -0.12030 / -0.12138`**, versus baseline F1 `-0.51746`.

This is a leakage-safe and multi-seed-supported improvement in WER, lexical overlap, and content overlap. It is modest: retrieval did not improve on the final test, no sentence is an exact reconstruction, and repetition/degeneration remain substantial. The test result is reporting-only and is not being used to retune this checkpoint or its sampling settings.

The five highest test BERTScore-F1 generations are:

1. `twenty nine years old and the big year of change` -> `i started think about the goal`: `0.19559`
2. `get back on the ground the exhilaration overwhelms the terror` -> `i start to get up to the train`: `0.14229`
3. `year at my new school it finally began to resemble` -> `high grade i found the feel interest of purpose`: `0.14039`
4. `nothing she got fed whenever she wanted she was ivy` -> `the started to know the the the was wanted`: `0.11168`
5. `moving to town he wanted to live in my valley` -> `young wants the people to attention to the subject`: `0.10329`

These are the best semantic matches, not necessarily the best lexical matches. They preserve fragments such as age/change, getting up, school/grade, or wanting, but they still miss much of the target event. The large BERTScore improvement therefore supports better semantic conditioning while the WER and retrieval results show that exact open-ended reconstruction is still poor.

The five highest word-overlap-F1 generations are:

1. `around we say dude is the thing home you know` -> `your sense is to caught up sticky the food thing`: word/content/BERT F1 `0.3000 / 0.1818 / -0.0033`
2. `get back on the ground the exhilaration overwhelms the terror` -> `i start to get up to the train`: `0.2222 / 0.2222 / 0.1423`
3. `journalist coming to the school and miss roberts rushing across` -> `will to know you hear the sound the`: `0.2222 / 0.0000 / -0.0062`
4. `start relating to each other differently across the racial divide` -> `the fire part to accent the the stickyy`: `0.2222 / 0.0000 / -0.0216`
5. `thing you ever seen like those spike lee movies where` -> `camp you like to keep the central mood thebased`: `0.2105 / 0.1538 / -0.1201`

The complete per-example lexical, content, WER, and BERTScore ranking is saved as `ranked_examples.json` in the locked evaluation directory.

## Phase 3: retain four delay tokens through conditioning

If a fused 384-D bottleneck remains limiting, use the existing ordered delay-token adapter:

```text
4 x 384 delayed blocks
  -> shared 384-to-512 projection + learned delay positions
  -> one small transformer encoder over four tokens
  -> compact learned-query resampler
  -> ELF context
```

Keep this adapter under roughly 5-10M trainable parameters, use dropout `0.1`, and distill its output toward the known-text context produced by perfect MiniLM. Compare it with the 1,536-parameter fusion using validation WER and content overlap.

## Phase 4: cautious ELF adaptation

Only after the predicted validation interface improves:

- Unfreeze the final context projection or use LoRA/adapters in the last 2-4 ELF blocks.
- Keep the base decoder and most diffusion blocks frozen.
- Use a smaller LR (`1e-6` to `5e-6`) than the semantic front end.
- Mix oracle and OOF rows and retain context-consistency regularization.
- Require improvement across at least three fixed validation seeds before considering test evaluation.

The previous full-ELF continuation worsened held-out test WER from `0.9897` to `1.0187`, so full unfreezing is not the next step.

## Phase 5: limited brain-side end-to-end training

The current NPZ bridge cannot backpropagate into MRI2SEM. A new raw-fMRI trainer is required for true end-to-end optimization:

1. Load raw 40,000-D fMRI segments and the pretrained MRI2SEM checkpoint.
2. Initially freeze the 40,000-to-2,048 encoder and train only the final `2048 -> 1536` projector plus the small delay fusion.
3. Retain the original MiniLM contrastive/cosine objective alongside semantic-interface and diffusion losses.
4. If stable, unfreeze only the last residual MLP block with a much smaller LR.
5. Use story-level OOF folds or nested validation; never expose the 107 test brain segments.

## Phase 6: locked final evaluation (completed 2026-08-30)

The following pre-test checklist was completed before test107 was opened:

- Choose one architecture and checkpoint from the 266-row predicted validation set.
- Lock sampling steps, CFG, seed policy, word cap, and all post-processing.
- Record the selection rule and hashes/paths of the chosen artifacts.
- Run one final evaluation on all 107 MRI2SEM predictions.
- Report raw and capped WER, edit counts, word/content overlap, BERTScore, retrieval, length/degeneration metrics, every generated sentence, and the oracle-to-predicted degradation.
- Rank qualitative examples separately by lexical F1, content F1, and BERTScore so semantically related paraphrases are not hidden by exact-token scoring. Test BERTScore is reporting-only and cannot be used for model or decoding selection.

The primary scientific goal is not merely a lower WER from shorter output. A successful model must improve content overlap and generated-text retrieval while keeping the brain test split fully untouched during development.

## 2026-08-31 ELF diffusion update

The first explicit-brain diffusion system now exceeds the frozen T5-prefix
val266 content-F1 point estimate: `0.047099` versus `0.043743`. It uses one
deterministic generation, a frozen MRI2SEM, decoder-only cross-attention in the
last two ELF blocks, and a train-only 1,536-D lexical MLP with a Top-10
one-shot scattered decode bias. Three fixed brain derangements score
`0.035891`, `0.042271`, and `0.031996`, confirming that the matched-brain
result is not merely an unconditional language-prior effect.

This is a content-first proof of concept rather than the overall language
winner: word F1 is `0.068849`, WER is `1.142105`, and BERTScore F1 is
`-0.098912`, all worse than T5-prefix. The paired bootstrap content interval
also crosses zero. Full details, paths, rejected dual-head audit, and package
provenance are recorded in `FMRI_LONG_EXPERIMENT_PROGRAM_2026-08-30.md`.
