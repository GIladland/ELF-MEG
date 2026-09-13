# MEG semantic + simulated D'Ascoli sentence-source ELF-B results

Date: 2026-09-12

## Verdict

The sentence-start idea works as an information channel, but the first fusion
mechanism is not the production design.

- A target-derived D'Ascoli sentence with 49.82% correct word positions is very
  informative by itself: direct output has word/content F1 `0.49909/0.45317`
  and generated-text Top-1/Top-5 `0.59091/0.77273`.
- The best trained ELF-B refinement starts from the complete source sentence,
  keeps full trust in every source token, and applies only `0.035` of the
  current semantic-context amplitude. It reaches word/content F1
  `0.36658/0.28839`, WER `0.66727`, and Top-1/Top-5
  `0.47273/0.69091`.
- Full-strength MEG-predicted ADA conditioning is destructive in this setup.
  At the same source accuracy it gives sum `0.19854` with unweighted source
  tokens and only `0.12450` with the proposed confidence-to-Gaussian blend.
- The best flow remains significantly worse than emitting the simulated
  D'Ascoli sentence directly. It retains `67.52%` of genuinely correct source
  positions but corrects only `0.18%` of wrong source positions.

The right next system is therefore a source-preserving, uncertainty-aware
editor: keep the full D'Ascoli sentence on the T5 manifold, expose confidence
and top-k posteriors as separate features, and add a zero-initialized semantic
correction gate inside ELF blocks. Do not use confidence to replace uncertain
token states with independent Gaussian noise.

## Contract

- Data: frozen nested-OOF `qc4wyals` 1,536-D predicted ADA for 2,652 training
  rows and the established final 110 Apple validation rows.
- D'Ascoli evidence: target-derived full ten-word simulations, not measured
  MEG word-classifier predictions.
- Correct/incorrect confidence: `Beta(8,2)` / `Beta(2,8)`.
- Wrong words: sampled from position-specific train-only vocabularies.
- ELF initialization: audited x1 ELF-B checkpoint and its frozen semantic
  adapter.
- Training: mixed assumed source accuracies `0.25/0.50/0.75/0.90`, independent
  10% semantic and source dropout, fixed seed 49.
- Evaluation: identical 110 rows, diffusion seed, and 110 T5 retrieval
  candidates. Extended null uses 100,000 one-to-one row permutations.
- Protected test26 was never opened.

For source T5 state `h`, confidence `c`, Gaussian prior `epsilon`, and target
state `x`, the tested transport was:

```text
s   = c*h + (1-c)*(2.5*epsilon)
z_t = t*x + (1-t)*s
v*  = (x-z_t)/(1-t)
```

The semantic condition remained in the existing 64-token prefix. The later
weight grid applied a scalar to that prefix and a confidence floor
`c' = floor + (1-floor)*c` while retaining zero-confidence padding.

## Training screen

Selection used validation word F1 + content F1 at 49.82% simulated word
accuracy. Retrieval was not part of checkpoint selection.

| Arm | Best step | Content F1 | Word F1 | Sum | WER |
|---|---:|---:|---:|---:|---:|
| LoRA-4, final 4 blocks, LR 2e-6 | 840 | 0.02959 | 0.08466 | 0.11424 | 0.98455 |
| **LoRA-8, final 4 blocks, LR 1e-5** | **1680** | **0.02809** | **0.09050** | **0.11859** | **0.97455** |
| Full final 2 blocks, LR 5e-7 | 840 | 0.02814 | 0.08263 | 0.11077 | 0.98000 |

LoRA-8 improved at every checkpoint in the sum (`0.11281`, `0.11512`,
`0.11631`, `0.11859`), whereas the larger full-block arm peaked at step 840.

## Primary 110-row controls at 49.82% source accuracy

| System | Content F1 | Word F1 | Sum | WER ↓ | Raw BERTScore | BLEU-1 | ROUGE-1 | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trained semantic-only | 0.01763 | 0.05985 | 0.07748 | 1.00545 | 0.81530 | 0.05581 | 0.06268 | 0.06364 | 0.14545 |
| Full semantic + confidence-weighted source | 0.02882 | 0.09568 | 0.12450 | 0.97636 | 0.82172 | 0.08978 | 0.10401 | 0.11818 | 0.30909 |
| Full semantic + unweighted source | 0.05895 | 0.13960 | 0.19854 | 0.93636 | 0.82433 | 0.13050 | 0.14904 | 0.15455 | 0.34545 |
| Sentence-only + confidence-weighted source | 0.22234 | 0.29909 | 0.52143 | 0.87364 | 0.83879 | 0.27627 | 0.31525 | 0.43636 | 0.61818 |
| Sentence-only + full source trust | 0.28461 | 0.36223 | 0.64683 | 0.69818 | 0.84464 | 0.35149 | 0.37176 | 0.42727 | 0.66364 |
| **Semantic scale 0.035 + full source trust** | **0.28839** | **0.36658** | **0.65497** | **0.66727** | **0.84693** | **0.35862** | **0.37377** | **0.47273** | **0.69091** |
| Direct simulated D'Ascoli sentence | 0.45317 | 0.49909 | 0.95226 | 0.50182 | 0.86351 | 0.49909 | 0.49701 | 0.59091 | 0.77273 |

The trained row-shuffled joint control has sum `0.08766` and Top-1/Top-5
`0.04545/0.16364`, near semantic-only. The sentence benefit is therefore
matched and row-specific, not a generic change in language-model behavior.

### Does transport training help?

At full semantic strength and 49.82% source accuracy:

| Condition | Untrained x1 sum | Trained LoRA-8 sum |
|---|---:|---:|
| Joint confidence-weighted | 0.10397 | 0.12450 |
| Joint unweighted | 0.17015 | 0.19854 |
| Sentence-only, confidence-weighted | 0.47787 | 0.52143 |
| Row-shuffled | 0.07445 | 0.08766 |
| Semantic-only | 0.07976 | 0.07748 |

Training improves all source-driven branches and slightly degrades the
semantic-only branch. For trained versus untrained weighted joint output, the
paired word-F1 gain is `+0.01658`, 95% CI `[0.00427,0.02942]`; the content-F1
and WER intervals include zero.

## Assumed D'Ascoli accuracy response

This curve keeps full semantic strength and the original confidence-weighted
source blend.

| Nominal accuracy | Realized accuracy | Untrained sum | Trained sum | Trained WER | Trained Top-1 | Trained Top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00000 | 0.07246 | 0.07868 | 1.00636 | 0.02727 | 0.15455 |
| 0.25 | 0.23091 | 0.08085 | 0.08212 | 0.99273 | 0.04545 | 0.16364 |
| 0.50 | 0.49818 | 0.10397 | 0.12450 | 0.97636 | 0.11818 | 0.30909 |
| 0.75 | 0.75545 | 0.15485 | 0.18330 | 0.93545 | 0.18182 | 0.37273 |
| 0.90 | 0.89727 | 0.18177 | 0.20039 | 0.91364 | 0.21818 | 0.42727 |
| 1.00 | 1.00000 | 0.20739 | 0.21890 | 0.90818 | 0.28182 | 0.48182 |

Under this deliberately simple corruption model, roughly 50% position
accuracy is where the full-strength weighted joint branch first shows a clear
word-F1 and WER advantage over semantic-only. Even a perfect source is badly
underused by that branch, proving that its plateau is an ELF fusion problem.

## Semantic strength and source-trust surface

At 49.82% source accuracy, every row and column below uses the same checkpoint
and random seed. Values are word F1 + content F1.

| Semantic scale \\ confidence floor | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.52143 | 0.58180 | 0.61527 | 0.62579 | 0.64683 |
| 0.1 | 0.31058 | 0.34526 | 0.38870 | 0.41107 | 0.43202 |
| 0.25 | 0.10933 | 0.11234 | 0.11055 | 0.12080 | 0.14495 |
| 0.5 | 0.09405 | 0.09324 | 0.10316 | 0.10415 | 0.10696 |
| 1 | 0.12450 | 0.12960 | 0.14789 | 0.16594 | 0.19854 |

Two patterns are unambiguous:

1. Increasing the confidence floor consistently helps: low-confidence words
   still carry sentence syntax and on-manifold T5 geometry. Replacing them
   with independent Gaussian states is harmful.
2. Scaling the existing semantic prefix is nonlinear and mostly destructive.
   This input-amplitude knob is affected by normalization inside ELF and is
   not a satisfactory modality gate.

A finer full-source-trust sweep found a narrow useful range:

| Semantic scale | Sum | WER ↓ | Top-1 | Top-5 |
|---:|---:|---:|---:|---:|
| 0.000 | 0.64683 | 0.69818 | 0.42727 | 0.66364 |
| 0.020 | 0.65134 | 0.68455 | 0.43636 | 0.67273 |
| 0.025 | 0.65240 | 0.68273 | 0.44545 | 0.65455 |
| 0.030 | 0.65325 | 0.68000 | 0.46364 | 0.66364 |
| **0.035** | **0.65497** | **0.66727** | **0.47273** | **0.69091** |
| 0.040 | 0.64570 | 0.67182 | 0.48182 | 0.67273 |
| 0.050 | 0.62670 | 0.67909 | 0.45455 | 0.65455 |
| 0.075 | 0.54098 | 0.71000 | 0.40909 | 0.59091 |
| 0.100 | 0.43202 | 0.77545 | 0.33636 | 0.52727 |

Against scale zero, scale 0.035 changes content/word F1 by only
`+0.00379/+0.00435`; both paired 95% intervals include zero. Its WER change is
`-0.03091`, 95% CI `[-0.05182,-0.01182]`, with bootstrap probability of
improvement `0.9996`. Treat `0.035` as a validation-selected diagnostic, not a
stable production hyperparameter.

## What the flow does to classified positions

Exact position diagnostics use all 1,100 validation word positions. The source
contains 548 correct and 552 wrong positions.

| Output | Exact position accuracy | Retain correct source | Correct wrong source | Copy wrong source | Mean words |
|---|---:|---:|---:|---:|---:|
| Full semantic, confidence-weighted | 0.03273 | 0.05474 | 0.01087 | 0.00000 | 9.15 |
| Full semantic, unweighted | 0.06364 | 0.11679 | 0.01087 | 0.01993 | 9.42 |
| Sentence-only, confidence-weighted | 0.17455 | 0.33212 | 0.01812 | 0.01449 | 11.02 |
| Sentence-only, full trust | 0.33000 | 0.66058 | 0.00181 | 0.39312 | 10.46 |
| Scale 0.035, full trust | 0.33727 | 0.67518 | 0.00181 | 0.38768 | 10.08 |

The final branch improves structure and retention but almost never repairs a
wrong classified position. Compared with it, direct source output improves
content F1 by `+0.16478` (95% CI `[0.13635,0.19417]`), word F1 by `+0.13251`
(`[0.11094,0.15521]`), and WER by `-0.16545`
(`[-0.19455,-0.13818]`).

## Significance and qualitative examples

For the selected scale-0.035 branch, every extended score is above its
one-to-one row-permutation null at the Monte Carlo floor `p=9.9999e-06`:

| Metric | Actual | Permutation mean | Favorable difference |
|---|---:|---:|---:|
| Content F1 | 0.28839 | 0.01099 | 0.27741 |
| Word F1 | 0.36658 | 0.03765 | 0.32893 |
| WER | 0.66727 | 1.01923 | 0.35195 |
| Raw BERTScore F1 | 0.84693 | 0.81245 | 0.03447 |
| BLEU-1 | 0.35862 | 0.03677 | 0.32186 |
| ROUGE-1 F1 | 0.37377 | 0.04334 | 0.33043 |

Its Top-1/Top-5 retrieval rates are `0.47273/0.69091`; exact binomial chance
tests give `p=3.51e-75/5.67e-75`.

Examples from the selected branch:

```text
target:    home of twenty five years and my jeep gets stolen
generated: home of twenty five years the my appearance gets nothing

target:    new york times business sections and the wall street journal
generated: new york times high guys put the wall street journal

target:    dangerous edgy pictures of women and he wants to shoot
generated: dangerous edgy act gonna women and he looks to shoot
```

These examples are consistent with the position analysis: the system preserves
many classified words and improves sentence shape, but semantic correction of
wrong words remains weak.

## Recommendation

### Use this implementation as a baseline, not the final fusion

Keep the new source-to-target transport path because it proves ELF-B can start
from a D'Ascoli sentence and because it supplies strong ablations. Change the
conditioning design before training on real classifier output:

```text
h_src = T5(D'Ascoli full sentence)              # always on-manifold
z_0   = h_src                                   # no confidence/Gaussian blend

v_src  = v(z,t | source memory, empty semantic)
v_joint= v(z,t | source memory, MEG ADA semantic)
v      = v_src + g_sem(t, confidence)*(v_joint-v_src)
```

- Implement `g_sem` on the semantic cross-attention **output/residual inside
  each target block**, initialized to zero. Do not scale prefix embeddings
  before layer normalization.
- Make the gate token-specific and larger for uncertain source positions, for
  example from `[1-c, entropy, top1-top2 margin]`.
- Keep `h_src` as separate source memory throughout the ODE so correct words
  cannot disappear after the first integration steps.
- Feed D'Ascoli top-k posterior word memory, confidence, entropy, and margin as
  features. Do not reduce uncertainty to a hard top-1 scalar.
- Add a high-confidence preservation loss and report correct-position
  retention, wrong-position correction, and false-change rate. The loss must
  still use the true training sentence as target; never train ELF to reproduce
  a wrong classifier guess as ground truth.
- Compare against direct D'Ascoli output on every run. A combined model is not
  useful merely because it beats semantic-only ELF.

An ordered masked-infill corrector is also a strong low-cost baseline: freeze
high-confidence source tokens and ask a T5 editor to replace only uncertain
positions using ADA semantic memory. The present results suggest this may fit
the task better than unconstrained full-sequence diffusion.

### Training with real D'Ascoli output

The next ELF training pack must use story-grouped OOF D'Ascoli predictions for
all 2,652 training rows and held-out predictions for the 110 validation rows.
Synthetic corruption can remain as augmentation, not the primary source.
Select one fusion checkpoint before any protected evaluation.

## Limitations

- The target-derived simulator inherits the true word order and sentence
  template, samples independent replacement errors, and uses deliberately
  well-separated confidence distributions. Real classifier errors will be
  correlated and less perfectly calibrated.
- The semantic scale, confidence floor, and checkpoint were selected on the
  same 110 validation rows. Their point estimates are validation diagnostics.
- The simulated D'Ascoli stream is target-derived, so no source-conditioned
  row in this report is a brain-decoding result. Only the ADA semantic stream
  is MEG-predicted.
- Protected test26 remains unopened.

## Artifacts

- Selected compact LoRA-8 checkpoint (ARC):
  `/data/engs-pnpl/glandau/elf-runs/qc4wyals_elfb_dascoli_lora8_mid_ep40_seed49_20260911/best.pt`
  (1.6 MB; recursively loads the audited x1 parent).
- Full primary tables: `meg_dascoli_sentence_flow_20260912/trained_matrix.csv`
  and `untrained_matrix.csv`.
- Weight surface: `meg_dascoli_sentence_flow_20260912/weight_grid.csv` and
  `weight_grid.md`.
- Extended metrics/nulls and ranked generations: all `*.permutation.json` and
  `*.ranked_generations.csv` files in the same directory.
- Paired bootstraps: `meg_dascoli_sentence_flow_20260912/bootstrap_*.json`.
- Position behavior: `meg_dascoli_sentence_flow_20260912/position_behavior.json`.
- D'Ascoli classifier handover:
  `DASCOLLI_WORD_CLASSIFICATION_HANDOVER_2026-09-11.md`.
