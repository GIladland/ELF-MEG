# MEG pure-semantic diffusion gap closure

## Contract

- Inference begins from noise and is conditioned only on the frozen
  `qc4wyals` 1,536-D predicted ADA vector.
- No D'Ascoli prediction, target word, T5-generated sentence, or T5-generated
  prefix is available to the model.
- ELF's native frozen `t5-small` latent representation and tokenizer are not a
  generative source and remain part of the existing diffusion architecture.
- Training and selection use the established 2,652 Apple training rows and 110
  validation rows. The protected 26-row test set remains unopened.
- The primary comparison uses the fixed ten-word output cap and the same 110
  retrieval candidates.

## Gap to close

| System | Content F1 | Word F1 | Sum | WER |
|---|---:|---:|---:|---:|
| T5 direct-projector winner | 0.02551 | 0.09943 | 0.12493 | 0.97727 |
| MEG-output-adapted ELF-M lexical comparator, corrected cap10 | 0.02414 | 0.08132 | 0.10546 | 0.98182 |
| Frozen-qc4wyals-ADA-only ELF-M control, corrected cap10 | 0.01800 | 0.07764 | 0.09564 | 0.97909 |
| Required uplift from the strict frozen-ADA control | +0.00751 | +0.02179 | +0.02929 | -0.00182 |

The main deficit remains lexical: approximately 74% of the required sum uplift
from the strict frozen-ADA control is word F1. Capped WER is already within
`0.00182` of the T5 point estimate.

The distinction between the two ELF rows is scientifically important. The
published lexical checkpoint was trained from raw MEG and its saved run
configuration reports 6,305,152 trainable parameters in the `qc4wyals`
`final_convs`/`projection_head`, plus a trainable semantic projector. It is a
valid brain-conditioned ELF comparator, but it is not a frozen predicted-ADA
input control. The strict run instead loads the already-exported normalized
1,536-D predictions and prevents any MEG-side or semantic-projector update.
Its strength-zero result is the baseline for every experiment in this report.

A row-level token audit sharpens that diagnosis. Relative to the capped ELF
winner, T5 recovers 72 additional matched token instances while ELF recovers
53 different instances. T5's largest exclusive gains are overwhelmingly
function words: `i` (14), `and` (11), `a` (6), `my` (5), `of` (4), and `the`
(4). T5 has a net matched-token advantage on 39 rows, ties on 40, and loses to
ELF on 31. This points to discrete decoder calibration/order, not missing
semantic content, as the immediate bottleneck.

## First experiment: decoder-only semantic cross-attention

The current ELF path concatenates a clean 64-token semantic prefix with the
noisy target sequence. Ordinary self-attention can underuse that weak prefix.
The new screen adds explicit target-to-semantic cross-attention in the final
ELF-M blocks.

The cross-attention output projection is zero-initialized, making the initial
model exactly equivalent to the selected pure-semantic winner. It is active
only for the discrete decoder objective. The original noise-to-latent flow,
exact-ADA-trained ELF-M weights, and selected `qc4wyals` semantic projector are
frozen. This separation is important: the earlier fMRI shared-flow experiment
damaged an already useful diffusion trajectory, whereas the decoder-only
correction improved conditioning without changing the flow.

Four finite arms vary:

| Arm | Adapted final blocks | Cross-attention LR | Content token weight | Pairing margin |
|---|---:|---:|---:|---:|
| `last2_lr1e4_pm0p5_ce1` | 2 | 1e-4 | 1.0 | 0 |
| `last4_lr1e4_pm0p5_ce1` | 4 | 1e-4 | 1.0 | 0 |
| `last2_lr3e5_pm1p5_ce1` | 2 | 3e-5 | 1.0 | 0 |
| `last2_lr1e4_pm0p5_ce2_pair0p05` | 2 | 1e-4 | 2.0 | 0.05 |

All arms use decoder probability 1, decoder loss weight 1, denoiser loss weight
0, full target masking, 20 epochs, seed 49, and validation checkpoint selection
by mean word/content F1. The first smoke showed that the legacy decoder mixing
mean `0.8` retains enough target latent signal to reduce decoder CE to `8e-6`,
leaving almost no semantic-conditioning gradient. The production arms therefore
use mixing means `-0.5` or `-1.5`, explicitly forcing the decoder to rely more on
the frozen predicted-ADA memory. Checkpoints store only the small trainable
cross-attention delta and retain the selected semantic winner as their parent.

- Launcher:
  `submit-jobs/train_qc4wyals_elfm_semantic_xattn_20260912.sbatch`
- Smoke job: `8791557_0` (`COMPLETED`, exit `0:0`)
- Superseded default-mixing array: `8791580[0-3%2]` (cancelled before start)
- Corrected mixing smoke: `8791591_0` (`COMPLETED`, exit `0:0`)
- Superseded A100/L40S requests: `8791608`, `8791614` (cancelled before start)
- Corrected in-sample-prediction array: `8791616[0-3%1]`
- Nested-OOF robustness follow-up: `8791872[2-3%1]`, dependency-gated after
  `8791616`

## Promotion rule

An arm is promoted only if its fixed cap10 word+content sum exceeds `0.12493`
while WER remains approximately `0.97727`. A promoted checkpoint must then pass:

1. identical val110 metric recomputation from saved row-level generations;
2. matched versus fixed rolled-ADA controls;
3. the existing 100,000 one-to-one row-permutation audit;
4. multiple fixed generation seeds if the point estimate remains competitive;
5. explicit confirmation that no D'Ascoli/T5-generated source and no protected
   test26 row entered training, selection, or evaluation.

## Frozen decode sweep

The selected pure ELF-M winner had only been decoded at 32 ODE steps, CFG 1.0,
and self-conditioning CFG 1.0 in the current comparison. A frozen 3-by-3 screen
tests ODE steps `{16,32,64}` and semantic CFG `{0.5,1.0,1.5}` at fixed seed 49.
This is a pure inference change: every row starts from noise and uses only its
predicted ADA condition.

- Launcher:
  `submit-jobs/eval_qc4wyals_elfm_puresem_decode_sweep_20260912.sbatch`
- Standalone-pack diagnostic array: `8791917[0-8%1]` (tasks 0-3 completed;
  the remainder was cancelled)
- Combined-pack 27-token-evaluation diagnostic chain: `8792011`-`8792022`
  (tasks through CFG-1/32 steps completed; the remainder was cancelled)
- Exact-interface position-prior array: `8792193[0-6%1]`, which keeps the
  published 27-token model/training configuration but samples a 22-token
  validation canvas. Its strength-zero cell is the frozen-ADA reproduction
  control.

The completed low-guidance controls show that output length alone is not the
missing ingredient:

| ODE steps | CFG | Capped Content F1 | Capped Word F1 | Sum | Capped WER | Mean words |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.5 | 0.01480 | 0.06830 | 0.08309 | 0.97909 | 8.19 |
| 32 | 0.5 | 0.01475 | 0.07056 | 0.08531 | 0.97909 | 8.30 |
| 64 | 0.5 | 0.01335 | 0.06811 | 0.08146 | 0.98000 | 8.17 |

The table rows are 27-slot diagnostics only. The original published run configuration
records `target_length=27` but `eval_target_length=22`: ELF-M was configured on
the 27-token training canvas and sampled only 22 validation target-token slots.
The standalone-pack array instead configured and sampled 22 slots, while the
first combined-pack correction configured and sampled 27. Independent 27-slot
paths agreed exactly at the CFG-1/32 reproduction point (sum `0.08439`, WER
`0.98636`). The corrected 27-model/22-evaluation strength-zero control reaches
content F1 `0.01800`, word F1 `0.07764`, sum `0.09564`, WER `0.97909`, and mean
capped length `9.57` words.

This control does not reproduce the MEG-output-adapted comparator's `0.10546`
sum, and it should not: it bypasses the comparator's fine-tuned MEG output
layers and supplies frozen qc4wyals ADA directly. A separate controlled run,
ARC job `8792250`, evaluated the pre-MEG-tuning exact-ADA flat projector. It is
decisively worse on frozen predictions: content F1 `0.01618`, word F1 `0.06154`,
sum `0.07772`, WER `0.99000`. The MEG-trained projector is therefore retained
as the stronger fixed base even though its MEG-side component is bypassed.

## Status

The launcher and compact-checkpoint support are implemented and shell-validated.
The ARC smoke successfully restored the exact-ADA-trained ELF-M generator and
the selected predicted-ADA semantic projector, enabled 8,925,576 trainable
cross-attention parameters in final blocks 22-23, froze every adapter tensor,
ran decoder-only optimization and generation, and wrote 35-MB trainable-delta
`best.pt`/`final.pt` checkpoints. Its near-zero decoder CE identified target
leakage into the training input, so the original full array was cancelled before
allocation and the logistic-normal decoder mixing controls were exposed for a
corrected smoke and screen. Corrected smoke `8791591_0` increased first-step
decoder CE from `0.000008` to `0.036014`, confirming a nontrivial training signal,
and completed generation/checkpointing successfully.

The first production arm, final-two-block attention at `1e-4`, peaked before
capping at word F1 `0.07568`, content F1 `0.02368`, sum `0.09937`, and WER
`1.02818`. Under the fixed ten-word cap it reaches word/content
`0.07594/0.02468`, sum `0.10062`, and WER `0.98636`: still a clear negative
relative to the capped pure baseline (`0.10546`, WER `0.98182`). The
final-four-block arm moved in the same direction and was stopped early. The
lower-rate arm was also stopped after a negative early validation point. On
nested-OOF inputs, the pairing/content arm's step-250/500 sums were
`0.09292/0.08695`, with WER `1.02909/1.03545`. Decoder-only cross-attention is
therefore rejected for gap closure.

This failure is consistent with a train/deployment condition mismatch: the
2,652 training ADA predictions from the deployed `qc4wyals` checkpoint are
in-sample, while val110 is held out. A dependency-gated follow-up will train the
same small cross-attention path on the existing 2,652 nested session-OOF ADA
predictions, which better match the held-out prediction distribution, while
retaining the identical deployment val110 tail.

Two redundant 1.97-GB evaluation checkpoint copies from the rejected nested-OOF
attention arm were removed; its JSON metrics, logs, and compact 35-MB best
delta remain. Together with the rejected residual-calibrator tensor this freed
about 5.7 GB without touching any established candidate checkpoint.

## Deployment-latent correction

The first attention arm also reveals a second mismatch: legacy decoder training
uses a synthetic corruption of the correct target-text latent, but deployment
decodes an imperfect latent produced from noise by the semantic-conditioned
flow. A second follow-up therefore caches one fixed 32-step frozen-flow target
latent for each of the 2,652 nested-OOF training ADA rows. It then trains the
decoder-only attention directly on those flow outputs and the training text.

The flow cache contains no validation or test row and no target-derived latent.
At validation, the model still starts from fresh noise and sees only the frozen
held-out `qc4wyals` ADA prediction. Thus the correction matches the decoder's
training input to deployment without introducing a sentence source.

- Launcher:
  `submit-jobs/train_qc4wyals_elfm_semantic_xattn_flowlatent_20260912.sbatch`
- Focused cache regression: `8791894` (`COMPLETED`, exit `0:0`), covering
  target-only shape, atomic creation, batching, and cache reuse.
- Production arms: `8791907[2-3%1]`, lower-rate arm 2 and pairing/content arm
  3, 10 epochs, nested-OOF train conditions, dependency-gated after the simpler
  OOF screen.

## Nested-OOF semantic residual calibration

The most direct correction leaves both the selected ELF-M generator and its
semantic-to-context network frozen. A 1.58M-parameter residual mapper is placed
in 1,536-D ADA space and initialized to exact identity, so its step-zero model
is the current pure-semantic winner. It is trained on nested session-OOF
predicted ADA paired with exact train ADA; the deployment val110 tail remains
held out.

Two finite controls separate semantic calibration from flow supervision:

| Arm | Trainable component | Primary supervision | Epochs | ARC job |
|---|---|---|---:|---:|
| `semobj` | residual ADA mapper only | exact-ADA cosine, context, contrastive and preservation losses | 40 | `8791952` |
| `flowjoint` | residual ADA mapper only | same semantic losses plus frozen denoiser and weak decoder loss | 20 | `8791955` (afterany `8791952`) |

- Reproducibility launcher:
  `submit-jobs/train_qc4wyals_elfm_semantic_residual_calibrator_20260912.sbatch`
- Initial submissions `8791942`/`8791944` stopped before optimization because
  compact checkpoints did not yet support trainable adapter deltas. The retry
  stores one ordinary best checkpoint per arm and no redundant final copy.
- The semantic-only retry was stopped after its first validation at step 250:
  training semantic cosine rose to about `0.86`, but held-out word/content F1
  fell to `0.060/0.006`, WER was `0.989`, and generated retrieval fell to
  Top-1/Top-5 `0.009/0.045`. Its rejected 1.9-GB checkpoint was deleted while
  metrics/logs were retained; the flow-supervised lower-rate arm remains.
- The flow-supervised arm was stopped after steps 250/500 produced word/content
  sums about `0.060`/`0.069`, respectively, both far below baseline. Its
  rejected 1.9-GB `best.pt` was removed while all JSON metrics and logs were
  retained.
- No target word, D'Ascoli proposal, or T5-generated source is used. At every
  validation decode, the mapper receives only held-out `qc4wyals` predicted
  ADA and ELF-M starts from noise.

## Pure-ELF candidate diversity

The three existing capped ELF specialists have a target-informed per-row
selection ceiling of word F1 `0.12086`, content F1 `0.05140`, sum `0.17227`,
and WER `0.97182`. It chooses the lexical/content/retrieval specialists on
`31/21/58` rows. This is not a deployable result because the diagnostic uses
the validation target to choose a candidate, but it establishes that the
pure-diffusion candidate set already contains enough complementary correct
words to clear T5.

A target-free control fits a ridge map from nested-OOF predicted ADA to frozen
RoBERTa sentence states on the 2,652 training texts, then selects among only
the three ELF candidates by validation predicted-ADA compatibility. Reference
texts are read only after selection for reporting. The selector is not a new
condition or sentence source, and no T5 candidate enters the pool.

- Launcher: `submit-jobs/rerank_meg_pure_elf_roberta_oof_mapper_20260912.sbatch`
- ARC job: `8791974` (`COMPLETED`, exit `0:0`)
- Best target-free point: length penalty `0.002`, word/content F1
  `0.07926/0.02568`, sum `0.10494`, WER `0.98636`. It remains just below the
  single lexical-composite baseline, so whole-sentence routing is rejected
  despite the high target-informed candidate ceiling.

## Nested-OOF flow LoRA

Because input calibration and decoder-only attention can both leave the
continuous transport mismatch unresolved, one conservative arm adapts only
rank-4 attention deltas in the final four ELF-M blocks. The selected semantic
projector remains frozen. Training conditions are nested-OOF predicted ADA;
validation receives deployment predicted ADA and begins from noise.

- Launcher: `submit-jobs/train_qc4wyals_elfm_nested_oof_lora_20260912.sbatch`
- ARC job: `8791989`
- Ten epochs, LR `3e-5`, denoiser weight `1.0`, weak decoder weight `0.25`,
  checkpoint selection by val110 word/content mean.
- Validation sums at steps 250/500/750 were approximately
  `0.083/0.073/0.080`; the run was stopped as a decisive negative. Its compact
  checkpoints and metrics remain.

## Direct semantic word evidence inside ELF

The T5 gap is overwhelmingly exact-word overlap, while the frozen ELF flow is
already close on content F1 and capped WER. A new bounded branch therefore
trains a small multi-label content-word head directly on the 2,652 nested-OOF
predicted ADA vectors. Validation uses only the held-out 110 deployment
predictions. The classifier is not permitted to emit a sentence: its top-k
scores become a sparse, soft bias on the frozen ELF decoder logits after the
noise-to-latent trajectory. Each selected word is biased at most once and ELF
retains authority to choose every token.

This remains a pure semantic-conditioned diffusion system: there is one
1,536-D predicted-ADA inference input, no D'Ascoli proposal, no generated T5
prefix or sentence, and no validation reference used in generation. The word
head is simply a trainable conditioning submodule of ELF, analogous to a
cross-attention adapter but targeted at the observed lexical bottleneck.

- Word-head launcher:
  `submit-jobs/train_qc4wyals_semantic_word_head_20260912.sbatch`
- Position-aware head launcher:
  `submit-jobs/train_qc4wyals_semantic_ordered_head_20260912.sbatch`
- Frozen decoder-bias sweep:
  `submit-jobs/eval_qc4wyals_elfm_semantic_word_bias_20260912.sbatch`
- Head screen: linear, 256-hidden, and 512-hidden content variants plus three
  compact ten-position variants, all with nested-OOF training, train-only
  vocabularies and word/position priors.
- Decode screen: an initial top-1 bias range `{0.25,0.5,1.0,2.0}` plus a
  top-3/top-5 high-strength range `{4,8,16,32}`, fixed 32-step flow, CFG 1.0,
  exact 27-token-model/22-token-evaluation val110 contract, followed by the
  fixed ten-word cap.

The content-head screen completed in ARC job `8792160`. The selected linear
head reaches val110 top-1 F1 `0.02883` and top-5 F1 `0.05922`; neither MLP
improved on its initialization prior. The position-aware screen completed in
job `8792178`. Its strongest MLP reaches standalone word F1 `0.16075` and WER
`0.92727`, but content F1 is zero. This is evidence that an ordered grammatical
prior can repair the observed function-word deficit, not a candidate result by
itself. It must improve the semantic diffusion output while preserving matched
content and matched-over-shuffled dependence.

The first decoder-bias submissions (`8792227`, `8792228`) failed before
inference because ARC still had an older lexical-wrapper module. No checkpoint
or metric was produced. After synchronizing the module, corrected ordered and
content-word arrays were submitted as `8792234` and `8792235`. Ordered additive
biases through `0.8` and top-1 content biases through `2.0` leave every decoded
sentence unchanged. The still-unstarted low-dose cells were therefore replaced
by prior-corrected ordered strengths `{1,2,4,8,16}` (`8792357`) and top-3/top-5
content strengths `{4,8,16,32}` (`8792374`).

The high-strength screen produces a real content gain. Top-3/strength-32 alone
reaches content F1 `0.03034`, word F1 `0.08163`, sum `0.11197`, WER `0.98545`.
Top-5/strength-32 reaches `0.03691/0.08594`, sum `0.12285`, at the same WER.
Thus the semantic word head nearly closes the sum gap and clearly exceeds T5's
content F1, but aggressive consecutive placement costs edit distance.

## Pure-ELF finalist

Combining the top-3/strength-32 semantic content head with the ordered head
adds complementary function-word information. Ordered strength `24` reaches
word F1 `0.09996`, content F1 `0.02968`, sum `0.12964`: `+0.00471` above the T5
sum. This is the first pure frozen-ADA/noise-start ELF candidate to cross the
primary overlap threshold. Its WER is `0.98727`, however, `0.01000` above T5,
so it is provisional rather than promoted.

- Combined launcher:
  `submit-jobs/eval_qc4wyals_elfm_semantic_combined_bias_20260912.sbatch`
  (`8792405`).
- Standard 100,000-row permutation audit: ARC job `8792490`.
- WER-targeted follow-up: replace consecutive content-word insertion with
  compatibility-based scattered placement, preserve the train-only position
  prior at strength `0.4`, and scan content/ordered strengths. Launcher:
  `submit-jobs/eval_qc4wyals_elfm_semantic_scattered_bias_20260912.sbatch`
  (`8792489`).

The provisional candidate's permutation audit is complete. Word F1 is
`0.09996` versus null mean `0.08618` (`p=0.01197`); raw BERTScore is `0.81804`
versus `0.81503` (`p=0.00001`). Content F1 is `0.02968` versus `0.02658`
(`p=0.22762`), and WER is `0.98727` versus `0.98640` (`p=0.64334`). Thus row
pairing contributes significant word and embedding-level signal, but the
absolute content/function-word improvement is partly a corpus prior and the
candidate is not WER-promotable yet.

Compatibility-based scattered placement resolves the point-estimate trade-off.
The bounded sweep gives:

| Content bias | Ordered bias | Word F1 | Content F1 | Sum | WER |
|---:|---:|---:|---:|---:|---:|
| 24 | 0 | 0.08103 | 0.02757 | 0.10860 | 0.98000 |
| 32 | 0 | 0.08168 | 0.03186 | 0.11354 | 0.98364 |
| 40 | 0 | 0.08158 | 0.03172 | 0.11330 | 0.98455 |
| 24 | 16 | 0.08749 | 0.02774 | 0.11522 | 0.98273 |
| 24 | 24 | 0.09858 | 0.02466 | 0.12324 | 0.98182 |
| **24** | **32** | **0.11710** | **0.02548** | **0.14258** | **0.97636** |
| 32 | 20 | 0.09868 | 0.03234 | 0.13102 | 0.98545 |
| 32 | 24 | 0.09909 | 0.02931 | 0.12840 | 0.98455 |

The bold cell is the current finalist. Relative to the T5 direct-projector
point estimate it improves word F1 by `0.01767`, essentially ties content F1
(`-0.00003`), improves the sum by `0.01765`, and improves WER by one edit
(`0.97636` versus `0.97727`). Its saved configuration confirms an empty raw
brain input, no D'Ascoli source, predicted-ADA `input_embeddings` as the sole
condition, fresh 32-step ODE sampling, and the fixed 22-token evaluation canvas
followed by the ten-word cap. The underlying diffusion checkpoint is ELF-M.

This is a validation-selected finalist, not a protected-test claim. ARC jobs
`8792605`/`8792614`, `8792606`, and `8792607` completed the 100,000-permutation
audit, two additional fixed generation seeds, and three held-out
semantic-vector rolls, respectively. Test26 remains unopened.

The permutation audit is complete. Word F1 remains above the one-to-one row
null (`0.11710` versus `0.10678`, `p=0.03209`); ROUGE-1 is significant at
`p=0.01516`, and raw BERTScore at `p=0.00111`. The primary sum is `0.14258`
versus null `0.13240` (`p=0.11070`), content F1 alone is null-level
(`p=0.50649`), and WER is directionally favorable but nonsignificant
(`p=0.13889`). The high null reflects the strong train-only grammatical prior,
so the significant word and embedding-level pairing tests are the relevant
evidence that the finalist is not condition-independent.

A separate 100,000-sample paired val-row bootstrap against T5 gives a sum
difference of `+0.01765`, 95% interval `[-0.01284, 0.04813]`, and probability
of a positive difference `0.8718`. The word-F1 probability is `0.9479`; the
one-edit WER advantage has probability `0.5301`. Accordingly this is a
deterministic validation leaderboard win, not statistically established
population-level superiority over T5.

The win is metric-specific rather than universal. The finalist also improves
BLEU-1 (`0.11623` versus `0.09563`) and ROUGE-1 (`0.12329` versus `0.10877`),
but T5 retains substantially higher raw BERTScore (`0.82821` versus `0.81420`).
The result closes the declared lexical/WER gap; it does not establish better
semantic fluency overall.

The two additional noise seeds also clear T5's overlap sum: seed 7 reaches
word/content `0.11867/0.02922`, sum `0.14789`, WER `0.97455`; seed 82 reaches
`0.10725/0.02652`, sum `0.13377`, WER `0.98545`. Across seeds 49, 7, and 82,
the mean is word/content `0.11434/0.02707`, sum `0.14141`, and WER `0.97879`.
All three seeds beat T5 on the primary sum, two of three beat it on WER, and
the mean WER remains approximately `0.98`.

The fixed held-out semantic rolls quantify condition dependence. Rolls 1, 37,
and 73 reach sums `0.13736`, `0.13513`, and `0.13322`, with WER `0.98182`,
`0.98545`, and `0.98091`. Their mean is sum `0.13524`, WER `0.98273`.
Correct pairing therefore adds `0.00734` sum and removes seven edit errors
relative to the roll mean. A paired bootstrap against the per-row roll mean
assigns probability `0.7798` to the sum improvement and `0.9272` to the WER
improvement; both 95% intervals include zero. The rolls themselves remain
strong because the train-only ordered/position priors carry common grammar.
Thus semantic pairing helps consistently, but much of the absolute lexical
score is decoder calibration rather than row-specific semantic information.

## Train-only positional decoder prior

Because the T5 advantage is concentrated in common grammatical words, a
separate seven-point screen adds a smoothed token-position log prior to ELF's
native decoder logits. The prior is estimated from only the selected 2,652
training rows; validation labels are never read to construct it. Per-position
log probabilities are centered at zero and clipped, so the term only penalizes
implausible alternatives rather than inserting a sentence. Strength zero is
an exact reproduction control, followed by `{0.025,0.05,0.10,0.20,0.40,0.80}`.

This is decoder calibration inside the diffusion model, not a language-model
generator: the continuous sample still begins at noise, follows the frozen
predicted-ADA-conditioned ELF flow, and is decoded once by ELF. Launcher:
`submit-jobs/eval_qc4wyals_elfm_position_prior_20260912.sbatch`.

The seven-point array completed successfully. Strength `0.4` is the best cell:
content F1 `0.01800`, word F1 `0.07866`, sum `0.09666`, WER `0.97818`. This is
only a `+0.00102` sum gain over strength zero and cannot close the T5 gap by
itself. Strength `0.8` gives a similar sum (`0.09672`) with slightly worse WER
(`0.97909`).

## Nested-OOF context-projector adaptation

The published MEG-output-adapted comparator establishes that the interface into
the frozen ELF-M generator has useful capacity, but it entangles changes to the
MEG decoder and the semantic projector. A stricter follow-up trains only the
semantic-to-ELF context projector on the 2,652 nested session-OOF predicted ADA
vectors. `qc4wyals` is never instantiated in this run; the saved 1,536-D vector
is the sole condition. Every ELF-M parameter remains frozen.

Three sequential arms compare learning rates `1e-6` and `3e-6` from the
MEG-adapted projector initialization and `1e-6` from the pre-MEG exact-ADA
projector. They use 12 epochs of frozen-flow/decoder supervision, a weak exact
context consistency loss, a small train-only content loss, and selection by
val110 word/content mean. The model still performs a fresh noise-start 32-step
flow for every validation row. Launcher:
`submit-jobs/train_qc4wyals_elfm_oof_context_projector_20260912.sbatch`; ARC
array `8792427[0-2%1]`. The initial short-partition submission `8792265` was
cancelled before allocation and moved intact to the six-hour RTX8000 queue.

The two MEG-initialized arms were stopped at their first held-out evaluation.
The `1e-6` arm fell to approximately word/content F1 `0.077/0.021`, WER
`1.040`; the `3e-6` arm fell to `0.069/0.010`, WER `1.046`. The exact-projector
initialization control was superseded before allocation once the decoder-only
finalist crossed both T5 targets. Context-projector adaptation is therefore
rejected for this gap-closure result.

## Decision

The declared val110 gap is closed. The selected seed-49 pure predicted-ADA
ELF-M system exceeds T5's word+content sum (`0.14258` versus `0.12493`) and
slightly improves WER (`0.97636` versus `0.97727`). Its three-seed mean remains
well above T5 on sum (`0.14141`) and approximately matches the WER target
(`0.97879`). No T5-generated text, D'Ascoli output, validation target token, or
test26 row is an inference condition.

The mechanism is decoder calibration inside ELF: a frozen train-only semantic
content head supplies three soft candidate words, scattered once into
compatible decoder positions at strength `24`; a frozen train-only ordered
head supplies soft positional token evidence at strength `32`; and the
train-only token-position prior has strength `0.4`. The continuous ELF-M sample
still starts from independent noise and follows the same 32-step
predicted-ADA-conditioned flow before one native decoder pass.

Promotion is limited to the declared lexical/WER validation objective. The
gain is stable enough to retain as the pure-diffusion finalist, but it is not a
uniform text-quality victory: raw BERTScore remains below T5, paired bootstrap
intervals versus T5 and semantic rolls include zero, and the strong language
prior produces a high permutation null. No additional val110 tuning is
recommended before a separately authorized protected evaluation.
