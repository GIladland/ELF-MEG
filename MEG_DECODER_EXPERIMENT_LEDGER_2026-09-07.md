# MEG semantic-to-text experiment ledger

## Evaluation contract

- Frozen brain encoder: `qc4wyals`.
- Development set: the same 110 Apple validation rows for every arm.
- Final test: 26 Birth of a Nation rows remain sealed.
- Report one generation per row from a single checkpoint; no reference-based reranking.
- Retrieval comparisons must use the same 110 candidates.
- Primary selection score: mean of word F1 and content-word F1.
- Also report WER, raw BERTScore, ROUGE-1, generated Top-1/Top-5, semantic
  Top-1/Top-5, and matched-minus-deranged margins.

## Locked diffusion reference

- Audit: `MEG_BEST_GENERATION_SIGNIFICANCE_2026-09-04.md`.
- Word F1: `0.06197`.
- Content-word F1: `0.02978`.
- WER: `1.00364`.
- Raw BERTScore F1: `0.81739`.
- Generated retrieval Top-1/Top-5: `0.03636 / 0.13636`.
- Incoming semantic retrieval Top-1/Top-5: `0.29091 / 0.45455`.

This is the locked audited reference, not necessarily the latest provisional
validation checkpoint. A new model is promoted only after the same complete
audit.

## Active diffusion branch

- Continue from the converged larger-data ELF-M/x16 semantic decoder.
- Compare raw MEG vectors, adapter-only, ELF LoRA, and true end-to-end tuning.
- Explore content/word-balanced losses while requiring a positive conditional
  margin and retaining retrieval.
- Record provisional checkpoints separately from audited winners.

## Active direct T5-prefix branch

- Setup report: `MEG_T5_PREFIX_BASELINE_2026-09-07.md`.
- Smoke job: `8749981` — completed.
- Oracle array: `8749995`.
- Oracle selector: `8749996`.
- Brain-stage array: `8749997`.
- Retrieval/BERTScore array: `8749998`.
- Summary: `8749999`.
- Permutation audit: `8750000`.

The exact-ADA oracle is a decoder ceiling and is never compared directly with
brain-conditioned results as evidence of a brain-decoding improvement.

## Promotion rule

A breakthrough requires a brain-conditioned single checkpoint to improve the
locked reference under the same val110 contract, principally word F1 and
content-word F1, without disguising a retrieval collapse. WER, BERTScore,
ROUGE-1, retrieval, and permutation/derangement controls determine whether the
gain is broad and credible.

## Status checkpoint — 2026-09-07 23:32 BST

The direct T5-prefix ladder has completed the oracle and seven of nine brain
arms. Its current best joint-objective arm is
`joint_projector_adapter_lora4_content_pair`:

- word F1 `0.08485`;
- content-word F1 `0.02797`;
- word + content composite `0.11282`;
- WER `0.98000`;
- conditional content margin `+0.01410`;
- adapted-semantic Top-1/Top-5 `0.13636 / 0.35455`.

Relative to the locked audited diffusion reference, the provisional T5 arm is
better on word F1, composite, and WER, but slightly worse on content F1 and
semantic retrieval. Its generated-text retrieval and BERTScore audit are still
dependency-blocked. It is therefore a near-breakthrough, not yet a promoted
winner.

The larger-ELF diffusion adapter search produced a provisional ELF-L residual
arm at word/content F1 `0.08562/0.02168`, WER `1.01727`; this improves the
composite but not content F1 or WER. The context-distillation matrix failures
were implementation failures (W&B artifact-staging quota and unbatched
temporal-attention evaluation), not scientific negatives. Both issues were
fixed and the finite matrix was relaunched:

| Stage | Retry job |
|---|---:|
| Context distillation | `8762065` |
| Robust-initialized distillation | `8762066` |
| Context selector | `8762067` |
| Post-context adapter/LoRA matrix | `8762068` |
| Direct temporal-context distillation | `8762069` |
| Temporal post-context matrix | `8762070` |
| Three-seed fixed-val110 evaluation | `8762071` |
| Aggregate report | `8762072` |

## Controlled diffusion-versus-T5 test — 2026-09-08

The earlier decoder comparison is not sufficient to establish an architecture
advantage: the x16 ELF-M diffusion oracle saw the larger augmented exact-ADA
corpus, whereas the initial T5-prefix oracle saw only the 2,652 brain-training
sentences. The provisional T5 result is therefore informative but not an
apples-to-apples diffusion comparison.

The matched T5 control now uses the same 86,538-row x16 exact ADA/text archive
as the diffusion oracle and approximately the same 43k optimizer-update scale.
Only this text-to-decoder oracle stage sees the augmented corpus. Every brain
arm still trains exclusively on the 2,652 `qc4wyals` training predictions and
is selected on the same 110 validation predictions; the 26-row test remains
sealed. Generation is capped at ten words, retrieval uses the same 110
candidates, and the final audit includes lexical, semantic, retrieval,
derangement, multi-seed, and paired-permutation results.

| Stage | Job |
|---|---:|
| x16 exact-ADA T5 oracle (two learning rates) | `8762211` |
| Oracle selector | `8762212` |
| Frozen/adapter/LoRA/joint brain ladder | `8762214` |
| Generated retrieval and BERTScore audit | `8762215` |
| Aggregate report | `8762221` |
| Paired permutation audit | `8762222` |

The primary claim under test is deliberately scoped: whether ELF diffusion
beats a frozen-T5-prefix decoder under matched known-text training conditions.
No result will be labelled a diffusion win unless it survives the identical
validation contract and the paired audit.

## Status checkpoint — 2026-09-08

All nine brain-training arms of the original, smaller-corpus T5-prefix ladder
have completed. The new provisional leader is `direct_projector` at epoch 120:

- word F1 `0.09943`;
- content-word F1 `0.02551`;
- word + content sum `0.12493`;
- WER `0.97727`;
- conditional content margin `+0.01638`;
- incoming semantic Top-1/Top-5 `0.29091 / 0.45455` on 110 candidates.

The retrieval/BERTScore audit for that ladder remains queued, and this result
still does not constitute a fair architecture comparison because its oracle
used less text than the x16 diffusion oracle.

Diffusion retry task `8762065_0` (`output_only` context distillation) completed
after 33,200 updates. Its teacher-context-selected checkpoint was step 1,000,
with word/content F1 `0.05500 / 0.01614`, WER `1.03000`, and generated-text
Top-1/Top-5 `0.01818 / 0.12727` on 110 candidates. This arm is a negative
result and is not promoted. The other context, projector, temporal, LoRA, and
post-distillation arms remain queued; the matched x16 T5 oracle and downstream
ladder also remain queued.

Eight previously submitted x16 MEG true-E2E arms (`8732753`) also completed.
They establish a partial, rather than broad, improvement:

| x16 E2E distinction | ELF/arm | Step | Word F1 | Content F1 | Sum | WER | Generated T1/T5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Best word+content sum | ELF-M, all MEG + projector, frozen ELF | 13,500 | 0.07192 | 0.02687 | 0.09879 | 1.05273 | 0.05455 / 0.11818 |
| Best generated retrieval | ELF-M, all MEG + projector + LoRA4 | 14,000 | 0.07233 | 0.02552 | 0.09786 | 1.04636 | 0.09091 / 0.16364 |
| Best content F1 | ELF-M, output + projector + LoRA4 | 1,500 | 0.06578 | 0.02860 | 0.09439 | 1.04455 | 0.04545 / 0.13636 |

Against the locked x1 diffusion reference (`0.06197/0.02978`, sum `0.09175`,
WER `1.00364`, generated T1/T5 `0.03636/0.13636`), x16 true-E2E improves
word F1, the joint sum, and—in one arm—generated retrieval. It does not improve
the content-F1 maximum or WER. Therefore none of the eight is an all-metric
promotion. Their available BERTScore files are baseline-rescaled (best F1
`-0.06408`) and are not directly comparable to the locked raw BERTScore.

## Nested OOF MEG -> ELF-B gap-closure ladder — 2026-09-08

This experiment measures how much of the loss between exact ADA conditioning
and raw `qc4wyals` MEG predictions can be recovered without fitting an ELF
adapter to in-sample MEG predictions. The earlier 2,652-row bridge source was
in-sample and is not valid for this question: the source MEG model had already
trained on those brain rows.

The replacement uses five deterministic, session-grouped outer folds. Each
fold trains a scratch qc4-like MEG2SEM model on four-fifths of the Tang/Apples
training sessions, uses a separate held-out inner session for checkpoint
selection, and exports predictions only for outer sessions that the fold model
never loaded. The five outer exports are assembled in original row order with
an exact-once coverage and metadata-alignment audit. The fixed external
auxiliary corpus and semantic-negative banks are shared across folds; no outer
brain rows are used by those components. The 26-row locked test set is absent
from all training and selection.

The downstream decoder is the exact-ADA ELF-B checkpoint:

`/data/engs-pnpl/glandau/elf-runs/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/best.pt`

All four ladder points use the identical 110-row validation set, 110 retrieval
candidates, ten-word generation cap, and five decoding seeds:

1. exact ADA -> frozen ELF-B (oracle/decoder ceiling);
2. raw held-out `qc4wyals` validation predictions -> frozen ELF-B;
3. OOF-trained flat adapter -> frozen ELF-B;
4. OOF-trained residual-identity adapter -> frozen ELF-B.

Primary metrics are word F1, content-word F1, their sum, WER, generated-text
Top-1/Top-5 retrieval, and mean rank. Gap closure is reported as
`(adapted - raw) / (exact - raw)` with the appropriate sign for WER and rank.
This is a validation experiment because the same 110 rows select adapter
checkpoints; it is not a final locked-test claim.

| Stage | ARC job |
|---|---:|
| Five nested OOF MEG folds | `8767086` |
| Assemble and audit OOF predictions | `8767087` |
| Flat and residual ELF-B adapters | `8767088` |
| Exact-ADA ELF-B evaluation | `8767089` |
| Raw-MEG ELF-B evaluation | `8767090` |
| Flat-adapter ELF-B evaluation | `8767091` |
| Residual-adapter ELF-B evaluation | `8767092` |
| Aggregate gap-closure report | `8767093` |

At 2026-09-08 13:06 BST, outer fold 0 was running without error; folds 1--4
were pending for priority. No numerical result is recorded until the complete
matched ladder has finished.

### Interim matched endpoints — 2026-09-08 15:37 BST

The five-seed exact-ADA and raw-MEG ELF-B evaluations have completed under the
same val110/110-candidate contract. These define the gap; they do not yet show
adapter closure.

| Conditioning | Word F1 | Content F1 | Sum | WER ↓ | Generated T1 | Generated T5 | Mean rank ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact ADA | 0.88894 | 0.89448 | 1.78342 | 0.13655 | 1.00000 | 1.00000 | 1.00 |
| Raw held-out MEG | 0.04601 | 0.01744 | 0.06345 | 1.04145 | 0.04545 | 0.14182 | 42.99 |

Across decoding seeds, exact-ADA word/content F1 standard deviations were
`0.00727/0.00616`; raw-MEG standard deviations were `0.00452/0.00317`.

Two of five nested OOF models have also completed. Their outer-session metrics
were:

| Fold | Outer rows/candidates | Cosine | Top-1 | Top-5 | Selected epoch |
|---|---:|---:|---:|---:|---:|
| 0 | 559 | 0.56904 | 0.05009 | 0.13953 | 85 |
| 1 | 520 | 0.68313 | 0.05769 | 0.18846 | 112 |

The remaining three OOF tasks were changed from a 24-hour long-queue request
to a 12-hour short-queue request after the first two completed in about two
hours each. The final adapted rows and gap-closure fractions remain pending.

### Interim two-fold adapter ladder

To obtain an earlier directional result without changing the definitive
five-fold experiment, folds 0 and 1 were assembled into an explicitly partial
OOF pack. The audit found 1,079 unique covered rows, zero duplicates, 1,573
intentionally missing source rows, and zero protected test rows. Combined
1,079-way OOF retrieval was Top-1 `0.03244`, Top-5 `0.10380`, matched cosine
`0.62402`; deployment val110 retrieval remains Top-1/Top-5
`0.29091/0.45455` with cosine `0.71742`.

| Interim stage | ARC job |
|---|---:|
| Flat and residual ELF-B adapters | `8767836` |
| Flat-adapter five-seed val110 evaluation | `8767837` |
| Residual-adapter five-seed val110 evaluation | `8767838` |
| Interim gap-closure report | `8767839` |

These rows are labelled interim because only 40.7% of the intended OOF source
rows are available. They may guide direction, but only the complete five-fold
ladder is eligible for the main gap-closure conclusion.

The interim ladder completed on 2026-09-08. Five-seed matched val110 results:

| Conditioning | Word F1 | Content F1 | Sum | WER ↓ | Generated T1 | Generated T5 | Mean rank ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact ADA | 0.8889 | 0.8945 | 1.7834 | 0.1365 | 1.0000 | 1.0000 | 1.00 |
| Raw MEG | 0.0460 | 0.0174 | 0.0634 | 1.0415 | 0.0455 | 0.1418 | 42.99 |
| OOF2 flat | **0.0511** | **0.0220** | **0.0731** | **1.0089** | 0.0309 | 0.1145 | 43.24 |
| OOF2 residual | 0.0462 | 0.0140 | 0.0603 | 1.0236 | 0.0418 | 0.1091 | **41.88** |

The flat adapter recovered `0.60%/0.52%/0.56%` of the raw-to-oracle gap in
word F1/content F1/their sum, and `3.60%` of the WER gap. It improved all
lexical metrics but regressed generated Top-1/Top-5 and mean rank. The residual
arm did not produce a useful joint improvement. Therefore the two-fold result
is a positive directional result for lexical adaptation, not an all-metric
success. The complete five-fold source is now available and remains the proper
test of whether broader OOF coverage preserves the lexical gain while restoring
retrieval.

### Full five-fold assembly and accelerated adapter chain

All five outer folds completed. The full assembly was run directly after its
small CPU scheduler job remained pending, and passed the definitive audit:

- `2,652 / 2,652` source rows covered exactly once;
- zero duplicated or missing rows;
- zero protected test rows;
- OOF matched cosine `0.65978`;
- OOF 2,652-way Top-1/Top-5 `0.01282 / 0.04676`;
- OOF mean/median rank `670.17 / 373`;
- deployment val110 semantic Top-1/Top-5 `0.29091 / 0.45455`.

The original still-pending assembly/dependent jobs (`8767087`, `8767088`,
`8767091`, `8767092`, `8767093`) were cancelled after the assembly job was
found to contain a validation-interface path error. No completed computation
was removed. The corrected post-assembly chain is:

| Full-data stage | ARC job |
|---|---:|
| Flat and residual ELF-B adapters | `8769446` |
| Flat five-seed val110 evaluation | `8769447` |
| Residual five-seed val110 evaluation | `8769448` |
| Full gap-closure report | `8769449` |

The full five-fold adapter ladder completed on 2026-09-08:

| Conditioning | Word F1 | Content F1 | Sum | WER ↓ | Generated T1 | Generated T5 | Mean rank ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact ADA | 0.8889 | 0.8945 | 1.7834 | 0.1365 | 1.0000 | 1.0000 | 1.00 |
| Raw MEG | 0.0460 | 0.0174 | 0.0634 | 1.0415 | **0.0455** | **0.1418** | 42.99 |
| Full OOF flat | **0.0498** | **0.0201** | **0.0699** | **0.9982** | 0.0418 | 0.1127 | **42.35** |
| Full OOF residual | 0.0437 | 0.0093 | 0.0530 | 1.0058 | 0.0291 | 0.1273 | 41.90 |

The flat adapter closes `0.45%/0.30%/0.38%` of the word/content/sum gap,
`4.78%` of the WER gap, and `1.50%` of the mean-rank gap. It improves word
F1, content F1, their sum, WER, and mean rank over raw MEG, but generated
Top-1/Top-5 regress. The residual arm is rejected. The full flat checkpoint is
the only OOF adapter promoted as an initialization candidate for controlled
E2E/LoRA training; it is not itself an all-metric winner.

### Full-OOF ELF-B true-E2E matrix — submitted 2026-09-08

The next controlled stage starts from the promoted full-OOF flat checkpoint,
the fixed `qc4wyals` deployment checkpoint, and raw MEG train windows. Six
arms isolate the contribution of each trainable component:

1. semantic projector only;
2. semantic projector plus ELF-B LoRA rank 4 in the last four blocks;
3. MEG2SEM output head plus semantic projector;
4. MEG2SEM output head plus projector plus ELF-B LoRA;
5. all MEG2SEM weights plus semantic projector;
6. all MEG2SEM weights plus projector plus ELF-B LoRA.

All arms fit the 2,652 training rows, select checkpoints only on val110 using
the mean of word F1 and content-word F1, and report lexical, WER, generated
retrieval, semantic-interface, and oracle-ADA diagnostics. The protected
test26 MEG rows remain absent. ARC array: `8770023`; W&B group:
`qc4wyals_oof5_elfb_truee2e_20260908`.

The full matrix completed on 2026-09-09. None of the six arms beat the
audited x1 ELF-B word-plus-content baseline (`0.09175`):

| Trainable components | Word F1 | Content F1 | Sum | WER ↓ | Gen T1 | Gen T5 | Mean rank ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Projector only | 0.06345 | 0.02201 | **0.08546** | 0.99455 | 0.02727 | 0.11818 | 41.53 |
| Projector + LoRA | **0.06374** | 0.01948 | 0.08322 | **0.98909** | 0.03636 | 0.10000 | 42.34 |
| Output head + projector | 0.06174 | 0.01871 | 0.08045 | 1.00000 | 0.04545 | 0.11818 | **39.53** |
| Output head + projector + LoRA | 0.05294 | 0.02424 | 0.07718 | 0.99909 | **0.04545** | **0.15455** | 40.11 |
| Full MEG + projector | 0.05507 | **0.02427** | 0.07935 | 1.00000 | 0.03636 | 0.12727 | 42.07 |
| Full MEG + projector + LoRA | 0.06017 | 0.02067 | 0.08085 | 0.99909 | 0.02727 | 0.11818 | 40.98 |

This clean E2E result says that adapting more of MEG2SEM or adding ELF-B LoRA
does not reliably recover additional lexical information. Projector-only is
the best clean word/content arm; the output-head-plus-LoRA arm is best only
for generated Top-5 retrieval.

### Audited-x1 plus nested-OOF hybrid repair — submitted 2026-09-09

The new OOF flat adapter is not itself the historical diffusion leader. The
audited x1 ELF-B checkpoint remains stronger on content F1 (`0.02978`) and
word-plus-content (`0.09175`). Because both checkpoints use the same ELF-B
decoder and a compatible flat semantic-context projector, the next test
preserves the audited x1 checkpoint as initialization and fine-tunes its
projector on all 2,652 nested session-OOF MEG predictions.

ARC array `8770647` tests four conservative frozen-ELF variants: low and
medium projector learning rates, stronger content losses, and stronger
exact-ADA context anchoring. Checkpoints are selected by word/content mean on
val110; MEG test26 remains absent. W&B group:
`qc4wyals_x1best_oof5_hybrid_20260909`.

The first two arms trained far beyond their early best points, then failed
while writing redundant periodic checkpoints because the shared ARC data
filesystem reached 100% capacity. Their retained best checkpoints are valid
but do not improve the baseline:

| Arm | Best step | Word F1 | Content F1 | Sum | WER ↓ | Gen T1 | Gen T5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low-LR balanced | 1,500 | 0.04775 | 0.01431 | 0.06205 | 0.99273 | 0.05455 | 0.12727 |
| Mid-LR balanced | 1,000 | 0.05270 | 0.01478 | 0.06748 | **0.97727** | 0.02727 | **0.15455** |

Only redundant evaluation checkpoint copies from these rejected arms were
removed; `best.pt`, metric JSONs, and logs were retained. The strong-content
and strong-context arms were resubmitted as HTC array `8772212` with periodic
checkpoint duplication disabled.

### Lexical complementarity metric extension — 2026-09-09

Future generation evaluations retain the historical exact-token word and
content-word F1 metrics and also report Porter-normalized companions. The
normalized scores capture regular morphology such as `friend/friends` and
`jump/jumped/jumping` without changing the historical leaderboard contract.
Evaluations now also report the total and fraction of decoded function words,
using the established content-stopword lexicon plus common contractions.

For the mid-LR hybrid val110 checkpoint, exact/stemmed word F1 is
`0.05270/0.05707`, and exact/stemmed content-word F1 is
`0.01478/0.02053`. It generated 758 tokens, of which 228 (`30.08%`) are
function words; the references contain 395/1,100 (`35.91%`).

Backfilling the same normalized metrics changes the provisional semantic
leader: the x16 ELF-M output-head + projector + LoRA checkpoint at step 1,500
has exact/stemmed word F1 `0.06578/0.07452` and exact/stemmed content F1
`0.02860/0.04072`. It is the current stemmed-content and stemmed-composite
leader, but remains provisional pending the locked multi-seed/permutation
audit. Within that x16 E2E matrix, all-MEG + projector has the strongest exact
composite (`0.09879`), while its LoRA counterpart is the generated-retrieval
leader (`0.09091/0.16364` Top-1/Top-5). The context-distillation result below
subsequently surpassed the former exact-composite score.

### Content/retrieval reconciliation program — submitted 2026-09-09

The context-distillation audit promoted
`meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906`
as the validation exact-word and exact-composite leader: word/content F1
`0.08071/0.02481`, sum `0.10551`, WER `1.03273`, and generated Top-1/Top-5
`0.03636/0.10000`. Its frozen exact-ADA context teacher improves lexical
decodability but does not preserve the all-MEG retrieval advantage.

Two controlled experiments now test whether those signals are complementary:

1. ARC array `8774835` evaluates nine full-checkpoint interpolation points
   from the stemmed-content specialist (alpha 0) to the generated-retrieval
   specialist (alpha 1). Both adapter and ELF LoRA tensors are interpolated;
   this is validation-only and never loads test26.
2. ARC array `8774836` starts from the robust context-distillation leader and
   trains four dual-teacher arms. Exact ADA teaches the ELF context, while a
   frozen copy of the all-MEG candidate teaches in-batch retrieval logits.
   Three arms route diffusion/lexical gradients only through the projector and
   optional ELF LoRA, leaving MEG2SEM to receive semantic losses only; the
   fourth is a joint-gradient control. The four variants are frozen ELF,
   LoRA4, LoRA4 with moderately stronger content losses, and joint LoRA4.

All dual-teacher arms train on the 2,652 training rows, select only on val110,
use the same 110 retrieval candidates, and exclude the protected test26. A
candidate is not an all-purpose promotion merely for improving one metric: the
review must report exact and stemmed word/content F1, WER, generated Top-1/Top-5,
and mean rank. The desired result beats the `0.10551` exact lexical sum while
recovering at least the content specialist's `0.04545/0.13636` retrieval; a
strict breakthrough would also approach or exceed `0.09091/0.16364`.

### Reconciliation status — 2026-09-09

Interpolation tasks alpha `0.000` through `0.375` completed. The strongest
observed interior point so far is alpha `0.375`: exact word/content F1
`0.06278/0.01973` (sum `0.08251`), stemmed word/content F1
`0.06925/0.02969`, WER `1.04273`, generated Top-1/Top-5
`0.03636/0.10909`, and mean/median rank `42.35/33`. Relative to the alpha-0
reevaluation, it improves exact word F1, exact content F1, both stemmed scores,
Top-1, and mean rank while tying Top-5 and WER. It does not yet beat either
historical specialist, so this is evidence of modest complementarity rather
than a new leader.

Tasks alpha `0.500` through `1.000` were interrupted by the shared
`/data/engs-pnpl` filesystem reaching 100% capacity. Five superseded,
retrainable ELF-L checkpoint tensors were removed while preserving their run
metrics and all current leaders, freeing approximately 15 GB. Only the failed
interpolation tasks were resubmitted as ARC array `8776080`; dual-teacher array
`8774836` remains queued. Test26 remains sealed.

### Three-candidate generation significance audit — 2026-09-10

ARC job 8778233 completed a common 100,000-row-permutation audit of the
lexical-composite, content-specialist, and retrieval-specialist saved val110
generations. Raw RoBERTa-large BERTScore F1, macro sentence BLEU-1, and
Porter-stemmed ROUGE-1 F1 were significant for all three models, including
after Bonferroni correction across all 12 requested tests. WER was not
significant for any candidate (p=0.06490, 0.52692, and 0.24681).

The lexical-composite model had the best absolute WER/BERTScore/BLEU-1/ROUGE-1
(1.03273/0.81985/0.07672/0.08733). The retrieval specialist had the largest
BERTScore increment above its own text-marginal-preserving null (+0.00327)
and retained its 110-way generated retrieval lead (0.09091/0.16364).
Full results and ranked generations are in
MEG_THREE_CANDIDATE_PERMUTATION_2026-09-10.md.
