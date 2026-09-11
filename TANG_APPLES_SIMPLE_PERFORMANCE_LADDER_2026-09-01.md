# Tang/Apples MEG-to-text: simple performance ladder

## What this ladder measures

Use one fixed set of 110 Apple validation sentences and one fixed ELF
checkpoint. Change only the source of its 1,536-dimensional ADA condition.
This separates the two main questions:

1. Can the overfit ELF reconstruct the text when it receives the exact ADA
   embedding?
2. How much does generation degrade when exact ADA is replaced by the output
   of the validation-selected `qc4wyals` MEG2SEM model?

Only after measuring that gap do we train an adapter, change ELF, or fine-tune
the MEG model.

## Fixed evaluation set and decoder

- Evaluation split: all 110 `qc4wyals` Apple validation rows.
- Exact semantic representation: normalized `text-embedding-ada-002`, 1,536
  dimensions.
- MEG representation: normalized ADA prediction from `qc4wyals`.
- ELF checkpoint for the current ladder: the immutable fresh-full ELF-B
  snapshot selected after the requested waiting window at step 35,000.
- Sampling: 32 diffusion steps, CFG 1.0, seed 49.
- No fitting occurs in either rung 1 or rung 2.

The 26 final Birth-of-a-Nation MEG predictions remain protected and are not
used in this validation comparison.

## Clean exact-ADA ceiling rerun

The `0.0751` word-F1 result below came from the earlier exact-ADA LoRA-8
checkpoint. A cleaner and substantially longer memorization control is now
running and will supersede that ceiling if successful:

- training ARC job: `8696138`;
- dependent all-2,788 evaluation: `8696143`;
- W&B run: `zyo6he48`;
- run name:
  `tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901`;
- initialization: stock pretrained ELF-B plus a fresh ADA adapter;
- trainable: all ELF-B parameters and the complete adapter; no LoRA;
- data: all 2,788 exact ADA/ten-word pairs, deliberately including the 136
  audit rows;
- schedule: 1,000 epochs / 44,000 updates, constant LR `1e-4`, no early stop;
- selection: lowest WER on the first 136 known training rows;
- final audit: free generation on all 2,788 known training rows.

Automatic downstream jobs are already chained:

- `8696168`: frozen exact-ADA versus raw-qc4wyals evaluation on the same 110
  validation rows using the newly selected `best.pt`;
- `8696169`: adapter/residual/robust/mixed/ELF-LoRA interface matrix, released
  only after both the all-2,788 audit and the exact/raw validation comparison
  complete successfully.

The first 136-row audit at step 1,000 reports WER `1.1765`, word F1 `0.0438`,
content-word F1 `0.0110`, generated-text Top-5 `0.1029`, and diffusion-latent
Top-1/Top-5 `0.022/0.081`. This is already better than the older ceiling on
WER, but not yet on overlap or retrieval; it is an early checkpoint at roughly
23 of the planned 1,000 epochs.

After the requested additional waiting window, step 35,000 was frozen as the
interim oracle while the original job continued toward step 44,000:

- checkpoint:
  `/data/engs-pnpl/glandau/elf-runs/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/checkpoints/eval_step_00035000_score_0.773228.pt`;
- 136-row word P/R/F1: `0.8426 / 0.8404 / 0.8388`;
- 136-row content P/R/F1: `0.8485 / 0.8452 / 0.8436`;
- WER: `0.2074`;
- exact sentence match: `0.2353`;
- generated-text Top-1/Top-5: `1.0 / 1.0`;
- diffusion-latent Top-1/Top-5: `1.0 / 1.0`.

Immediate evaluations from this immutable snapshot:

- `8697436`: exact ADA free generation on all 2,788 known rows (batch-8 rerun
  after the original batch-64 evaluation exhausted GPU memory);
- `8697334`: exact ADA versus raw qc4wyals on the same 110 validation rows,
  completed;
- `8697437`, `8697343`, and `8697344`: post-hoc BERTScore jobs;
- `8697337`: new-oracle MEG-interface matrix.

The corrected chunked all-2,788 audit completed as `8697577` (the evaluator
previously ignored its generation batch size and attempted a 30.3-GiB logits
allocation). Across the complete known-text corpus, step 35,000 reaches word
F1 `0.8022`, content-word F1 `0.8006`, WER `0.2520`, and exact-sentence match
`0.2245`. The smaller 110-row validation slice is easier and yields the higher
numbers tabulated below.

This is a memorization-capacity ceiling, not a held-out generalization result.
After it selects a checkpoint, that exact checkpoint must be rerun on exact
ADA and raw qc4wyals validation inputs with identical sampling to replace
rungs 1 and 2 below.

## Ladder

| Rung | Input to frozen ELF | What it isolates | Status |
|---|---|---|---|
| 1. Exact-ADA ceiling | True ADA vector for each validation sentence | ELF/ADA-to-text reconstruction error | Complete: ARC `8695645_0` |
| 2. Raw-MEG pipeline | Raw `qc4wyals` predicted ADA vector | Additional MEG2SEM distribution/interface error | Complete: ARC `8695645_1` |
| 3. Improved interface | Calibrated/adapter/robustly trained MEG condition | Recoverable interface error | Queued after rungs 1–2 |
| 4. E2E adaptation | Adapter, ELF LoRA/blocks, and optionally MEG2SEM | Remaining jointly recoverable error | Queued after rung 3 selection |

## Final step-43,000 results on all 110 validation rows

### Generated text

| Metric | Exact ADA → ELF | Raw MEG2SEM → ELF | Raw retention / change |
|---|---:|---:|---:|
| Word precision | 0.9015 | 0.0434 | 4.81% retained |
| Word recall | 0.8891 | 0.0436 | 4.91% retained |
| Word F1 | **0.8931** | **0.0434** | **4.86% retained** |
| Content-word precision | 0.9013 | 0.0193 | 2.14% retained |
| Content-word recall | 0.9005 | 0.0175 | 1.94% retained |
| Content-word F1 | **0.8985** | **0.0181** | **2.02% retained** |
| WER | 0.1336 | 1.0445 | +0.9109 worse |
| Generated-text Top-1 | 1.0000 | 0.0364 | 3.64% retained |
| Generated-text Top-5 | 1.0000 | 0.1364 | 13.64% retained |
| Generated-text mean rank | 1.00 | 41.97 | +40.97 worse |
| Generated-text median rank | 1 | 33 | +32 worse |
| BERTScore precision (rescaled) | 0.8021 | -0.1015 | -0.9036 |
| BERTScore recall (rescaled) | 0.8108 | -0.0692 | -0.8800 |
| BERTScore F1 (rescaled) | **0.8066** | **-0.0842** | -0.8908 |

### Diffusion latent retrieval

This ranks each target T5 latent by the frozen diffusion loss before converting
the latent to free text.

| Metric | Exact ADA → ELF | Raw MEG2SEM → ELF | Raw retention / change |
|---|---:|---:|---:|
| Top-1 | 1.0000 | 0.0545 | 5.45% retained |
| Top-5 | 1.0000 | 0.1364 | 13.64% retained |
| Mean rank | 1.00 | 37.83 | +36.83 worse |
| Median rank | 1 | 33 | +32 worse |

## Current diagnosis

The diagnosis is now much cleaner:

1. **ELF/ADA-to-text is a strong memorization oracle.** On exact ADA, the
   frozen step-43,000 decoder reaches `0.8931` word F1, `0.8985` content-word
   F1, WER `0.1336`, and perfect generated-text and latent retrieval.
2. **The raw MEG-to-ELF interface is the dominant bottleneck.** Replacing only
   exact ADA with `qc4wyals` predictions reduces word F1 to `0.0434` and
   content-word F1 to `0.0181`; latent Top-1 falls from `1.0` to `0.0545`.
   Rescaled RoBERTa-large BERTScore F1 falls from `0.8066` to `-0.0842`.

The strongest validation improvement available at the time of this update is
the true-MEG frozen-`qc4wyals` + trainable projector + ELF-wide rank-8 LoRA
checkpoint at step 3,250. It reaches word F1 `0.0521`, content-word F1
`0.02834`, WER `0.9909`, and generated Top-1/Top-5 `0.0727/0.1727`. This is a
`56.3%` relative content-word gain over the raw pipeline, but it is still a
validation candidate rather than the final model. Mixed exact/predicted,
empirical-residual robustness, BatchNorm-safe E2E, and full-ELF adaptation
arms are still being evaluated before one checkpoint is locked.

The next experiments therefore hold the successful decoder fixed first and
learn only the train-split MEG-to-ELF correction. Restricted ELF LoRA/last
blocks and later true MEG2SEM E2E tuning are measured only after that adapter
baseline, so each recovery can be attributed.

A two-epoch diagnostic on cached train predictions already shows that the gap
is recoverable without touching the protected validation rows: the small
identity-residual correction improves raw word F1 `0.0495 -> 0.0578`, content
F1 `0.0150 -> 0.0211`, and WER `1.0309 -> 0.9845`. This is only a screen from
the interim step-35,000 oracle, not the final selected model.

## Simple gap accounting

Taking exact target text as F1 `1.0` and WER `0.0`:

| Quantity | Lost before MEG (exact ADA → ELF) | Extra loss from raw MEG input |
|---|---:|---:|
| Word F1 | 0.1069 | 0.8497 |
| Content-word F1 | 0.1015 | 0.8804 |
| WER above zero | 0.1336 | 0.9109 |

This decomposition says the clean diffusion/text decoder is no longer the
main failure: the raw brain-to-condition interface creates most of the final
loss and preserves only 2.02% of the exact-ADA content-word F1.

## Improvement stages and what is trainable

| Stage | MEG2SEM | ADA correction / semantic adapter | ELF |
|---|---|---|---|
| 1. Exact ceiling | Not used | Frozen after exact-ADA training | Frozen after exact-ADA LoRA training |
| 2. Raw pipeline | Frozen `qc4wyals` predictions | Frozen exact-ADA adapter | Frozen exact-ADA ELF |
| 3a. Residual correction | Frozen | Train zero-initialized `1536→1536` residual only | Frozen |
| 3b. Adapter-only | Frozen | Train semantic adapter on MEG-train predictions | Frozen |
| 3c. Robust/mixed adapter | Frozen | Train on exact ADA + MEG predictions + train-only residual noise | Frozen |
| 4a. Restricted E2E | Frozen | Train | Train ELF LoRA or last two blocks |
| 4b. Full-chain E2E | Train last MEG2SEM layers at a very low LR | Train | Train ELF LoRA |

Every stage is selected on the 110 validation rows. The final 26 brain vectors
are evaluated once only after choosing a validation winner. The reported delta
from stage 2 to stage 3 measures adapter recovery; stage 3 to stage 4 measures
the additional value of joint diffusion/ELF or full-chain training.

## Qualitative outputs

- Exact-ADA validation generations:
  `tang_apples_val110_exact_ada_to_elf_target_generated_2026-09-01.csv`
- Raw-qc4wyals validation generations:
  `tang_apples_val110_raw_qc4wyals_to_elf_target_generated_2026-09-01.csv`
- Row-aligned target/exact-ADA/raw-MEG comparison:
  `tang_apples_val110_exact_vs_raw_meg_to_elf_2026-09-01.csv`

## Metrics to report at every rung

- Semantic input cosine, Top-1, Top-5, nDCG, rank, and prediction variance.
- Generated-text semantic Top-1, Top-5, mean/median rank.
- Word precision, recall, and F1.
- Content-word precision, recall, and F1.
- WER.
- Rescaled RoBERTa-large BERTScore precision, recall, and F1.
- Target/generated CSV and ranked qualitative examples.

## How to read the gaps

- `perfect text - rung 1`: loss inside the ELF diffusion/decoding stage.
- `rung 1 - rung 2`: loss introduced by replacing exact ADA with raw MEG2SEM
  predictions.
- `rung 3 - rung 2`: recovery from a leakage-safe adapter or calibration.
- `rung 4 - rung 3`: recovery attributable to joint/E2E fine-tuning.

Absolute metric differences and relative retention (`later / earlier`) will be
reported. If rung 1 is weak, improving MEG2SEM cannot fix the main bottleneck;
the ADA-to-text decoder must be repaired first. If rung 1 is strong but rung 2
collapses, the priority is the MEG-to-ELF interface.

## Already-known semantic input gap

Before text generation, the raw `qc4wyals` validation output has:

- cosine: `0.7174`;
- semantic Top-1: `0.2909`;
- semantic Top-5: `0.4545`;
- nDCG: `0.4877`.

These numbers show usable brain-derived ADA signal, but they do not establish
that the frozen diffusion model can turn that signal into text. Rungs 1 and 2
measure that missing link directly.
