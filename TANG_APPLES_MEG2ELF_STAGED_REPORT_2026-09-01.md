# Tang/Apples `qc4wyals` MEG → ADA → ELF staged report

## Evaluation contract

- ELF may deliberately memorize all 2,788 exact ADA/text pairs, including the
  known Birth-of-a-Nation target text.
- The `qc4wyals` MEG2SEM model is fit on 2,652 training rows and selected on
  110 validation rows.
- The 26 Birth-of-a-Nation MEG vectors are absent from interface/E2E fitting
  and validation selection. They will be evaluated once, after a single
  checkpoint is locked.
- Primary selection metric: validation content-word F1; secondary metrics are
  word F1, BERTScore F1, WER, generated-text retrieval, and semantic-interface
  stability.

## Fixed components

- MEG2SEM W&B run: `qc4wyals`
- MEG2SEM run name:
  `tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82`
- MEG2SEM checkpoint:
  `/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt`
- Clean exact-ADA ELF-B run:
  `tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901`
- Selected exact-ADA checkpoint: step 43,000, available as the run's `best.pt`.
- Generation: 32 diffusion steps, CFG 1.0, ten-word targets.

## Stage 1 — exact ADA → memorized ELF ceiling

### All 2,788 known text rows

| Metric | Value |
|---|---:|
| Word precision / recall / F1 | 0.8626 / 0.8559 / **0.8576** |
| Content precision / recall / F1 | 0.8689 / 0.8571 / **0.8601** |
| WER | **0.1807** |
| Exact sentence match | 0.3583 |

### Same 110-row validation slice used below

| Metric | Value |
|---|---:|
| Word precision / recall / F1 | 0.8967 / 0.8858 / **0.8894** |
| Content precision / recall / F1 | 0.8999 / 0.8954 / **0.8954** |
| WER | **0.1371** |
| Exact sentence match | 0.4036 |
| Generated Top-1 / Top-5 | 1.0000 / 1.0000 |
| Latent Top-1 / Top-5 | 1.0000 / 1.0000 |
| Rescaled RoBERTa-large BERTScore P / R / F1 | 0.8021 / 0.8108 / **0.8066** (seed 49) |

Interpretation: the deliberate ADA-to-text memorization ceiling is strong.
The remaining failure cannot be attributed mainly to ELF capacity.

## Stage 2 — raw `qc4wyals` MEG predictions → identical frozen ELF

The generation values below are means across fixed seeds 49, 101, 202, 303,
and 404. The semantic-input and diffusion-latent retrieval values are
deterministic for the supplied condition and checkpoint.

| Metric | Raw MEG pipeline | Retained from exact-ADA ceiling |
|---|---:|---:|
| Word precision / recall / F1 | 0.0458 / 0.0465 / **0.0460** | 5.17% F1 |
| Content precision / recall / F1 | 0.0179 / 0.0175 / **0.0174** | 1.95% F1 |
| WER | **1.0415** | +0.9044 absolute error |
| BERTScore precision / recall / F1 | -0.0980 / -0.0744 / **-0.0851** | -0.8916 F1 absolute |
| Generated Top-1 / Top-5 | 0.0455 / 0.1418 | 4.55% / 14.18% |
| Generated mean / median rank | 42.99 / 36.60 | +41.99 / +35.60 worse |
| Diffusion-latent Top-1 / Top-5 | 0.0545 / 0.1364 | 5.45% / 13.64% |
| Diffusion-latent mean / median rank | 37.83 / 33.00 | +36.83 / +32.00 worse |

Before text generation the same raw predictions still have validation cosine
`0.7174`, semantic Top-1 `0.2909`, Top-5 `0.4545`, and nDCG `0.4877`.
Therefore the semantic signal is real, but the heavily memorized ELF oracle is
very brittle to the off-manifold prediction distribution.

### Strict raw-Pareto ELF interface

The robust flat-context checkpoint at step 11,250 is the current
retrieval-safe winner:

`/data/engs-pnpl/glandau/elf-runs/tang_apples_qc4wyals_robust_residualaug_adapter_from_tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901_robustrebuild_finalbest43k_ev250_topk1_seed49_20260901/checkpoints/eval_step_00011250_score_0.406216.pt`

| Metric | Raw `qc4wyals` | Robust flat context | Directional result |
|---|---:|---:|---|
| Word F1 | 0.0460 | **0.0514** | better |
| Content F1 | 0.0174 | **0.0214** | better |
| WER | 1.0415 | **1.0198** | better |
| BERTScore F1 | -0.0851 | **-0.0743** | better |
| Generated Top-1 / Top-5 | 0.0455 / 0.1418 | **0.0491 / 0.1618** | both better |
| Generated mean / median rank | 42.99 / 36.60 | **41.16 / 35.60** | both better |
| Diffusion-latent Top-1 / Top-5 | 0.0545 / 0.1364 | **0.0727 / 0.1636** | both better |
| Diffusion-latent mean / median rank | 37.83 / 33.00 | **37.23 / 32.00** | both better |
| Semantic ADA cosine / Top-1 / Top-5 | 0.7174 / 0.2909 / 0.4545 | 0.7174 / 0.2909 / 0.4545 | exactly tied |

These are five-seed means for stochastic generation and deterministic values
for the semantic and latent interfaces. The flat adapter's
`project_semantic()` is the identity, verified by byte-identical exported
validation arrays, so it cannot inflate or damage the original MEG2SEM
retrieval. It learns only how ELF converts the unchanged ADA vector into
conditioning context. This checkpoint therefore improves every measured
downstream criterion while exactly retaining the raw semantic retrieval.

The overnight objective is now to widen this margin. Saved-checkpoint and
continuation sweeps are selected on validation only; the 26 protected MEG
test vectors remain unopened.

## Stage 3 — validation-selected residual adapter

The stable interface checkpoint is:

`/data/engs-pnpl/glandau/elf-runs/tang_apples_qc4wyals_pred_residual_identity_curriculum_from_tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901_finalbest43k_ev200_topk_short_seed49_20260901/checkpoints/eval_step_00001800_score_0.408155.pt`

It was fit on the 2,652 MEG-training predictions only. `qc4wyals` and all ELF
parameters remained frozen; only the `1536 -> 1536` identity-residual semantic
adapter was trainable. Training used a 20-epoch semantic-only curriculum,
followed by lightly weighted diffusion and decoder losses. The 110 validation
rows selected the checkpoint and inference recipe; no protected test MEG row
was loaded.

### Complete raw-to-adapter comparison

| Metric | Raw, 32 steps | Adapter, 32 steps | Adapter, selected 64 steps |
|---|---:|---:|---:|
| Word precision | 0.0458 | 0.0720 | **0.0755** |
| Word recall | 0.0465 | 0.0691 | **0.0733** |
| Word F1 | 0.0460 | 0.0701 | **0.0738** |
| Content precision | 0.0179 | 0.0233 | **0.0260** |
| Content recall | 0.0175 | 0.0201 | **0.0233** |
| Content F1 | 0.0174 | 0.0213 | **0.0242** |
| WER | 1.0415 | **1.0120** | 1.0195 |
| BERTScore precision | -0.0980 | -0.0797 | **-0.0763** |
| BERTScore recall | -0.0744 | -0.0715 | **-0.0695** |
| BERTScore F1 | -0.0851 | -0.0746 | **-0.0719** |
| Generated Top-1 | 0.0455 | 0.0291 | **0.0473** |
| Generated Top-5 | 0.1418 | 0.1618 | **0.1709** |
| Generated mean rank | 42.99 | **40.48** | 41.26 |
| Generated median rank | 36.60 | 32.00 | **30.80** |
| Diffusion-latent Top-1 | **0.0545** | 0.0182 | 0.0182 |
| Diffusion-latent Top-5 | 0.1364 | **0.1545** | 0.1455 |
| Diffusion-latent mean rank | **37.83** | 42.54 | 42.52 |
| Diffusion-latent median rank | **33.00** | 39.00 | 39.00 |

Relative to raw input, the selected 64-step recipe improves word F1 by
`60.4%`, content F1 by `38.9%`, generated Top-5 by `20.5%`, and BERTScore F1
by `+0.0131` absolute. WER decreases by `0.0220`. Generated Top-1 changes only
from `0.0455` to `0.0473`; it is not a meaningful improvement.

The adapter is a decoder-interface correction rather than a globally better
ADA decoder. Its matched ADA cosine improves from `0.7174` to `0.7490`, but
ADA-space Top-1/Top-5 decline from `0.2909/0.4545` to `0.1545/0.3727`.
Likewise, diffusion-latent retrieval does not improve consistently even though
free-text overlap, BERTScore, and generated-text Top-5 do. This distinction is
why all three retrieval layers are reported separately.

The sampler result is also sharply localized: 56 steps gives word/content F1
`0.0477/0.0200`; 64 steps with CFG `1.25` gives `0.0472/0.0199`; and 64 steps
with CFG `1.5` gives `0.0467/0.0215`. Thus the fixed winning inference recipe
is 64 steps and CFG `1.0`, selected from five-seed validation means rather than
per-example reranking.

Machine-readable full table:
`tang_apples_qc4wyals_raw_vs_adapter_val110_2026-09-01.csv`.

### Five-seed controls that did not beat the adapter

| Control | Word F1 | Content F1 | WER | Generated Top-1 / Top-5 | Latent Top-1 / Top-5 |
|---|---:|---:|---:|---:|---:|
| Robust residual-augmentation adapter, step 11,250 | 0.0514 | 0.0214 | 1.0198 | 0.0491 / 0.1618 | **0.0727 / 0.1636** |
| Robust residual augmentation + ELF LoRA-8, step 500 | 0.0535 | 0.0193 | 1.0247 | 0.0455 / 0.1436 | 0.0545 / 0.1818 |
| Robust residual augmentation + full ELF, step 750 snapshot | 0.0540 | 0.0196 | 1.0256 | 0.0455 / 0.1564 | 0.0364 / **0.1727** |
| Robust residual augmentation + full ELF, final selected step 10,000 | 0.0527 | 0.0148 | 1.0282 | 0.0364 / 0.1636 | 0.0545 / **0.2000** |
| Train-only Procrustes, alpha 0.25, no learned ELF interface | 0.0504 | 0.0143 | 1.0382 | 0.0345 / 0.1327 | 0.0636 / 0.1636 |
| Mixed exact/predicted adapter, step 1,600 | 0.0473 | 0.0190 | **0.9996** | not selected | not selected |
| Oracle-step-33k ELF-wide LoRA-8 | 0.0565 | 0.0183 | 1.0215 | not selected | not selected |
| Strong-content LoRA-8, step 1,250, 16 steps/CFG 0.5 | 0.0427 | 0.0098 | not selected | not selected | not selected |

The robust adapter is especially informative: it has the strongest
diffusion-latent Top-1/Top-5 in this table but weaker generated content than
the selected residual adapter. This independently confirms that latent
retrieval and useful free generation are not interchangeable objectives.

The full-ELF step-750 snapshot looked promising on the training process's
single validation draw (`0.0280` content F1), so it was frozen and evaluated
across all five fixed generation seeds before the parent run completed. Its
mean fell to `0.0196`; BERTScore F1 was `-0.0829`, also below the selected
residual adapter's `-0.0719`. This candidate is therefore rejected rather
than promoted from a favorable single diffusion draw.

The completed full-ELF run later selected step 10,000 from an even better
single draw (`0.0291` content F1). Its five-seed content mean fell further to
`0.0148`, despite the strongest diffusion-latent Top-5 in this comparison
(`0.2000`). BERTScore F1 was `-0.0778`. The completed LoRA-8 arm was likewise
rejected: its five-seed word/content F1 were `0.0535/0.0193`, BERTScore F1 was
`-0.0809`, and generated Top-1/Top-5 were `0.0455/0.1436`.

On the aligned seed-49 generations, a 50,000-resample paired bootstrap gives
the selected 64-step adapter a word-F1 delta of `+0.0231`, 95% CI
`[+0.0079, +0.0383]`, and probability of improvement `0.9987`. The content-F1
delta is `+0.0044`, CI `[-0.0082, +0.0171]`, probability `0.7572`; the
BERTScore-F1 delta is `+0.0030`, CI `[-0.0142, +0.0201]`. Thus word overlap is
the clearest row-level improvement, while content and BERT improvements are
positive in the five-seed means but not yet conclusive on one paired seed.

### Gap accounting

Using the five-seed means, exact ADA reaches word/content F1
`0.8894/0.8954`; raw MEG reaches `0.0460/0.0174`; and the selected adapter
reaches `0.0738/0.0242`. Therefore:

- raw MEG introduces an additional word-F1 loss of `0.8434` and content-F1
  loss of `0.8780` after the already-trained ELF oracle;
- the adapter recovers `0.0278` word F1, or `3.30%` of that raw-interface gap;
- it recovers `0.0068` content F1, or only `0.77%` of the content gap;
- it removes `0.0220`, or `2.43%`, of the raw pipeline's excess WER over the
  exact-ADA oracle.

So the adapter is a measurable improvement, but most of the MEG-to-exact-ADA
distribution gap remains unresolved.

## Assumption traceback

The best exact-ADA checkpoint is not automatically the most robust brain
checkpoint. A raw-MEG sweep across the same oracle trajectory found:

| Oracle step | Word F1 | Content F1 | WER |
|---:|---:|---:|---:|
| 5,000 | 0.0397 | 0.0147 | 1.2700 |
| 16,000 | **0.0554** | 0.0155 | 1.0655 |
| 25,000 | 0.0450 | 0.0189 | 1.0500 |
| 33,000 | 0.0486 | **0.0217** | **1.0282** |
| 43,000 | 0.0434 | 0.0181 | 1.0445 |

Step 33,000 is therefore being tested as a more content-robust initialization
for frozen-MEG + ELF LoRA adaptation.

## Stage 4 — final protected evaluation

Pending. The current retrieval-safe five-seed leader is the robust flat-context
step-11,250 checkpoint documented above. The residual step-1,800 adapter with
64 diffusion steps remains the text-overlap leader, but it is not eligible for
the all-metric objective because it degrades semantic and diffusion-latent
retrieval. The protected evaluation remains sealed while the overnight
validation program runs.

### Overnight training outcome (2026-09-02)

All five continuations completed all 12,600 scheduled updates; none was
early-stopped.  Their best *single-seed training-trajectory draws* are shown
below.  The entries within a row generally occur at different checkpoints,
so this is a capacity/trajectory diagnostic, not a claim that one checkpoint
simultaneously achieved every value.

| Arm | W&B | Best word F1 (step) | Best content F1 (step) | Best WER (step) | Best generated Top-5 (step) |
|---|---|---:|---:|---:|---:|
| context-preserving adapter | `vrcxgfh5` | 0.064260 (8,000) | 0.025113 (1,000) | 1.018182 (1,750) | 0.181818 (1,000) |
| context + strong content losses | `1pxdtq03` | 0.063721 (10,000) | 0.022783 (4,750) | 1.017273 (3,000) | 0.163636 (7,250) |
| frozen adapter + ELF LoRA-4 | `z7r0vwso` | 0.060595 (750) | 0.025229 (750) | 1.018182 (500) | **0.190909** (8,500) |
| frozen adapter + last two ELF blocks | `tofw3dos` | 0.061970 (12,000) | 0.024766 (6,000) | **1.015455** (7,250) | **0.190909** (10,500) |
| context adapter + ELF LoRA-4 | `z3cti6w7` | **0.064317** (7,750) | **0.026500** (3,500) | 1.016364 (1,500) | 0.172727 (7,500) |

The trajectories genuinely plateaued: the best content checkpoints occurred
at steps 750--6,000, not in the final 15%, and later validation draws did not
establish a rising content frontier.  The one late word maximum at step 12,000
was accompanied by weaker content and is retained for the fixed-cache screen.

The fresh-stock control (`pgbqdaly`) also completed its entire 42,000-update
schedule.  Its validation-selected checkpoint was decisively worse across
five seeds: word F1 `0.017783`, content F1 `0.005778`, WER `1.379455`,
BERTScore F1 `-0.200032`, generated Top-1/Top-5 `0.014545/0.060000`, and
latent Top-1/Top-5 `0.009091/0.067273`.  This rejects the assumption that a
fresh ELF-B can jointly learn the exact-text memory and noisy MEG interface
from this mixture.  Starting from the completed exact-ADA memory is essential.

The training evaluation used one stochastic generation draw.  Consequently,
the 29 unique checkpoints retained by the per-metric manifests are being
re-evaluated with the immutable target-latent cache and seed 49 in ARC array
job `8705196`.  Only a fixed-cache Pareto candidate will receive five-seed
BERTScore evaluation.  The active ARC maintenance reservation delayed this
screen; Slurm currently projects 2026-09-03 00:20 BST unless a GPU frees
earlier.  Until that screen completes, robust step 11,250 remains the only
confirmed all-metric winner.

### Information-loss localization

As a diagnostic upper bound, each raw validation MEG prediction was matched
against the 110 exact ADA vectors for the same known-text validation set, and
the nearest vector's sentence was returned.  This is closed-set retrieval,
not free generation, and therefore is not a competing decoder.  It reaches:

| Diagnostic | Top-1 / Top-5 | Word F1 | Content F1 | WER |
|---|---:|---:|---:|---:|
| raw MEG ADA -> nearest known exact ADA text | 0.290909 / 0.454545 | 0.348182 | 0.302932 | 0.704545 |
| raw MEG ADA -> nearest training-only exact ADA text | not paired-set retrieval | 0.066364 | 0.016243 | 0.990909 |
| robust step 11,250 free diffusion (five seeds) | generated 0.049091 / 0.161818 | 0.051372 | 0.021402 | 1.019818 |

This sharpens the diagnosis.  When the candidate text memory is available,
the unchanged MEG vector contains far more target-specific information than
ELF currently extracts in free generation.  The dominant remediable loss is
therefore the brittle ADA-to-ELF context/generation interface.  The weak
training-only nearest-neighbour content score also shows that the ADA geometry
is not a general lexical sentence space; an effective solution must exploit
the known exact-ADA/text memory without turning the final protected decision
into test-set selection.

### Overnight protocol

The overnight search uses the following fixed rules:

1. Every candidate is compared with raw `qc4wyals` and robust step 11,250.
2. All candidates use the same immutable validation target-latent cache.
   Duplicate fixed-cache evaluations of step 11,250 produced identical
   generations and metrics, confirming the corrected evaluation contract.
3. A candidate must improve word/content F1, BERTScore, WER, generated-text
   retrieval, and diffusion-latent retrieval without reducing raw semantic
   retrieval. A flat context adapter ties the latter exactly by construction.
4. Training runs their complete scheduled 12,600 updates. A run is extended
   if its best checkpoint lies in the final 15% of updates or its smoothed
   validation frontier is still improving; it is not stopped because of one
   noisy diffusion sample.
5. Screening uses validation seed 49. Only Pareto candidates receive the full
   five-seed evaluation and BERTScore computation.
6. Model selection uses validation only. The 26 protected MEG vectors are
   opened once, after one checkpoint and sampler are locked.

The first continuation array is ARC job `8699922`:

| Task | Variant | Trainable components | Outcome |
|---:|---|---|---|
| 14 | context-preserving continuation | flat context adapter | completed 12,600 |
| 15 | context + stronger content losses | flat context adapter | completed 12,600 |
| 16 | frozen adapter + LoRA-4 | ELF LoRA only | completed 12,600 |
| 17 | frozen adapter + final two ELF blocks | two ELF blocks | completed 12,600 |
| 18 | context + LoRA-4 | context adapter and ELF LoRA | completed 12,600 |

Each run starts from robust step 11,250, uses exact/text rows plus train-only
empirical MEG residual augmentation, and excludes protected test predictions.
The context-preserving arms directly target the largest observed failure:
raw ADA retrieval remains useful, but the frozen exact-ADA ELF context is too
brittle to the predicted-vector neighbourhood.

### Fixed-cache sampler screen

The first inference-only screen held the checkpoint, validation examples,
seed (`49`), and target-latent cache fixed.  This isolates the diffusion
sampler from training and cache-construction noise.  The step-11,250 baseline
uses 32 steps and CFG 1.0.

| Steps | CFG | Word F1 | Content F1 | WER | Generated Top-1 / Top-5 | Generated mean / median rank | Latent Top-1 / Top-5 | Latent mean / median rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.5 | 0.051916 | 0.014760 | **1.009091** | 0.036364 / **0.181818** | 45.609 / 44 | 0.072727 / 0.163636 | 37.227 / 32 |
| 16 | 1.0 | **0.056338** | 0.025204 | 1.020000 | **0.054545** / 0.172727 | 40.382 / 33 | 0.072727 / 0.163636 | 37.227 / 32 |
| 24 | 0.75 | 0.052978 | 0.013995 | 1.011818 | 0.018182 / 0.100000 | 41.655 / 37 | 0.072727 / **0.172727** | **37.200** / 32 |
| **32** | **1.0** | 0.054218 | 0.026719 | 1.027273 | 0.045455 / 0.172727 | **40.036 / 30** | 0.072727 / 0.163636 | 37.227 / 32 |
| 48 | 1.0 | 0.053412 | **0.027073** | 1.023636 | 0.045455 / 0.154545 | 40.882 / 32 | 0.072727 / 0.163636 | 37.227 / 32 |
| 64 | 1.0 | 0.050290 | 0.025454 | 1.024545 | 0.045455 / 0.154545 | 41.636 / 41 | 0.072727 / 0.163636 | 37.227 / 32 |
| 64 | 0.75 | 0.045189 | 0.016307 | 1.020909 | 0.027273 / 0.136364 | 43.218 / 36 | 0.072727 / 0.163636 | 37.227 / 32 |
| 20 | 1.0 | 0.053349 | 0.025520 | 1.020000 | 0.045455 / 0.163636 | 40.873 / 34 | 0.072727 / 0.163636 | 37.227 / 32 |
| 24 | 1.0 | 0.051528 | 0.024022 | 1.022727 | 0.045455 / 0.145455 | 40.800 / 35 | 0.072727 / 0.163636 | 37.227 / 32 |
| 28 | 1.0 | 0.052358 | **0.026935** | 1.023636 | 0.045455 / 0.172727 | 40.882 / 34 | 0.072727 / 0.163636 | 37.227 / 32 |
| 36 | 1.0 | 0.052453 | 0.024122 | 1.025455 | 0.045455 / **0.181818** | 40.118 / 32 | 0.072727 / 0.163636 | 37.227 / 32 |
| 40 | 1.0 | 0.050404 | 0.026736 | 1.024545 | 0.045455 / 0.163636 | 41.618 / 37 | 0.072727 / 0.163636 | 37.227 / 32 |
| 32 | 0.9 | 0.050429 | 0.014410 | 1.020909 | 0.027273 / 0.145455 | 43.364 / 39 | 0.072727 / 0.172727 | 37.200 / 32 |

No tested sampler strictly dominates the 32-step baseline.  For example,
28 steps gives the best content F1, but loses word F1 and generated mean rank;
36 steps improves generated Top-5, but loses both overlap scores.  The sampler
remains fixed at 32 steps/CFG 1.0 unless a training candidate later shows a
different five-seed optimum.

Those jobs were moved in place from the reservation-delayed `long` partition
to `short`, preserving their exact environments, and their reservations were
reduced from 12 hours to a conservative 4 hours so they can backfill overnight:

| Arm | Training task | Projected start at scheduling | Automatic evaluator |
|---|---:|---:|---:|
| Robust residual augmentation + ELF-wide LoRA-8 | `8698337_6` | 2026-09-01 22:14 BST | `8699545` |
| Robust residual augmentation + full ELF | `8698338_10` | 2026-09-01 22:48 BST | `8699546` |
| Fresh stock ELF-B + robust mixture + full ELF | `8699157_10` | 2026-09-01 23:41 BST | `8699547` |

Each dependent evaluator resolves the arm's saved `best.pt`, generates the
same five validation seeds, computes BERTScore, and writes one multiseed
summary JSON. A training timeout can still be evaluated if it left a valid
`best.pt`; a failed run without a checkpoint is rejected explicitly.

After these summaries exist, one checkpoint will be chosen from validation
only and run once on the 26 protected Birth-of-a-Nation MEG rows. This section
will then contain its checkpoint, complete metrics, deltas against stages
1–2, BERTScore, and target/generated examples.
