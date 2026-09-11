# Tang/Apples ADA → ELF → MEG Progress (2026-09-01)

## Objective

Train an ELF diffusion decoder for exact `text-embedding-ada-002` vectors from
the Tang/TheMoth/Apples corpus, then connect the validation-selected
`qc4wyals` MEG2SEM decoder to it without training on the final 26
Birth-of-a-Nation MEG predictions.

The user explicitly allows the semantic-to-text diffusion model to train on all
corpus text, including the 26 final target sentences. This is therefore a
known-text diffusion test. The protected boundary is the brain interface: the
26 MEG-derived test vectors remain evaluation-only.

## Inputs

- Exact ADA corpus: 2,652 train + 110 validation + 26 test = 2,788 rows.
- Known-text ELF pack:
  `/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1/tang_apples_ada002_knowntext_all2788_testfirst.npz`
- MEG2SEM run: `qc4wyals`
- Run name: `tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82`
- Checkpoint:
  `/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt`
- Previously exported final-test predictions:
  `/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/utils/final_results/tang_themoth_apples_ada002_cosine0831_best_predictions.npz`

## Phase 1: Exact-ADA diffusion ceiling

ARC array job: `8695189`

W&B group: `tang_apples_ada2text_oracle_matrix_20260901`

Six single-checkpoint arms:

1. ELF-B full tuning, standard loss.
2. ELF-B full tuning, content-weighted CE + content recall + content precision.
3. ELF-B LoRA rank 8, content losses.
4. ELF-B final four blocks, content losses.
5. ELF-B semantic-adapter only.
6. ELF-L full tuning, content losses.

All arms use 2,788 exact ADA/text pairs, 32 train/sampling steps, a cosine LR
schedule, and checkpoint selection by content-word overlap. The first 26 rows
provide the deterministic known-text diffusion audit.

Current state at the time of this update:

- arms 0 and 1 running;
- arms 2–5 queued;
- arm 0 passed its data/config audit and began training;
- confirmed dimensions: `(2788, 1536)`;
- confirmed full-tuning parameters: 104,579,940 ELF + 140,549,120 semantic adapter;
- first evaluation is scheduled at step 1,000.

Subsequent live update:

- full-standard reached step 4,000: exact-ADA latent retrieval Top-1/Top-5
  `0.192/0.423`, but generated content-word F1 remained `0.003` and WER
  `1.988`;
- full-content reached step 4,000: retrieval `0.192/0.462`, content-word F1
  `0.012`, WER `1.988`;
- LoRA reached step 2,000: retrieval `0.038/0.192`, word F1 `0.030`,
  content-word F1 `0.003`, WER `1.296`;
- these are warm-up checkpoints, not accepted oracle ceilings. Training is
  continuing toward step 13,200.

Latest oracle update:

- full ELF-B tuning was stopped after it showed a clear failure mode: latent
  retrieval kept improving while free generation collapsed into repeated
  tokens such as `hello`, `sort`, `craft`, and `minimize`;
- the ELF-B LoRA-8 arm is the current exact-ADA leader at step 8,000:
  generated-text T5 Top-1/Top-5 `0.1154/0.5000`, mean rank `7.31`, word F1
  `0.0651`, content-word F1 `0.0496`, and WER `1.3692` on the 26 known texts;
- the later step-11,000 checkpoint gives the best balanced retrieval/WER
  point so far: generated-text Top-1/Top-5 `0.1923/0.5385`, word F1 `0.0607`,
  content-word F1 `0.0335`, and WER `1.2923`;
- post-hoc, baseline-rescaled RoBERTa-large BERTScore F1 is still poor:
  `-0.1295` at step 8,000 and `-0.1337` at step 11,000. This confirms that
  the apparent retrieval improvement has not yet become high-quality language;
- the best LoRA WER so far is `1.2962` at step 2,000, while step 6,000 has the
  best word F1 (`0.0705`). This confirms that checkpoint selection must remain
  multi-metric rather than using training loss alone;
- an exact-26 adapter/LoRA booster initialized from the collapsed full-tuning
  branch did not repair the collapse. A clean pretrained-ELF adapter control
  is running instead.

The oracle is not yet accepted: it proves usable conditioning and retrieval,
but the text is still too degenerate for the requested ADA-to-text ceiling.

## Phase 2: Canonical qc4wyals exports

Exporter:
`scripts/export_qc4wyals_split_predictions.py`

It imports the canonical MEG2SEM checkpoint reconstruction and prediction
functions rather than recreating model logic. It exports normalized and raw
predicted/exact ADA vectors plus aligned sentence/session/run/start metadata and
retrieval metrics for train, validation, and test.

ARC jobs:

- `8695224`: expedited 10-minute V100 attempt, running;
- `8695303`: one-hour generic-GPU fallback, queued;
- `8695204`: four-hour L40S fallback, queued.

The expedited export completed successfully in 8m50s; both redundant fallbacks
were cancelled.

Canonical semantic metrics:

| Split | Rows | Cosine | Top-1 | Top-5 | nDCG |
|---|---:|---:|---:|---:|---:|
| train | 2,652 | 0.7907 | 0.5830 | 0.9570 | 0.8098 |
| validation | 110 | 0.7174 | 0.2909 | 0.4545 | 0.4877 |
| protected test | 26 | 0.7081 | 0.1923 | 0.5385 | 0.4942 |

Only one successful export is needed; redundant queued jobs will be cancelled
after success.

## Phase 3: Leakage-safe interface packs

Builder:
`scripts/build_qc4wyals_elf_interface_packs.py`

Prepared outputs include:

- predicted train + validation-tail pack;
- row-aligned exact ADA target pack;
- standalone protected test-prediction and exact-test packs;
- mixed exact-all + predicted-train + predicted-validation-tail pack;
- robust pack using randomly reassigned empirical train-only MEG residuals at
  scales 0.5, 1.0, and 1.5.

The final 26 predicted MEG vectors are never concatenated into a training pack.

All interface packs were built successfully by ARC job `8695321` in 57
seconds. The empirical-residual robustness pack has 13,506 rows.

## Phase 4: Brain-to-ADA calibration

ARC matrix prepared in:
`submit-jobs/train_qc4wyals_ada_mapper_matrix.sbatch`

Five validation-selected variants:

1. linear;
2. MLP-512;
3. identity-initialized residual linear;
4. identity-initialized residual MLP-512;
5. identity-initialized residual MLP-1536.

Training uses 2,652 train predictions. Early stopping/model selection uses only
the 110 validation predictions. The 26 test predictions are evaluated only
after checkpoint selection.

Quick mapper array `8695332` completed; every arm early-stopped before 120
epochs:

| Mapper | Val Top-1 | Val Top-5 | Val cosine | Test Top-1 | Test Top-5 | Test cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw qc4wyals | 0.2909 | 0.4545 | 0.7174 | 0.1923 | 0.5385 | 0.7081 |
| linear | 0.1636 | 0.4455 | 0.7609 | 0.1923 | **0.6538** | 0.7548 |
| MLP-512 | 0.1545 | 0.4000 | **0.7912** | 0.2308 | 0.5385 | **0.7847** |
| residual linear | 0.2000 | 0.4364 | 0.7647 | 0.2308 | 0.6154 | 0.7560 |
| residual MLP-512 | 0.2455 | **0.4636** | 0.7748 | **0.2692** | 0.5000 | 0.7651 |
| residual MLP-1536 | 0.2182 | 0.4364 | 0.7770 | 0.2308 | 0.5385 | 0.7659 |

No single mapper dominates. Raw predictions preserve validation Top-1;
residual MLP-512 improves protected-test Top-1; linear improves protected-test
Top-5; plain MLP-512 maximizes cosine. All four will be compared through the
same frozen diffusion checkpoint, because semantic retrieval and diffusion
generation need not prefer the same geometry.

Lightweight validation calibration job `8695352` also completed:

- validation-nDCG winner: 64-NN uniform residual correction at alpha 0.5;
- protected-test cosine/Top-1 after that correction: `0.7640 / 0.2308`;
- a diagonal-affine cosine winner reached test cosine `0.8599` but collapsed
  variance and hurt retrieval, so it is retained only as a collapse control.

## Phase 5: ELF connection matrix

After the exact-ADA oracle winner is identified, compare:

1. zero-shot frozen oracle on raw qc4wyals predictions;
2. frozen oracle after validation-selected linear/ridge calibration;
3. frozen oracle after validation-selected residual MLP calibration;
4. semantic-adapter-only end-to-end tuning on predicted train rows;
5. ELF LoRA end-to-end tuning;
6. limited final-block ELF tuning;
7. mixed exact-ADA + predicted-train training;
8. empirical train-residual robustness augmentation;
9. staged semantic-alignment curriculum followed by diffusion/text loss.

No training or hyperparameter selection will use the protected 26 MEG test
vectors.

## Phase 6: T5 teacher-context traceback

The original flat adapter directly learns a `1536 -> 64 x 512` mapping with
about 140.5M trainable parameters. This conflates two questions: whether ELF
can reconstruct the Tang/Apples text from a language-aligned context, and
whether ADA can be mapped into that context.

A new diagnostic therefore preserves the exact T5 token-latent sequence as
the ELF condition prefix:

- exporter: `scripts/export_text_t5_context_npz.py`;
- frozen pretrained-ELF diagnostic job: `8695465`;
- output pack:
  `/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1/tang_apples_ada002_knowntext_all2788_t5context.npz`.

If the T5-context ceiling is strong, the next stage will learn a compact
ADA-to-T5-context mapper and then train it on exact ADA plus qc4wyals
train/validation predictions. If the ceiling is weak, the failure is inside
the ELF reconstruction/decode contract and must be fixed before any MEG
interface training is meaningful.

## Evaluation contract

Report both the 110-row validation selection performance and the single final
26-row Birth-of-a-Nation evaluation, including:

- semantic retrieval Top-1/Top-5/nDCG/rank/cosine/variance;
- generated-text word precision/recall/F1;
- generated-text content-word precision/recall/F1;
- WER;
- BERTScore precision/recall/F1;
- target/generated CSV and ranked examples;
- exact-ADA versus MEG-predicted-ADA degradation for each checkpoint.

## Simple 110-row validation ladder

To make the bottlenecks directly attributable, ARC job `8695645` evaluated
the selected merged step-8,000 LoRA checkpoint twice with identical sampling:
first with exact ADA and then with raw `qc4wyals` ADA predictions.

| Metric | Exact ADA | Raw qc4wyals |
|---|---:|---:|
| Diffusion-latent Top-1 / Top-5 | 0.3818 / 0.6909 | 0.0182 / 0.0636 |
| Diffusion-latent mean / median rank | 6.26 / 2 | 44.97 / 39 |
| Generated-text Top-1 / Top-5 | 0.0455 / 0.1909 | 0.0273 / 0.0818 |
| Word P / R / F1 | 0.0681 / 0.0891 / 0.0751 | 0.0356 / 0.0464 / 0.0392 |
| Content-word P / R / F1 | 0.0360 / 0.0485 / 0.0399 | 0.0063 / 0.0075 / 0.0067 |
| WER | 1.3464 | 1.3527 |
| Rescaled BERTScore P / R / F1 | -0.1628 / -0.0573 / -0.1098 | -0.2300 / -0.1126 / -0.1709 |

Thus the exact-ADA diffusion ceiling is itself weak, while the raw brain
interface retains only 16.8% of its content-word F1 and 9.2% of its latent
Top-5. Both stages are bottlenecks; the brain-interface loss is particularly
large for content-specific conditioning.

The complete explanation, run paths, loss interpretation, and local CSV links
are in `TANG_APPLES_SIMPLE_PERFORMANCE_LADDER_2026-09-01.md`.

## Fresh full-ELF exact-ADA memorization replacement

The previous ladder uses a rank-8 LoRA checkpoint and therefore is not the
final answer to the exact-ADA capacity question. A new clean run starts from
stock pretrained ELF-B with a fresh 1,536-D ADA adapter, updates every ELF-B
and adapter parameter, and deliberately trains on all 2,788 known text pairs:

- training job `8696138`, W&B `zyo6he48`;
- 1,000 epochs / 44,000 updates, no early stopping;
- WER selection over the first 136 known training rows;
- dependent all-corpus generation job `8696143`.
- dependent exact-ADA/raw-qc4wyals validation ladder `8696168`;
- new-oracle interface matrix `8696169`, dependent on both ceiling audits.

At step 1,000, the first 136-row known-text audit reached WER `1.1765`, word
F1 `0.0438`, content-word F1 `0.0110`, generated Top-5 `0.1029`, and latent
Top-1/Top-5 `0.022/0.081`. It improves WER early but has not yet surpassed the
old LoRA checkpoint's overlap/retrieval; training continues to step 44,000.

The requested additional waiting window selected immutable step 35,000 as the
interim oracle: word F1 `0.8388`, content-word F1 `0.8436`, WER `0.2074`, exact
sentence match `0.2353`, and perfect generated/latent Top-1 on the 136 known
audit rows. The original training job remains active through step 44,000.

Snapshot evaluation chain:

- all-2,788 exact-ADA generation: batch-8 rerun `8697436` after the original
  batch-64 attempt exhausted GPU memory;
- same-checkpoint exact/raw qc4wyals validation ladder: completed as `8697334`;
- BERTScore: `8697437`, `8697343`, and `8697344`;
- leakage-safe interface matrix: `8697337`.

The completed 110-row ladder using the exact same frozen step-35,000 decoder
and sampling settings is:

| Metric | Exact ADA | Raw qc4wyals | Raw retention |
|---|---:|---:|---:|
| Word P / R / F1 | 0.8482 / 0.8491 / **0.8465** | 0.0494 / 0.0500 / **0.0495** | **5.85%** |
| Content P / R / F1 | 0.8612 / 0.8527 / **0.8543** | 0.0152 / 0.0152 / **0.0150** | **1.76%** |
| WER | **0.2000** | **1.0309** | +0.8309 |
| Generated Top-1 / Top-5 | 1.0000 / 1.0000 | 0.0273 / 0.1091 | 2.73% / 10.91% |
| Latent Top-1 / Top-5 | 1.0000 / 1.0000 | 0.0545 / 0.0909 | 5.45% / 9.09% |
| Generated mean / median rank | 1.00 / 1 | 41.42 / 33 | +40.42 / +32 |

This reverses the earlier diagnosis from the weak LoRA oracle: exact-ADA
diffusion is now strong, and nearly all remaining loss is introduced when raw
brain-predicted ADA is fed into the sharply memorized conditioning interface.
The priority is therefore adapter/distribution correction, followed by
restricted ELF adaptation and only then true MEG2SEM-backprop E2E tuning.

The chunked all-2,788 step-35,000 audit completed with word F1 `0.8022`,
content-word F1 `0.8006`, WER `0.2520`, and exact sentence match `0.2245`.
Generation now honors the configured batch size; the first two audit attempts
failed because all rows were decoded in one logits tensor.

A two-epoch cached-prediction screen gives the first interface recovery:

| Arm | Word F1 | Content F1 | WER | Generated Top-5 |
|---|---:|---:|---:|---:|
| Raw qc4wyals, no fitting | 0.0495 | 0.0150 | 1.0309 | 0.1091 |
| Full flat adapter, 2 epochs | 0.0532 | **0.0226** | 1.0191 | 0.1091 |
| 1536-D identity residual, 2 epochs | **0.0578** | 0.0211 | **0.9845** | **0.1545** |

The original exact-ADA run continued improving after the requested waiting
window and completed all 44,000 steps. The retained winner is step 43,000:
word F1 `0.8811`, content F1 `0.8908`, WER `0.1471`, exact-sentence match
`0.4044`, and perfect generated/latent Top-1 and Top-5 on the 136-row audit.
On all 2,788 known corpus rows, the same `best.pt` reaches word F1 `0.8576`,
content F1 `0.8601`, WER `0.1807`, and exact-sentence match `0.3583`.

The final frozen 110-row validation ladder is:

| Metric | Exact ADA | Raw qc4wyals | Raw retention |
|---|---:|---:|---:|
| Word P / R / F1 | 0.9015 / 0.8891 / **0.8931** | 0.0434 / 0.0436 / **0.0434** | **4.86%** |
| Content P / R / F1 | 0.9013 / 0.9005 / **0.8985** | 0.0193 / 0.0175 / **0.0181** | **2.02%** |
| WER | **0.1336** | **1.0445** | +0.9109 |
| Generated Top-1 / Top-5 | 1.0000 / 1.0000 | 0.0364 / 0.1364 | 3.64% / 13.64% |
| Latent Top-1 / Top-5 | 1.0000 / 1.0000 | 0.0545 / 0.1364 | 5.45% / 13.64% |
| Latent mean / median rank | 1.00 / 1 | 37.83 / 33 | +36.83 / +32 |

Serious final-checkpoint optimization jobs are now submitted:

- cached-prediction adapter/residual/mixed/robust/LoRA/last-block matrix:
  `8697735` (validation every 200 updates, 12-hour short partition);
- true raw-MEG E2E matrix spanning frozen MEG2SEM, trainable MEG2SEM, ELF
  LoRA, and last-two-block tuning: `8697736` (validation every 250 updates,
  12-hour short partition);
- post-hoc BERTScore for the all-2,788 ceiling and final exact/raw validation
  runs: `8697705`, `8697718`, and `8697719`.

True raw-MEG training was also validated end to end in smoke job `8697572`:
the `qc4wyals` Lightning checkpoint reconstructs with no missing/unexpected
keys, raw MEG is passed through MEG2SEM and the ADA projector into ELF, and
gradients can be restricted to the projector, extended into MEG2SEM, or sent
through ELF LoRA/selected final blocks. Frozen submodules are explicitly kept
in inference mode so BatchNorm statistics and dropout cannot silently drift.

Once complete, the selected checkpoint will be evaluated in this order:

1. all 2,788 exact-ADA training rows for the memorization ceiling;
2. exact ADA on the 110 validation rows;
3. raw qc4wyals ADA predictions on those same 110 rows;
4. corrected residual, adapter-only, robust/mixed, ELF-LoRA, and full-chain
   variants, each reported as a delta from steps 2 and 3.
