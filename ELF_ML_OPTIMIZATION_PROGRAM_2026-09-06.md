# ELF-M/L semantic-to-text optimization program

## Objective and selection rule

Train stable ELF-M and ELF-L semantic-to-text oracle checkpoints, then compare
raw brain input, adapter-only, ELF LoRA, brain-head E2E, and E2E+LoRA systems.
The primary checkpoint score is now:

```text
0.5 * (word F1 + content-word F1)
```

WER, conventional and rescaled BERTScore, generated-text Top-1/Top-5, mean and
median rank, and incoming semantic retrieval remain required report metrics.

## Leakage boundary

- Exact semantic/text pairs may include every target sentence for the explicit
  known-text oracle/memorization experiment.
- MRI val266 and MEG val110 brain measurements may be used for model selection.
- MRI test107 and MEG test26 brain measurements remain absent from training and
  validation and are opened once for the final promoted systems.
- No reference-based candidate reranking is allowed for a promoted single
  checkpoint.

## Locked reference results

### Exact MRI MiniLM oracle, fixed 266 known sentences

| Decoder | Word F1 | Content F1 | WER | Generated Top-1 / Top-5 |
|---|---:|---:|---:|---:|
| ELF-B, converged | 0.7910 | 0.7958 | 0.2658 | 0.9962 / 0.9962 |
| ELF-M, stopped while rising | 0.5260 | 0.5103 | 0.5677 | 0.9511 / 0.9887 |
| ELF-L, collapsed optimization | 0.0031 | 0.0043 | 1.1278 | 0.0075 / 0.0376 |

The previous ELF-M audit reached its best word/content result at its final
45,420th update, so it was not converged. The previous ELF-L result is an
optimization failure, not a model-capacity estimate.

### Brain-conditioned development baselines

| Modality | Split | Word F1 | Content F1 | WER | Generated Top-1 / Top-5 |
|---|---|---:|---:|---:|---:|
| MRI best single diffusion checkpoint | val266 | 0.06885 | 0.04710 | 1.14211 | 0.01128 / 0.04887 |
| MEG best confirmed x1 adapter | val110 | 0.06197 | 0.02978 | 1.00364 | 0.03636 / 0.13636 |

## Stage 1: recover oracle capacity

ARC array `8731559` compares four exact-MiniLM training arms; task 0 was moved
to the free interactive GPU as replacement job `8732341`, while tasks 1--3
remain in the L40S queue:

1. ELF-M continuation, constant `8e-5`, 180 epochs.
2. ELF-M continuation, cosine `5e-5`, 180 epochs.
3. Fresh ELF-L, constant `2e-5`, 130 epochs.
4. Fresh ELF-L, cosine `1e-5`, 130 epochs.

Only `best.pt` is retained per arm to control storage. Exact val266 evaluation
array `8731560` starts after all candidate arms finish and evaluates every arm
that produced a checkpoint. Raw MRI input evaluation array `8731737` uses the
same checkpoints and the same 266 candidates.

## Stage 1b: high-capacity models on larger corpora

The exact-only MiniLM runs are capacity controls. They feed a required x8
high-data continuation (`8732570`) using 96,784 unique semantic/text pairs and
181,736 physical rows, versus 12,098 rows at x1. Each recovered ELF-M/L arm is
continued with a conservative constant or cosine schedule. Fixed exact, raw
MRI, and adapter comparisons are dependency-gated as `8732571`, `8732572`, and
`8732573`.

The ADA/MEG M/L oracle search (`8732289`) already trains on x16: 44,608 unique
pairs and 86,538 physical rows, versus 2,788 rows at x1. Thus every promoted
large decoder must have both a matched exact-only capacity measurement and a
larger-corpus checkpoint; downstream selection cannot promote an x1-only M/L
model without explicitly showing that the larger-data model lost on the fixed
validation set.

### Live results, 2026-09-06

The x16 ADA search is complete. ELF-M with the cosine schedule is the clear
oracle leader on the fixed 110-row audit: word F1 `0.91088`, content F1
`0.90541`, WER `0.11818`, and generated-text Top-1/Top-5 `1.000/1.000`.
ELF-M constant reached `0.90135/0.89644/0.13727`; both ELF-L schedules were
substantially worse (`0.785--0.788` word F1). This is evidence for a useful
data-scaling gain in ELF-M, but not in ELF-L under the tested optimization.

The first matched frozen-adapter checkpoint (ELF-M constant, step 750) improves
its raw-MEG control as follows:

| Input to the same ELF-M | Word F1 | Content F1 | WER | Generated T1 / T5 | Diffusion T1 / T5 | Diffusion mean rank |
|---|---:|---:|---:|---:|---:|---:|
| Raw qc4wyals MEG2SEM vector | 0.05355 | 0.01506 | 1.04182 | 0.04545 / 0.09091 | 0.04545 / 0.16364 | 39.27 |
| Trained flat adapter, step 750 | 0.06632 | 0.02421 | 1.03545 | 0.06364 / 0.15455 | 0.04545 / 0.18182 | 34.30 |

Thus the saved adapter improves every listed quantity except diffusion Top-1,
which ties rather than improves. Compared with the previous all-time x1 MEG
baseline (`0.06197` word, `0.02978` content, `1.00364` WER), it establishes a
new word-F1 result but has not yet won content F1 or WER. The remaining adapter,
LoRA, and E2E arms must therefore continue.

## Stage 2: adapter search

ARC array `8731738` starts after Stage 1. For every oracle candidate it compares:

- residual MLP fusion of the four delayed MiniLM predictions;
- ordered four-delay transformer residual fusion.

ELF remains frozen. Training excludes the 266-row validation tail and all
test107 brain vectors. Checkpoints are selected by the joint word/content
score; large periodic checkpoints are disabled.

## Stage 3: diffusion and brain-side adaptation

Promote the best ELF-M and ELF-L oracle/adapter paths, then compare on the same
fixed validation rows:

1. frozen ELF + adapter only;
2. frozen adapter + ELF LoRA;
3. brain semantic-output head + adapter, frozen ELF;
4. brain semantic-output head + adapter + final-block ELF LoRA;
5. controlled full-chain E2E with discriminative learning rates.

Current MEG x16 pilot jobs already cover ELF-M LoRA (`8730354`) and true E2E
with/without final-four-block LoRA (`8732102`; corrected resubmission of cancelled
job `8730355`, selecting checkpoints by mean word/content F1). Their protected test26 split is
not loaded.

The legacy E2E array `8732102` was cancelled after it was confirmed to initialize
from the weaker September 4 adapter checkpoint. Replacement array `8737697`
repeats the controlled frozen-ELF versus final-four-block LoRA comparison while
initializing directly from the new x16 cosine ELF-M winner
`elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt`
(locked exact-ADA word/content F1 `0.9109/0.9054`).

Replacement array `8737697` completed. On the identical MEG val110 candidate
set, the final-four-block LoRA arm improved over the old half-trained-ELF-M E2E
checkpoint on every reported metric:

| E2E initialization/arm | Word F1 | Content F1 | WER | Generated T1 / T5 | Diffusion T1 / T5 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| Old September 4 ELF-M, frozen ELF | 0.06097 | 0.02307 | 1.02000 | 0.02727 / 0.09091 | 0.03636 / 0.15455 | 38.16 |
| New cosine ELF-M, frozen ELF | 0.06714 | 0.02283 | 1.01727 | 0.02727 / 0.14545 | 0.04545 / 0.12727 | 37.30 |
| New cosine ELF-M + rank-4 final-four-block LoRA | **0.07033** | **0.02476** | **1.01909** | **0.04545 / 0.14545** | **0.04545 / 0.16364** | **37.31** |

This establishes that better oracle inversion capacity transfers to the brain
path, especially for total word F1. It does not yet beat the older x1 MEG
baseline on content F1 (`0.02978`) or WER (`1.00364`). The exact-ADA audit also
drops to roughly `0.84--0.85` during brain adaptation, showing that semantic
interface adaptation is trading away some of the new oracle's `0.91` capacity.

Using the primary content-weighted composite `word F1 + content F1`, the old
E2E checkpoint scores `0.08404` and the new cosine-ELF-M LoRA checkpoint scores
`0.09509`, an absolute gain of `0.01105` or `13.15%`. Row-wise, 35/110 examples
improve, 39 tie, and 36 worsen. A paired sign-flip test gives `p=0.224` and a
row-bootstrap 95% interval of `[-0.0160, 0.0397]`, so this remains directional
rather than statistically secure. The full paired audit is
`meg_e2e_previous_vs_new_coselfm_lora_val110_20260906.csv`.

Because the LoRA arm reached its best word/content checkpoint at the final
6,640th update rather than plateauing, continuation job `8740622` resumes that
exact E2E state for 100 additional epochs. It retains only the best
word/content checkpoint, uses the same learning rates/objective for a clean
duration test, and continues to exclude protected MEG test26.

Continuation `8740622` completed and is now plateau-confirmed. Its best
checkpoint occurred at step 1,000 (`word=0.06926`, `content=0.02618`, composite
`0.09543`); training continued through step 33,200 without a new record. This
is a small all-time MEG composite improvement over the prior x1 baseline
`0.09175` (`+4.0%`), but it is not significant across the 110 paired rows
(one-sided sign-flip `p=0.404`, bootstrap 95% interval for the delta
`[-0.0261, 0.0332]`). The improvement over the half-trained ELF-M E2E control
is larger (`+13.56%`) but also not yet significant (`p=0.226`). The continuing
constant-ELF-M residual-adapter arm has reached a provisional word-F1 record
of `0.07663`, but remains open and has lower content F1 (`0.01878`).

The ADA/MEG oracle-capacity continuation is job `8732289` (four arms: ELF-M
constant/cosine and ELF-L constant/cosine). It starts from the strongest x16
checkpoints, uses conservative learning rates, retains only each arm's best
mean-word/content checkpoint, and loads no predicted MEG vector.

Its dependency-gated comparisons are exact-semantic evaluation `8732388`, raw
MEG-vector evaluation `8732389`, and adapter search `8732390`. The adapter
search tests both the established full flat context adapter and a
same-width residual semantic mapper while freezing ELF; both select on the
same val110 mean word/content score.

The first high-data LoRA matrices are also dependency-gated:

- MRI/MiniLM x8: `8732720`, after adapter array `8732573`;
- MEG/ADA x16: `8732721`, after adapter array `8732390`.

For ELF-M and ELF-L separately, a strict promotion utility selects the best
completed adapter by mean word/content F1 and requires both
`best_metrics.json` and a checkpoint. Each matrix then compares rank-4
final-four-block and rank-8 all-block LoRA, both as LoRA-only controls and as
joint adapter+LoRA training. The selector cannot fall back to an older x1
checkpoint.

True brain-model E2E matrices are dependency-gated on the same promoted
high-data adapters:

- MRI/MiniLM x8: `8732752`, comparing MRI2SEM projector versus projector plus
  final residual block, each with frozen ELF and with rank-4 final-four-block
  LoRA;
- MEG/ADA x16: `8732753`, comparing the qc4wyals semantic output head versus
  all MEG2SEM parameters at a tenfold smaller learning rate, each with frozen
  ELF and with rank-4 final-four-block LoRA.

The MRI brain path is initialized from the surviving native 1,536-D MRI2SEM
`model.pt`; the promoted x8 checkpoint initializes only ELF and its semantic
adapter. The MEG E2E loader now reconstructs either the flat or
residual-identity semantic projector, so promotion does not silently exclude
the residual winner. Both matrices retain only the validation-selected best
checkpoint and keep MRI test107/MEG test26 brain data sealed.

## Stage 4: loss and decoding search

After architecture/schedule selection, vary one lever at a time:

- content-token CE weight;
- content bag-of-words recall;
- hallucinated-content precision penalty;
- condition-pairing margin;
- semantic alignment/contrastive/geometry preservation;
- CFG and sampling-step settings using target-free validation selection.

Promotion requires the highest joint word/content score from one checkpoint,
with all secondary metrics and row-level generations retained for audit.

## Experiment-closure rule

A run completing its requested epochs is not, by itself, a closed experiment.
Every promoted training path must be labelled as one of `running`,
`budget-stopped`, or `plateau-confirmed`. Plateau confirmation requires:

1. the best validation `word F1 + content F1` checkpoint is not in the final
   ten evaluation audits;
2. the last ten audits show no material new composite improvement (absolute
   threshold `0.002`), with WER and retrieval inspected for conflicting trends;
3. exact-semantic/oracle performance is checked for degradation during brain
   adaptation;
4. if the best checkpoint remains near the end, resume that exact model state
   with a controlled continuation before declaring closure.

This rule applies to oracle ELF, adapter, LoRA, and true E2E runs. Scheduled
training duration is therefore a minimum opportunity to reach closure, not the
definition of convergence.
