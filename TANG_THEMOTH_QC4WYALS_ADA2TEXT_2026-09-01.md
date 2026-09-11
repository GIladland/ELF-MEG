# qc4wyals MEG -> ADA -> ELF diffusion experiment

## Primary brain model

- W&B run: `qc4wyals`
- Run name: `tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82`
- Checkpoint: `/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt`
- Export: `/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/utils/final_results/tang_themoth_apples_ada002_cosine0831_best_predictions.npz`

The residual/collapsed model is retained only as a later control. `qc4wyals` is
the primary decoder input because it combines retrieval signal, cosine, and
healthy prediction variance.

## Data contract

- 26 Birth-of-a-Nation rows.
- Exact normalized ADA: `exact_ada002_birthofanation26.npz`.
- Normalized `qc4wyals` MEG predictions: `qc4wyals_predicted_ada002_birthofanation26.npz`.
- ELF is intentionally allowed to overfit all 26 target texts from exact ADA.
- The 26 MEG-predicted vectors remain evaluation-only. Therefore this measures
  the MEG-to-ADA interface plus a known-text diffusion decoder, not unseen-text
  generalization.

## Frozen zero-shot audit

Initialization was the ELF-B exact-ADA known-text model trained on the separate
12,098-sentence fMRI corpus:

`/data/engs-pnpl/glandau/elf-runs/fmri_knowntext_ada002_elfb_long_ep300_testfirst_k64_seed49_20260830/best.pt`

| Input | Retrieval Top-1 | Top-5 | Content-word F1 | Word F1 | WER |
|---|---:|---:|---:|---:|---:|
| Exact ADA | 1.0000 | 1.0000 | 0.1128 | 0.1167 | 1.2692 |
| `qc4wyals` MEG-predicted ADA | 0.0769 | 0.2308 | 0.0073 | 0.0223 | 1.5000 |

Predicted-to-exact cosine is healthy (`mean=0.7081`, `median=0.7061`) and the
prediction/target per-dimension variance ratio is `1.1399`. The failure is thus
not simple prediction collapse. Exact semantic retrieval is already perfect,
while free generation and robustness to the MEG prediction error remain poor.

## Submitted training

ARC array `8694758` trains only on the 26 exact ADA/text pairs:

1. `adapter_only`: frozen ELF, train ADA adapter.
2. `lora4_content`: adapter plus rank-4 attention/MLP LoRA in the final four ELF blocks.
3. `last2_content`: adapter plus full tuning of the final two ELF blocks.

The content-aware arms retain positional decoder CE, upweight content tokens,
reward target content coverage, and penalize hallucinated content tokens.

## Robust overfit follow-up

`scripts/augment_semantic_npz_noise.py` constructs tangent-space perturbations
from exact ADA only. It retains the clean vectors and spans noise norms
`0.10,0.20,0.35,0.50,0.75,1.00`, corresponding to increasingly broad angular
neighborhoods. This follow-up compares:

1. full-ELF exact-text overfit;
2. noisy-neighborhood adapter tuning;
3. noisy-neighborhood LoRA tuning.

Every resulting checkpoint must be selected first on exact-ADA generation, then
evaluated once with the identical 26 `qc4wyals` prediction vectors and fixed
seed/sampling schedule. Report retrieval, content precision/recall/F1, word
precision/recall/F1, WER, BERTScore, and all target/generated pairs.

## First fine-tuning results

The exact-ADA selected checkpoints were evaluated on the untouched normalized
`qc4wyals` predictions with seed 49 and 32 diffusion steps. The shared
`/data/engs-pnpl` filesystem was full during evaluation, so evaluation ran with
node-local temporary outputs and streamed metrics; the trained best checkpoints
remain intact.

| Checkpoint | Content F1 | Content P | Content R | Word F1 | WER | Top-1 | Top-5 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen zero-shot | 0.0073 | 0.0054 | 0.0112 | 0.0223 | 1.5000 | 0.0769 | 0.2308 | 11.73 |
| Adapter-only | 0.0048 | 0.0038 | 0.0064 | 0.0125 | 1.4654 | 0.1154 | 0.2308 | 11.54 |
| LoRA-4 content | **0.0110** | **0.0082** | **0.0167** | **0.0226** | 1.5115 | **0.1154** | 0.2308 | 11.50 |
| Last-two-block content | 0.0033 | 0.0024 | 0.0055 | 0.0180 | 1.4769 | 0.0769 | **0.3462** | **10.77** |

No checkpoint is a convincing MEG-to-text success. LoRA is the narrow content
winner, adapter-only has the lowest WER, and last-two-block tuning has the best
Top-5/mean rank. The conflicting metric winners and visibly unrelated,
degenerate generations show that exact-ADA overfit alone does not bridge the
MEG prediction distribution shift.
