# MEG ADA → T5-prefix baseline

## Question

Does direct T5-prefix generation recover more usable text from the same
`qc4wyals` MEG-predicted ADA vectors than ELF diffusion?

## Locked data contract

- MEG2SEM: frozen `qc4wyals` (`tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82`).
- Semantic space: 1,536-D ADA002.
- Training: 2,652 rows.
- Development evaluation: the fixed 110-row Apple validation split.
- Protected final evaluation: 26 Birth of a Nation rows; neither their brain
  vectors nor their targets are present in this graph.
- Generation: one deterministic generation per row, ten-word cap.
- Retrieval: the same 110 candidates for every validation arm.

The available `qc4wyals` training predictions are in-sample outputs, whereas
the 110 validation predictions are held out. This is evaluation-leakage safe,
but not the stronger story/session-OOF train-noise match used by the final fMRI
T5-prefix work. OOF MEG predictions remain a possible follow-up.

## Experimental ladder

1. Three exact-ADA oracle learning rates train the same 16-token T5-small
   prefix architecture on train2652 and evaluate unseen exact-ADA val110.
2. The best oracle is copied into one immutable shared checkpoint.
3. The same checkpoint is evaluated with raw `qc4wyals` predictions.
4. Eight controlled brain-stage alternatives test identity-residual adapters,
   content-position CE/pairing loss, T5 cross-attention LoRA, joint prefix
   tuning, direct projector tuning, and LoRA-only tuning.
5. Every arm receives post-hoc raw/rescaled BERTScore and 110-way generated
   retrieval.
6. The best positive-condition-margin arm receives a 100,000-permutation
   lexical/ROUGE/BERTScore audit and exact retrieval chance tests.

Checkpoint selection uses `0.5 × (word F1 + content-word F1)`. Matched versus
three fixed derangements is reported, and a promoted brain arm must retain a
positive content margin.

## ARC graph

| Stage | Job |
|---|---:|
| GPU/data/save-reload smoke | `8749981` — completed successfully |
| Exact-ADA oracle LR array | `8749995` |
| Immutable oracle selector | `8749996` |
| Raw/adapter/LoRA/joint brain array | `8749997` |
| Retrieval and BERTScore array | `8749998` |
| Ladder report | `8749999` |
| Winner permutation/significance audit | `8750000` |

Status at the final 2026-09-07 check: the exact-ADA oracle array is valid and
pending for ordinary HTC scheduler priority; every downstream stage is held by
its intended dependency. Query these jobs with `--clusters=htc`.

Output root:

```text
/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907
```

The smoke's intentionally undertrained one-epoch oracle produced exact-ADA
val110 word/content F1 `0.1905/0.1357`. Passing raw `qc4wyals` vectors through
that same prefix produced word/content F1 `0.0645/0.0130`, WER `0.9800`, and a
positive matched-minus-deranged content margin `+0.0058`. These are contract
checks only, not baseline results; production trains the oracle for 100 epochs
and selects it before any brain-stage comparison.
