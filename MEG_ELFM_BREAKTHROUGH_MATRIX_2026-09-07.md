# MEG → ELF-M Breakthrough Matrix — 2026-09-07

## Locked references

- Exact-ADA ELF-M: `elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906`
- MEG2SEM: `qc4wyals` / `tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82`
- Training/selection: MEG train2652 / validation110.
- Protected MEG test26 is absent from every job in this matrix.
- Current generation records to beat: word F1 `0.07663`, content F1 `0.02978`, composite `0.09543`, WER `1.0036`.

## Finite experiment matrix

| Stage | ARC job | What it tests |
|---|---:|---|
| Semantic interpolation preparation | `8749761` | Builds aligned raw→exact ADA validation conditions. Completed successfully. |
| Five-point interpolation evaluation | `8749762` | Estimates how much semantic recovery ELF-M needs before text quality changes. |
| Exact-context distillation | `8749763` | Output-only, projector-only, joint-output/projector, and conservative full-MEG variants. |
| Residual-augmented initialization | `8749764` | Repeats joint context distillation from the empirical MEG-residual-trained projector; waits for `8732390_2`. |
| Context winner selection | `8749765` | Selects highest held-out teacher-context cosine across the five completed arms. |
| Post-context E2E/LoRA matrix | `8749766` | Frozen ELF, LoRA-4, strong-content LoRA-4, frozen-adapter LoRA-4, LoRA-8, and conservative full-MEG variants. |
| Direct temporal context distillation | `8749767` | Bypasses the single ADA point and maps temporal MEG directly into 64 ELF context tokens. |
| Temporal post-distillation | `8749768` | Adapter-only, LoRA-4, and LoRA-8 strong-content variants. |
| Three-seed fixed validation | `8749769` | Re-evaluates all nine final models on identical val110 candidates using seeds 49/101/202, including retrieval and BERTScore. |
| Aggregate summary | `8749770` | Produces per-model multi-seed JSON summaries. |

## Decision rule

Rank final checkpoints by mean `(word F1 + content F1)`, while reporting word F1,
content F1, WER, BERTScore, generated-text retrieval, latent retrieval, exact-ADA
retention, and teacher-context cosine separately. A claimed breakthrough must improve
the three-seed mean rather than one stochastic generation pass.

The matrix is deliberately finite. No additional reactive sweeps should be submitted
before these results are collected and compared.
