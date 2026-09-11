# MEG ADA decoder comparison: Vec2Text, T5 prefix, and ELF diffusion

## Contract

- Identical 110 Apple validation rows and 110 retrieval candidates.
- MEG semantic source: frozen `qc4wyals` 1,536-D ADA predictions.
- Diffusion comparators: the lexical-composite winner plus optional content and retrieval specialists.
- T5 comparator: the optional direct-projector winner from the original 2,652-row prefix ladder.
- Rows labelled exact-ADA are semantic-inversion ceilings, not brain-conditioned results.
- Vec2Text: pretrained ADA-002 hypothesizer/corrector, 20 recursive steps, beam width 4.
- Primary comparison: ten-word output cap; native Vec2Text length is retained as a diagnostic.
- Null: 100,000 one-to-one row permutations; lower WER and higher other scores are favorable.
- Protected test26 remains unopened.

## Actual performance

| System | Content F1 | Word F1 | Sum | WER ↓ | Raw BERTScore | BLEU-1 | ROUGE-1 | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELF diffusion: lexical-composite winner | 0.02481 | 0.08071 | 0.10551 | 1.03273 | 0.81985 | 0.07672 | 0.08733 | 0.03636 | 0.10000 |
| ELF diffusion: content specialist | 0.02860 | 0.06578 | 0.09439 | 1.04455 | 0.81809 | 0.06188 | 0.07530 | 0.04545 | 0.13636 |
| ELF diffusion: retrieval specialist | 0.02552 | 0.07233 | 0.09786 | 1.04636 | 0.81900 | 0.06870 | 0.07697 | 0.09091 | 0.16364 |
| T5 prefix: direct-projector winner | 0.02551 | 0.09943 | 0.12493 | 0.97727 | 0.82821 | 0.09563 | 0.10877 | 0.02727 | 0.14545 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | 0.02271 | 0.05206 | 0.07477 | 1.06818 | 0.81100 | 0.05124 | 0.05806 | 0.04545 | 0.13636 |
| Vec2Text: predicted qc4wyals ADA, native length | 0.01116 | 0.03272 | 0.04389 | 4.09364 | 0.76613 | 0.02178 | 0.03486 | 0.01818 | 0.10000 |
| T5 prefix: exact-ADA oracle | 0.24788 | 0.32694 | 0.57481 | 0.86636 | 0.85599 | 0.31384 | 0.34307 | 0.67273 | 0.90909 |
| Vec2Text: exact ADA ceiling, 10-word cap | 0.97543 | 0.97601 | 1.95144 | 0.06727 | 0.98922 | 0.97431 | 0.97653 | 1.00000 | 1.00000 |
| Vec2Text: exact ADA ceiling, native length | 0.97538 | 0.97410 | 1.94948 | 0.08091 | 0.98910 | 0.96436 | 0.97461 | 1.00000 | 1.00000 |

## Interpretation

- **Best brain-conditioned lexical/fluency system:** T5 direct-projector. It
  has the highest word F1 and word+content sum, lowest WER, and highest raw
  BERTScore, BLEU-1, and ROUGE-1.
- **Best brain-conditioned content F1:** the ELF diffusion content specialist
  (`0.02860`, versus T5's `0.02551`).
- **Best generated-text retrieval:** the ELF diffusion retrieval specialist
  (Top-1/Top-5 `0.09091/0.16364`).
- **Vec2Text on predicted MEG ADA:** real but weaker lexical generation than
  the T5 and diffusion leaders. Its capped Top-1/Top-5 (`0.04545/0.13636`) is
  nevertheless competitive with the diffusion content specialist.
- **Clean-ADA diagnosis:** pretrained Vec2Text nearly perfectly reconstructs
  exact ADA but drops to word/content F1 `0.05206/0.02271` on predicted ADA.
  The dominant failure is therefore robustness to the off-manifold MEG2SEM
  prediction distribution, not intrinsic invertibility of ADA.
- Native-length Vec2Text overgenerates badly (WER `4.09364`); the matched
  ten-word cap is necessary for a useful comparison.

The systems do not have identical training budgets: the reported T5 winner
comes from the original 2,652-row prefix ladder, whereas the ELF-M diffusion
specialists use the larger x16 program. All point estimates above are on the
same 110 rows and candidates, but this remains a validation comparison rather
than a protected-test claim.

## Actual mean versus permutation mean

| System | Metric | Actual | Permutation mean | Favorable difference | One-sided p |
|---|---|---:|---:|---:|---:|
| ELF diffusion: lexical-composite winner | Content F1 | 0.02481 | 0.00856 | 0.01625 | 6.99993e-05 |
| ELF diffusion: lexical-composite winner | Word F1 | 0.08071 | 0.05543 | 0.02528 | 9.9999e-06 |
| ELF diffusion: lexical-composite winner | WER | 1.03273 | 1.03850 | 0.00577 | 0.0648994 |
| ELF diffusion: lexical-composite winner | Raw BERTScore F1 | 0.81985 | 0.81726 | 0.00259 | 1.99998e-05 |
| ELF diffusion: lexical-composite winner | BLEU-1 | 0.07672 | 0.05266 | 0.02406 | 9.9999e-06 |
| ELF diffusion: lexical-composite winner | ROUGE-1 F1 | 0.08733 | 0.05900 | 0.02834 | 9.9999e-06 |
| ELF diffusion: content specialist | Content F1 | 0.02860 | 0.00854 | 0.02007 | 9.9999e-06 |
| ELF diffusion: content specialist | Word F1 | 0.06578 | 0.04717 | 0.01861 | 0.000879991 |
| ELF diffusion: content specialist | WER | 1.04455 | 1.04466 | 0.00011 | 0.526925 |
| ELF diffusion: content specialist | Raw BERTScore F1 | 0.81809 | 0.81506 | 0.00303 | 9.9999e-06 |
| ELF diffusion: content specialist | BLEU-1 | 0.06188 | 0.04458 | 0.01730 | 0.000929991 |
| ELF diffusion: content specialist | ROUGE-1 F1 | 0.07530 | 0.05043 | 0.02487 | 5.99994e-05 |
| ELF diffusion: retrieval specialist | Content F1 | 0.02552 | 0.00894 | 0.01659 | 4.99995e-05 |
| ELF diffusion: retrieval specialist | Word F1 | 0.07233 | 0.04908 | 0.02325 | 4.99995e-05 |
| ELF diffusion: retrieval specialist | WER | 1.04636 | 1.04896 | 0.00260 | 0.246808 |
| ELF diffusion: retrieval specialist | Raw BERTScore F1 | 0.81900 | 0.81572 | 0.00327 | 9.9999e-06 |
| ELF diffusion: retrieval specialist | BLEU-1 | 0.06870 | 0.04616 | 0.02254 | 2.99997e-05 |
| ELF diffusion: retrieval specialist | ROUGE-1 F1 | 0.07697 | 0.05271 | 0.02426 | 4.99995e-05 |
| T5 prefix: direct-projector winner | Content F1 | 0.02551 | 0.00937 | 0.01614 | 0.000129999 |
| T5 prefix: direct-projector winner | Word F1 | 0.09943 | 0.07032 | 0.02911 | 9.9999e-06 |
| T5 prefix: direct-projector winner | WER | 0.97727 | 0.98490 | 0.00762 | 0.0269097 |
| T5 prefix: direct-projector winner | Raw BERTScore F1 | 0.82821 | 0.82366 | 0.00455 | 9.9999e-06 |
| T5 prefix: direct-projector winner | BLEU-1 | 0.09563 | 0.06823 | 0.02740 | 1.99998e-05 |
| T5 prefix: direct-projector winner | ROUGE-1 F1 | 0.10877 | 0.07710 | 0.03167 | 9.9999e-06 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | Content F1 | 0.02271 | 0.00893 | 0.01378 | 0.000439996 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | Word F1 | 0.05206 | 0.03668 | 0.01538 | 0.00166998 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | WER | 1.06818 | 1.07093 | 0.00275 | 0.173588 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | Raw BERTScore F1 | 0.81100 | 0.80758 | 0.00342 | 9.9999e-06 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | BLEU-1 | 0.05124 | 0.03547 | 0.01577 | 0.00100999 |
| Vec2Text: predicted qc4wyals ADA, 10-word cap | ROUGE-1 F1 | 0.05806 | 0.04017 | 0.01789 | 0.000769992 |
| Vec2Text: predicted qc4wyals ADA, native length | Content F1 | 0.01116 | 0.00621 | 0.00496 | 0.0203698 |
| Vec2Text: predicted qc4wyals ADA, native length | Word F1 | 0.03272 | 0.02532 | 0.00740 | 0.00811992 |
| Vec2Text: predicted qc4wyals ADA, native length | WER | 4.09364 | 4.11298 | 0.01934 | 0.000309997 |
| Vec2Text: predicted qc4wyals ADA, native length | Raw BERTScore F1 | 0.76613 | 0.76335 | 0.00278 | 9.9999e-06 |
| Vec2Text: predicted qc4wyals ADA, native length | BLEU-1 | 0.02178 | 0.01711 | 0.00467 | 0.0227198 |
| Vec2Text: predicted qc4wyals ADA, native length | ROUGE-1 F1 | 0.03486 | 0.02794 | 0.00692 | 0.0170998 |
| T5 prefix: exact-ADA oracle | Content F1 | 0.24788 | 0.01168 | 0.23620 | 9.9999e-06 |
| T5 prefix: exact-ADA oracle | Word F1 | 0.32694 | 0.06476 | 0.26218 | 9.9999e-06 |
| T5 prefix: exact-ADA oracle | WER | 0.86636 | 0.98388 | 0.11752 | 9.9999e-06 |
| T5 prefix: exact-ADA oracle | Raw BERTScore F1 | 0.85599 | 0.82406 | 0.03194 | 9.9999e-06 |
| T5 prefix: exact-ADA oracle | BLEU-1 | 0.31384 | 0.06222 | 0.25162 | 9.9999e-06 |
| T5 prefix: exact-ADA oracle | ROUGE-1 F1 | 0.34307 | 0.07033 | 0.27274 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, 10-word cap | Content F1 | 0.97543 | 0.01855 | 0.95688 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, 10-word cap | Word F1 | 0.97601 | 0.06552 | 0.91049 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, 10-word cap | WER | 0.06727 | 0.98301 | 0.91573 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, 10-word cap | Raw BERTScore F1 | 0.98922 | 0.82261 | 0.16661 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, 10-word cap | BLEU-1 | 0.97431 | 0.06542 | 0.90889 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, 10-word cap | ROUGE-1 F1 | 0.97653 | 0.07053 | 0.90600 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, native length | Content F1 | 0.97538 | 0.01865 | 0.95673 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, native length | Word F1 | 0.97410 | 0.06530 | 0.90879 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, native length | WER | 0.08091 | 1.01384 | 0.93293 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, native length | Raw BERTScore F1 | 0.98910 | 0.82244 | 0.16666 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, native length | BLEU-1 | 0.96436 | 0.06463 | 0.89973 | 9.9999e-06 |
| Vec2Text: exact ADA ceiling, native length | ROUGE-1 F1 | 0.97461 | 0.07030 | 0.90431 | 9.9999e-06 |

`Favorable difference` is actual minus null for overlap/BERTScore/BLEU/ROUGE, and null minus actual for WER.
