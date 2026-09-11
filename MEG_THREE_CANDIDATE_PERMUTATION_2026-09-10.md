# MEG diffusion candidate permutation audit

## Contract

- Models: lexical-composite, content-specialist, and retrieval-specialist MEG
  diffusion checkpoints.
- Data: the identical 110 Apple validation rows for every model. The protected
  26-row test set was not opened.
- Generation: one saved generation per row; no reference-based reranking.
- Null: 100,000 random one-to-one reassignments of generated sentences to
  target rows. This preserves both text marginals while destroying row-level
  MEG/text correspondence.
- Tests: one-sided in the favorable direction (lower WER; higher BERTScore,
  BLEU-1, and ROUGE-1).
- BERTScore: raw, non-rescaled RoBERTa-large layer 17. A high absolute score is
  expected even under the shuffled null, so matched-minus-null is important.
- BLEU-1: macro sentence BLEU-1 using modified unigram precision and the
  standard brevity penalty.
- ROUGE-1: macro unigram F1 after Porter stemming.

## Results

| Candidate | Metric | Observed | Shuffled mean | Shuffled 95% interval | Favorable difference | One-sided p |
|---|---|---:|---:|---:|---:|---:|
| Lexical composite | WER ↓ | **1.03273** | 1.03850 | [1.03182, 1.04455] | 0.00577 lower | 0.06490 |
| Lexical composite | Raw BERTScore F1 ↑ | **0.81985** | 0.81726 | [0.81614, 0.81842] | +0.00259 | 0.000020 |
| Lexical composite | BLEU-1 ↑ | **0.07672** | 0.05266 | [0.04223, 0.06349] | +0.02406 | 0.000010 |
| Lexical composite | ROUGE-1 F1 ↑ | **0.08733** | 0.05900 | [0.04778, 0.07067] | +0.02834 | 0.000010 |
| Content specialist | WER ↓ | 1.04455 | 1.04466 | [1.03818, 1.05091] | 0.00011 lower | 0.52692 |
| Content specialist | Raw BERTScore F1 ↑ | 0.81809 | 0.81506 | [0.81396, 0.81623] | +0.00303 | 0.000010 |
| Content specialist | BLEU-1 ↑ | 0.06188 | 0.04458 | [0.03452, 0.05501] | +0.01730 | 0.000930 |
| Content specialist | ROUGE-1 F1 ↑ | 0.07530 | 0.05043 | [0.03942, 0.06188] | +0.02487 | 0.000060 |
| Retrieval specialist | WER ↓ | 1.04636 | 1.04896 | [1.04273, 1.05455] | 0.00260 lower | 0.24681 |
| Retrieval specialist | Raw BERTScore F1 ↑ | 0.81900 | 0.81572 | [0.81462, 0.81687] | **+0.00327** | 0.000010 |
| Retrieval specialist | BLEU-1 ↑ | 0.06870 | 0.04616 | [0.03599, 0.05691] | +0.02254 | 0.000030 |
| Retrieval specialist | ROUGE-1 F1 ↑ | 0.07697 | 0.05271 | [0.04143, 0.06458] | +0.02426 | 0.000050 |

All nine BERTScore/BLEU-1/ROUGE-1 tests survive Bonferroni correction across
the entire 12-test table (0.05 / 12 = 0.00417). None of the three WER tests
is significant. Thus, the matched MEG/text signal is real and reproducible in
semantic and unigram similarity, but WER remains too substitution- and
length-dominated to detect it reliably.

The lexical-composite checkpoint has the best absolute value for all four
reported metrics, including the lowest WER. The retrieval specialist has the
largest BERTScore matched-minus-shuffled effect and, separately, the strongest
110-way generated retrieval.

## Generated-text retrieval

These are exact binomial chance tests on the same 110 candidates, not row
permutation tests.

| Candidate | Top-1 | p | Top-5 | p |
|---|---:|---:|---:|---:|
| Lexical composite | 4/110 (0.03636) | 0.01843 | 11/110 (0.10000) | 0.01167 |
| Content specialist | 5/110 (0.04545) | 0.00345 | 15/110 (0.13636) | 0.000143 |
| Retrieval specialist | **10/110 (0.09091)** | **7.91e-8** | **18/110 (0.16364)** | **2.45e-6** |

## Artifacts

- [Lexical-composite ranked generations](meg_three_candidate_permutation_20260910/lexical_composite_ranked_generations.csv)
- [Content-specialist ranked generations](meg_three_candidate_permutation_20260910/content_specialist_ranked_generations.csv)
- [Retrieval-specialist ranked generations](meg_three_candidate_permutation_20260910/retrieval_specialist_ranked_generations.csv)
- [Lexical-composite machine-readable audit](meg_three_candidate_permutation_20260910/lexical_composite.json)
- [Content-specialist machine-readable audit](meg_three_candidate_permutation_20260910/content_specialist.json)
- [Retrieval-specialist machine-readable audit](meg_three_candidate_permutation_20260910/retrieval_specialist.json)

Analysis job: ARC 8778233.
