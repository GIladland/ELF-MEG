# Best MEG -> ELF generation significance audit

## Contract

- Run: `elf_semantic_scaling_meg_ada_x1_trained_adapter_val110_seed49_20260902`
- Checkpoint: `/data/engs-pnpl/glandau/elf-runs/elf_semantic_scaling_meg_ada_x1_trained_adapter_val110_seed49_20260902/best.pt`
- Data: fixed 110-row Apple validation split; protected test26 remains unopened.
- Decode: one saved generation per row, seed 49, no reference-based reranking.
- Null: 100,000 random one-to-one reassignments of the 110 generated sentences to the 110 targets. This preserves both text distributions but destroys the MEG/text row correspondence.
- Tests are one-sided in the favorable direction. The minimum attainable Monte Carlo p-value is `1 / 100001`.

## Results

| Metric | Observed | Shuffled mean | Shuffled 95% interval | One-sided p |
|---|---:|---:|---:|---:|
| Content-word F1 | **0.02978** | 0.01021 | [0.00380, 0.01789] | **0.000010** |
| Word F1 | **0.06197** | 0.05058 | [0.03956, 0.06222] | 0.02759 |
| ROUGE-1 F1, Porter-stemmed | **0.06694** | 0.05400 | [0.04253, 0.06603] | 0.01811 |
| WER | 1.00364 | 1.00714 | [1.00000, 1.01364] | 0.19083 |
| Raw BERTScore F1, RoBERTa-large layer 17 | **0.81739** | 0.81462 | [0.81353, 0.81576] | **0.000010** |

The old BERTScore value `-0.08197` is the same result after language-baseline rescaling. The conventional non-rescaled value is `0.81739`. Its absolute magnitude should not be interpreted as 82% reconstruction: shuffled sentence pairings already score `0.81462`. The matched increment is small (`+0.00277`) but reliably above this corpus-preserving null.

With a five-test Bonferroni threshold of `0.01`, content-word F1 and raw BERTScore remain significant; word F1 and ROUGE-1 are nominally significant, and WER is not significant.

## Retrieval chance tests

These are exact one-sided binomial chance tests, not row-permutation tests. All generated-text retrieval uses the same 110 candidates.

| Retrieval | Hits | Observed | Chance | p |
|---|---:|---:|---:|---:|
| Generated-text Top-1 | 4 / 110 | 0.03636 | 0.00909 | 0.01843 |
| Generated-text Top-5 | 15 / 110 | 0.13636 | 0.04545 | 0.000143 |
| Incoming semantic Top-1 | 32 / 110 | 0.29091 | 0.00909 | 1.27e-38 |
| Incoming semantic Top-5 | 50 / 110 | 0.45455 | 0.04545 | 3.09e-37 |

## Ten best generations by content-word F1

| Row | Target | Generated | Matching content | Content F1 |
|---:|---|---|---|---:|
| 72 | us military you will forever be an al qaeda target | nation to back the ehlateral the military | military | 0.2222 |
| 15 | eyes closed and her mouth open and blood was trickling | dog was the underneath a blood and it | blood | 0.2222 |
| 62 | the street he turned left and instead of turning around | go home and do instead of he and the stop | instead | 0.2000 |
| 20 | kept getting louder and louder the silence then i finally | started getting hurting back to my whiy | getting | 0.1818 |
| 18 | this nonstop loop of my stepfather almost like that cliche | bobppy start whiping up into the hill like | like | 0.1818 |
| 59 | i wanted i asked to go on vacation to see | go home avan a year up now wests | go | 0.1818 |
| 26 | took us a long time to heal my stepfather broke | the home detached and she took to home her wedding job | took | 0.1667 |
| 80 | best to look artsy to fit in but even though | keep up all of people in hit at night and even what like | even | 0.1667 |
| 27 | one nursing twenty four hours around the clock at home | a front side brunite's home by | home | 0.1667 |
| 31 | multiple surgeries on the arm every time the arm got | oil piecemate a time to take of of drink | time | 0.1667 |

## Reproducible artifacts

- Full 110-row target/generation CSV: `meg_best_x1_adapter_target_generated_2026-09-04.csv`
- Machine-readable results: `meg_best_x1_adapter_generation_significance_2026-09-04.json`
- Analysis implementation: `scripts/analyze_generation_permutation.py`
