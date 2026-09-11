# Best fMRI -> ELF generation significance audit

## Contract

- System: `fmri_elf_content_first_val266_20260831`
- Checkpoint: `/data/engs-pnpl/glandau/elf-runs/fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_lexgate_top5_frompartstep37_top10_prior0p1_t2_scattered_b40_val266_generationonly/packaged_system/fmri_elf_content_first_val266_20260831.pt`
- Data: fixed 266-row predicted validation split; protected test107 remains unopened for this model-selection audit.
- Decode: one saved generation per row, no reference-based reranking.
- Null: 100,000 random one-to-one reassignments of the 266 generated sentences to the 266 targets. This preserves both text distributions but destroys the fMRI/text row correspondence.
- Tests are one-sided in the favorable direction. The minimum attainable Monte Carlo p-value is `1 / 100001`.

## Results

| Metric | Observed | Shuffled mean | Shuffled 95% interval | One-sided p |
|---|---:|---:|---:|---:|
| Content-word F1 | **0.04710** | 0.03804 | [0.03264, 0.04357] | **0.000690** |
| Word F1 | **0.06885** | 0.05705 | [0.05067, 0.06359] | **0.000240** |
| ROUGE-1 F1, Porter-stemmed | **0.07309** | 0.05989 | [0.05334, 0.06656] | **0.000070** |
| WER | 1.14211 | 1.14482 | [1.14023, 1.14925] | 0.13650 |
| Raw BERTScore F1, RoBERTa-large layer 17 | **0.81453** | 0.81255 | [0.81193, 0.81320] | **0.000010** |

The previously reported BERTScore F1 `-0.09891` is this same result after language-baseline rescaling. The conventional non-rescaled score is `0.81453`. Randomly paired sentences already score `0.81255`, so the conditional matched increment is `+0.00198`; the high absolute score is not an 81% reconstruction rate.

At a five-test Bonferroni threshold of `0.01`, content-word F1, word F1, ROUGE-1, and raw BERTScore remain significant. WER is not significant.

## Retrieval chance tests

These are exact one-sided binomial chance tests, not row-permutation tests. Generated-text retrieval uses the same 266 candidates for every row.

| Retrieval | Hits | Observed | Chance | p |
|---|---:|---:|---:|---:|
| Generated-text Top-1 | 3 / 266 | 0.01128 | 0.00376 | 0.07995 |
| Generated-text Top-5 | 13 / 266 | 0.04887 | 0.01880 | **0.00180** |

## Ten best generations by content-word F1

| Row | Target | Generated | Matching content | Content F1 |
|---:|---|---|---|---:|
| 137 | television set and i find the new york mets now | know went to new york in now new york go get really | new, now, york | 0.3750 |
| 255 | know you get a little momentum going and one day | the see up around little like know really back | know, little | 0.3077 |
| 226 | helmut newton they said you know helmut newton the guy | said know reallya one in i ess gete back | know, said | 0.2857 |
| 245 | little money i think a lot of people did around | little end of people us getin know now really two back one | little, people | 0.2667 |
| 122 | dad one evening i took a two mile walk to | dining room we people the one camp two back around | one, two | 0.2667 |
| 192 | years old i got really sick um and no one | two know cause said it really school one last years last now | one, years | 0.2500 |
| 254 | dance and criticizing my hairdo and one night he said | soft life and now one two she started little really said back know | one, said | 0.2500 |
| 48 | accidentally i found myself straddling two lives a new one | constant now fighting my journey back my one two time | one, two | 0.2500 |
| 42 | first police detective i spoke with said get ready kid | back one little kid get likely to know us now really | get, kid | 0.2500 |
| 17 | one day we get a phone call from our case | new press get now back two know one people to say us | get, one | 0.2500 |

## Reproducible artifacts

- Full 266-row target/generation/metric CSV: `fmri_best_diffusion_target_generated_metrics_2026-09-04.csv`
- Machine-readable results: `fmri_best_diffusion_generation_significance_2026-09-04.json`
- Analysis implementation: `scripts/analyze_generation_permutation.py`
