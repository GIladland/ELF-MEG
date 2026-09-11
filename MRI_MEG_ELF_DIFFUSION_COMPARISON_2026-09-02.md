# MRI/MEG -> ELF diffusion comparison

## Models and aggregate results

All brain results below are from one generation per row, without reference-based reranking. The MRI development winner has not been opened on test107; the MEG result uses the 110-row Apple validation split and leaves the 26 final examples protected.

### fMRI

Best current validation model: single-checkpoint MiniLM-conditioned ELF diffusion package

```text
/data/engs-pnpl/glandau/elf-runs/fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_lexgate_top5_frompartstep37_top10_prior0p1_t2_scattered_b40_val266_generationonly/packaged_system/fmri_elf_content_first_val266_20260831.pt
```

| Input/system | Split | Word F1 | Content F1 | WER | BERTScore F1 | Generated Top-1 / Top-5 | Mean / median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| Exact MiniLM -> overfit ELF-B replay | locked 107-row known-text audit | 0.4466 | 0.4229 | 0.6841 | — | 0.9159 / 0.9720 | 1.64 / 1 |
| Best brain-conditioned ELF diffusion | val266 | 0.0688 | **0.0471** | 1.1421 | -0.0989 | 0.0113 / 0.0489 | 113.55 / 109 |

The exact-MiniLM row outputs were archived only for the 107-row locked audit, not val266. Therefore the five val266 examples immediately below show the current validation winner, and the following table gives the available *matched* oracle-versus-brain comparison on the same 107 audit sentences. No split is silently mixed.

Five fixed, evenly spaced val266 rows:

| Row | Target | Best fMRI -> diffusion prediction |
|---:|---|---|
| 0 | eyes shut or change a diaper with one hand tied | movie not really one us two whole new people now years |
| 66 | laura walking around in mexico spending my money and even | scientist years littlee really people know now one two she said |
| 133 | stay i left baghdad early the next morning going back | said to go back two one years know really now promise |
| 199 | long enough ago i was fourteen a long time ago | little of around now it until a one thousand years |
| 265 | a shirt progress not perfection and i never got worse | get they know now of a gay people even really one little |

Matched exact-MiniLM oracle versus locked brain audit examples:

| Row | Target | Exact MiniLM -> overfit ELF | Locked fMRI -> ELF |
|---:|---|---|---|
| 0 | thirty years old never met brought anybody home i never | thirty seven i home met no me i never never ever | the wanted to want to know the the the the |
| 26 | teaching climbing in northern new eng new england well i | teachinging to northern new land lland lg | town i i really really loved the garment transportation |
| 53 | street and across the street at a coffee shop called | market shop across the street at a shopping cafe | plan d into the campus because the the |
| 80 | that the teachers at my school and miss roberts finally | that the teacher at my school eth stevens | the to make the the the the conversation |
| 106 | hair and she goes oh doctor saint heller the plastic | hair and she goes oh doctor god heller the plastic | mind you't the good feeling the wantede |

The locked brain audit model in this matched table is the earlier leakage-safe reference (`eval_fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829_unseenbrain_direct1536_adapteronly_test107_20260829`): word/content F1 `0.0463/0.0063`, WER `0.9897`, Top-1/Top-5 `0.0093/0.0654`. It is shown only because it has a saved row-aligned oracle replay; it is not the newer val266 content winner.

Ten best val266 rows by content-word F1 for the current MRI diffusion winner:

| Row | Content F1 | Overlapping content words | Target | Prediction |
|---:|---:|---|---|---|
| 137 | 0.375 | new, now, york | television set and i find the new york mets now | know went to new york in now new york go get really |
| 255 | 0.308 | know, little | know you get a little momentum going and one day | the see up around little like know really back |
| 226 | 0.286 | know, said | helmut newton they said you know helmut newton the guy | said know reallya one in i ess gete back |
| 122 | 0.267 | one, two | dad one evening i took a two mile walk to | dining room we people the one camp two back around |
| 245 | 0.267 | little, people | little money i think a lot of people did around | little end of people us getin know now really two back one |
| 17 | 0.250 | get, one | one day we get a phone call from our case | new press get now back two know one people to say us |
| 42 | 0.250 | get, kid | first police detective i spoke with said get ready kid | back one little kid get likely to know us now really |
| 48 | 0.250 | one, two | accidentally i found myself straddling two lives a new one | constant now fighting my journey back my one two time |
| 192 | 0.250 | one, years | years old i got really sick um and no one | two know cause said it really school one last years last now |
| 219 | 0.250 | around, one | one nursing twenty four hours around the clock at home | back around in addition from people really two one know and everything |

## MEG

Best confirmed all-metric model: frozen `qc4wyals` MEG2SEM and frozen ELF-B, with the robust flat context adapter selected at step 11,250:

```text
/data/engs-pnpl/glandau/elf-runs/tang_apples_qc4wyals_robust_residualaug_adapter_from_tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901_robustrebuild_finalbest43k_ev250_topk1_seed49_20260901/checkpoints/eval_step_00011250_score_0.406216.pt
```

| Input/system | Split | Word F1 | Content F1 | WER | BERTScore F1 | Generated Top-1 / Top-5 | Mean / median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| Exact ADA -> overfit ELF-B | val110, 5-seed mean | **0.8894** | **0.8954** | **0.1371** | 0.8066* | 1.0000 / 1.0000 | 1.00 / 1 |
| `qc4wyals` -> robust adapter -> ELF diffusion | val110, 5-seed mean | 0.0514 | 0.0214 | 1.0198 | -0.0743 | 0.0491 / 0.1618 | 41.16 / 35.6 |

`*` Exact-ADA BERTScore is the seed-49 value. Example generations and the top ten below also use seed 49; its brain content F1 is `0.0267` and generated Top-1/Top-5 is `0.0455/0.1727`.

Five fixed, evenly spaced val110 rows, with exactly aligned targets:

| Row | Target | Exact ADA -> overfit ELF | Best MEG -> diffusion prediction |
|---:|---|---|---|
| 0 | years old i got really sick um and no one | years old i got really sick um and no one went | sloweding and to my minding about |
| 27 | one nursing twenty four hours around the clock at home | one nursing forty four hours around the clock at home | a step brother build up and from one home park |
| 54 | they speak they swallow letters one time a soldier asked | they speak they swallow letters one time a soldier asked | country put an end very weak people and the you said categories |
| 81 | wrong way to make art now this makes me extremely | wrong way to make art now this makes me extremely | real world is keep the way new and never know |
| 109 | home of twenty five years and my jeep gets stolen | home of twenty five years and jeep gets stolen | donor donations from north america thinking about big stuff |

Ten best val110 rows by content-word F1 for the MEG model (seed 49):

| Row | Content F1 | Overlapping content words | Target | Prediction |
|---:|---:|---|---|---|
| 57 | 0.308 | military, police | iraqi police iraqi military local mayors leaders with top us | security police and as senior military now at |
| 27 | 0.286 | home, one | one nursing twenty four hours around the clock at home | a step brother build up and from one home park |
| 45 | 0.286 | away, driver | driver sped away trying to get to safety not knowing | television clip the driver is parked running away and wildly |
| 5 | 0.200 | delivering | offering to give thanks for delivering my child home safely | of marching year delivering |
| 7 | 0.182 | long | long enough ago i was fourteen a long time ago | about thirty minutes there it was thirty the which was long |
| 98 | 0.182 | years | committed relationship for the past eighteen years with his boyfriend | beties likefve joaky years later |
| 93 | 0.167 | things | anything but things at home start to get a little | things infinite and feels up but it's prettyly understand |
| 71 | 0.154 | stay | stay i left baghdad early the next morning going back | home last my stay as i had my flight we supposed |
| 23 | 0.154 | got | parents got taken somewhere else my arm you know because | new front sidewing it got up a muth umham |
| 62 | 0.154 | turning | the street he turned left and instead of turning around | go back and again keep keep turning his back and his apparently |

## Bottom line

Both modalities show the same bottleneck: the corpus-overfit semantic-to-text diffusion decoder can reconstruct from exact MiniLM/ADA, but most sentence-specific information is lost when exact semantics are replaced by brain-predicted vectors. The MEG result is the cleaner demonstration because its oracle and brain rows are fully aligned on the same validation split and its selected adapter improves the raw pipeline without sacrificing the fixed semantic-vector retrieval input.

## Semantic-corpus scaling experiment (started 2026-09-02)

The next experiment tests whether a wider same-corpus semantic-to-text inverse reduces that exact-to-brain drop. It keeps every exact sentence as an anchor and adds nested phrase-recombined sentences from the same source corpus. The scaled models continue from the converged exact-input ELF-B checkpoint, receive approximately the same number of additional optimizer updates in a first stage of about 11k--12k updates, and are compared at increasing *unique* corpus sizes. Validation is measured every 1,000 updates; a scale is continued only when its final three checkpoints show a real non-trivial WER improvement, avoiding both premature stopping and arbitrary extra epochs:

- MRI/MiniLM: `x1`, `x4`, `x8`; fixed 266-row validation.
- MEG/ADA: `x1`, `x4`, `x16`; fixed 110-row validation.

Prepared audited corpora:

- MRI `x4`: 48,392 unique pairs (36,294 augmented), 84,952 physical anchor-balanced rows.
- MRI `x8`: 96,784 unique pairs (84,686 augmented), 181,736 physical rows.
- MEG `x4`: 11,152 unique pairs (8,364 augmented), 19,626 physical rows.
- MEG `x16`: 44,608 unique pairs (41,820 augmented), 86,538 physical rows.
- Exact-text overlap between the generated augmentation and the original pairs: zero in every arm.

ARC graph: MRI ELF training `8708585`, MEG ELF training `8708586`, raw ladders `8708587`/`8708588`, trained-adapter ladders `8708589`/`8708590`, matched `x1` controls `8708591`--`8708593`, final report `8708594`.

Every scale uses the same three-rung ladder: exact semantic input, untouched brain-predicted input, and a trained brain-to-ELF adapter. Each rung reports word/content F1, WER, BERTScore, generated-text Top-1/Top-5 and ranks, plus semantic retrieval where available. The 266 MRI and 110 MEG validation brain vectors remain evaluation/selection data; MRI test107 and MEG test26 remain sealed. Exact validation text is intentionally trainable because this is a known-text diffusion-capacity experiment.

The automatically generated result table will be written to `ELF_SEMANTIC_SCALING_LADDER_2026-09-02.md` after the dependency chain completes.

## Direct brain-vector scaling result (2026-09-03)

The completed two-point audit changes only the ELF semantic-to-text corpus/checkpoint. Within each
modality, the brain predictions, validation rows, candidate pool, seed, and 32-step diffusion sampler
are identical. No trained adapter or reference reranking is used. Generated-text retrieval is 266-way
for MRI and 110-way for MEG; the protected MRI test107 and MEG test26 remain unopened.

| Modality | ELF scale | Unique pairs | Exact content F1 | Exact WER | Direct word F1 | Direct content F1 | Direct WER | BERT F1 | Generated Top-1 / Top-5 | Mean / median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MRI / MiniLM | x1 | 12,098 | 0.7958 | 0.2658 | **0.0646** | **0.0089** | **1.0060** | **-0.0760** | **0.0188 / 0.0940** | **94.55 / 77** |
| MRI / MiniLM | x4 | 48,392 | 0.7659 | 0.3305 | 0.0586 | 0.0086 | 1.0568 | -0.1162 | 0.0150 / 0.0639 | 95.42 / 82 |
| MEG / ADA | x1 | 2,788 | 0.8917 | 0.1455 | **0.0454** | 0.0155 | **1.0455** | **-0.0803** | **0.0455 / 0.1182** | 41.90 / 37 |
| MEG / ADA | x16 | 44,608 | 0.8876 | 0.1409 | 0.0421 | **0.0194** | 1.0664 | -0.0969 | 0.0364 / 0.1091 | 41.94 / **33** |

This is not evidence of a general positive scaling law. MRI x4 is flat on content F1 and significantly
worse on WER and BERTScore. MEG x16 has a small content-F1 increase (`+0.0039`), but its paired 95%
interval crosses zero (`[-0.0061, +0.0143]`), while WER and BERTScore worsen significantly. Incoming
brain-vector retrieval is unchanged by construction: MRI Top-1/Top-5 `0.0338/0.1278`; MEG
`0.2909/0.4545`. The result therefore points to distribution mismatch at the brain-to-ELF interface,
not insufficient text-pair count alone.

The full tables, retention ratios, and 20,000-sample paired intervals are in
`ELF_DIRECT_BRAIN_TO_SCALED_DIFFUSION_2026-09-03.md`. MRI x8 and MEG x4 recovery runs remain queued
to fill the intermediate points after GPU maintenance; they are not used to overstate the current
two-point conclusion.

## Trained larger-ELF interface experiment (submitted 2026-09-03)

The next comparison holds each completed ELF checkpoint fixed and trains the strongest leakage-safe
brain-to-ELF interface recipe for that modality. MRI x4 uses the residual delayed-MiniLM mapper with
cosine, contrastive, context, rank-distillation, raw-anchor, geometry, content-BOW, content-precision,
and condition-pairing objectives. MEG x16 exactly reproduces the successful robust-residual-augmentation
recipe: full flat context adapter, dropout `0.05`, batch `64`, LR `2e-5`, 60 epochs, and the established
alignment/content weights. The canceled older ladder had accidentally restricted MEG to its input
projection; this resubmission corrects that mismatch.

The larger checkpoints run first, followed by x1 controls through the identical code path. Selection
uses val266/val110, while MRI test107 and MEG test26 remain sealed. The redundant quadratic internal
ELF retrieval loop is disabled during training, but incoming semantic retrieval and generated-text
Top-1/Top-5 remain enabled.

- Larger trained interfaces: ARC `8713469` (task 0 MRI x4; task 3 MEG x16).
- Matched x1 controls: ARC `8713470` (tasks 4 and 5), dependency-gated after the larger runs.
- Updated full ladder report: ARC `8713471`.

The jobs are pinned to two free L40S slots on `htc-g083`, outside the active maintenance-node set.
They are queued behind the scheduler's current reservation/backfill boundary; all required checkpoints,
training packs, exact alignment targets, and cached T5 latents were verified before submission.
