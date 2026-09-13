# Closing the MEG T5-prefix versus ELF diffusion gap

Date: 2026-09-12

## Current answer

Under a scorer-aligned cap of at most ten lexical tokens, the frozen
T5-to-ELF-B confidence-gated editor closes the original T5 gap and exceeds a
stronger T5 decoding source. The frozen balanced arm uses confidence threshold
`0.20`, flow end time `0.25`, and predicted-ADA semantic scale `0.10`.

| System | Content F1 | Word F1 | Sum | WER |
|---|---:|---:|---:|---:|
| Original T5 direct projector | 0.02551 | 0.09943 | 0.12493 | 0.97727 |
| ELF-M lexical composite | 0.02414 | 0.08132 | 0.10546 | 0.98182 |
| Frozen T5, improved beam-4 decode | 0.02669 | 0.10398 | 0.13067 | 0.97909 |
| **T5-to-ELF-B confidence-gated editor** | **0.02834** | **0.10508** | **0.13343** | **0.97727** |

The balanced hybrid exceeds the original T5 sum by `0.00849` and the improved
T5 source by `0.00276`; it matches the original T5 WER and improves the source
by two edit operations (`1075` versus `1077` errors over 1,100 reference
words). The paired improvement over the stronger source is concentrated: two
rows improve word F1 and WER, one of those improves content F1, and no row is
worse on these metrics. The 100,000-row-resampling bootstrap intervals touch
zero because most rows tie, so this is a deterministic validation point win,
not yet a population-level significance claim.

For reference, the standalone ELF-M lexical-composite model remains the pure
diffusion model that gets to approximately `0.98` WER.

These capped values preserve source spelling and punctuation but define a
word with the evaluator regex
`[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?`. The ranked candidate CSVs are restored to
dataset index order before use. The original reports used a looser decoding
contract, so this is a new matched diagnostic rather than a silent rewrite of
the old leaderboard.

## Experiments started

### 1. Corrected four-system audit

- Slurm job: `8790906` (array 0-3)
- Inputs: T5 direct projector plus lexical, content, and retrieval ELF
  candidates, all capped by the same evaluator-visible token definition.
- Outputs: `/data/engs-pnpl/glandau/elf-runs/meg_t5_elf_metriccap10_audit_20260912`
- Evaluation: 100,000 row permutations, raw RoBERTa-large BERTScore,
  BLEU-1, ROUGE-1, word/content F1, and WER.

### 2. Target-free T5/ELF candidate selector

- Slurm job: `8790905`
- Candidates per row: T5 direct, ELF lexical, ELF content, ELF retrieval.
- Selector input: nested session-OOF `qc4wyals` predicted ADA on train2,652
  and frozen predicted ADA on val110.
- Selector target space: RoBERTa-large layer-17 mean sentence states.
- Map fitting: ridge regression trained only on OOF train predictions and
  their train texts; ridge strength selected with a story-grouped held-out
  train split.
- Validation selection is target-free: candidate cosine to the mapped brain
  vector, optionally with a fixed ten-word length prior.
- Output: `/data/engs-pnpl/glandau/elf-runs/meg_t5_elf_roberta_oof_mapper_rerank_val110_20260912`

Validation targets are loaded only after candidate selection to report the
result. The target-informed candidate oracle remains a diagnostic ceiling and
is not used by this selector.

The selector completed and did not beat the T5 default. Its best point uses a
`0.0002` fixed length penalty and obtains content/word F1
`0.02441/0.09421` (sum `0.11861`) with WER `0.97727`. It chooses T5 on 59
rows and one of the ELF candidates on 51 rows. This is evidence that noisy
whole-sentence routing throws away too many correct T5 outputs.

### 3. T5 sentence as the ELF flow starting point

- Slurm job: `8790913` (array 0-5)
- Source sentence: the T5 direct-projector validation prediction.
- Semantic condition: the same frozen `qc4wyals` predicted ADA vector.
- Flow checkpoint: the existing ELF-B LoRA-8 source-flow winner, trained on
  train-only simulated sentence corruptions.
- Source trust: full, because the prior D'Ascoli experiment showed that
  replacing uncertain on-manifold source states with independent Gaussian
  states was harmful.
- Semantic scale ladder: `0`, `0.02`, `0.035`, `0.05`, `0.1`, `1.0`.
- Output prefix:
  `/data/engs-pnpl/glandau/elf-runs/qc4wyals_elfb_t5source_lora8_..._20260912`

This is a validation-only transfer test. It asks whether a flow trained to
edit corrupted full sentences can preserve T5's syntax and use MEG semantics
to repair uncertain content. It does not yet train on out-of-fold T5 source
sentences, so a negative result will not reject the stronger proposed editor.

The completed full-flow results are negative but informative. After applying
the same lexical cap, the best arm reaches sum `0.12153`, WER `0.98273` at
semantic scale `0.05`. Scale `0.035` reaches sum `0.12014`, WER exactly
`0.98000`. Both are much closer to T5 than standalone ELF-M, but neither beats
the unedited T5 source.

| Semantic scale | Content F1 | Word F1 | Sum | WER |
|---:|---:|---:|---:|---:|
| 0.000 | 0.02023 | 0.09734 | 0.11756 | 0.98182 |
| 0.020 | 0.02039 | 0.09525 | 0.11564 | 0.98182 |
| 0.035 | 0.02356 | 0.09658 | 0.12014 | 0.98000 |
| **0.050** | **0.02482** | **0.09671** | **0.12153** | **0.98273** |
| 0.100 | 0.01913 | 0.09349 | 0.11262 | 0.98364 |
| 1.000 | 0.01847 | 0.07315 | 0.09162 | 0.98636 |

### 4. Partial transport as an edit-strength control

- Slurm job: `8790928` (array 0-11)
- Flow end times: `0`, `0.05`, `0.1`, `0.25`, `0.5`, `1.0`.
- Semantic scales: `0.05`, `1.0`.

At semantic scale `0.05`, the best capped partial result so far is end time
`0`: sum `0.12229`, WER `0.98091`. The ELF diffusion decoder changes the T5
latent even with no ODE step, so merely stopping transport early cannot supply
a true identity path.

### 5. Train-only lexical-probe selector

- Slurm job: `8790944` (completed).
- Result: sum `0.10596`, WER `0.98273`; worse than always choosing T5.
- Probe recall: `0.02546` at 10 and `0.14745` at 100.
- Interpretation: individual word directions in the OOF predicted-ADA space
  are too weak to route among four complete generated sentences.

### 6. Frozen T5 source decoding sweep

- Slurm job: `8791146` (completed).
- No weights changed and no per-row reference choice was made.
- Global validation winner: beam `4`, minimum `10` new tokens, maximum `20`,
  length penalty `0.8`, evaluator-aligned ten-word cap.
- Result: content F1 `0.02669`, word F1 `0.10398`, sum `0.13067`, WER
  `0.97909`.

This exceeds the original T5 point estimate (`0.12493`) by `0.00573` and is
the first result in this program above the old line. The paired bootstrap
intervals still include zero: word-F1 difference `+0.00455`, 95% CI
`[-0.00780, 0.01753]`; content-F1 difference `+0.00118`, 95% CI
`[-0.00826, 0.00997]`. It is therefore a validation-selected decoding
improvement, not yet a strong architecture claim. It becomes the new source
and target-free reference for confidence-gated ELF editing.

### 7. Source-preserving confidence gate

Implementation now supports two independent uses of T5 confidence:

1. the source latent always remains on the T5 manifold;
2. after ELF decoding, source subword tokens above a global confidence
   threshold are hard-copied back, while lower-confidence positions may be
   edited by the semantic flow.

The T5 confidence exporter records normalized selected-beam transition
probabilities and aggregates constituent SentencePiece probabilities to each
word by geometric mean. It does not read target text to form confidence.
Threshold, flow end time, and semantic scale are global hyperparameters chosen
on val110 metrics; they never use a row's target to route or edit that row.
Consequently all results below are validation-selected and test26 remains the
only valid confirmation set.

The completed confidence and transport ladders give the following matched
validation points after fixing the evaluator-visible ten-word contract:

| Semantic scale | Flow end | Copied source tokens | Content F1 | Word F1 | Sum | WER | Top-1 | Top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.25 | 89.48% | 0.02834 | 0.10499 | 0.13333 | 0.97818 | 0.01818 | 0.14545 |
| **0.10** | **0.25** | **89.48%** | **0.02834** | **0.10508** | **0.13343** | **0.97727** | 0.01818 | 0.12727 |
| 0.25 | 0.25 | 89.48% | 0.02834 | 0.10523 | 0.13357 | 0.97818 | 0.01818 | 0.12727 |
| 0.50 | 0.25 | 89.48% | 0.02678 | 0.10415 | 0.13093 | 0.98000 | 0.02727 | 0.13636 |
| 1.00 | 0.25 | 89.48% | 0.02834 | 0.10509 | 0.13344 | 0.97818 | 0.02727 | 0.16364 |

Scale `0.25` has the highest overlap sum, while scale `0.10` is frozen as the
balanced winner because it also attains the best WER. The scale-zero ablation
already improves the source through low-confidence latent projection by the
ELF decoder. Scale `0.10` adds a second favorable edit, demonstrating a small
incremental effect of the predicted-ADA-conditioned transport.

The clearest content correction is validation row 61:

- target: `passenger seat the driver was listening to really loud music`
- T5 source: `driver's bed and i took my car home`
- hybrid: `driver's seat and i took my car home`

The source confidence for `bed` is `0.124`, so it is editable under the global
`0.20` gate. On row 15, scale `0.10` also removes a low-confidence trailing
`with`, improving WER without target-informed routing.

A source-contract bug was found and fixed during this audit. A spaced hyphen
in one T5 proposal was counted as a whitespace word, causing its true tenth
lexical word to be dropped. External proposals now preserve punctuation and
cap by evaluator-visible lexical words; target latent length expands when
needed so hard-copying is a true identity operation. No target label is used
to choose or apply this correction.

### 8. Frozen winner audit

- Slurm job: `8791247` (array 0-1; completed).
- Arms: improved frozen T5 source and balanced T5-to-ELF-B hybrid.
- Evaluation: identical 110 rows/candidates, T5 retrieval, raw
  RoBERTa-large layer-17 BERTScore, BLEU-1, ROUGE-1, and 100,000 one-to-one
  row permutations.
- Outputs:
  `/data/engs-pnpl/glandau/elf-runs/meg_t5_beam4_winner_audit_20260912` and
  `/data/engs-pnpl/glandau/elf-runs/meg_t5_elfb_confcopy_winner_audit_20260912`.

| System | Content F1 | Word F1 | Sum | WER | Raw BERTScore | BLEU-1 | ROUGE-1 | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved frozen T5 source | 0.02669 | 0.10398 | 0.13067 | 0.97909 | 0.82888 | 0.10145 | 0.11387 | 0.02727 | 0.15455 |
| **Balanced T5-to-ELF-B hybrid** | **0.02834** | **0.10508** | **0.13343** | **0.97727** | **0.82901** | **0.10232** | **0.11579** | 0.01818 | 0.12727 |

The hybrid improves all six generation-quality point estimates; generated-text
retrieval is the explicit tradeoff. Against the one-to-one row-permutation
null, the hybrid's one-sided p-values are content F1 `0.000130`, word F1
`0.000010`, WER `0.039110`, raw BERTScore `0.000010`, BLEU-1 `0.000010`, and
ROUGE-1 `0.000010`. The improved T5 source has WER p `0.088359`, so the
hybrid crosses the predeclared `0.05` null threshold on WER.

Paired bootstrap comparisons against the stronger source remain appropriately
conservative. The 95% intervals for hybrid minus source are content F1
`[0.00000, 0.00496]`, word F1 `[0.00000, 0.00322]`, WER
`[-0.00455, 0.00000]`, raw BERTScore `[-0.00051, 0.00080]`, BLEU-1
`[-0.00003, 0.00266]`, and ROUGE-1 `[0.00000, 0.00470]`. This supports a
validation dominance result but not a claim that the small gain will
generalize without a protected evaluation.

## Person-name behavior

Person names are now a tracked row-level diagnostic. The fixed 110-row text is
lowercased, so every target and generation row was manually reviewed and the
annotations are bound to the exact paired text by SHA-256. Places,
organizations, brands, deities, demonyms, and eponymous syndrome names are not
counted. On a named target, exact recovery, wrong-name substitution, and name
deletion are mutually exclusive.

| System | Named target -> any name | Exact name | Wrong-name substitution | Name deletion | False-name insertion |
|---|---:|---:|---:|---:|---:|
| ELF-M lexical composite | 0/7 (0.00%) | 0/7 (0.00%) | 0/7 (0.00%) | 7/7 (100.00%) | 1/103 (0.97%) |
| Improved frozen T5 source | 3/7 (42.86%) | 0/7 (0.00%) | 3/7 (42.86%) | 4/7 (57.14%) | 1/103 (0.97%) |
| Current T5-to-ELF-B hybrid | 3/7 (42.86%) | 0/7 (0.00%) | 3/7 (42.86%) | 4/7 (57.14%) | 1/103 (0.97%) |

The T5 source has a strong name-presence signal without identity recovery:
named targets produce some person name on `42.86%` of rows, compared with
`0.97%` on unnamed targets (post-hoc one-sided Fisher exact `p=0.000631`). The
hybrid inherits all four generated names unchanged from T5, so the current ELF
edit does not yet add name recovery. Reproducible inputs and outputs are
`meg_t5_elf_gap_closure_20260912/person_name_annotations_val110.json`,
`name_audit_*.json`, and `scripts/audit_person_name_behavior.py`.

## Expanded confidence and edit-policy search

Two staged, validation-only searches were launched without opening test26:

- ARC job `8792728`: 12 raw T5 word-confidence thresholds crossed with 11
  semantic scales at flow end `0.25` (132 arms). The earlier successful
  partial-flow ladder fixed threshold `0.20`, so this measures a previously
  missing interaction rather than repeating the old screen.
- ARC job `8792740`: five continuous source-latent preservation weights at
  editable positions, crossed with four semantic scales and three flow end
  times (60 arms). High-confidence output tokens remain exact copies; this
  tests whether uncertain positions benefit from retaining partial T5 latent
  geometry instead of choosing only between hard copy and unrestricted edit.
- ARC job `8792781`: preserve every T5 function-word token and expose only
  low-confidence content words to editing, crossed over four thresholds, five
  semantic scales, and three flow endpoints (60 arms).
- ARC job `8792768` (completed): re-export the identical frozen beam-4 source
  with teacher-forced probability, selected-versus-alternative margin, and
  normalized inverse-entropy confidence. The generations and targets are
  byte-for-byte identical to the frozen source. Teacher probability is
  effectively redundant with transition probability (`r=0.999996`), whereas
  margin and inverse entropy are distinct (`r=0.870` and `0.908`).
- ARC job `8792870`: test six quantile-aligned thresholds for both margin and
  inverse entropy at four semantic scales (48 arms). The redundant
  teacher-probability grid is deliberately omitted.

The frozen current hybrid and all earlier candidates remain untouched. The
predeclared selection procedure is:

1. retain the Pareto frontier across content F1, word F1, WER, generated-text
   Top-1, and Top-5;
2. run raw BERTScore, BLEU-1, and ROUGE-1 for every frontier arm and every arm
   that exceeds the current overlap sum or lowers its WER;
3. promote a balanced winner only if its WER is no more than one edit worse
   than the current `1075/1100`, and at least four of the six text-quality
   metrics (content F1, word F1, WER, BERTScore, BLEU-1, ROUGE-1) are no worse;
4. among eligible arms, maximize word+content F1, breaking exact ties by WER,
   BERTScore, ROUGE-1, BLEU-1, Top-5, and Top-1 in that order; report a separate
   retrieval specialist if it differs.

All promoted arms will receive the same 100,000 one-to-one row-permutation
audit, paired comparison against the frozen T5 and current hybrid, qualitative
row differences, and the person-name audit above.

## Promotion criteria

A result is useful only if it improves T5's word+content sum `0.12493` without
materially worsening WER `0.97727`, or improves ELF's content/retrieval axis
while retaining its corrected near-T5 WER. The balanced hybrid satisfies this
validation criterion. No protected test26 row is loaded for development,
selection, or reporting.

The next architecture step, once real D'Ascoli proposals arrive, is to train
the editor on out-of-fold full-sentence proposals and calibrated token
confidences. A zero-initialized residual semantic edit gate inside each ELF
block remains preferable to treating low-confidence source tokens as
independent Gaussian noise; it preserves the identity path learned by this
successful validation experiment.
