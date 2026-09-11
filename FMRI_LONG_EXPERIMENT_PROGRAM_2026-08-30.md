# Long fMRI-to-Text Experiment Program

## Frozen reference

The machine-readable reference is `fmri_minilm_baseline_2026-08-30.json`. The current system is a known-text experiment: the ELF-B decoder and its exact-MiniLM context projector were trained on all 12,098 target sentences, including the 107 final sentences, while the 107 fMRI vectors remained unseen until the single locked evaluation.

The frozen reference pipeline is:

```text
fMRI -> frozen MRI2SEM MLP -> predicted 4 x 384 delayed MiniLM
     -> mean + residual MLP(1536, 512, 384)
     -> frozen known-text context projector -> frozen corpus-overfit ELF-B
```

The residual bridge alone was trained for 20 epochs on 11,725 story-level OOF MRI2SEM predictions. Its selected checkpoint occurred at step 1,000 (epoch 2.73), so longer optimization must be tested with explicit regularization and learning-rate schedules rather than assuming that more constant-LR steps will help.

## Evaluation contract

- All architecture, schedule, ADA, ELF-size, and end-to-end decisions use only the 11,725 training rows and 266 predicted validation rows.
- Validation decoding uses fixed seeds 7, 23, and 101.
- Primary metrics are content-word F1, word F1, generated-text retrieval mean rank/Top-5, and post-hoc validation BERTScore. WER is secondary.
- Retain Pareto-optimal checkpoints instead of choosing only the lowest WER checkpoint.
- Test107 stays closed during this program. After all comparisons, exactly one preselected candidate may be evaluated once against the frozen reference.
- The known-text assumption is reported explicitly; a future unseen-text split is a separate experiment.

## Experiment families

### A. Long-schedule MiniLM controls

Run the current residual bridge for 100 epochs with checkpoint evaluations every 1,000 steps and retain the best checkpoints independently by content overlap, word overlap, retrieval rank, structured quality, and WER.

Compare:

1. Constant LR `1e-4` as the matched long control.
2. Warmup plus cosine decay from `1e-4` to `1e-6`.
3. Residual width 1,024 with dropout `0.1` and cosine decay.
4. A two-stage curriculum: semantic alignment first, then diffusion plus semantic regularization.

### B. Delay-preserving architecture

Compare the selected residual MLP against a residual delay-token transformer that starts as the exact four-block mean, learns ordered-delay interactions, and continues to use the frozen known-text context projector. This avoids replacing the proven 384-D MiniLM-to-ELF interface with a randomly initialized 64-token projector.

### C. Staged end-to-end fMRI training

Use raw 40,000-D fMRI training/validation segments and load the pretrained MRI2SEM checkpoint.

1. Freeze the MRI2SEM trunk; jointly train its final `2048 -> 1536` projection and the semantic bridge.
2. If stable, unfreeze the last MRI2SEM residual block at a 10-20x smaller LR.
3. Keep the known-text context projector and ELF frozen initially.
4. Then compare unfreezing only the final 1-2 ELF blocks with a very small LR.

The objective retains MRI2SEM semantic contrastive/cosine supervision together with exact-MiniLM alignment, context consistency, and diffusion losses. Validation remains story held out.

### D. Embedding and decoder scale

Run a controlled 2x2 comparison:

| Semantic family | Decoder |
|---|---|
| delayed MiniLM | ELF-B |
| delayed MiniLM | ELF-L |
| ADA-002 | ELF-B |
| ADA-002 | ELF-L |

ELF-L requires a new known-text oracle checkpoint; ELF-B checkpoints cannot be loaded into ELF-L. ADA also requires an exact ten-word ADA oracle decoder and a comparable fMRI-to-ADA segment model. The existing ADA MRI2SEM run is useful as a preliminary ceiling (`seg10` test Top-1 `0.0561`, Top-5 `0.2150`) but uses only 2,048 voxels and three response lags, so it is not a fair replacement for the current 10,000-voxel/four-lag MiniLM setup.

### E. Decision analysis

For each validation checkpoint report:

- word/content F1 with paired bootstrap intervals;
- counts of improved/tied/worsened rows;
- generated-text retrieval Top-1/Top-5/mean/median rank;
- RoBERTa-large baseline-rescaled BERTScore;
- semantic-interface cosine and retrieval;
- repetition, length, and degeneration metrics;
- parameter count, wall time, peak memory, and training stage.

Select from the Pareto frontier. A model that gains overlap by producing longer generic text, or improves BERTScore while losing semantic retrieval, is not an unqualified improvement.

## Executable experiment matrix

The program is implemented in `submit-jobs/submit_fmri_long_program.sh`. It launches the following development jobs while keeping test107 closed:

| Job | Variants | Training/evaluation boundary |
|---|---:|---|
| `fmri_minilm_long_bridge_val.sbatch` | 5 | 100-epoch fixed-OOF sweep: constant LR, cosine LR, wider/dropout MLP, ordered delay-token transformer, and 20-epoch semantic-first curriculum |
| `fmri_minilm_rawbrain_e2e_projector_val.sbatch` | 2 | True raw-fMRI E2E stage 1: jointly train the MRI2SEM final projector and either residual MLP or delay-token bridge |
| `prepare_fmri_knowntext_ada002.sbatch` | 1 | Create exact row-aligned ADA-002 versions of the all-12,098 and train11725/val266 corpora |
| `fmri_knowntext_oracle_scale_semantic.sbatch` | 3 | Decoder/interface ceilings: MiniLM+ELF-L, ADA+ELF-B, ADA+ELF-L; existing MiniLM+ELF-B completes the 2x2 grid |
| `fmri_rawbrain_e2e_scale_semantic_val.sbatch` | 3 | Raw-fMRI validation for MiniLM+ELF-L, ADA+ELF-B, and ADA+ELF-L |

For raw-fMRI E2E, the code loads only `train_x` and `val_x`, verifies row identity through `story/start_tr/stop_tr`, and never reads `test_x`. MRI2SEM’s final `2048 -> 1536` projector uses LR `1e-5`; the zero-initialized semantic residual uses LR `1e-4`; the MRI2SEM trunk and known-text decoder are frozen. ADA starts from the pretrained 1,536-D MiniLM brain projector and adapts it toward exact 1,536-D ADA supervision.

All long jobs retain candidate checkpoints separately for content overlap, word overlap, generated-text Top-5/mean rank, sentence structure, and WER. BERTScore is added post hoc to the complete 266-row generation files for the retained candidates. The final MRI2SEM residual block and any ELF blocks are deliberately not unfrozen in the first wave: that second stage is gated on stable validation gains from the projector-only E2E stage, preventing a large trainable model from obscuring which interface change helped.

## Submitted ARC graph

Submitted on 2026-08-30 after 29 ARC unit tests passed and raw-fMRI/text metadata matched exactly:

- `8685696[0-4]`: five-way 100-epoch fixed-OOF MiniLM bridge/schedule sweep.
- `8685697[0-1]`: two-way 100-epoch true raw-fMRI MiniLM+ELF-B E2E-projector sweep.
- ADA corpus preparation is complete and validated: all-corpus `(12,098, 1,536)` and train/validation `(11,991, 1,536)`, with `11,725` train and `266` validation rows.
- `8685740[0-2]`: MiniLM/ADA and ELF-B/ELF-L known-text oracle grid.
- `8685741[0-2]`: corresponding raw-fMRI E2E validation grid, dependent on all of `8685740`.
- `8687778`: final post-hoc BERTScore, consolidated CSV/JSON/Markdown metrics, and brain-validation Pareto report; dependent on `8685696`, repaired E2E job `8687777`, and `8685741`.

Audit note: `8685698` completed all 12,098 ADA API embeddings but failed in the metadata join because its source archive lacked story labels. The join was repaired using the exact, verified 11,991-row sentence suffix and its resulting archive passed shape, sentence-order, split-count, and ZIP-integrity checks. Never-runnable or redundant queued dependency jobs (`8685709`, `8685710`, `8685732`–`8685736`) were cancelled; no training result was discarded and no API call was repeated.

The final test107 fMRI examples are not part of any submitted training, validation, or checkpoint-selection job.

## Interim status: 2026-08-30 16:20

- All five fixed-OOF 100-epoch jobs (`8685696[0-4]`) completed successfully in about 1.5 hours each.
- Raw-fMRI delay-token E2E completed successfully (W&B `ujh33py7`). Raw-fMRI residual-MLP task 0 lost a race while both array tasks created the same target-latent cache; it was resubmitted alone as `8687777` and is running from the now-complete cache.
- All three oracle-scale jobs (`8685740[0-2]`) are running. At this snapshot: MiniLM+ELF-L is at epoch ~24/100, ADA+ELF-B ~168/300, and ADA+ELF-L ~8/100.
- Brain-only partial BERTScore/report job `8687805` failed immediately because the environment did not expose the separately installed `bert-score` package. The report launcher now adds the pinned `bert-score==0.3.13` package root to `PYTHONPATH`; replacement job `8687882` is pending. Final report job is `8687778`.

The strongest completed candidate so far is true raw-fMRI E2E with the ordered delay-token bridge, selected on content overlap at step 35,000 / epoch 95.5:

| Validation-266 metric | Prior 20-epoch bridge, content-selected step 3,000 | New raw-fMRI E2E delay tokens | Change |
|---|---:|---:|---:|
| Content overlap F1 | 0.01058 | **0.02053** | +94% |
| Word overlap F1 | 0.06267 | **0.06636** | +5.9% |
| Generated Top-1 | 0.01128 | 0.01128 | tied |
| Generated Top-5 | 0.04887 | **0.05263** | 13 -> 14 of 266 |
| Mean rank | 112.49 | **101.79** | 10.70 ranks better |
| Median rank | 105 | **89** | 16 ranks better |
| Semantic cosine | 0.13895 | **0.19887** | +43% |
| WER | **0.9812** | 1.0320 | worse; secondary metric |

This is an interim single-seed validation result, not a test107 result. BERTScore and the residual-MLP E2E comparison are still pending.

## Stage 2: train through ELF

The selected stage-2 baseline is frozen in `fmri_minilm_stage2_baseline_2026-08-30.json`: raw-fMRI E2E with the residual ordered-delay transformer at step 35,000. Its validation-266 BERTScore F1 is `-0.07718`. Test107 remains closed.

Stage 2 continues from that exact checkpoint rather than reinitializing the MRI projector, delay mapper, context projector, or ELF weights. Three 100-epoch, story-balanced arms are compared:

| Arm | Trainable ELF parameters | ELF LR |
|---|---|---:|
| full ELF-B | all ELF-B parameters | `2e-6` |
| attention LoRA | rank-8 LoRA on every attention QKV/output projection | `1e-5` |
| attention+MLP LoRA | rank-8 LoRA on every attention and SwiGLU projection | `1e-5` |

All arms also continue the MRI2SEM final projector at `5e-6` and residual delay-token mapper at `5e-5`. The MRI2SEM trunk and T5 encoder stay frozen. Training uses only train11725, checkpoint selection uses only val266, and the 107 test brain vectors are never loaded. Primary comparison metrics remain content-word F1, word F1, BERTScore F1, generated Top-5, and mean/median retrieval rank.

Implementation validation job `8688007` completed successfully: 35 ARC tests passed, including zero-initialization equivalence, frozen-base/trainable-LoRA parameter checks, gradient propagation, base-checkpoint-to-LoRA remapping, semantic adapters, raw-fMRI composition, BERTScore reporting, and metric collection.

- Training array: `8688014[0-2]` (submitted; all tasks gated on validation job `8688007`).
- Post-hoc BERTScore and consolidated report: `8688015`, dependent on the complete training array.
- Report destination: `/data/engs-pnpl/glandau/elf-runs/fmri_stage2_elf_e2e_lora_report_20260830`.

Adaptive stop: all-parameter ELF-B task `8688014[0]` was stopped after step 21,000 when content F1 had fallen to `0.00967` and WER risen to `1.33459`, substantially worse than its retained step-5,000 best (`0.01666` content). Tasks 1/2 continue. The downstream ensemble dependency now names tasks 1/2 explicitly, so the intentional stop cannot block reporting.

Second adaptive stop: attention-only rank-8 task `8688014[1]` was stopped at epoch 63 after its latest content F1 fell to `0.01245` and its retained best remained `0.01609`, dominated by the content-focused arms. Task 2 (attention+MLP rank 8) remains the long standard-LoRA control. All task-1 generations remain in the later target-free candidate pool.

### Content-first adaptive branch

The first step-1,000 checkpoints from full ELF-B and attention-LoRA did not beat the frozen content baseline (`0.01092` and `0.01171` versus `0.02053`). They remain early and continue training. In parallel, a content-first branch directly upweights decoder cross-entropy on target subword tokens overlapping non-stopword content words. The branch uses content-token weight `3`, total decoder-loss weight `2`, and zero conditioning dropout, while retaining the semantic alignment objectives. This tests a loss-level content lever rather than waiting only on architecture and scale.

- Content-objective validation job `8688149`: completed, 38 ARC tests passed.
- Content-focused LoRA array `8688187[1-2]`: attention rank-8 and attention+MLP rank-8, pending/running as resources allow.
- Content-focused BERTScore/report job: `8688190`.

### True full-chain E2E branch

The first stage-2 matrix deliberately kept the pretrained MiniLM-to-ELF context projector frozen. A second, stricter E2E matrix now trains the complete path from the MRI2SEM final projector through the ordered-delay mapper and the full `384 -> 64 x 512` context projector, with optional ELF LoRA. All arms initialize exactly from the frozen baseline and run for 150 story-balanced epochs:

| Arm | ELF update | Content objective |
|---|---|---|
| full adapter | frozen ELF-B | standard |
| full adapter + LoRA | attention+MLP rank 8 | standard |
| full adapter + LoRA | attention+MLP rank 16 | content-token weight 2, decoder weight 1.5 |
| full adapter | frozen ELF-B | content-token weight 2, decoder weight 1.5 |

- Training array: `8688309[0-2]`, plus isolated content arm `8688350[3]`.
- BERTScore/consolidated report: `8688364`, dependent on both jobs.
- The checkpoint-ensemble dependency was extended to include `8688309` and `8688350`, so these periodic generations enter the same target-free semantic reranking pool.

### Accelerated scale dependencies

The original raw-brain scale array waited for all three oracle jobs, which would unnecessarily hold the nearly-complete ADA+ELF-B branch behind the slower ELF-L jobs. It was replaced by per-arm dependencies:

- `8688334[1]`: corrected ADA+ELF-B raw-brain E2E. The initial `8688216[1]` was stopped after about two minutes when its startup log showed that it had begun before the stronger-projector patch reached ARC; it had not reached the first validation checkpoint.
- `8688361[0]`: corrected MiniLM+ELF-L raw-brain E2E after oracle `8685740[0]`.
- `8688362[2]`: corrected ADA+ELF-L raw-brain E2E after oracle `8685740[2]`.
- `8688219`: consolidated scale report after all three accelerated arms.

The redundant pending array `8685741` and obsolete report `8687778` were canceled before they started; no training output was removed.

The accelerated raw-brain scale arms initialize from the frozen validation-trained delay-token baseline checkpoint rather than the original MRI2SEM `model.pt`. This reuses the stronger `2048 -> 1536` projector while keeping the large MRI2SEM trunk frozen; the same train11725/val266 contract applies.

Slurm snapshots submitted scripts. Therefore the never-started cached submissions `8688217`, `8688218`, `8688300`, `8688301`, and `8688314` were canceled and replaced by the corrected IDs above after their on-disk launchers changed. The replacement is configuration repair before execution, not a result-driven rerun.

### Frozen-baseline sampler sweep

A fast val266-only sweep evaluates the frozen stage-2 baseline across sampling seeds `{7, 23, 49, 101}` and CFG `{0.5, 1.0, 1.5}` at 32 ODE steps. This separates model improvement from sampling variance and can identify a stronger content-first inference configuration without using test107.

- Sampler array: `8688289[0-11]` (four seeds by three CFG values, maximum four concurrent GPUs).
- BERTScore and consolidated sampler report: `8688290`, dependent on the complete sampler array.
- Report destination: `/data/engs-pnpl/glandau/elf-runs/fmri_stage2_sampler_report_20260830`.
- Semantic-export validation: `8688296`; exact ARC GPU-environment test.
- Target-free semantic reranker: `8688297`, dependent on the sampler array and validation. It selects among all 12 generations by cosine similarity between each generated candidate's exact MiniLM embedding and the model-predicted MiniLM vector; reference text is not used for selection.
- Reranked BERTScore/consolidated report: `8688298`, dependent on the reranker.
- Checkpoint-ensemble semantic reranker: `8688363`, dependent on the continuing standard E2E/LoRA arm, content-focused LoRA array, and all four full-chain arms. Its pool includes every periodic val266 generation from those leakage-safe training runs plus the 12 sampler outputs.
- Checkpoint-ensemble BERTScore/consolidated report: `8688365`, dependent on the ensemble reranker.
- Early completed-checkpoint semantic ensemble: `8688321`. This uses only already finished validation generations from the frozen baseline, five 100-epoch OOF bridge runs, and the stopped residual-MLP raw-brain arm; it computes BERTScore in the same job and can produce an early result while long branches continue.

The weaker residual-MLP raw-brain arm was stopped after step 23,000 once its best retained checkpoint remained below the delay-token baseline (content F1 `0.01536`, word F1 `0.06061`, WER `1.03459`). Its checkpoints and metrics remain preserved; stopping it released a GPU for the content-focused and sampler branches.

### Content-recovery diagnostics and coverage-loss branch

Three target-free inference diagnostics failed to recover the missing content signal:

- strict retrieval from the 11,725-row training text bank: content F1 `0.01459`, word F1 `0.06128`, WER `0.98872`;
- semantic-cosine reranking over sampler candidates: content F1 `0.01748`;
- a train-only lexical probe used to rerank 234 generated candidates: content F1 `0.01585`, with predicted-content recall only `0.0895` even at 100 words.

These results indicate that candidate selection is not the main bottleneck. A new differentiable content-coverage objective therefore supplements position-wise decoder CE. For every target content subword, it maximizes the probability that the token occurs at least once anywhere in the decoded sequence. It uses training labels only and never reads val targets during optimization or any test107 rows.

At the 2026-08-30 18:55 snapshot, the strongest non-baseline full-chain candidate is the rank-16 attention+MLP LoRA/content arm at content F1 `0.01948` (step 6,000), about 5.2% below the frozen `0.02053` baseline. The standard full-chain arm retained `0.01872` at step 13,000, and the bridge-only content arm retained `0.01900` at step 22,000. These are promising but are not declared wins.

Two new 150-epoch full-chain variants add content-presence loss weight `0.1` on top of content-token weight `2` and decoder weight `1.5`:

| Arm | ELF update | Full trainable path |
|---|---|---|
| content coverage | frozen ELF-B | MRI projector + delay mapper + complete context projector |
| content coverage + LoRA | attention+MLP rank 16 | MRI projector + delay mapper + complete context projector + ELF LoRA |

The implementation has focused unit coverage for anywhere-in-sequence reward and differentiable zero behavior when no content tokens are marked. These arms are launched only after the ARC GPU-environment test passes. Checkpoint selection remains content F1 on val266, with BERTScore and WER used in that priority order; test107 remains closed.

- ARC validation `8688398`: completed; all focused tests passed.
- Coverage-loss training `8688400[4-5]`: submitted for the frozen-ELF and rank-16 LoRA variants.
- The initially updated report/ensemble submissions (`8688403`–`8688405`) were later superseded before execution when the stronger coverage-weight pair was added. Current IDs are recorded below. Slurm snapshots launch scripts, so every superseded still-pending job was canceled before it consumed compute.

Adaptive resource reallocation: standard attention+MLP LoRA `8688014[2]`, content-weighted attention+MLP LoRA `8688187[2]`, and standard full-chain rank-8 LoRA `8688309[1]` were stopped after their latest content F1 values declined to `0.01077`, `0.01369`, and `0.01613`; retained bests were only `0.01684`, `0.01813`, and `0.01801`. No checkpoint or metric was deleted. The stronger content-attention, full-chain frozen-ELF, and full-chain rank-16 content arms continue, while the released GPUs are reassigned to coverage-loss jobs.

The plain full-chain frozen-ELF arm `8688309[0]` was subsequently stopped after its latest content F1 fell to `0.01507` from retained `0.01872`; the rank-16 content arm remains the stronger full-chain control.

Because the observed unscaled coverage loss was only about `0.03`–`0.10`, weight `0.1` contributes much less than decoder CE. A second controlled pair at coverage weight `1.0` was submitted as `8688428[6-7]` (frozen ELF-B and rank-16 attention+MLP LoRA). The refreshed full-chain report is `8688429`; refreshed ensemble/report are `8688430`/`8688431`. The no-coverage frozen-ELF content control `8688350[3]` was stopped after declining from retained content F1 `0.01761` to `0.01540` at step 7,000, freeing its GPU for the stronger sweep. All its outputs remain preserved.

Initial weight-1 results: the frozen-ELF arm reached content F1 `0.01899`, word F1 `0.06806`, WER `1.02820` at step 1,000. This does not yet beat the primary content baseline, but it beats baseline word F1 (`0.06636`) and is close enough for continued training. Its immutable step-1,000 snapshot is being evaluated at seed 101 / CFG 1.5 in sampler job `8688452[11]`. The weight-0.1 pair was stopped after later content fell to `0.01228`/`0.01203`; the stronger pair continues.

The seed-101/CFG-1.5 sampler check reduced content to `0.01275`, so a broad sampler sweep was rejected. The full-adapter frozen-ELF weight-1 arm was stopped after its next three validations declined to `0.01552`, `0.01531`, and `0.01048`; its step-1,000 best and immutable snapshot remain. Rank-16 LoRA weight-1 continues through the later-recovery window seen in its no-coverage control.

The early full-adapter peak followed by rapid decline suggests overfitting in the approximately 142M-parameter context projector. Low-capacity schedule job `8688474[3-4]` therefore freezes the context projector and ELF-B, trains only the MRI final projector and ordered-delay residual mapper, and applies coverage weight `1.0`. Task 3 uses the standard mapper/brain LRs (`5e-5`/`5e-6`); task 4 uses `1e-5`/`1e-6`. Its initial report/ensemble submissions were superseded before execution as the focused local grid expanded.

The low-capacity schedule confirmed that smaller LR is essential. Standard LR produced content F1 `0.00902`; low LR reached `0.01959` at step 1,000, then declined. A staged low-LR continuation from the full-adapter step-1,000 snapshot reached only `0.01650` then `0.01523`. Coverage weight `2.0` reached `0.01600`; coverage `0.5` reached `0.01917`. The latter job then encountered a transient W&B API timeout after safely writing its checkpoint; training code now catches telemetry exceptions so logging cannot invalidate future local results.

Ultra-low mapper/brain LR (`2e-6`/`2e-7`) reached content F1 `0.02002`, word F1 `0.06670` at step 2,000; midpoint LR (`5e-6`/`5e-7`) reached content F1 `0.02006`, word F1 `0.06778` at step 1,000. Both are new word-overlap improvements but remain just below primary content baseline `0.02053`. Later checkpoints declined and the corresponding best checkpoints remain preserved.

Final objective-localization job `8688583[9-10]` tests midpoint LR with either content-token CE weight `3` / decoder weight `2`, or no semantic alignment/contrastive/context auxiliaries. This asks whether the remaining lexical gap is caused by insufficient content pressure or loss competition while keeping the context projector and ELF frozen. The intermediate target-free semantic ensemble over prior candidates reached only content F1 `0.01867` and was rejected.

Task 9 produced the first checkpoint above the frozen primary content metric at step 1,000: content F1 `0.020913` versus `0.020534` (+1.85%). Its word F1 is `0.064950`, WER `1.034586`, and BERTScore F1 `-0.080750`, so it is retained as a **content-specialist candidate**, not promoted to the overall baseline. Paired validation-row bootstrap (20,000 samples) gives a content difference of `+0.000379`, 95% interval `[-0.003480, 0.004323]`; 13 rows improve, 240 tie, and 13 worsen. Steps 2,000--5,000 all lost content, and the dominated continuation was stopped while preserving its step-1,000 `best.pt`.

A local follow-up grid `8688617[11-13]` now asks whether that narrow content gain can coexist with better BERTScore and WER. It compares: (11) 2.5x smaller input-side learning rates; (12) half-strength order-invariant content coverage; and (13) a smaller total decoder coefficient. The context projector and ELF-B remain frozen in all three, semantic auxiliaries remain active, and test107 remains unopened. Exact BERTScore for retained checkpoints is computed post hoc; candidate selection remains on val266 only.

All three local schedules peaked below the content baseline (`0.01995`, `0.01853`, and `0.02016`) and then declined, so they were stopped with checkpoints preserved. Two higher-level follow-ups are active:

- checkpoint interpolation `8688642[0-4]` evaluates alphas `{0.10, 0.25, 0.50, 0.75, 0.90}` between the immutable baseline and content specialist. Construction uses only weights, never reference text, and evaluates val266 at the fixed seed;
- recovery E2E/LoRA `8688659[14-16]` initializes from the content specialist and compares rank-4 attention+MLP LoRA restricted to the final two ELF blocks against a low-LR full context-projector update. ELF LoRA LR is `1e-6`, mapper LR `1e-6`, and brain-projector LR `1e-7`, substantially smaller and narrower than the rejected broad-LoRA arms.

The ADA+ELF-L oracle was adaptively stopped at step 43,000 after retaining a strong step-42,000 ceiling: content F1 `0.03430`, word F1 `0.06651`, Top-5 `0.22430`. Its 3.17-GB best checkpoint was copied to a size-verified immutable snapshot, and leakage-safe raw-fMRI ADA+ELF-L validation was launched immediately as `8688653[2]`. This reallocates the remaining GPU time from oracle-only optimization to the actual brain interface; test107 brain remains closed.

The MiniLM+ELF-L oracle ceiling improved to content F1 `0.03962` at step 39,000, almost twice the real-brain baseline. To avoid waiting several more hours for all 100 oracle epochs, that 3-GB best checkpoint was copied to an immutable, size-verified step-39,000 snapshot. Accelerated real-fMRI MiniLM+ELF-L job `8688444[0]` loaded that snapshot successfully and verified raw inputs `(11,725, 40,000)` train / `(266, 40,000)` validation. It trains only the MRI final projector plus 1.97M-parameter residual semantic mapper, freezes ELF-L and the MRI trunk, and never loads test107. The never-started dependency-gated duplicate `8688361[0]` was canceled.

Its first real-fMRI ELF-L checkpoint is poor despite the strong oracle ceiling: content F1 `0.00531`, word F1 `0.03880`, WER `1.36541`, generated Top-5 `0.02632`, and semantic-interface cosine `0.20940`. It continues briefly to test whether projector/residual adaptation closes the geometry gap; this result is not a candidate improvement.

The second ELF-L checkpoint remained poor (content F1 `0.00713`, word F1 `0.03522`, WER `1.31654`), so the arm was stopped and its outputs preserved. The strong oracle ceiling therefore does not transfer through the current brain interface.

## Late-stage LoRA recovery and validation leader

The first narrow recovery uses rank-4 attention+MLP LoRA only in ELF-B blocks 10--11. It starts from the content-specialist checkpoint, keeps the large MRI2SEM trunk frozen, and updates the MRI final projector, ordered delay mapper, and LoRA weights at low learning rates. A second recovery starts from that arm's step-3,000 checkpoint and raises semantic alignment/contrastive/context weights while keeping the same narrow trainable path.

The second recovery's step-1,000 checkpoint is the strongest individual generator found in the late search:

| Metric | Frozen stage-2 baseline | Recovery step 1,000 | Change |
|---|---:|---:|---:|
| Content-word F1 | 0.020534 | **0.023116** | +12.6% |
| Word F1 | 0.066359 | **0.066611** | +0.4% |
| WER | 1.031955 | **1.031579** | slightly better |
| Generated Top-5 | 0.052632 | **0.060150** | 14 -> 16 of 266 |
| BERTScore F1 | **-0.077178** | -0.084597 | worse |

This checkpoint is retained as the best single-generator content specialist, not the overall leader, because its BERTScore regressed. A 50,000-sample paired bootstrap estimated content delta `+0.002582` with probability of improvement `0.853`; the interval `[-0.002193, 0.007459]` includes zero.

Several attempts to combine the complementary checkpoints in weight space failed. Baseline/content-checkpoint interpolation and merged-LoRA/baseline soups all fell below the content baseline. Fixed sampler variations of the earlier content and LoRA checkpoints also failed. ADA+ELF-L and MiniLM+ELF-L retained high oracle ceilings but did not transfer through raw fMRI. These rejected branches remain preserved and are not counted as improvements.

### Train-only RoBERTa reranker

The final successful lever operates after generation and never uses a validation reference for per-row selection:

1. Fit a ridge map from exact MiniLM-384 to RoBERTa-large layer-17 mean embeddings using only train11725.
2. Select ridge `10` on a story-group-heldout subset of train11725; held-out cosine is `0.98443`.
3. For each validation brain row, score five already generated candidates by cosine to the mapped RoBERTa vector.
4. Add a fixed structural prior of `0.001 * abs(generated_word_count - 10)` because every task target is defined as exactly ten words.

The candidate pool contains the frozen baseline step 35,000, first-recovery steps 2,000/3,000, and second-recovery steps 250/1,000. Their final selection counts are `81/71/44/44/26`. Validation targets are read only after all row selections to report metrics. This is therefore a validation-selected multi-candidate decoding system rather than a single-checkpoint generator.

| Validation-266 metric | Frozen stage-2 baseline | New leader | Change |
|---|---:|---:|---:|
| Content-word F1 | 0.020534 | **0.023005** | +0.002471 / +12.0% |
| Word F1 | 0.066359 | **0.069709** | +0.003350 / +5.0% |
| BERTScore F1 | -0.077178 | **-0.060364** | +0.016813 |
| WER | 1.031955 | **1.030827** | -0.001128 |

Post-hoc T5-small retrieval for the same frozen generations gives Top-1 `0.0000`, Top-5 `0.06391`, mean rank `98.33`, and median rank `78`. Relative to the single-checkpoint baseline, Top-1 falls from 3/266 to 0/266, while Top-5 improves from 14/266 to 17/266 and mean/median rank improve from `101.79/89`. This mixed retrieval result is reported rather than hidden; the promotion is driven by the requested content/BERTScore/WER priority.

Paired bootstrap over 50,000 validation-row resamples gives:

- BERTScore delta 95% CI `[0.01037, 0.02330]`, probability better `1.000`;
- content delta CI `[-0.00171, 0.00691]`, probability better `0.875`;
- word delta CI `[-0.00250, 0.00925]`, probability better `0.869`;
- WER delta CI `[-0.00752, 0.00526]`, probability better `0.650`.

Thus the BERTScore gain is robust on the paired validation rows, whereas the smaller lexical/WER gains are positive means but remain uncertain at this sample size. The original single-checkpoint stage-2 baseline remains immutable. The new system is frozen separately in `fmri_stage2_validation_leader_2026-08-30.json`, with complete generations, BERTScore rows, bootstrap report, ranked examples, and a target/generated CSV. Test107 remains sealed and has never been loaded by this search; it is reserved for one later confirmatory evaluation of the frozen system.

ARC provenance for this phase:

- `8688829[18]`: second semantic-strength recovery; stopped after later decline, retaining step 1,000;
- `8688927`: train-only RoBERTa mapper and five-candidate reranker;
- `8688935`: fixed ten-word length-penalty grid;
- `8688946`--`8688948`: exact BERTScore for the three eligible length penalties;
- `8688960[4,5,7,11]`: additional 16-step sampler diversity from the recovery checkpoint, still validation-only at the time this leader was frozen.

### OOF-noise-matched content-first operating point

The exact-MiniLM-to-RoBERTa mapper above still has a train/inference geometry mismatch: its train inputs are perfect MiniLM vectors, while its validation input is brain-predicted. A second mapper instead uses the 11,725 story-OOF predicted delayed MiniLM vectors as train inputs, mean-pools their four ordered delay blocks from 1,536 to 384 dimensions, and maps those noisy inputs to the same train-text RoBERTa targets. Its validation input is the previously frozen 266-row prediction ensemble from the same OOF archive. Test rows are explicitly absent from that archive.

A very small fixed ten-word penalty (`0.0001`) yields the highest content-first operating point:

| Validation-266 metric | Frozen baseline | OOF content-first | Balanced leader |
|---|---:|---:|---:|
| Content-word F1 | 0.020534 | **0.023666** | 0.023005 |
| Word F1 | 0.066359 | 0.068828 | **0.069709** |
| BERTScore F1 | -0.077178 | -0.060453 | **-0.060364** |
| WER | 1.031955 | 1.031955 | **1.030827** |
| T5 Top-5 | 0.052632 | 0.056391 | **0.063910** |
| T5 mean rank | 101.79 | 100.58 | **98.33** |

The OOF variant is frozen as `fmri_stage2_content_first_leader_2026-08-30.json` because the requested priority is content first. It improves content by `+0.003132` / 15.3%, improves BERTScore by `+0.016724`, and does not worsen WER. Its 50,000-sample paired bootstrap probabilities of beating baseline are `0.922` for content and `1.000` for BERTScore; BERTScore CI is fully positive (`[0.01054, 0.02301]`), while the content CI still crosses zero (`[-0.00120, 0.00771]`).

The exact-input version remains the recommended balanced operating point because it wins word overlap, BERTScore, WER, T5 Top-5, and mean/median T5 rank while giving up only `0.000661` content F1. Both systems have T5 Top-1 `0/266`, below the baseline's `3/266`; this weakness is explicitly retained in the report. No test107 decision was made from either validation system.

Adding all 12 completed frozen-baseline sampler outputs to form a 17-candidate pool was rejected. The OOF/exact mappers peaked at content F1 `0.01858`/`0.01884`, both below the original baseline and far below the five-candidate leaders. Their global length penalties could improve WER as low as approximately `1.001`, but only by sacrificing the primary content metric. The jobs were stopped after their complete metric grids were written, before redundant BERTScore work, and all outputs remain available for audit.

## 2026-08-31: single-checkpoint content/WER audit and next experiments

The immutable single-checkpoint reference for this search is the second
rank-4 recovery checkpoint at step 1,000:

```text
fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000/checkpoints/eval_step_00001000_score_0.403006.pt
```

Its val266 metrics are content-word F1 `0.023116`, word F1 `0.066611`, WER
`1.031579`, generated Top-5 `0.060150`, and BERTScore F1 `-0.084597`.
Test107 remains sealed.

The first assumption audit produced four important results:

1. Fixed output caps of 8/9/10 words reduce WER to
   `0.980827/0.981203/0.987594`, but content F1 falls to
   `0.021533/0.021069/0.022415`. Length control alone therefore cannot meet
   the joint objective.
2. Even with perfect reordering of words already present in the generated
   outputs, the vocabulary-limited WER floor is approximately `0.984586`.
   Reaching WER `0.90` requires recovering many new correct word instances,
   not merely deleting insertions.
3. A train-only exact-MiniLM lexical probe reaches top-10 content recall
   `0.38349`, while the same prototype method on brain-predicted MiniLM reaches
   only `0.01995`. Most lexical information is lost before diffusion.
4. The residual ordered-delay transformer preserves four tokens internally,
   but mean-pools them to one 384-D vector before the proven
   `384 -> 64 x 512` context network. The four delays were therefore not
   actually retained as separate ELF conditioning slots.

A bidirectional decoder objective was implemented: target-content coverage
rewards recall, and content precision penalizes likely content words absent
from the target. The balanced precision arm (`8689226_20`) declined
monotonically to content F1 `0.01973` by step 750. The milder arm
(`8689199_19`, W&B `tj1b49f6`) reached content F1 `0.018997` and WER
`1.030075` at step 250. Both were stopped. This rejects decoder precision as
a standalone fix: it reduces hallucination slightly but suppresses genuine
content more strongly.

Two complementary changes are now active:

- direct train-only lexical ranking on the predicted 384-D semantic vector,
  with a semantic-only fast-training mode (`8689331[23-26]`,
  `8689347[27-29]`, and V100 short probe `8689379_28`);
- a zero-initialized residual projection that adds the four centered 384-D
  delay blocks to the first four ELF context slots. Step zero is exactly the
  reference model, so any change is attributable to learned delay retention.
  Long jobs are `8689363[30-31]`; short probe `8689364_30` compares the plain
  delay branch against the same branch plus semantic lexical ranking.

Per-delay train-only lexical diagnostic `8689378` measures whether useful
content signal is concentrated in a particular delay or becomes visible only
under score-level fusion. Test arrays are not loaded by any of these jobs.

The completed per-delay audit used a 5,000-word train-only vocabulary and
found top-10 content recall `0.34288` for exact ten-word MiniLM, but only
`0.05953`--`0.06631` even for the four oracle delayed MiniLM blocks. The
brain-predicted blocks reached `0.00678`--`0.00980`; mean/max score fusion did
not materially improve them. This exposes a target-contract mismatch in
addition to brain noise: MRI2SEM was trained toward broad delayed semantic
windows, while ELF is asked for a content-maximized exact ten-word window.

Direct-contract jobs `8689384[32-33]` (short V100) and `8689385[32-33]`
(long fallback) therefore initialize a single 384-D MRI head by averaging the
four contiguous pretrained output heads, load the existing exact-MiniLM
context network, and fine-tune only against the exact ten-word MiniLM target.
Task 32 tests semantic alignment alone; task 33 adds train-only lexical
ranking. This remains a single-checkpoint model and does not load test107.

The first direct-contract evaluations clarified the next handoff. Semantic
alignment alone at step 250 increased validation semantic Top-5 from
`0.08271` to `0.10902` and generated-text Top-5 from `0.06015` to `0.08647`.
This was a genuine validation-only retrieval improvement, although its frozen
ELF output still had only `0.01264` content F1. By step 500, semantic Top-5
had regressed to `0.08647`; the immutable step-250 state was therefore copied
to `semantic_retrieval_step250_snapshot.pt`. Adding the fixed lexical
prototype objective reached semantic Top-5 `0.10150` and content F1 `0.01539`
at step 250, below the clean direct objective, so that branch was stopped.

The explicit four-delay context branch also failed to beat the reference.
Across steps 250--2,000, its best content F1 was `0.02157` at step 750. The
branch was stopped because later points remained below `0.020`, demonstrating
that retaining the four delayed features inside ELF is insufficient when the
delayed feature targets themselves are weakly related to the exact phrase.

Three single-checkpoint E2E recovery arms (`8689459[34-36]`) now start from
the preserved direct-384 step-250 state. They compare (a) semantic-protected
last-two-block rank-4 LoRA, (b) the same path with stronger content CE and
coverage, and (c) a frozen improved brain head with only the residual context
mapper and LoRA trainable. All select on val266 content F1 and never load
test107.

The target-timing audit found that the content-maximized ten-word phrase is
not the literal temporal-center phrase in most rows: only `39.85%` of val266
targets contain the nearest center word, the selected-window midpoint is
`3.07 s` from center on average, and the maximum offset is `8.50 s`. The
reference checkpoint obtains content F1 `0.02734` for targets within 2 seconds
of center, versus `0.01879` for offsets between 2 and 5 seconds. CPU job `8689540`
therefore prepares two train11725/val266 semantic auxiliaries without loading
predicted test brain vectors: an exactly centered ten-word MiniLM target and a
MiniLM target spanning every word in the full 20-second brain interval.

Finally, the lexical-probe program fits a train-only supervised multilabel lexical
head on the frozen MRI hidden state, a linear hidden-state control, and the
current 1536-D MRI semantic output. This probe bypasses ELF and asks whether
the MRI encoder retains held-out-story content that the semantic bridge loses.
If hidden-state lexical prediction is materially stronger, the next model will
add a learned content bottleneck and decoder-token guidance inside one saved
checkpoint; if it is not, the temporally aligned semantic target becomes the
primary brain-side correction.

The first random-initialized hidden-state MLP probe completed in devel job
`8689544`. Its best top-5 content F1 was `0.04153` and it found at least one
correct word in `18.42%` of val rows. A mandatory train-frequency-only control
was stronger: top-5 content F1 `0.05430` and nonzero fraction `24.06%` from
the fixed words `know/one/really/two/back`. The random head therefore does not
establish brain-specific lexical information and will not be integrated into
ELF. Replacement array `8689631[0-2]` and devel probe `8689632_0` initialize
exactly at the frequency control (zero feature weights plus train log-prior
bias) and train only a residual brain-dependent correction. Only validation
improvement above the epoch-zero prior counts as evidence of lexical signal.

The completed prior-residual hidden-state probe (`8689632_0`) found only a
very small transient lift: epoch-zero top-5 F1 was `0.05430`, epoch one reached
`0.05651`, and all later epochs ultimately overfit.  This `+0.00221` change is
not yet accepted as brain-specific evidence.  Three target-permutation runs
(`8689657[3-5]`) preserve the exact train word frequencies while destroying
the brain/text pairing; they quantify the amount of apparent lift expected
from optimizer noise and repeated epoch selection.  Linear-hidden and
semantic-1536 controls run separately in `8689660[1-2]`.

This control exposed a broader evaluation problem.  On the same val266 target
rows, emitting the fixed train-prior phrase `know one really two back` for
every input gives content-word F1 `0.04217` and WER `0.98459`, compared with
the nominal single-checkpoint reference's `0.02312` and `1.03158`.  Therefore,
content F1 and WER alone can reward a language prior that ignores fMRI.  From
this point, promotion requires all of the following: improvement over the
single-checkpoint reference, a matched-vs-deranged-brain advantage, and a
comparison against the train-only constant-language null.  Test107 remains
sealed; none of these controls load it.

### Full-window target correction and lexical residual

The permutation controls completed and confirmed a small but real conditional
lexical trace. The prior-initialized hidden MLP reached top-5 F1 `0.05651`,
while three shuffled-pairing controls reached `0.05456`, `0.05430`, and
`0.05430`. A linear hidden head reached `0.05639`; the existing 1,536-D
semantic output reached `0.05769`; and the improved direct-384 semantic head
reached `0.05848`. The signal is weak, but it is reproducibly above the
train-frequency null and is slightly more accessible after semantic
projection than in the raw MRI hidden state.

The temporal-target audit identified a substantially better brain-side
contract. The direct-384 step-250 prediction was evaluated against three
MiniLM targets on val266:

| Semantic target | cosine | Top-1 | Top-5 | mean rank | median rank |
|---|---:|---:|---:|---:|---:|
| exact content-selected 10 words | 0.16331 | 0.03008 | 0.10902 | 79.20 | 59 |
| literal centered 10 words | 0.16814 | 0.03008 | 0.09398 | 82.08 | 64 |
| complete 20-second interval | **0.18306** | **0.03383** | **0.17669** | **68.74** | **42** |

Thus the mean-pooled 20-second fMRI input is much more identifiable as the
full-window semantic content than as the post-hoc content-maximized 10-word
phrase. The full-window and exact target embeddings have cosine `0.5314` and
set-word F1 `0.3895`, so they remain related but are not interchangeable.

A lexical head trained on full-window content and evaluated on the exact
ten-word target emits five words per row with content F1 `0.04865` and WER
`0.98459`. Across 20 derangements of the predicted rows, mean content F1 is
`0.04330` and the maximum is `0.04717`; matched brain therefore beats every
derangement, though only by a small margin. The exact-window head reaches
content F1 `0.04574` versus deranged mean `0.04157`. These word-list decoders
are useful evidence but are not fluent final generators: the exact head has
BERTScore F1 `-0.13168`, below the ELF reference's `-0.08460`.

Mapping either lexical score vector back to one retrieved train sentence was
rejected. Exact-head retrieval produced content F1 `0.02985`, WER `0.99173`;
full-window-head retrieval produced content F1 `0.00881`, WER `0.99135`.
Lexical evidence should condition generation directly, not snap the output to
memorized text.

Directly adding the predicted word-prototype mixture to the MiniLM vector
reduced semantic retrieval at every epoch and was rejected. The active
alternative keeps the proven MiniLM vector unchanged, converts the frozen
head's top-k train-only word prototypes through the ELF context network, and
adds them as a separate context stream through a zero-initialized 64-slot
gate. At step zero this is exactly the direct-384 checkpoint. Only the gate,
residual semantic mapper, and rank-4 LoRA in the last two ELF-B blocks train.
Exact-head job `8689716[38]` and full-window-head jobs
`8689725[38]`--`8689727[38]` are validation-only long runs; smoke job
`8689728[38]` checks the branch at step 100.

The next lexical experiments remove the train-frequency logit prior before
top-k and add a matched-vs-deranged ranking loss. For a target word, the
ranking comparison uses a brain row whose training target does not contain
that word, so the objective cannot be solved by a constant word-frequency
prior. Jobs `8689730`--`8689739` sweep prior subtraction for the existing
exact/full-window heads. Jobs `8689740[6]`--`8689746[6]` train pairwise
residual heads, including broad-window supervision with checkpoint selection
on the exact validation phrase. All selection remains on val266; test107 has
not been loaded.

### Overnight temporal and ordered-lexical program (2026-08-31)

The first temporal-index audit failed because the new reconstruction treated
`stop_tr` as an exclusive slice endpoint. MRI2SEM's segment builder records
the last included TR, so a ten-TR segment has `stop_tr - start_tr + 1 == 10`.
After correcting the reconstruction, an audit over train11725 and val266
recovers exactly ten consecutive rows for every segment. It indexes no test
key. This bug affected only the not-yet-trained temporal branch, not the
established mean-pooled baseline.

The lag representation also contains substantial duplication. Ten segment
rows each hold four response lags, producing 40 labeled time-by-lag vectors,
but their chronological union contains only 13 unique response TRs
(`t-3 ... t+9`). The temporal matrix therefore compares both representations:

- three from-scratch 40-token models with exact/full-window auxiliary weights
  `0`, `0.5`, and `1.0` (`8689823`--`8689825`);
- one from-scratch 13-unique-TR control with full-window weight `0.5`
  (`8689884`);
- four zero-initialized temporal-residual models (`8689882[0-3]`) that start
  exactly at the frozen direct-384 prediction. These compare 40 versus 13
  tokens, fixed versus full-to-exact curriculum, and 64 versus 128 hidden
  dimensions.

The zero-output residual design is deliberately lower risk than replacing
MRI2SEM: epoch zero is numerically identical to the established direct-384
semantic model. A focused ARC GPU check (`8689902`) passed both tokenizations,
step-zero identity, gradient-flow, and ordered-decoder prior initialization.

Both full-window lexical-to-ELF context arms evaluated so far were dominated.
Pairwise-head top-5 content remained only `0.015` through step 1,000, with WER
`1.031`; the earlier pair-0.3 arm reached the same `0.0154`. They were stopped
after four evaluations, preserving all checkpoints. The top-8 arm remains as
the last context-gate control, but this architecture is no longer the primary
route.

A target-free substitution diagnostic establishes that the pairwise lexical
head contains useful conditional signal even though ELF ignores it. Starting
from the immutable baseline generation, replacing up to five existing content
positions from right to left with the head's ordered word predictions gives:

| val266 metric | immutable baseline | lexical substitution |
|---|---:|---:|
| Content-word F1 | 0.023116 | **0.050510** |
| Word F1 | 0.066611 | **0.082848** |
| WER | 1.031579 | **1.030451** |

Across 200 derangements of lexical predictions relative to brain rows,
content F1 has mean `0.039108` and maximum `0.047019`; matched brain exceeds
every derangement by at least `0.00349`. This is strong evidence for
brain-specific lexical information, but it is not promoted as the final
model: it composes a separate lexical head with a heuristic substitution
rule, and it barely changes WER.

The bag-of-words target itself explains that WER failure. A new ten-position
word decoder starts exactly at train-only per-position word-frequency priors,
then learns a brain-dependent residual with position-wise cross-entropy and a
matched-versus-mismatched ranking loss whose prior bias cancels. Linear and
MLP semantic heads, a stronger pairwise loss, and a hidden-2048 control are
queued as `8689910[0-3]`, with fast probe `8689911`. Every epoch reports the
epoch-zero prior, multiple prior-subtraction operating points, matched brain,
and fixed derangements. This branch directly tests whether the recovered
content can be placed in target order and therefore lower WER. Test107 remains
sealed.

### First single-checkpoint beat and OOF-to-T5 diagnosis (2026-08-31)

The ordered position head by itself did not recover enough content. Its
train-only positional prior reached WER `0.95977` with zero content overlap;
after training, the best independent position-head configurations remained
below the immutable content baseline. Integrating the already frozen
pairwise content head as a broadcast lexical bias was more successful. The
checkpoint bundles both heads and selects only on val266. Its best operating
point (`content strength 0.5`, position-prior subtraction `0.25`) gives:

| val266 metric | immutable fluent baseline | integrated ordered-content checkpoint |
|---|---:|---:|
| Content-word F1 | 0.023116 | **0.025960** |
| Word F1 | 0.066611 | 0.051650 |
| WER | 1.031579 | **0.977444** |
| Deranged content F1, mean / max | not available | 0.017996 / 0.021584 |
| Matched-minus-deranged content margin | not available | **0.007964** |

This is the first leakage-safe single-checkpoint result that beats the locked
content metric while improving WER. It remains a rigid ten-position lexical
decoder rather than the desired fluent system. The related heuristic
position-prior/lexical hybrid has content F1 `0.05003`, WER `0.97331`, and
matched brain beats all 200 derangements, but its BERTScore F1 is `-0.16327`;
it is evidence and an upper-bound construction, not the final model.

A frozen-T5 prefix decoder localizes the fluent bottleneck. With exact
MiniLM384 input, a learned 16-token prefix reaches content F1 `0.22049` and
WER `0.92669` at oracle epoch 10; the best observed oracle WER is `0.90827`
at epoch 15. Replacing exact MiniLM with the current in-sample-trained brain
representation collapses to content F1 `0.01853`, WER `1.02105`, despite a
positive derangement margin. The decoder is therefore capable; the main
failure is the brain-to-semantic interface.

The first T5 brain phase used in-sample MRI2SEM train outputs but genuinely
held-out validation outputs. That train/validation noise mismatch is now
removed. New jobs consume the audited robustness archive directly: 11,725
story-level OOF delayed MiniLM predictions for training and the separate
266-row prediction ensemble for validation. Test rows are absent. Mean pooling
the four OOF delay blocks gives validation cosine `0.08381`, Top-1 `0.03383`,
Top-5 `0.12782`, mean rank `70.88`, and median rank `47.5` against exact
MiniLM. A leakage-safe OOF lexical head on these features reaches top-5
set-F1 `0.05647` after one linear-head epoch.

The active fluent matrix separates three assumptions:

- OOF mean versus all four ordered delay blocks;
- frozen oracle prefix plus an identity-initialized residual calibrator versus
  direct prefix-projector fine-tuning;
- joint token/semantic optimization versus a semantic-only warm-up followed
  by low-LR token training.

Every calibrator evaluation now logs exact-MiniLM retrieval as well as text
metrics and fixed derangements. A further single-checkpoint branch bundles the
OOF lexical head into the T5 checkpoint and applies a target-free vocabulary
logit bias during beam search. It sweeps top-3/top-5/top-8 and bias strengths
`0.5`, `1.0`, and `2.0`; matched and deranged brains receive the identical
procedure. This is intended to inject the proven conditional content signal
without replacing fluent T5 generation with word-list post-processing.

The earlier bidirectional content recall/precision loss is rejected as a
training route: target-set rewards were too easy to solve through frequent
content priors and did not retain brain dependence. Current lexical branches
instead use story-OOF inputs, matched-versus-deranged controls, prior
subtraction, and target-free generation-time conditioning. Test107 remains
sealed until one configuration is frozen.

### Objective-corrected fluent overnight program (2026-08-31)

Longer frozen-T5 training by itself did not close the lexical gap. The best
completed plain long OOF run retained content F1 `0.01883` with WER `0.98421`;
the first capped rank-4/8 LoRA probes reached content F1 `0.02017`, WER
`0.98684`, and a positive matched-minus-deranged margin, but their best state
was still the semantic warm-up rather than the subsequent token stage. This
shows that ordinary teacher-forced CE can improve the language prior while
reducing dependence on the brain condition.

The next T5 grid therefore changes the objective rather than merely adding
epochs. It adds (1) positional CE upweighting only on target content-token
positions, (2) a matched-versus-rolled-brain sequence-NLL margin, and (3) an
optional train-time frozen lexical keyword context that removes the prior
generation-only train/inference mismatch. These are single-checkpoint,
story-OOF, target-free-at-inference mechanisms. Validation smoke job `8690152`
passed the weighted-NLL, keyword, ordered-delay, and LoRA gradient contracts;
short objective grid `8690155[27-34]` is queued.

A second audit found that the cosine-dominant semantic warm-up is partly
collapsing sample identity: raw OOF mean features have exact-MiniLM Top-5
`0.12782`, whereas warm-up checkpoints commonly fall to `0.056`--`0.079`
despite higher mean cosine. No-warm-up and contrastive-dominant alternatives
are queued as `8690189[39-42]`.

The 20-second full-window target remains more brain-aligned than the selected
ten-word phrase. A story-OOF linear lexical head is being trained on full-window
content but selected against the exact val266 phrase (`8690195_2`). Its
downstream T5 keyword/logit-bias probes are dependency-gated in
`8690196[35-38]`. Segment identity is verified by story/start/stop metadata
when auxiliary text differs from the exact phrase. The older full-window
ELF-context arms `8689761_38`, `8689784_38`, and `8689831_38` were adaptively
stopped after 34--52 epochs: their best rounded content F1 was only
`0.019`--`0.020`, and all retained checkpoints remain available.

All of these runs keep `test107` sealed. Promotion still requires content F1
above `0.023116`, a positive margin over all fixed brain derangements, and a
single bundled checkpoint; the rigid ordered-content checkpoint at `0.025960`
remains the current strict single-checkpoint winner until a fluent model beats
it.

The first fluent promotions then completed. Both checkpoints bundle the OOF
lexical head and use one deterministic generation per brain row:

| val266 checkpoint | content F1 | word F1 | WER | deranged mean / max | margin |
|---|---:|---:|---:|---:|---:|
| `prefix16_oofordered_semwarm_keyword3_shortprobe` | **0.027278** | 0.046969 | 0.984962 | 0.015899 / 0.018692 | **0.011379** |
| `prefix16_oofordered_semwarm_lexk5_b2p0` | 0.026409 | **0.082238** | **0.980075** | 0.012667 / 0.015561 | **0.013742** |

Both beat the immutable fluent baseline's content F1 `0.023116`, beat every
fixed brain derangement, and improve WER from `1.031579`. The keyword-3 arm is
the content leader, while the bias-2 arm is currently the more balanced fluent
checkpoint because it also improves word overlap substantially. BERTScore jobs
`8690226` and `8690227` are pending. Neither model is frozen for test yet:
objective-corrected, decoding-sweep, no-collapse, full-window-head, and T5-base
validation arms remain active.

### Fluent validation leaders and rejected assumptions (2026-08-31, overnight)

Longer keyword conditioning and the temporally broader lexical supervision
produced two further single-checkpoint validation improvements. The current
content leader bundles the frozen OOF lexical head, oracle MiniLM-to-T5 prefix,
and identity-started brain calibrator in one checkpoint; its selected state is
semantic-warm-up epoch 1, before token-stage fine-tuning:

| val266 checkpoint | content F1 | word F1 | WER | deranged mean / max | margin |
|---|---:|---:|---:|---:|---:|
| locked fluent baseline | 0.023116 | 0.066611 | 1.031579 | -- | -- |
| `prefix16_oofordered_semwarm_keyword5_lexb0p5` | 0.034222 | 0.041404 | 0.986466 | 0.025412 / 0.029557 | 0.008811 |
| `prefix16_oofordered_full20lex_contentce2_pair0p1_keyword5train_lexb1_shortprobe` | **0.034850** | 0.040689 | 0.988346 | 0.023663 / 0.027606 | **0.011187** |

The latter improves content F1 by 50.8% relative to the locked fluent
baseline and exceeds every fixed shuffled-brain control. It is not yet opened
on test107: coverage-aware decoding, minimum-length, BERTScore, T5-base, and
paired-bootstrap validation checks remain active.

The balanced bias-2 checkpoint remains the strongest fluent compromise at
content F1 `0.026409`, word F1 `0.082238`, WER `0.980075`, and BERTScore F1
`-0.030838` (versus the locked baseline BERTScore F1 `-0.084597`). The first
short keyword-3 checkpoint had poor BERTScore F1 `-0.374943`, largely because
it generated empty or very short strings. Minimum-generation-length sweeps are
therefore being evaluated as a decoding contract rather than inferred from
WER alone.

Several assumptions are now rejected or narrowed:

- positional content-token CE reached oracle content F1 about `0.23` but
  collapsed to roughly `0.009`--`0.012` after the brain stage; decoder capacity
  is not the limiting factor;
- direct train-memory nearest-neighbour text retrieval reached only content F1
  `0.01317` (conditional margin `0.00514`), so retrieval is not a replacement
  decoder;
- same-story `[-10, 0, +10]` TR lexical context selected the prior-only epoch
  zero (top-5 set-F1 `0.05430`), below the independent current-row head, so
  adjacent segment concatenation did not recover extra lexical signal;
- no-warm-up/contrastive-dominant arms preserved semantic rank somewhat better
  but did not beat the fluent content leaders.

The static lexical logit bias is now also being challenged. A bundled
coverage-aware logits processor removes a predicted token's lexical bonus once
that token has appeared, preventing the same keyword from being rewarded at
every decoding step. Fixed-checkpoint sweeps compare static versus one-shot
bias, keyword/no-keyword context, prior subtraction, and minimum lengths. They
still produce one deterministic generation per row and do not rerank multiple
candidate generations.

An additional train-IDF audit prevents the nominal content objective from
hiding generic-word gains. IDF weights and the common-word exclusion set are
estimated from train11725 only. The locked baseline, balanced bias-2 model,
and two nominal content leaders score as follows:

| val266 system | ordinary content F1 | train-IDF content F1 | noncommon content F1 |
|---|---:|---:|---:|
| locked baseline | 0.023116 | 0.023740 | **0.017807** |
| balanced bias-2 | 0.026409 | 0.023575 | 0.012049 |
| keyword-5 bias-0.5 long | 0.034222 | 0.029563 | 0.004144 |
| full-window content leader | **0.034850** | **0.030063** | 0.006552 |

Thus the nominal leaders do carry more weighted overlap, but most of the gain
comes from train-common content words (`know`, `one`, `go`, `said`, and related
forms). They are retained as valid improvements under the locked metric, but
not treated as a solved semantic decoder. New linear lexical heads weight
positive targets by train-only IDF powers `0.5` and `1.0`, select on IDF-weighted
top-5 set-F1, and compare exact versus full-20-second supervision. This is an
explicit attempt to recover informative nouns/verbs rather than optimize a
frequency shortcut.

### One-shot content leader and overnight assumption audit (2026-08-31)

A deterministic fixed-checkpoint decoding sweep found a stronger one-shot
lexical rule for the full-window checkpoint. Each of the five brain-predicted
lexical tokens receives a generation bonus only until it first appears; the
same frozen checkpoint and one global rule are used for every row. There is no
candidate pool or reference-based reranking.

| val266 system | content F1 | word F1 | WER | BERTScore F1 | deranged mean / max | margin |
|---|---:|---:|---:|---:|---:|---:|
| locked fluent baseline | 0.023116 | **0.066611** | 1.031579 | -0.084597 | -- | -- |
| full-window one-shot bias 1.5 | **0.036790** | 0.040839 | **0.987218** | **-0.082104** | 0.022987 / 0.029550 | **0.013803** |

The content increase is `+59.2%` relative to the locked baseline. A paired
50,000-resample validation bootstrap gives content delta `+0.013674`, 95% CI
`[+0.002726,+0.024977]`, probability better `0.99268`; WER delta is
`-0.044361` with probability better `1.0`. The small BERTScore improvement is
uncertain (probability better `0.655`, CI crosses zero), and ordinary word F1
falls. This is therefore the current content-first leader, not yet the final
balanced promotion. Test107 remains sealed.

Train-IDF content F1 also rises to `0.031647`, but noncommon content F1 is only
`0.006866`. Frequent words still dominate, so the nominal improvement is real
under the locked metric but not evidence of high-specificity lexical recovery.

Further assumption checks:

- IDF-positive weighting selected essentially the same early exact lexical
  head; it did not improve Top-5 set-F1. Reweighting targets alone is rejected.
- Preserving all four delayed MiniLM blocks as a 1,536-D lexical feature made
  every ordered head select epoch zero, identical to a shuffled-target null.
  The apparent score was a language prior, not delayed brain information.
- Lexical confidence and Top-1/Top-2 gaps were not monotonically related to
  held-out hits. Confidence gating is rejected; useful signal is distributed
  weakly across the Top-5 set.
- Perfect MiniLM oracle decoding reaches WER `0.8996` and content F1 about
  `0.22`, so the T5 decoder has adequate ceiling. The main bottleneck is the
  brain-to-prefix interface.
- Full 20/10/30 schedules usually peak in early semantic warmup and then lose
  content. New schedules train longer at lower warmup/brain learning rates,
  remove contrastive false negatives in one arm, and test true joint
  calibrator+prefix-projector updates with optional T5 cross-attention LoRA.

### Corrected decoding finalists and frequency-shortcut traceback (2026-08-31)

The corrected fixed-checkpoint sweeps (colon-delimited Slurm grids, avoiding
comma truncation in `--export`) produced three finalists. Prompt labels were
removed by a fixed, target-free cleanup before scoring the keyword systems.

| val266 system | content F1 | word F1 | WER | BERTScore F1 | IDF content F1 | noncommon content F1 | shuffled max | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| locked fluent baseline | 0.023116 | 0.066611 | 1.031579 | -0.084597 | 0.022893 | **0.018102** | -- | -- |
| exact-head balanced, top-8/bias-3 one-shot | 0.040694 | **0.081652** | **0.985714** | **-0.084606** | 0.036160 | 0.009235 | 0.025282 | **0.020145** |
| full-window top-5/bias-3 one-shot | 0.043732 | 0.040072 | 0.987970 | -0.123425 | 0.037727 | 0.005717 | 0.030858 | 0.014874 |
| full-window top-8/bias-4 one-shot | **0.044367** | 0.037337 | 0.988722 | -0.103146 | **0.038959** | 0.005905 | 0.039322 | 0.010126 |

The high ordinary-content scores are not yet a sufficient promotion signal.
The full-window systems emit corpus-common words (`know`, `one`, `get`, `go`,
`time`) hundreds of times and lose both BERTScore and noncommon-word overlap.
The exact-head balanced system is the provisional scientific leader: it
improves ordinary content by 76%, word overlap by 22.6%, and WER by 0.0459,
keeps BERTScore essentially unchanged, and has the strongest matched-minus-
shuffled margin. It is not frozen while the informative-content traceback is
active.

A paired 50,000-resample validation bootstrap confirms that its ordinary
metrics are not a few-row accident: content delta `+0.017578`, 95% CI
`[+0.005244,+0.030143]`, probability better `0.99770`; word-F1 delta
`+0.015042`, CI `[+0.001718,+0.028472]`, probability better `0.98702`; WER
delta `-0.045865`, CI `[-0.056391,-0.035338]`, probability better `1.0`.
BERTScore delta is effectively zero (`-0.000009`, wide CI), which is why the
model is a lexical/WER promotion rather than a semantic-similarity promotion.

The provisional fallback is now serialized as one deterministic packaged
checkpoint at
`/data/engs-pnpl/glandau/elf-runs/fmri_minilm_t5_prefix_20260831/packaged_system/provisional_balanced_valonly.pt`
(SHA-256 `e79be49a2a2fee18cb2a54d82200cff2c52fc103c84c4d9555dc926a18049e7f`,
47,511,763 bytes). A fresh package-only val266 evaluation reproduced content
F1 `0.0406944872`, word F1 `0.0816524209`, and WER `0.9857142857` exactly.
This is a fallback artifact, not the final test freeze; test107 remains sealed
while the informative-content and cross-target runs execute.

Two target-free controls now test the shortcut directly. First, stronger
subtraction of the train-only lexical prior is swept without changing model
weights. Second, a new prior ceiling prevents the lexical logits processor
from boosting words whose bundled train probability exceeds a global
threshold. Jobs `8690591`--`8690594` test these mechanisms separately on the
exact and full-window checkpoints; all are smoke-gated and test107 remains
sealed.

The positional decoder loss is also being revised. Earlier unordered
set-recall/precision objectives could be satisfied by frequent content priors.
New runs instead retain positional teacher-forced CE and multiply only target
content positions by occurrence-normalized IDF estimated from train11725.
The exponent-0.5/exponent-1.0 exact-head arms and an exponent-0.5 full-window
arm run for 30 oracle epochs plus 10 semantic-warmup and 50 brain epochs in
array `8690613[57-59]`, behind smoke job `8690611`. Selection still requires
a positive matched-versus-shuffled margin and one deterministic generation
from one checkpoint.

### Overnight reconciliation and low-capacity delayed calibration (2026-08-31)

All scheduler status checks now explicitly use `--clusters=htc`. ARC's default
accounting database contained historical jobs with reused numeric IDs, which
could falsely make current HTC jobs appear complete. Artifact existence plus
HTC accounting is now required; missing files are never treated as results.

The stronger prior-ceiling finalist slightly improves WER and BERTScore but
does not replace the content-first leader:

| val266 system | content F1 | word F1 | WER | BERTScore F1 | shuffled max | margin |
|---|---:|---:|---:|---:|---:|---:|
| provisional balanced | **0.040694** | **0.081652** | 0.985714 | -0.084606 | 0.025282 | 0.020145 |
| prior ceiling | 0.040254 | 0.081007 | **0.984962** | **-0.081568** | 0.021950 | **0.021486** |
| beam 4 | 0.037125 | 0.078255 | 0.986090 | pending | 0.023447 | 0.018641 |
| beam 8 | 0.034869 | 0.075949 | 0.986466 | pending | 0.026534 | 0.015258 |

The 1.9M-parameter residual calibrator is now a traced assumption rather than
a default conclusion. The balanced checkpoint selected semantic-warmup epoch
one, and later calibration usually degraded validation content. A new
`DelayWeightedCalibrator` therefore learns only four global delay weights plus
a diagonal affine transform (772 trainable scalars), starts as the exact
normalized mean of delays 1--4, and never accesses validation targets during
training. Five long arms test learning rates `3e-4`, `1e-4`, and `3e-5`, plus
exact versus full-20-second oracle semantic targets. GPU smoke job `8690789`
passed; array `8690793[62-66]` is smoke-gated and followed by deterministic
val266 sweeps, BERTScore, informative-content audits, and shuffled-brain
controls. Test107 remains sealed.

The lexical decoder also discarded all score magnitudes and used only a fixed
linear rank bonus. A second target-free traceback keeps the same frozen brain,
prefix, T5, and lexical-head weights but converts the train-prior-adjusted
Top-K logits into temperature-controlled softmax bonuses. This is not
row-wise candidate reranking: one global weighting rule produces one
deterministic generation for every row. GPU smoke `8690814` passed. Sweep
`8690817` tests temperatures `0.25/0.5/1/2`, Top-K `5/8`, three prior ceilings,
and several global strengths; jobs `8690818`--`8690820` finalize the leading
record, compute BERTScore, and rebuild the leakage-safe leaderboard. Promotion
still requires matched content above every fixed derangement.

### Score-aware and dual-head validation finalists (2026-08-31)

The score-aware sweep and a new fixed dual-head traceback both surpassed the
provisional ordinary-content score. The dual rule takes a global convex
mixture of the exact-window lexical logits (`0.75`) and full-20-second lexical
logits (`0.25`), mixes their train-only log priors by the same weights, and
uses only the 2,297 words shared by the two train vocabularies. It is still one
deterministic generation per row: there is no candidate generation pool,
row-wise reranking, or target access at inference.

| val266 system | content F1 | word F1 | WER | BERTScore F1 | IDF content F1 | noncommon content F1 | shuffled max | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| provisional ranked bias | 0.040694 | 0.081652 | 0.985714 | -0.084606 | 0.036160 | 0.009235 | 0.025282 | 0.020145 |
| score-aware softmax | 0.042591 | 0.080906 | **0.985338** | -0.085751 | **0.037986** | **0.009729** | 0.025677 | 0.021156 |
| exact/full20 dual head | **0.042651** | **0.082336** | 0.987218 | **-0.083609** | 0.037648 | 0.009235 | **0.024420** | **0.022320** |
| dual head + score-aware | **0.043743** | **0.082678** | 0.987218 | -0.088278 | **0.038637** | **0.009729** | 0.027914 | **0.022447** |

The dual-head candidate beat all five fixed derangements. Its ordinary-content
gain is accompanied by higher train-IDF content than the provisional system,
but its noncommon content is unchanged; the gain should therefore be described
as stronger weak lexical recovery, not recovery of substantially rarer words.
The score-aware candidate is only `0.000060` lower in ordinary content, has
better WER, and leads both informative-content audits; the dual head has
better word F1 and BERTScore. A paired 50,000-row bootstrap treats the ordinary
content difference as a tie (95% CI `[-0.004369,+0.004490]`, dual better
probability `0.506`). The dual-minus-score-aware WER delta is `+0.001880`, also
uncertain at this sample size. A paired BERTScore bootstrap and the compact
combination of both inference levers remain pending, so no final winner is
frozen yet.

The packager and packaged evaluator now support embedding the secondary
lexical-head tensors and vocabulary inside the final `.pt`. If the dual system
wins, this preserves the strict single-physical-checkpoint contract rather
than relying on a second file at evaluation. Test107 remains sealed.

The compact combination of both inference levers is the current content-first
validation leader. It uses the same `0.75/0.25` exact/full20 head mixture, but
weights the adjusted Top-8 scores by a softmax at temperature `2.0`, applies a
global strength of `2.5`, subtracts `0.25` of the train prior, and excludes
words with train prior above `0.1`. It beat all five fixed derangements and
improves ordinary content, word F1, train-IDF content, and the conditional
margin over the provisional checkpoint; its BERTScore and WER are slightly
worse. The content gain over the provisional checkpoint is `+0.003049` with
paired-bootstrap better probability `0.94468` and 95% interval
`[-0.000654,+0.007002]`.

A self-contained provisional package reproduced the sweep exactly on a fresh
val266-only evaluation:

`/data/engs-pnpl/glandau/elf-runs/fmri_minilm_t5_prefix_20260831/packaged_system/provisional_combined_valonly.pt`

It is 52,197,101 bytes with SHA-256
`2f19bdcb1f0ca4ef603b25642a8d123b859b6f55f3a36b1110f94ec89d9699f8`.
The exact reproduced metrics are content F1 `0.0437433181`, word F1
`0.0826777017`, and WER `0.9872180451`. This remains a validation-only
provisional package while the remaining low-capacity calibrator sweeps finish;
test107 is still sealed.

### Frozen winner and one-pass test107 result (2026-08-31)

After every scheduled long-training, cross-target, score-aware, dual-head, and
delay-calibrator arm was reconciled, the dual-head plus score-aware system
remained the validation content leader. It was frozen before test access at:

`/data/engs-pnpl/glandau/elf-runs/fmri_minilm_t5_prefix_20260831/packaged_system/frozen_content_first_val266_20260831.pt`

The package contains both lexical heads, the brain calibrator/projector, any
trained decoder-side tensors, and the global decode rule in one physical
checkpoint. Its size is 52,197,325 bytes and SHA-256 is
`5530ec2d6bbfe04d45918adb8ab3add1e18678f2f44f112068945aaf25b0e790`.
A package-only pre-test val266 run reproduced content F1 `0.0437433181`, word
F1 `0.0826777017`, and WER `0.9872180451` exactly.

Only then was test107 opened once, in job `8691017`. No model, checkpoint, or
decode setting was changed afterward. The confirmatory result is:

| post-freeze test107 metric | value |
|---|---:|
| content-word precision / recall / F1 | 0.019059 / 0.017034 / **0.017756** |
| word precision / recall / F1 | 0.061824 / 0.059813 / **0.060702** |
| WER | 0.988785 |
| BERTScore P / R / F1, RoBERTa-large baseline-rescaled | -0.057110 / -0.073912 / **-0.064609** |
| train-IDF content F1 | 0.015145 |
| noncommon content F1 | 0.006750 |

Relative to the earlier legitimate frozen-brain test result (content F1
`0.01018`, word F1 `0.05977`, WER `0.97570`, BERTScore F1 `-0.12138`), the new
single checkpoint improves content substantially, word overlap slightly, and
BERTScore substantially, while WER regresses. Thus the overnight objective
succeeded for the declared content-first priority but did not reach the
aspirational `0.90` WER. The validation content effect also shrank on test,
which is important evidence against claiming that the validation improvement
fully generalized.

The immutable test artifacts are:

- `packaged_system/frozen_content_first_val266_20260831_test107_metrics.json`
- `packaged_system/frozen_content_first_val266_20260831_test107_bertscore.json`
- `packaged_system/frozen_content_first_val266_20260831_test107_informative.json`
- `packaged_system/frozen_content_first_val266_20260831_test107_ranked_top10.json`
- `packaged_system/frozen_content_first_val266_20260831_test107_target_generated.csv`

Local inspection copies are `fmri_frozen_content_first_test107_ranked_top10_2026-08-31.json`
and `fmri_frozen_content_first_test107_target_generated_2026-08-31.csv`.

### ELF diffusion challenge to the T5-prefix baseline (started 2026-08-31)

The next development target is the frozen T5-prefix validation score, not its
already-opened test result. Promotion requires one ELF diffusion checkpoint to
exceed val266 content F1 `0.0437433181`; word F1, WER, BERTScore, and
matched-versus-deranged brain controls remain mandatory secondary checks. The
107 test rows are not loaded during this development phase.

The traced architectural difference is explicit conditioning: T5 decoder
tokens cross-attend to the brain-derived prefix, whereas the previous ELF
bridge concatenated condition and target tokens and allowed ordinary
self-attention to ignore the weak condition. ELF now has optional dedicated
target-to-brain cross-attention adapters in selected final blocks. Their output
projections are bias-free and initialized to exact zero, so enabling the new
path reproduces the imported diffusion checkpoint exactly and a zero/CFG brain
condition remains a no-op even after training.

The first screen freezes MRI2SEM, the recovered four-delay context adapter,
the ELF-B base, and its existing rank-4 LoRA weights. Only the new
cross-attention tensors train. Six arms vary the final `2/4/6` adapted blocks,
learning rate (`3e-5`, `1e-4`, or `3e-4`), and an optional matched-versus-rolled
condition margin loss. Frozen modules run deterministically. Development uses
train11725 and val266 only.

- Local compatible-Torch regression suite: 28/28 focused tests passed, including
  cross-attention checkpoint preservation, strict round-trip loading,
  zero-memory invariance, lexical-gate gradients, and deterministic frozen
  paths. The original ARC regression request `8691962` never ran and was
  cancelled after remaining pending; priority-QoS replacements were rejected
  for insufficient account credit rather than model failure.
- Long ARC training array: `8691965[0-5]` (standard-QoS L40S; cancelled after
  the A100 early arm established that this shared-flow formulation was harmful)
- Four-epoch early screen: `8692137[0-2]` (standard-QoS A100)
- Launcher: `submit-jobs/fmri_elf_brain_cross_attention_val.sbatch`
- Initialization: recovery-2 ELF checkpoint at validation step `1000`

No arm is promoted from training loss alone. The validation winner must pass a
fixed derangement audit showing that its lexical gain depends on the matched
brain row, followed by BERTScore and qualitative inspection. Only after this
architecture either clears or convincingly fails the target will capacity be
expanded or the brain adapter be jointly fine-tuned.

The first planned follow-up is already dependency-gated behind the early
screen rather than consuming GPUs concurrently. A train11725-only 1,536-D
semantic lexical probe has val266 Top-5 content F1 `0.0576936`; its top words
therefore contain enough held-out lexical signal to clear the T5 score in
principle, although merely emitting that word list would not be acceptable
text generation. Replacement array `8692277[0-3%2]` injects its frozen word prototypes into
the ELF condition through a separately zero-initialized 64-value gate and the
new cross-attention path. Two arms isolate lexical memory; two also test a
fixed brain-conditioned final-token bias. The frozen MRI2SEM and semantic
paths remain in evaluation mode even while the tiny lexical gate trains.
This remains one diffusion generator per input, not candidate reranking, and
uses no validation targets during fitting.

#### Shared-flow result and decoder-only correction

The first corrected A100 arm, `8692137_1`, completed four epochs and selected
step `1468`. It reached content F1 `0.0121394`, word F1 `0.0489422`, WER
`1.03947`, generation-retrieval Top-1 `0.0112782`, Top-5 `0.0263158`, mean
rank `102.805`, and median rank `89`. This is substantially worse than both
the imported ELF recovery checkpoint (roughly `0.0203` content F1) and the
frozen T5-prefix target (`0.0437433`). Earlier steps degraded in the same
direction, so this was treated as an architectural negative control rather
than evidence to train longer.

Loss tracing showed that the new attention was shared between a comparatively
large denoising objective and the small decoder CE objective. Consequently it
changed an already recovered diffusion trajectory while learning little
lexical use of the brain condition. The corrected mode applies brain
cross-attention only when ELF's discrete decoder head is active and assigns no
denoiser loss to those parameters. It also restores the condition prefix to
its clean value during decoder training, matching the condition memory used
throughout diffusion sampling. Regression tests verify that nondecoder flow
outputs remain exactly unchanged after the cross-attention weights are made
nonzero, decoder logits do change, and only condition positions are restored
in the decoder latent.

- Decoder-only screen: `8692276[0-3%2]`, eight epochs on standard-QoS L40S,
  varying final 2/4 blocks, pairing loss, and `1e-4`/`3e-5` learning rates.
- Decoder-only task-0 scheduling race: `8692287_0`, identical clean-memory
  task on standard-QoS V100; the unstarted duplicate will be cancelled after
  one production allocation becomes live.
- Decoder-only train11725 lexical continuation: `8692277[0-3%2]`, dependency
  gated after `8692276`.
- Independent low-rate shared-flow control: `8692212_3`, retained so the
  decoder-only conclusion is not confounded solely with learning rate.

The packed dataset's `VAL_NUM_EXAMPLES=266` value defines the physical split,
not only evaluation cost. A diagnostic override briefly demonstrated that
changing it would move held-out rows into training; that smoke was stopped
before optimization and the launcher now rejects any non-266 split override.
Smoke evaluations may reduce `EVAL_NUM_EXAMPLES`, but cannot alter the locked
train11725/val266 boundary.

The val266 promotion and fixed brain-derangement requirements are unchanged;
test107 remains sealed for this entire diffusion-development phase.

### Diffusion content winner versus T5-prefix (2026-08-31)

The explicit-brain ELF diffusion program produced a deterministic single-model
validation winner. It uses the recovery-2 ELF checkpoint, decoder-only brain
cross-attention in the final two blocks, a train11725-only supervised lexical
head over the frozen 1,536-D MRI2SEM output, and a one-shot scattered lexical
bias. It produces one generation per brain row; there is no candidate
generation, oracle selection, or reranking.

The exact source checkpoint is:

```text
/data/engs-pnpl/glandau/elf-runs/
fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_
lexgate_top5_partitioned_dev0p1eval64_fixrepeat4/final.pt
```

The decode rule is lexical Top-10, temperature `2.0`, train-prior subtraction
`0.1`, no maximum-prior exclusion, bias strength `40`, scattered one-shot
placement, and partitioned four-delay context. MRI2SEM and the imported
ELF/LoRA weights are frozen; the source run trained only the decoder-side
cross-attention/gate path, while the lexical MLP was fitted separately using
train11725 only. Model and decode selection used val266 only.

| leakage-safe val266 metric | frozen T5-prefix | ELF diffusion | diffusion minus T5 |
|---|---:|---:|---:|
| content-word F1 | 0.043743 | **0.047099** | **+0.003356** |
| content precision / recall | — | 0.042114 / 0.054868 | — |
| word F1 | **0.082678** | 0.068849 | -0.013828 |
| WER | **0.987218** | 1.142105 | +0.154887 |
| BERTScore F1 | **-0.088278** | -0.098912 | -0.010633 |
| T5 retrieval Top-1 / Top-5 | — | 0.011278 / 0.048872 | — |
| T5 retrieval mean / median rank | — | 113.545 / 109 | — |

Thus diffusion beats T5-prefix on the declared primary point estimate by
`7.7%` relative, but it is not an across-the-board language-quality win. A
paired 50,000-sample bootstrap gives a content-F1 difference interval of
`[-0.007767,+0.014302]` and probability `0.72882` that diffusion is better.
The 266-row evidence is therefore promising but not statistically decisive.
Diffusion is significantly worse in word F1 under this bootstrap and is also
worse in WER and mean BERTScore.

The result passes the essential matched-brain control. Fixed roll-1, roll-17,
and roll-83 derangements obtain content F1 `0.035891`, `0.042271`, and
`0.031996`; their mean is `0.036719`. Matched brain conditioning is better than
every derangement and has a `+0.010380` margin over their mean. This rules out
the interpretation that the measured content score comes only from an
unconditional language prior.

A second lexical head trained on the full-20-second target representation was
audited before freezing. Its best candidate-set ceiling moved from `0.056141`
to only `0.056839`, while conditional separation fell from `0.010599` to
`0.007233`; at the selected decode setting it was worse. It was rejected, so
the final system remains the simpler single-head model.

The reload-verified single-checkpoint package is:

```text
/data/engs-pnpl/glandau/elf-runs/
fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_
lexgate_top5_frompartstep37_top10_prior0p1_t2_scattered_b40_val266_generationonly/
packaged_system/fmri_elf_content_first_val266_20260831.pt
```

It contains the ELF checkpoint, brain adapter/cross-attention tensors, lexical
MLP and vocabulary, reconstructed train-only word prior, deterministic decode
configuration, matched/deranged validation provenance, BERTScore, and
retrieval metadata. The 107-row brain test was not loaded for any diffusion
training, selection, packaging, or validation in this phase.

The final enriched package is 1,893,042,242 bytes with SHA-256
`a4031717b2f8b5095998fd45bb922b43d7b3385ce5b9b3aeefd1ea7b56b4fe54`.
Focused package-era regression job `8692722` passed all 20 ARC tests covering
ELF brain cross-attention, the MRI2SEM lexical bridge, and one-shot token/word
bias. Four pure-Python ranking/bootstrap tests also pass locally.

Local qualitative artifacts are:

- `fmri_elf_diffusion_content_winner_val266_target_generated_2026-08-31.csv`
- `fmri_elf_diffusion_content_winner_val266_ranked_examples_2026-08-31.json`

These results establish a content-first diffusion proof of concept, not a
claim that diffusion has replaced T5-prefix overall. The next scientific task
is to preserve this brain-dependent content advantage while recovering word
precision, ordering, and fluency so that WER and BERTScore no longer regress.
