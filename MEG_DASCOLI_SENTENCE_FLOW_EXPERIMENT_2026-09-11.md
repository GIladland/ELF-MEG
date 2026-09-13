# MEG semantic + simulated D'Ascoli sentence-source ELF-B experiment

Date started: 2026-09-11

## Question

If D'Ascoli supplies a complete ordered ten-word sentence and calibrated
confidence for every word, does an ELF-B flow trained to refine that sentence
improve on the audited semantic-only qc4wyals decoder?  What D'Ascoli word
accuracy is needed before the joint system helps?

This is a target-derived validation sensitivity/ceiling experiment.  It is not
a brain-conditioned lexical result.  Protected test26 is absent.

## Fixed data and initialization

- Semantic input: the 2,652 train + 110 validation nested-OOF qc4wyals ADA pack.
- Validation: the established final 110 Apple rows.
- ELF initialization: audited x1 ELF-B best checkpoint.
- Semantic adapter: loaded from that checkpoint and frozen.
- Target latent and source sentence latent: the checkpoint's frozen T5-small
  encoder and original target sequence length.
- Checkpoint selection: validation word/content mean only.
- Checkpoint storage: trainable ELF deltas plus an explicit reference to the
  immutable audited x1 parent; the frozen ~638 MB model/adapter state is not
  duplicated per arm.

## Conditional transport

For source T5 state `h`, token confidence `c`, Gaussian prior `epsilon`, and
true target T5 state `x`:

```text
s   = c*h + (1-c)*(2.5*epsilon)
z_t = t*x + (1-t)*s
v*  = (x-z_t)/(1-t)
```

Sampling starts from `s` at `t=0` and follows the learned vector field to
`t=1`.  The MEG-predicted ADA semantic context occupies the existing clean
64-token prefix throughout.  Confidence is aligned from each whitespace word
to all overlapping T5 subwords; EOS receives the row-mean confidence and pads
remain at zero.

Training independently drops 10% of D'Ascoli sources and 10% of semantic
conditions.  This includes semantic-only, sentence-only, and unconditional
examples and prevents a compulsory-copy solution.

## Simulation

- Full ten-word hypotheses are produced at assumed per-word correctness
  probabilities 0.25, 0.50, 0.75, and 0.90 during training.
- Correct-word confidence is sampled from `Beta(8,2)`; incorrect-word
  confidence is sampled from `Beta(2,8)`.
- Wrong words are sampled from a position-specific vocabulary constructed only
  from the selected training rows, with a train-only global fallback.
- Evaluation accuracy masks are nested across 0/25/50/75/90/100%, so each
  higher level adds correct words to the lower level rather than drawing an
  unrelated corruption.
- Two source rows contain the audited surface form `that''s`; classifier
  positions therefore follow whitespace segmentation, which preserves the
  ten-word data contract.

## Training screen

| Arm | Trainable ELF parameters | ELF LR | Epochs |
|---|---|---:|---:|
| `lora4_low` | rank-4 attention+MLP LoRA, final 4 blocks | 2e-6 | 40 |
| `lora8_mid` | rank-8 attention+MLP LoRA, final 4 blocks | 1e-5 | 40 |
| `full_last2` | full final 2 blocks plus ELF projection/head | 5e-7 | 40 |

## Validation matrix

- Joint confidence-weighted at assumed accuracy 0, 0.25, 0.50, 0.75, 0.90,
  and 1.00.
- At 0.50: sentence-only, joint unweighted, joint row-shuffled, and
  semantic-only.
- The same matrix first runs on the unchanged audited x1 checkpoint to measure
  what sentence-start sampling can do without transport training.
- Final scoring uses word F1, content F1, their sum, WER, generated T5
  Top-1/Top-5, and qualitative examples.  The selected result will then receive
  the same extended lexical/significance scoring used by current winners.

## ARC jobs

| Purpose | Job | State at launch |
|---|---:|---|
| Real-checkpoint one-step smoke | 8788048 | completed, exit 0, 00:06:14 |
| Compact parent+delta reload test | 8788194 | completed, exit 0, 00:01:58 |
| Three-arm training screen | 8788262 | completed, exit 0 |
| Untrained x1 control matrix | 8788289 | completed, all 10 tasks exit 0 |
| Trained LoRA-8 control matrix | 8790325 | completed, all 10 tasks exit 0 |
| Semantic/source weight grid | 8790349 | completed, all 25 tasks exit 0 |

The initial L40S allocations (`8787976` and `8787979`) were canceled after the
scheduler projected starts on 18 September.  Their replacements above use the
same commands, seeds, and validation contract; only the accelerator class and
Slurm partition changed.

## Results

The one-step/one-row smoke successfully loaded the real audited checkpoint and
adapter, encoded both latent sequences, executed a source-to-target training
step, generated text, and saved metrics/checkpoint artifacts.  Its one-row
scores are not estimates and are not reported as results here.

ARC's shared `/data/engs-pnpl` filesystem reached 100% during launch.  The two
disposable smoke checkpoint blobs created by this work were removed after
their exit-0 logs and metrics were preserved, freeing about 1.2 GB locally
(about 4 GB was available after concurrent cleanup).  Full training uses
compact delta checkpoints to avoid multiplying the frozen parent state.

The independent compact-checkpoint round trip completed with exit code 0.  It
restored a parent model and adapter, applied one child trainable tensor, and
verified that the child tensor changed while untouched model and adapter
parameters remained exactly equal to the parent.

Full val110 results are reported in
`MEG_DASCOLI_SENTENCE_FLOW_RESULTS_2026-09-12.md`. The concise outcome is that
sentence-start refinement works, but confidence-to-Gaussian mixing and the
full-strength semantic prefix are harmful. The selected diagnostic uses full
source trust and semantic scale `0.035` (word/content sum `0.65497`), while the
direct simulated D'Ascoli sentence remains better (`0.95226`). The next model
should preserve the source sentence and apply uncertainty-gated semantic edits
inside ELF blocks.
