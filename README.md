# PyTorch ELF

PyTorch version of [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938).

## Installation

Create a conda environment named `elf` and install the dependencies:

```bash
conda create -n elf python=3.10 -y
conda activate elf
pip install -r requirements.txt
```

Then log in to WandB to track your experiments if needed:

```bash
wandb login YOUR_WANDB_API_KEY
```

## ARC Setup

This repo now has a working ARC setup for both the standard ELF workflow and
the PNPL LibriBrain smoke test.

### BrainDiffusion ARC env policy

Use the versioned node-local env for GPU experiments:

```bash
/tmp/$USER-braindiffusion-elf-torch213
```

The setup script stamps the env with:

```text
braindiffusion-elf-torch213-py310-20260803
```

Launchers only reuse the env when the stamp matches; otherwise they rebuild it
from `requirements.txt`. Keep executable Torch/CUDA libraries on node-local
`/tmp`, not in the persistent `/data` env. We saw `import torch` fail or hang
from the old `/data/engs-pnpl/glandau/BrainDiffusion/elf-env` because large
Torch shared objects can fail to map from NFS.

Large caches and model / dataset downloads still live on:

```bash
/data/engs-pnpl/glandau/elf-cache
```

This includes the Triton kernel cache; GPU launchers should set
`TRITON_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/triton-cache` so compiled
kernels do not consume the ARC home-directory quota.

Do not submit new GPU jobs against the old persistent env unless Torch import
has been verified on a compute node. It is acceptable for lightweight commands
that do not import Torch.

Persistent shared env path, for lightweight/non-GPU use only:

```bash
/data/engs-pnpl/glandau/BrainDiffusion/elf-env
```

Two env patterns are supported. For GPU runs, prefer the node-local env: large
Torch/CUDA shared libraries can be slow or fail to map from the `/data` NFS
env, while caches and downloaded artifacts still stay persistent on `/data`.

1. Persistent shared env on `/data` for lightweight commands, or only after
   verifying Torch import on a compute node:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/engs-pnpl/glandau/BrainDiffusion/elf-env
cd /data/engs-pnpl/glandau/BrainDiffusion/ELF
pip install -r requirements.txt
```

2. Node-local env rebuilt on `/tmp` from the same `requirements.txt`
   recommended for GPU jobs:

```bash
bash scripts/arc_setup_elf_env.sh
```

The node-local setup script keeps the executable env on
`/tmp/$USER-braindiffusion-elf-torch213`, stamps it with the dependency version,
and sends Hugging Face, torch, pip, conda, and wandb caches to `/data`.

### ARC smoke tests

ELF-B unconditional generation smoke test:

```bash
srun --clusters=htc --partition=short --qos=priority \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=01:00:00 \
  bash /data/engs-pnpl/glandau/BrainDiffusion/ELF/scripts/arc_run_owt_b_demo.sh
```

PNPL LibriBrain MEG + sentence smoke test:

```bash
srun --clusters=arc --partition=interactive --cpus-per-task=4 --mem=32G --time=00:25:00 \
  bash /data/engs-pnpl/glandau/BrainDiffusion/ELF/scripts/arc_run_libribrain_probe.sh
```

Tiny LibriBrain MEG-to-text overfit check:

```bash
LIBRIBRAIN_DATA_PATH=/path/to/libribrain \
srun --clusters=htc --partition=short --qos=priority \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=02:00:00 \
  bash /data/engs-pnpl/glandau/BrainDiffusion/ELF/scripts/arc_run_meg_context_overfit.sh \
  --books 1 --num-examples 8 --steps 2000
```

Expected successful probe output looks like:

```text
dataset_type=_FlexibleSemanticDataset
num_examples=2101
batch=0 meg_shape=(2, 306, 750) mask_shape=(2, 750) semantic_shape=(2, 1024)
lengths=[750, 750]
sample[0] ... 'A Study in Scarlet by Sir Arthur Conan Doyle'
sample[1] ... 'This is a LibriVox recording'
```

Sherlock SVA MiniLM + MEG export for MEG2SEM-style training:

```bash
cd /data/engs-pnpl/glandau/BrainDiffusion/ELF
sbatch submit-jobs/export_sherlock_sva_meg2sem_minilm.sbatch
```

Default output:

```text
/data/engs-pnpl/glandau/elf-cache/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz
```

The packed NPZ stores `meg`, `meg_time_mask`, `meg_lengths`, `input_embeddings`
/ `minilm_embeddings`, `sentence`, and `rows`. The wrapper also writes
MiniLM semantic sidecars under `embeddings_ada` at:

```text
/data/engs-pnpl/glandau/elf-cache/sherlock_sva/meg2sem/minilm_sidecars
```

For direct MEG2SEM loading of those sidecars, use `embedding_type=ADA` with
`embedding_dim=384` and point the semantic-vector root at that sidecar tree.

Directly route a trained MEG2SEM MiniLM checkpoint into ELF:

```bash
cd /data/engs-pnpl/glandau/BrainDiffusion/ELF
sbatch submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch
```

The job defaults to an eval-only pass for W&B run `dt8z7o55` using the
retrieval/nDCG-selected epoch 852 checkpoint:

```text
/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/dt8z7o55/checkpoints/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=852-val_loss=26.45.ckpt
```

It predicts MiniLM vectors from MEG, maps them through the saved MiniLM-to-ELF
K64 semantic projector, and generates/evaluates with the paired ELF checkpoint.
To match the original val-loss monitor instead, override `MEG2SEM_CHECKPOINT`
with the epoch 49 checkpoint.

To fine-tune instead of only evaluating:

```bash
EVAL_ONLY=0 TRAIN_MEG2SEM=1 UNFREEZE_ELF=1 \
  sbatch submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch
```

For the full end-to-end run with close monitoring enabled:

```bash
sbatch submit-jobs/sherlock_sva_meg2sem_minilm_e2e_finetune_dt8z7o55.sbatch
```

That wrapper trains MEG2SEM, the MiniLM-to-ELF semantic projector, and ELF
itself by default. It logs interface diagnostics every 5 steps, generation and
retrieval every 50 steps, and keeps the top 3 eval checkpoints.

### ARC training

For training on ARC, use a GPU allocation and let the launcher/setup script
activate the versioned node-local env:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
cd /data/engs-pnpl/glandau/BrainDiffusion/ELF
bash scripts/arc_setup_elf_env.sh
```

Expected setup output includes:

```text
TMP_ENV_ROOT=/tmp/$USER-braindiffusion-elf-torch213
ENV_STAMP=braindiffusion-elf-torch213-py310-20260803
torch 2.13.0+cu130
cuda_available True
```

Single-GPU training:

```bash
bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Single-node multi-GPU training:

```bash
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

## Converted Checkpoints

We provide PyTorch-converted versions of the official JAX checkpoints on HuggingFace:

| Model | Task | Params | HuggingFace Repo |
| --- | --- | --- | --- |
| ELF-B | OpenWebText (unconditional) | 105M | [embedded-language-flows/ELF-B-owt-torch](https://huggingface.co/embedded-language-flows/ELF-B-owt-torch) |
| ELF-M | OpenWebText (unconditional) | 342M | [embedded-language-flows/ELF-M-owt-torch](https://huggingface.co/embedded-language-flows/ELF-M-owt-torch) |
| ELF-L | OpenWebText (unconditional) | 652M | [embedded-language-flows/ELF-L-owt-torch](https://huggingface.co/embedded-language-flows/ELF-L-owt-torch) |
| ELF-B | XSum (summarization) | 105M | [embedded-language-flows/ELF-B-xsum-torch](https://huggingface.co/embedded-language-flows/ELF-B-xsum-torch) |
| ELF-B | WMT14 De-En (translation) | 105M | [embedded-language-flows/ELF-B-de-en-torch](https://huggingface.co/embedded-language-flows/ELF-B-de-en-torch) |

These are pulled automatically via `--checkpoint_path <hf-repo-id>` — no manual download needed.

## Reference Results

The PyTorch port targets parity with the JAX reference numbers from the
paper. Small differences (≲1 PPL, ≲0.5 BLEU/ROUGE) are expected due to bf16
vs. JAX TPU numerics and sampling stochasticity.

**Unconditional generation (OpenWebText), expected:**

| Model | Sampling | Gen. PPL ↓ | Entropy ↑ |
| --- | --- | --- | --- |
| ELF-B (105M) | 32-step SDE | 24.1 | 5.15 |
| ELF-M (342M) | 64-step SDE | 21.7 | 5.18 |
| ELF-L (652M) | 64-step SDE | 23.3 | 5.28 |

Gen. PPL is computed under a frozen GPT-2 Large; entropy is unigram entropy
over the generated tokens. Default sampling configs
(`src/configs/sampling_configs/uncond_sampling_configs.yml`) use SC-CFG=3 and
γ=1.5 (32-step) or γ=1.0 (64-step).

**Conditional generation (ELF-B), expected on the validation set:**

| Task | Metric | Reference (paper, test) | Validation |
| --- | --- | --- | --- |
| WMT14 De-En | BLEU ↑ | 26.4 | ≈ 26.7 |
| XSum | ROUGE-1 ↑ | 36.0 | ≈ 36.3 |
| XSum | ROUGE-2 ↑ | 12.2 | ≈ 12.5 |
| XSum | ROUGE-L ↑ | 27.8 | ≈ 28.1 |

Default conditional sampling
(`src/configs/sampling_configs/cond_sampling_configs.yml`): 64-step ODE,
CFG=2, SC-CFG=1.

The paper numbers were computed on TPU v5p-64; numbers from this PyTorch port
on 8× L40S / H200 should land within sampling noise (typically <1 PPL or
<0.5 metric points).

## Training

Launch single-GPU training:

```bash
bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Launch multi-GPU (single-host) training:

```bash
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Available training configs:

- `src/configs/training_configs/train_owt_ELF-B.yml` — ELF-B on OpenWebText
- `src/configs/training_configs/train_owt_ELF-M.yml` — ELF-M on OpenWebText
- `src/configs/training_configs/train_owt_ELF-L.yml` — ELF-L on OpenWebText
- `src/configs/training_configs/train_de-en_ELF-B.yml` — WMT14 De-En machine translation
- `src/configs/training_configs/train_xsum_ELF-B.yml` — XSum abstractive summarization

**Estimated wall-clock:** ~4 h per epoch on 8× H200 (OpenWebText, ELF-B,
global batch size 512, bf16). The default ELF-B OWT run is 5 epochs.

## Evaluation

Run evaluation against the converted checkpoints on HuggingFace. We recommend
passing `use_bf16=true` (matches the bf16 autocast used at training time) and
`use_compile=true` (wraps the eval model in `torch.compile`) for a ~3–4×
speedup on consumer GPUs:

**Unconditional generation (OpenWebText):**

```bash
# ELF-B (105M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true

# ELF-M (342M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-M.yml \
    --checkpoint_path embedded-language-flows/ELF-M-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true

# ELF-L (652M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-L.yml \
    --checkpoint_path embedded-language-flows/ELF-L-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true
```

**Conditional generation (XSum / WMT14 De-En):**

```bash
# XSum (ROUGE)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_xsum_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-xsum-torch \
    --config_override use_bf16=true --config_override use_compile=true

# WMT14 De-En (BLEU)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_de-en_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-de-en-torch \
    --config_override use_bf16=true --config_override use_compile=true
```

### Eval config flags

| Flag | Default | What it does |
| --- | --- | --- |
| `use_bf16` | `true` | Wraps the sampling forward in `torch.amp.autocast('cuda', dtype=bfloat16)`. Mirrors the training-time precision; output heads stay fp32. |
| `use_compile` | `false` | Wraps the eval model in `torch.compile`. First batch is slower due to tracing; subsequent batches run materially faster. |

Both flags are also editable in the YAML config under the same names. You can also run the standalone
PPL script afterwards:

```bash
python scripts/eval_ppl.py \
    --input outputs/<run>/<sampling_dir>/all_generated_*.jsonl \
    --batch_size 16
```
