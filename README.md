# Token-Level Guided Discrete Diffusion for Membrane Protein Design

[**Shrey Goel**](https://shreygoel09.github.io/)\* and [**Pranam Chatterjee**](https://www.chatterjeelab.com/)

![MemDLM diagram](./memdlm_schematic.png)

**arXiv preprint:** [link TBD]

Reparameterized diffusion models (RDMs) have recently matched autoregressive methods in protein generation, motivating their use for challenging tasks such as designing membrane proteins, which possess interleaved soluble and transmembrane (TM) regions.

We introduce ***Membrane Diffusion Language Model (MemDLM)***, a fine-tuned RDM-based protein language model that enables controllable membrane protein sequence design. MemDLM-generated sequences recapitulate the TM residue density and structural features of natural membrane proteins, achieving comparable biological plausibility and outperforming state-of-the-art diffusion baselines in motif scaffolding tasks by producing:

- Lower perplexity
- Higher BLOSUM-62 scores
- Improved pLDDT confidence

To enhance controllability, we develop ***Per-Token Guidance (PET)***, a novel classifier-guided sampling strategy that selectively solubilizes residues while preserving conserved TM domains. This yields sequences with reduced TM density but intact functional cores.

Importantly, MemDLM designs validated in TOXCAT β-lactamase growth assays demonstrate successful TM insertion, distinguishing high-quality generated sequences from poor ones.

Together, our framework establishes the first experimentally validated diffusion-based model for rational membrane protein generation, integrating *de novo* design, motif scaffolding, and targeted property optimization.

---

## Repository Authors

- <u>[Shrey Goel](https://shreygoel09.github.io/)</u> – undergraduate student at Duke University
- <u>[Pranam Chatterjee](mailto:pranam@seas.upenn.edu)</u> – Assistant Professor at University of Pennsylvania

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
   - [MeMDLM diffusion model](#1-memdlm-diffusion-model)
   - [Solubility classifier](#2-solubility-classifier)
   - [Oligomerization classifier](#3-oligomerization-classifier)
   - [Multipass classifier](#4-multipass-classifier)
5. [Sampling](#sampling)
   - [Unconditional generation](#1-unconditional-generation)
   - [Solubilization (PET)](#2-solubilization-pet)
   - [Desolubilization (PET)](#3-desolubilization-pet)
   - [Multipass generation](#4-multipass-generation)
   - [Oligomerization](#5-oligomerization)
6. [Citation](#citation)

---

## Installation

Clone the repository and install it in editable mode from the repository root:

```bash
git clone https://github.com/<your-org>/MeMDLM_v2.git
cd MeMDLM_v2
pip install -e .
pip install -r requirements.txt
```

Log in to Weights & Biases (used for experiment logging during training):

```bash
wandb login
# or: export WANDB_API_KEY=<your-key>
```

---

## Configuration

All hyperparameters live in YAML files under `src/configs/`. Before running anything, set `base_dir` in each config to your local clone path:

```yaml
base_dir: /path/to/MeMDLM_v2
```

| Config file | Purpose |
|---|---|
| `src/configs/lm.yaml` | General diffusion language model utils (mainly for training MemDLM) |
| `src/configs/solubility.yaml` | Train/eval per-residue solubility classifier + PET sampling hyperparameters |
| `src/configs/desolubilize.yaml` | Hyperparameters for performing reverse solubilization task ("desolubilization") |
| `src/configs/oligo.yaml` | Train/eval oligomerization classifier + LaMBO-2 sampling hyperparameters |
| `src/configs/multipass.yaml` | Train/eval TM-segment classifier + LaMBO-2 sampling hyperparameters |

**Training modes** are controlled by `training.mode` in each config:

| Value | Behavior |
|---|---|
| `train` | Run training from scratch |
| `test` | Load `best_model.ckpt` and evaluate on the test set |
| `resume_from_checkpoint` | Resume MeMDLM training from `checkpointing.resume_ckpt_path` (`lm.yaml` only) |

Checkpoints are saved to `checkpoints/<wandb.name>/best_model.ckpt`. After training the diffusion model, ensure `lm.ft_evoflow` in the classifier and sampling configs matches the `wandb.name` from `lm.yaml`.

---

## Data Preparation

Place your datasets under `<base_dir>/data/`. Expected CSV formats:

### MeMDLM & solubility classifier (`data/train.csv`, `data/test.csv`, `data/val.csv`)

| Column | Description |
|---|---|
| `Sequence` | Amino acid sequence. For the solubility classifier, use **uppercase** for soluble residues and **lowercase** for TM residues. For MeMDLM pretraining, sequences are uppercased automatically. |

### Oligomerization classifier (`data/olig_clf/`)

| Column | Description |
|---|---|
| `Sequence` | Input sequence |
| `Binary Label` | `0` or `1` oligomerization label |

### Multipass classifier (`data/multipass/`)

| Column | Description |
|---|---|
| `Sequence` | Input sequence |
| `TM_segments` | Integer count of predicted TM segments |

---

## Training

All training jobs are launched from the **repository root**. We recommend running long jobs with `nohup` so they persist after logout. Create a `logs/` directory first:

```bash
mkdir -p logs
```

### 1. MeMDLM diffusion model

**Config:** `src/configs/lm.yaml`  
**Entry point:** `src/lm/memdlm/main.py`

Set `training.mode: train` and adjust `training.devices` to the number of GPUs available.

```bash
nohup python src/lm/memdlm/main.py > logs/memdlm_train.out 2>&1 &
```

Key settings in `lm.yaml` (do not change unless reproducing a new experiment):

| Parameter | Default |
|---|---|
| Base model | `fredzzp/EvoFlow-650M-context-3070` |
| `training.max_steps` | 5000 |
| `training.n_layers` | 3 (unfrozen transformer layers) |
| `optim.lr` | 4e-5 |
| `data.batch_size` | 32 |
| `lm.num_diffusion_timesteps` | 500 |

Checkpoint saved to: `checkpoints/<wandb.name>/best_model.ckpt`

To evaluate a trained checkpoint, set `training.mode: test`.

```bash
nohup python src/lm/memdlm/main.py > logs/memdlm_test.out 2>&1 &
```

---

### 2. Solubility classifier

**Config:** `src/configs/solubility.yaml`  
**Entry point:** `src/guidance/solubility/main.py`

Trains a per-residue ESM-based classifier that predicts soluble vs. TM residues. This checkpoint is required for PET solubilization and desolubilization sampling.

Set `training.mode: train`.

```bash
nohup python src/guidance/solubility/main.py > logs/solubility_train.out 2>&1 &
```

Key settings:

| Parameter | Default |
|---|---|
| `training.max_steps` | 3000 |
| `model.num_layers` | 4 |
| `optim.lr` | 3e-5 |
| `data.batch_size` | 32 |

Checkpoint saved to: `checkpoints/<wandb.name>/best_model.ckpt`

---

### 3. Oligomerization classifier

**Config:** `src/configs/oligo.yaml`  
**Entry point:** `src/guidance/oligo/main.py`

Set `training.mode: train`.

```bash
nohup python src/guidance/oligo/main.py > logs/oligo_train.out 2>&1 &
```

Key settings:

| Parameter | Default |
|---|---|
| `training.max_steps` | 3000 |
| `model.num_layers` | 1 |
| `optim.lr` | 1e-4 |
| `data.batch_size` | 64 |
| `data.max_seq_len` | 54 |

Checkpoint saved to: `checkpoints/<wandb.name>/best_model.ckpt`

---

### 4. Multipass classifier

**Config:** `src/configs/multipass.yaml`  
**Entry point:** `src/guidance/multipass/main.py`

Predicts the number of TM segments in a sequence. Required for multipass-guided generation.

Set `training.mode: train`.

```bash
nohup python src/guidance/multipass/main.py > logs/multipass_train.out 2>&1 &
```

Key settings:

| Parameter | Default |
|---|---|
| `training.max_steps` | 3000 |
| `model.num_layers` | 4 |
| `optim.lr` | 3e-5 |
| `data.batch_size` | 32 |

Checkpoint saved to: `checkpoints/<wandb.name>/best_model.ckpt`

---

### Recommended training order

```
1. MeMDLM (lm.yaml)
       ↓
2. Classifiers in any order:
   • solubility.yaml
   • oligo.yaml
   • multipass.yaml
```

Update `lm.ft_evoflow` in all downstream configs to match the `wandb.name` produced by step 1.

---

## Sampling

Sampling scripts generate sequences and write CSVs under `<base_dir>/results/`. Each script loads the fine-tuned MeMDLM checkpoint (`checkpoints/<lm.ft_evoflow>/best_model.ckpt`) and, where applicable, the corresponding classifier checkpoint (`checkpoints/<wandb.name>/best_model.ckpt`).

Run all sampling jobs from the **repository root**:

```bash
mkdir -p logs
```

### 1. Unconditional generation

**Config:** `src/configs/lm.yaml`  
**Script:** `src/sampling/unconditional_generator.py`

Generates *de novo* membrane protein sequences from a fully masked prior.

```bash
nohup python src/sampling/unconditional_generator.py > logs/unconditional_sample.out 2>&1 &
```

**Output:** `results/denovo/<wandb.name>/<date>_multipass/seqs_with_ppl.csv`

Columns: `Generated Sequence`, `ESM PPL`, `MeMDLM PPL`

---

### 2. Solubilization (PET)

**Config:** `src/configs/solubility.yaml`  
**Script:** `src/sampling/pet_generator.py`

Uses Per-Token Guidance to redesign **uppercase (soluble)** positions in a scaffold while preserving **lowercase (TM)** residues. Update the input CSV path in `pet_generator.py` (default: `results/heme/cybtx.csv`) to point to your scaffold sequences.

```bash
nohup python src/sampling/pet_generator.py > logs/solubilize_sample.out 2>&1 &
```

**Output:** `results/heme/<lm.ft_evoflow>/solubilize/<date>/<prior-params>/infilled_seqs.csv`

---

### 3. Desolubilization (PET)

**Config:** `src/configs/desolubilize.yaml`  
**Script:** `src/sampling/desolubilize_generator.py`

The inverse of solubilization: redesigns **lowercase (soluble)** positions while preserving **uppercase (TM)** residues. Uses the same solubility classifier checkpoint. Update the input CSV path in `desolubilize_generator.py` (default: `results/heme/4d2.csv`).

```bash
nohup python src/sampling/desolubilize_generator.py > logs/desolubilize_sample.out 2>&1 &
```

**Output:** `results/heme/<lm.ft_evoflow>/desolubilize/<date>/<prior-params>/infilled_seqs.csv`

---

### 4. Multipass generation

**Config:** `src/configs/multipass.yaml`  
**Script:** `src/sampling/multipass_generator.py`

Generates sequences guided toward a target multipass TM topology using the multipass classifier.

```bash
nohup python src/sampling/multipass_generator.py > logs/multipass_sample.out 2>&1 &
```

**Output:** `results/multipass/<wandb.name>/<date>/lamb=<reg_strength>_tau=<sampling_temperature>/seqs_with_ppl.csv`

Columns: `Generated Sequence`, `ESM PPL`, `MeMDLM PPL`, `Pred TM Segments`

---

### 5. Oligomerization

**Config:** `src/configs/oligo.yaml`  
**Script:** `src/sampling/olig_generator.py`

Redesigns sequences to increase predicted oligomerization propensity using the oligomerization classifier (NOS guidance).

```bash
nohup python src/sampling/olig_generator.py > logs/oligo_sample.out 2>&1 &
```

**Output:** `results/oligo/<wandb.name>/<date>/seqs_with_ppl.csv`

Columns: `Original Sequence`, `Generated Sequence`, `OG Olig Value`, `New Olig Value`, `Olig Increase`, `ESM PPL`, `MeMDLM PPL`, `MemDLM Blosum`

---

### Sampling quick-reference

| Task | Config | Script | Classifier required |
|---|---|---|---|
| Unconditional | `lm.yaml` | `unconditional_generator.py` | No |
| Solubilization | `solubility.yaml` | `pet_generator.py` | Solubility |
| Desolubilization | `desolubilize.yaml` | `desolubilize_generator.py` | Solubility |
| Multipass | `multipass.yaml` | `multipass_generator.py` | Multipass |
| Oligomerization | `oligo.yaml` | `olig_generator.py` | Oligomerization |

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{goel2026memdlm,
  title   = {Token-Level Guided Discrete Diffusion for Membrane Protein Design},
  author  = {Goel, Shrey and Chatterjee, Pranam},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

If you enjoyed this repo, please cite it and star the repository. We appreciate your support!
