# Ophthalmic Diffusion Classifier

**Author:** Maximilian Paul
**Institution:** Johannes Gutenberg University Mainz
**Context:** Bachelor's thesis — *Exploring Generative AI Models for Ophthalmic Disease Classification*

This repository extends the official implementation of

> **Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free** (MIDL 2025)
> Gian Favero\*, Parham Saremi\*, Emily Kaczmarek, Brennan Nichyporuk, Tal Arbel
> Original repository: https://github.com/faverogian/med-diffusion-classifier

for **Optic Disc Drusen (ODD) classification on fundus images**, using a conditional diffusion model as a classifier (via reconstruction error, following the original framework) and comparing it against a discriminative ResNet50 baseline.

The original Favero et al. README is preserved unmodified at the bottom of this file. Everything above it is specific to this thesis.

---

## What this project adds

Starting from the original two-dataset (CheXpert/ISIC) framework, this thesis:

- **Adapts the pipeline to fundus images** and introduces a two-stage training strategy: large-scale unconditioned pretraining on public fundus images, followed by fine-tuning on a private, clinically-sourced Optic Disc Drusen dataset.
- **Compares fine-tuning against training from scratch** directly on the Drusen data, to assess whether the pretraining stage is actually beneficial for this task.
- **Adds a ResNet50 discriminative baseline** for a like-for-like comparison with the diffusion classifier.
- **Introduces k-fold cross-validation** for all three of the above (finetuned, from-scratch, baseline), since the private Drusen dataset is small — a single train/valid/test split would not give a robust performance estimate.
- **Adds an uncertainty quantification analysis**: the diffusion classifier's Monte Carlo majority vote is used to derive a per-sample uncertainty estimate, and accuracy is reported as a function of how much of the most uncertain data is filtered out — reproducing an analysis from Favero et al. on this new task.
- **Adds an AUC metric** and continuous-score support throughout the classification pipeline.
- **Provides a fully Dockerized, reproducible execution environment** for the clinical workstation, including unattended (detached) execution with e-mail progress notifications.
- **Splits the work across two Git branches** for two different compute environments (see below), since large-scale pretraining and clinical fine-tuning happened on different infrastructure with different constraints.

Every one of these changes is documented in detail — see [Documentation map](#documentation-map) below for where.

---

## Repository & branch structure

The thesis workflow spans two environments, tracked as two branches:

```text
              Original Favero et al. Framework
                            │
                            ▼
                Common Fundus Adaptations
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
        `drusen-mogon` branch      `drusen` branch
        MOGON HPC, SLURM           Clinical PC, Docker
        Large-scale unconditioned  Fine-tuning / from-scratch
        fundus pretraining         training on Drusen data,
               │                   cross-validation, baseline,
               │                   uncertainty analysis
               └────────── checkpoint ─────┘
```

These are **not** two independent copies of the framework — they are two consecutive stages of one pipeline, kept on separate branches because they use different Python versions, dependencies, and execution environments (no Docker on the HPC side; SLURM job scripts instead of `docker compose`). See [CHANGELOG.md, Section 5](CHANGELOG.md#5-branch-relationship) for the full rationale.

**If you are trying to reproduce the thesis results, you want the `drusen` branch** — it contains everything needed to fine-tune the (already-pretrained) diffusion model, train it from scratch for comparison, train the baseline, run cross-validation for all three, and reproduce the uncertainty analysis, all via Docker.

---

## Documentation map

Rather than duplicating information, each concern has exactly one authoritative document:

| Document | Covers |
|---|---|
| **README.md** (this file) | High-level overview, what the project does, where to find everything else |
| [**DOCKER.md**](DOCKER.md) | Complete setup, configuration, and command reference for running anything in this repo (training, fine-tuning, inference, explanation, all cross-validation variants) via Docker on the clinical PC |
| [**CHANGELOG.md**](CHANGELOG.md) | Exhaustive, line-by-line record of every modification relative to the original Favero et al. framework, organized by branch/topic — the primary reference for *what was changed and why* |
| [**dataset/splits/SPLITS.md**](dataset/splits/SPLITS.md) | What each dataset split CSV contains, how it was generated, and which script produced it |
| [**results/RESULTS.md**](results/RESULTS.md) | Layout of the `results/` directory: logs, plots, and metrics for every experiment (baseline, fine-tuned, from-scratch, single-run and cross-validated) |

**If you only read one other document, read `DOCKER.md`** — it is the practical entry point for actually running anything.

---

## Quick start

Everything runs in Docker on the clinical PC (`drusen` branch). Full details, prerequisites, and the complete command reference are in [DOCKER.md](DOCKER.md); in short:

```bash
docker compose build      # once
docker compose up -d      # runs whatever is currently configured in scripts/run.sh
docker compose logs -f    # follow progress
```

**What runs is configured by editing two files directly** — `scripts/run.sh` (model/task/cross-validation selection) and `scripts/unet/fundus-unet.sh` (diffusion-classifier-specific settings). This keeps configuration in exactly one place per concern. A few examples (see [DOCKER.md, Section 5](DOCKER.md#5-what-each-configuration-runs) for the complete list, including cross-validation, resuming, and advanced options):

| Goal | Set in `scripts/run.sh` |
|---|---|
| Fine-tune the diffusion classifier (single run) | `MODEL=unet`, `FUNCTION=finetune`, `CROSS_VALIDATION=0` |
| Train the diffusion classifier from scratch (single run) | `MODEL=unet`, `FUNCTION=train`, `CROSS_VALIDATION=0` |
| Train the ResNet50 baseline (single run) | `MODEL=baseline`, `FUNCTION=train`, `CROSS_VALIDATION=0` |
| 5-fold cross-validation for any of the above | as above, plus `CROSS_VALIDATION=1` |
| Generate counterfactual explanations | `MODEL=unet`, `FUNCTION=explain`, `CROSS_VALIDATION=0` |

---

## Reproducibility

- **Environment**: pinned via Docker (`Dockerfile`, `docker-compose.yml`) — Python 3.11.11, CUDA 12.4, all dependencies in `requirements.txt`. No manual environment setup is needed on the clinical PC; see [DOCKER.md](DOCKER.md).
- **Data & checkpoints stay local**: mounted as Docker volumes, never copied into the image — see [DOCKER.md, Section 6](DOCKER.md#6-notes).
- **Deterministic splits**: all dataset splits (single-split and 5-fold cross-validation) are generated by versioned scripts in `dataset/splits/`, documented in [SPLITS.md](dataset/splits/SPLITS.md) — re-running a split script with the same input reproduces the same CSVs.
- **Cross-validation everywhere it matters**: fine-tuning, from-scratch training, and the baseline are all evaluated via the same 5-fold procedure, reported as mean ± standard deviation rather than a single train/valid/test split, given the small size of the private Drusen dataset (111 original images).

---

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE)), inherited from the original Favero et al. repository.

## Citation

If you use this code, please cite the original paper this work builds on:

```bibtex
@misc{favero2025conditionaldiffusionmodelsmedical,
      title={Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free},
      author={Gian Mario Favero and Parham Saremi and Emily Kaczmarek and Brennan Nichyporuk and Tal Arbel},
      year={2025},
      eprint={2502.03687},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.03687},
}
```

---
The original Favero et al. README is preserved unmodified below.

# **Medical Diffusion Classifier: Official PyTorch Implementation**  

**Venue**: MIDL 2025  
**Paper**: Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free  
**Authors:**  Gian Favero\*, Parham Saremi\*, Emily Kaczmarek, Brennan Nichyporuk, Tal Arbel  
**Institution(s):**  Mila - Quebec AI Institute, McGill University

<p align="center">
<a href="https://arxiv.org/abs/2502.03687" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2410.05203-b31b1b.svg?style=flat" /></a>
<a href="https://faverogian.github.io/med-diffusion-classifier.github.io/" alt="webpage">
    <img src="https://img.shields.io/badge/Webpage-darkviolet" /></a>
<img src="https://img.shields.io/github/license/faverogian/med-diffusion-classifier" />
<img src="https://views.whatilearened.today/views/github/faverogian/med-diffusion-classifier.svg" />
  
<p align="center">
<picture>
  <img src="https://faverogian.github.io/med-diffusion-classifier.github.io/static/images/architecture.png">
</picture>
</p>

## Requirements

* Use Linux (recommended) for best performance, compatibility, and reproducibility.
* All testing, training, inference completed with A100 NVIDIA GPUs (single or multiple).
* 64-bit Python 3.10 and PyTorch 2.6. See https://pytorch.org for PyTorch install instructions.
* Python virtual environment (recommended) to manage libraries, packages for this repository.

## Getting Started

First, clone this repository.

### Installing Packages

Required packages are provided in the `requirements.txt` file and can be installed using the following command:

```bash
pip install -r requirements.txt
```

FlashAttention can be installed for faster inference time (especially for DiT models). The wheel file can be downloaded from [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1). We used `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` for our CUDA and Torch version. After downloading the wheel file, install it with:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Data Preparation

We use the CheXpert and ISIC datasets in our paper. Our train/validation/test CSV files for both datasets are in the `splits` folder.

### Downloading Model Weights

All trained models can be downloaded from [this link](https://drive.google.com/drive/folders/1x7CKrbS8pxS45EXzpUKhpusBYCLgH82Y?usp=drive_link). Alternatively, use the following command with `gdown`:

```bash
gdown --folder 1x7CKrbS8pxS45EXzpUKhpusBYCLgH82Y
```

### Configuration

Before using the models, modify the `scripts/run.sh` file:

- `PROJECT_ROOT`: Absolute path to the root directory of the diffusion-classifier repository.
- `DATA_ROOT`: Absolute path to the data directory containing `isic_mel_balanced/`, `chexpert/`, and `sd_isic_chexpert/` folders.
- `INFERENCE_CHECKPOINT_FOLDER`: Absolute path to the directory where the downloaded model weights are stored.

If you want to use CometML for experiment tracking, set the COMET variables in `run.sh`. Additionally, you must set `USE_COMET=1` to enable tracking.

All training/inference hyperparameters are defined in their corresponding bash scripts. For example, the CheXpert-UNet hyperparameters are specified in `scripts/unet/chexpert-unet.sh`.

Below, we describe various use cases that are easily achievable with simple customizations to our code. In any case, launching the desired experiment is done via `bash scripts/run.sh` from the parent folder of the repository.

## Using Pre-Trained Models

Scripts to run inference with all models are provided in the `scripts` folder. However to launch each script you only have to modify the `run.sh` file to select which model and data you want to run. For instance, to run the UNet model's inference on the CheXpert dataset, you'll need to set `MODEL=unet`, `DATA=chexpert`, `FUNCTION=inference`. 

### Baselines

Baseline classifiers for both datasets can be evaluated using scripts in `scripts/run.sh`. You can change the `VARIANT` and `BACKBONE` environment variables to run different models. Available models:

| VARIANT                              | BACKBONE     |
|--------------------------------------|--------------|
| resnet18                             | resnet       |
| resnet50                             | resnet       |
| efficientnet_b0                      | efficientnet |
| efficientnet_b4                      | efficientnet |
| swin_base_patch4_window7_224         | swin         |
| vit_base_patch16_224                 | vit          |
| vit_small_patch16_224                | vit          |

To evaluate on CheXpert, modify the `run.sh` file:

```bash
export MODEL="baseline" 
export FUNCTION="inference"
export DATA="chexpert" 
```

Or on ISIC:

```bash
export MODEL="baseline"
export FUNCTION="inference"
export DATA="chexpert" 
```

### Inference: Diffusion Models

Use the following instructions to run inference with diffusion models.

For faster inference, set the `FLASH_ATTENTION` variable to `true` for DiT and UNet models.

**CheXpert-UNet:**
```bash
export MODEL="unet"
export FUNCTION="inference"
export DATA="chexpert" 
```

**ISIC-UNet:**
```bash
export MODEL="unet"
export FUNCTION="inference"
export DATA="isic" 
```

**CheXpert-DiT:**
```bash
export MODEL="dit"
export FUNCTION="inference"
export DATA="chexpert" 
```

**ISIC-DiT:**
```bash
export MODEL="dit"
export FUNCTION="inference"
export DATA="isic" 
```

**CheXpert-StableDiffusion:**
```bash
export MODEL="sd"
export FUNCTION="inference"
export DATA="chexpert" 
```

**ISIC-StableDiffusion:**
```bash
export MODEL="sd"
export FUNCTION="inference"
export DATA="dit" 
```

### Counterfactual Generation

Counterfactual generation is currently supported only for UNet models. To generate counterfactuals for the UNet models, modify the script to run `explain.py` instead of `inference.py`. This can be easily done by changing the `FUNCTION` value to `explain`:

```bash
export MODEL="unet"
export FUNCTION="explain"
export DATA="chexpert" 
```

To improve visual quality, increase `SAMPLING_STEPS` to at least 256 in the unet scripts (`scripts/unet/chexpert-unet.sh` and `scripts/unet/isic-unet.sh`). `CFG_W` refers to the classifier-free guidance scale.

The images will be saved in the `inference_images` directory located within the experiment folder.

## Training Models

Similar to counterfactual generation, training different models is as simple as changing the `FUNCTION` to `train`. For example, to train the UNet model on CheXpert data you should use the following environment variables:

```
export MODEL="unet"
export FUNCTION="train"
export DATA="chexpert" 
```

**Note:** The Stable Diffusion model is jointly trained on both ISIC and CheXpert datasets. To train on a single dataset, modify the `metadata.jsonl` placed in `data/sd_isic_chexpert` file and adjust the training data folder accordingly.

In order to train the Stable Diffusion model, the diffusers package should be installed [from source](https://huggingface.co/docs/diffusers/installation#install-from-source). To do this you can run the following command:

```
pip install git+https://github.com/huggingface/diffusers
```

The output directory for the Stable Diffusion model can be set in its `train.sh` script, while other models will save their checkpoints within their respective experiment folders.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

```bibtex
@misc{favero2025conditionaldiffusionmodelsmedical,
      title={Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free}, 
      author={Gian Mario Favero and Parham Saremi and Emily Kaczmarek and Brennan Nichyporuk and Tal Arbel},
      year={2025},
      eprint={2502.03687},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.03687}, 
}
```
