# Results Directory Overview

This file documents the structure of the `results` directory on this branch.

## General layout

- `results/fundus-unet/pretraining/`: Contains the results of the Phase-1 large-scale, single-class fundus pretraining run on MOGON.

Note: Phase-2 fine-tuning, training-from-scratch, the ResNet50 baseline, cross-validation, and uncertainty-quantification results (and their `RESULTS.md`) live on the `drusen` branch instead — this branch only covers Phase-1 pretraining.

## fundus-unet/pretraining

- `results/fundus-unet/pretraining/plots/`
  - `plot_loss_pretrain_mogon.py` / `pretrain_loss.png`: train/validation reconstruction loss over training epochs — the only monitored signal for this single-class pretraining run, since classification metrics are undefined with a single class (see CHANGELOG.md Section 1.7).
  - `plot_metrics_pretrain_mogon.py` / `accuracy.png`, `f1.png`, `precision.png`, `recall.png`, `all_metrics.png`, `metrics_grid.png`: accuracy/F1/precision/recall curves from an earlier two-class training run on the common `fundus` (Kaggle) split, kept here for reference alongside the pretraining loss.
- `results/fundus-unet/pretraining/training_images/`: sample reconstructions saved during evaluation epochs (`inactive/` subfolder only — single-class pretraining has no conditioning class to distinguish `active`/`inactive`).

## Raw job logs

The raw SLURM stdout/stderr logs for every pretraining job submission (including resumed runs after job time limits) are kept as-is in `scripts/logs/` (`odc_<jobid>.out` / `.err`), rather than duplicated or cleaned up here — see `scripts/odc_fundus_job.sbatch` for the job definition that produced them.
