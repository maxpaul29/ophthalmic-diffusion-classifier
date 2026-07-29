# Changelog

All modifications relative to the upstream repository [faverogian/med-diffusion-classifier](https://github.com/faverogian/med-diffusion-classifier) (Favero et al., MIDL 2025) are documented here.

The implementation was developed and executed in two separate computational environments:

1. **MOGON HPC environment** – used primarily for large-scale pretraining on publicly available fundus images.
2. **Clinical PC environment** – used for fine-tuning the pretrained model on the clinical Drusen dataset.

The two environments are maintained in separate Git branches (`drusen-mogon` and `drusen`) because they use different Python versions, dependency configurations, and execution environments. The changes are therefore documented separately where they are specific to one environment.

---

# 1. Common Modifications

The following modifications are shared between the MOGON and Clinical PC implementations or represent conceptual extensions of the original Favero et al. framework.

## 1.1 Adaptation to Fundus Image Classification

The original framework was adapted from the existing ISIC-based implementation to support fundus image classification.

A dedicated dataset implementation was created:

```text
dataset/fundus.py
```

The dataset loader was adapted with respect to:

* class names,
* dataset paths,
* CSV file paths.

The overall dataset loading logic remains based on the original framework.

## 1.2 Fundus UNet Experiment Pipeline

A dedicated experiment directory was created:

```text
experiments/fundus-unet/
```

The following components were adapted or introduced:

```text
train.py
finetune.py
inference.py
explain.py
```

The scripts are based on the corresponding existing implementations and were adapted for the fundus dataset. The finetune script was created as a new script, based on the train script for ISIC Dataset.

The modifications include:

* dataset-specific imports,
* function names,
* dataset references,
* configuration handling,
* checkpoint handling.

This experiment folder, is called by a new introduced `fundus-unet.sh` file, created into

```text
scripts/
```

The modifications, which differ the script from `isic-unet.sh`, include:

* experiment folder reference,
* dataset reference,
* introduction of a finetuning option,
* configuration handling,
* creation of a pretrained checkpoint reference.

Also in 

```text
scripts/run.sh
```

the option for finetuning and for selecting the fundus dataset was included.


## 1.3 AUC Evaluation Metric

The original evaluation framework was extended to include the Area Under the ROC Curve (AUC).

### Metric Implementation

`utils/metrics.py` was extended with:

* a new AUC metric,
* a `requires_scores` property to distinguish metrics requiring continuous prediction scores.

### Diffusion Classifier

The diffusion classifier was modified to optionally return continuous classification scores.

The classification procedure was extended with a `return_scores` parameter.

The resulting scores are passed to the evaluation procedure and used to calculate the AUC.

AUC was integrated into:

* training (isic, chexpert, fundus),
* fine-tuning (fundus),
* inference (isic, chexpert, fundus).

The reported evaluation metrics therefore include:

* accuracy,
* F1-score,
* precision,
* recall,
* AUC.

## 1.4 Fine-Tuning and Checkpoint Resumption

The fine-tuning implementation was extended to support continued training from existing checkpoints.

Fine-tuning can be resumed either from:

* the latest available checkpoint, or
* a user-specified checkpoint.

This allows training to be continued without restarting the complete fine-tuning process.

## 1.5 Evaluation and Visualization

Additional utilities were introduced to facilitate the analysis of training and evaluation results.

These include:

* metric plotting,
* loss plotting,
* evaluation with different numbers of diffusion sampling steps (*TODO*).

The inference and explanation procedures were also adapted to support the corresponding experiment configurations and checkpoints.

## 1.6 Dataset Split Prefix

The dataset handling was extended with a configurable:

```text
split_prefix
```

parameter.

This parameter determines which CSV split files are used for a specific experiment.

The parameter was propagated through the fundus training, fine-tuning, inference, and explanation pipelines.

This mechanism allows different dataset splits to be selected without modifying the underlying dataset implementation and to select automatically the matching training metric (loss vs. classification metrics).

## 1.7 Single-Class Fundus Pretraining

A two-stage training strategy was introduced.

The first stage consists of pretraining on a large dataset containing unconditioned fundus images.

Because this dataset contains only one class, the standard classification-based checkpoint selection of the original framework cannot be applied.

The diffusion classifier was therefore extended to support training with:

```text
metrics = None
```

The following modifications were introduced:

* handling of missing checkpoint trackers,
* epoch-level training-loss calculation,
* validation-loss calculation,
* skipping of unnecessary classification passes,
* validation-loss-based best-checkpoint selection,
* continued saving of the latest checkpoint.

This allows the diffusion model to be pretrained without calculating classification metrics for the single-class dataset, which is automatically selected based on the previous created `split_prefix` parameter.

---

# 2. MOGON HPC Branch

The MOGON branch contains all modifications required for large-scale pretraining and execution on the MOGON high-performance computing system.

The MOGON environment was used primarily for pretraining the diffusion model on a large collection of publicly available fundus images.

The MOGON branch does **not** use Docker for training execution.

---

## 2.1 Python and Dependency Configuration

The MOGON implementation uses:

```text
Python 3.12.3
```

This version was selected because it was the available Python version on the MOGON system.

---

## 2.2 MOGON Job Execution

MOGON-specific SLURM job scripts were introduced to execute training and inference jobs on the cluster.

The job scripts were adapted with respect to:

* paths,
* dataset locations,
* checkpoint locations,
* execution commands,
* cluster-specific environment configuration.

A dedicated job script was introduced to execute the repository scripts on MOGON.

---

## 2.3 Dataset and Checkpoint Transfer

Because datasets and pretrained weights were transferred to MOGON from an external environment, additional procedures were introduced for:

* downloading datasets,
* downloading pretrained model weights,
* transferring files to MOGON,
* synchronizing datasets with the cluster filesystem.

---

## 2.4 Runtime Dataset Staging

To reduce access to the shared Lustre filesystem during training, the dataset handling was adapted to copy the required data to the local scratch directory of the current SLURM job:

```text
/localscratch/${SLURM_JOB_ID}/dataset
```

The dataset is subsequently unpacked at runtime.

The relevant scripts and job configurations were adapted to support this workflow.

---

## 2.5 Large-Scale Fundus Pretraining

A large-scale pretraining dataset was created from publicly available fundus datasets.

The resulting MOGON splits contains 426,371 images, referenced in new created csv files:

```text
splits/pretrain-mogon-train.csv
splits/pretrain-mogon-valid.csv
splits/pretrain-mogon-test.csv
```

The splits contain approximately:

* 341,096 training images,
* 85,274 validation images,
* 85,274 test images.

The pretraining configuration was adapted to the larger dataset.

The number of training epochs, evaluation frequency and evaluation batches were adjusted to match the experiment details and to reduce computational costs.

---

# 3. Clinical PC Branch

The Clinical PC branch contains the modifications required for fine-tuning the pretrained diffusion model on the clinical optic disc drusen dataset.

The Clinical PC environment differs from the MOGON environment in terms of:

* Python version,
* dependency configuration,
* available hardware,
* execution environment.

Unlike the MOGON branch, the Clinical PC branch uses **Docker** to provide a reproducible training environment.

---

## 3.1 Python and Dependency Configuration

The Clinical PC branch uses a Python version compatible with the Docker-based training environment: Python 3.11.11

The Python version and dependencies are therefore defined by the Docker configuration rather than by the MOGON HPC environment.

---

## 3.2 Docker-Based Execution Environment

A dedicated containerized training environment was introduced for the Clinical PC.

The following files were added:

```text
Dockerfile
docker-compose.yml
.dockerignore
docker.md
```

The Docker environment defines:

* the Python version,
* required Python dependencies,
* GPU support,
* the training environment,
* the execution configuration.

This ensures that the Drusen fine-tuning experiments can be reproduced independently of the host system configuration.

---

## 3.3 Drusen Dataset Preparation

The Clinical PC branch introduces the dataset preparation pipeline for optic disc drusen.

Dedicated scripts were created into `datasets/` and `datasets/splits` for:

* Drusen-specific data augmentation,
* dataset splitting,
* generating dataset splits without augmentation.

The resulting dataset configuration is used for the subsequent fine-tuning experiments.

---

## 3.4 Two-Stage Training Strategy

The Clinical PC branch uses the pretrained model obtained from the MOGON pretraining stage.

The training procedure therefore consists of:

```text
MOGON:
Large-scale unconditioned fundus pretraining
                │
                ▼
Clinical PC:
Fine-tuning on healthy + Drusen images
```

The Clinical PC branch is therefore dependent on the pretrained checkpoint generated by the MOGON branch.

---

## 3.5 Drusen Fine-Tuning Dataset

For fine-tuning, a balanced dataset containing:

* 890 healthy and 890 Drusen fundus images for training,
* 11 healthy and 11 Drusen fundus images for validation,
* 11 healthy and 11 Drusen fundus images for testing,

was used.

This data was referenced in the new created csv files:
```text
splits/drusen-train.csv
splits/drusen-valid.csv
splits/drusen-test.csv
```

The dataset split is selected using the configurable `split_prefix` mechanism introduced in the common fundus pipeline.

The number of training epochs, evaluation frequency, batch size and gradient accumulation were adjusted to match the experiment and to reduce computational costs.

---

## 3.6 Fine-Tuning Configuration

The fine-tuning configuration was adapted to the smaller clinical dataset and available GPU memory.

The configuration uses:

```text
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
```

This results in an effective batch size of approximately 128, with a small adaption in `diffusion_classifier.py` to accumulate also over the EMA.

The fine-tuning environment was additionally configured to use FlashAttention where supported, which was loaded in the docker file.

Also, a few lines were added in `diffusion_classifier.py` to empty cuda cache, and reduce memory usage.

To run `explain.py`a further fix was nessesary, to adapt the numbers of iterations to the configured batch size.

---

## 3.7 Checkpoint Selection for Drusen Classification

The checkpoint selection strategy was adapted for the Drusen classification task.

During the initial fine-tuning experiments, recall was used as the best-checkpoint selection metric.

The selection criterion was subsequently changed to the F1-score to provide a more balanced consideration of precision and recall for the drusen experiment. This is also in line with the used metric by Favero et al.

Drusen-specific checkpoint naming and checkpoint folder creation was introduced to distinguish the resulting models from other fundus experiments.

---

## 3.8 Clinical PC Training Monitoring

Because fine-tuning was performed on a local clinical workstation, additional notification and monitoring functionality was introduced.

This includes:

* training completion notifications,
* periodic training updates,
* additional logging,
* progress monitoring.

The following components were introduced or adapted:

```text
entrypoint_with_notify.sh
notify_email.py
```

The Docker configuration was adapted accordingly.

---

## 3.9 K-Fold Cross-Validation for Drusen Fine-Tuning

Given the limited number of available Drusen cases (111 original images), a single fixed train/valid/test split was considered insufficient to provide a robust estimate of test performance. A k-fold cross-validation procedure was therefore introduced for the fine-tuning stage.

### Dataset Splitting

A dedicated split generation script was added:

```text
dataset/splits/create_drusen_cv_splits.py
```

The script partitions the original Drusen images into k folds at the group level, so that augmented variants of the same original image are never split across folds. For each fold, one part is used as the test set, one as the validation set, and the remaining folds are used for training (including their augmented variants). Class balance and evaluation restricted to original, non-augmented images are preserved as in the existing single-split logic.

### Fold-Aware Configuration

The fine-tuning configuration (`fundus-unet.sh`) was extended with a configurable `FOLD` parameter. When set, it overrides `SPLIT_PREFIX` (selecting `drusen-fold{FOLD}` instead of the fixed `drusen` split) and `CHECKPOINT_FOLDER` (pointing inference at the corresponding fold's archived checkpoint). The default, non-CV single-split workflow remains unchanged when `FOLD` is unset.

### Run Orchestration

To orchestrate the resulting k independent fine-tuning and inference runs, a new script was introduced:

```text
scripts/run_drusen_cv.sh
```

which, for each fold, runs fine-tuning followed by inference, and archives the fold's checkpoint out of the shared experiment folder before the next fold starts, so that subsequent folds do not overwrite previous results. Because the experiment folder and the archive location reside on separate Docker volumes, archiving is implemented as an idempotent move (clearing any pre-existing target first) rather than a plain rename, so it can be safely re-run after an interrupted attempt. A `START_FOLD` parameter additionally allows resuming a cross-validation run from a specific fold, so already-completed and archived folds are not repeated.

`inference.py` was extended to additionally write the evaluation results to a JSON file alongside the checkpoint used for the run, so they can be collected programmatically. A new aggregation script was introduced:

```text
experiments/fundus-unet/aggregate_cv_results.py
```

which collects the per-fold test results and computes the mean and standard deviation across all folds, providing the final reported classification performance.

### Unattended Execution

Because cross-validation involves several consecutive fine-tuning runs and is intended to run unattended on the clinical workstation, a dedicated notification wrapper was added:

```text
scripts/entrypoint_cv_with_notify.sh
```

mirroring the existing `entrypoint_with_notify.sh` (log capture, hourly progress e-mails, final completion e-mail), but invoking `run_drusen_cv.sh` instead of a single `run.sh` call. It was kept as a separate script rather than a conditional branch inside `entrypoint_with_notify.sh`, so the default single-run workflow is left completely unaffected and cross-validation is only ever triggered by explicitly invoking this script. `docker-compose.yml` was extended with `K_FOLDS` and `PRETRAIN_CHECKPOINT` environment variables so these can be configured via `.env` instead of being passed on every invocation.

### Bug Fix: Environment Variable Propagation in `run.sh`

While integrating the orchestration script, it was discovered that `scripts/run.sh` unconditionally re-exported `MODEL`, `FUNCTION`, `DATA`, `BACKBONE`, and `VARIANT` to fixed literal values, silently discarding any values already exported by a calling script (such as `run_drusen_cv.sh` setting `MODEL=unet`). As a result, orchestrated runs unintentionally fell back to the baseline classifier configuration. These assignments were changed to `"${VAR:-default}"`, so a value pre-set by a calling script is respected, while direct, manual invocation of `run.sh` keeps its previous default behaviour.

---

## 3.10 Uncertainty Quantification for Drusen Classification

Favero et al. note that the diffusion classifier inherently produces an uncertainty estimate for each prediction, since majority voting already draws N per-sample votes from the (ε, λ) evaluations required for classification (Eq. (4)). This uncertainty was not previously exposed by the codebase and was added to evaluate whether filtering out uncertain predictions improves accuracy on the remaining Drusen test data, reproducing the corresponding analysis from the paper.

`DiffusionClassifier.classify()` (`diffusion/diffusion_classifier.py`) was extended with a `return_uncertainty` option. When enabled, it additionally returns, per sample, the Bernoulli variance `p * (1 - p)` of the winning class's vote share over the N evaluations already tallied for majority voting — 0 for a unanimous vote, 0.25 for an evenly split one. `evaluate()` and `inference()` were extended with a matching `collect_uncertainty` option that records, per test sample, the true label, predicted class, correctness, and this uncertainty value; all other call sites were updated for the resulting additional return value, with no change in behaviour when the option is left disabled.

This is exposed as a `UNCERTAINTY_ESTIMATION` flag in `scripts/unet/fundus-unet.sh` (default `false`) and, when enabled, `inference.py` additionally writes a `uncertainty_predictions.json` file alongside the checkpoint's evaluation results. A new plotting script was added:

```text
experiments/fundus-unet/plot_uncertainty_filtering.py
```

which sorts test samples by decreasing uncertainty, progressively removes the most uncertain ones, and recomputes accuracy on the remaining data at each step, reproducing the "removed data vs. accuracy" plot of Favero et al.

This analysis is currently only used on the clinical PC branch's plain single-split (80/10/10) Drusen classification run, not on the cross-validation folds, since it is intended as a qualitative check of whether the model's confidence is meaningful rather than as part of the reported cross-validated performance.

---

## 3.11 Cross-Validation for Training from Scratch and Unified CV Invocation

To assess whether Mogon Phase-1 pretraining meaningfully improves Drusen classification performance, a cross-validated training-from-scratch baseline was added, so it can be compared to the pretrained-and-finetuned CV results (Section 3.9) on the same folds rather than on a single split.

A dedicated orchestration script was introduced:

```text
scripts/run_drusen_scratch_cv.sh
```

which mirrors `run_drusen_cv.sh`, but calls `train.py` instead of `finetune.py` for each fold, without a pretrained checkpoint, and archives its checkpoints under a separate `drusen-unet-scratch` location so the two CV runs never share or overwrite each other's checkpoints. `scripts/unet/fundus-unet.sh` was extended with a `DRUSEN_MODEL_DIR` variable to control this archive location. While integrating this, it was discovered that `PRETRAINED_CHECKPOINT` was unconditionally re-exported to its default value in `fundus-unet.sh`, silently discarding an explicitly empty value set by a calling script; the assignment was changed from `"${VAR:-default}"` to `"${VAR-default}"`, which only substitutes the default when the variable is entirely unset.

To invoke the resulting three cross-validation variants (pretrained finetuning, training from scratch, baseline classifier) consistently, `scripts/run.sh` was extended with a `CROSS_VALIDATION` flag. When set to `1`, it dispatches to the matching orchestration script based on the selected `MODEL`/`FUNCTION` instead of running the normal single-run workflow; `CROSS_VALIDATION=0` (default) leaves the existing behaviour completely unchanged. Since each orchestration script itself calls `run.sh` again once per fold to perform the actual train/inference run, `CROSS_VALIDATION` is explicitly reset to `0` before dispatching, so this inherited-environment re-entry takes the normal single-run path instead of re-triggering the same orchestration script from fold 0 indefinitely.

---

# 4. Baseline Classifier

A separate discriminative baseline pipeline was implemented to provide a comparison with the diffusion-based classifier.

The baseline implementation is maintained separately from the diffusion-based UNet pipeline, in both branches, but runned on the Clinical Environment.

A dedicated fundus classifier configuration was created and adapted from the existing classifier implementation:

```text
scripts/baseline-classifier/fundus-classifier.sh
```

The baseline pipeline, was extended to:

```text
experiments/fudus-classifier/
```

Containing:

* train.py,
* inference.p,

both adapted from the scripts in `isic-classifier`, with modifications containing:

* dataloader adaption,
* configurable dataset split prefixes,
* AUC evaluation,
* fix of the AUC metric receiving hard predictions instead of continuous scores.

A ResNet50-based baseline using ImageNet-pretrained weights was evaluated with this setting.

## 4.1 K-Fold Cross-Validation for the Baseline Classifier

To allow a like-for-like comparison with the diffusion classifier's cross-validated performance (Section 3.9), the k-fold cross-validation procedure was extended to the baseline classifier, reusing the same fold splits.

Unlike the diffusion classifier, the baseline has no separate pretraining and fine-tuning stages: each fold trains directly from ImageNet-pretrained weights, so no shared pretrain checkpoint needs to be passed between folds.

`scripts/baseline-classifier/fundus-classifier.sh` was extended with the same `FOLD`-aware override of `SPLIT_PREFIX` and `CHECKPOINT_FOLDER` introduced for the diffusion classifier. `experiments/fundus-classifier/inference.py` was extended to additionally write its evaluation results to a JSON file alongside the checkpoint used for the run, matching the diffusion classifier's inference output.

A dedicated orchestration script was introduced:

```text
scripts/run_baseline_cv.sh
```

which, for each fold, trains the baseline from scratch and runs inference on the held-out test split, archiving the fold's checkpoint out of the shared experiment folder before the next fold starts, using the same idempotent-move pattern as `run_drusen_cv.sh`. Since the baseline's checkpoint directories are named after the selected backbone variant (`checkpoint_<variant>` / `best_checkpoint_<variant>`) rather than the diffusion classifier's fixed `checkpoints` / `best_checkpoint`, the existing aggregation script was extended with a `--checkpoint-subdir` option so it can locate and aggregate either classifier's fold results:

```text
experiments/fundus-unet/aggregate_cv_results.py
```

---

# 5. Branch Relationship

The two branches represent different stages of the overall experimental pipeline.

The intended workflow is:

```text
                 Original Favero et al. Framework
                              │
                              ▼
                  Common Fundus Adaptations
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          MOGON Branch              Clinical PC Branch
          Python 3.12.3             Python 3.11.11
          No Docker                 Docker environment
          HPC / SLURM               Local GPU workstation
                 │                         │
                 ▼                         │
      Large-scale unconditioned            │
      fundus pretraining                   │
                 │                         │
                 └─────── Checkpoint ──────┘
                              │
                              ▼
                    Drusen Fine-Tuning
                              │
                              ▼
                    ODD Classification
```

The MOGON branch and Clinical PC branch should therefore not be interpreted as two independent implementations of the complete framework.

Instead, they represent **two consecutive stages of the experimental workflow**:

1. **MOGON:** large-scale pretraining of the diffusion model on unconditioned fundus images.
2. **Clinical PC:** transfer of the pretrained model and fine-tuning on the clinical Drusen dataset.

The different Python versions, dependency configurations, and execution environments are intentional and are documented separately to ensure reproducibility.

---

# 6. Summary of Environment-Specific Modifications

| Component             | MOGON Branch                      | Clinical PC Branch                |
| --------------------- | --------------------------------- | --------------------------------- |
| Purpose               | Large-scale pretraining           | Drusen fine-tuning                |
| Environment           | MOGON HPC                         | Local clinical workstation        |
| Execution             | SLURM job scripts                 | Docker Compose                    |
| Docker                | No                                | Yes                               |
| Python                | 3.12.3                            | 3.11.11                           |
| Dataset               | Large-scale public fundus dataset | Clinical healthy + Drusen dataset |
| Training data         | ~400,000 fundus images            | 890 healthy + 890 Drusen images   |
| Training strategy     | unconditioned pretraining         | Fine-tuning of pretrained model   |
| Batch size            | 128                               | 16                                |
| Gradient accumulation | 1                                 | 8                                 |
| Effective batch size  | ~128                              | ~128                              |
| Checkpoint selection  | Validation loss                   | Classification metric             |
| Main objective        | Learn fundus image representation | Adapt to ODD classification       |
| Hardware adaptation   | HPC / GPU memory                  | Local GPU memory                  |
| Dataset staging       | MOGON local scratch               | Local filesystem / Docker volume  |
| Notifications         | HPC/job monitoring                | Docker-based notification scripts |

---

# 7. Overview of Changes Relative to the Original Framework

The following modifications distinguish the adapted implementation from the original Favero et al. framework:

1. Adaptation from the original datasets to fundus image classification.
2. Creation of a dedicated fundus dataset loader.
3. Creation of a dedicated fundus UNet training, fine-tuning, inference, and explanation pipeline.
4. Addition of AUC as an evaluation metric.
5. Addition of continuous prediction-score handling for AUC calculation.
6. Extension of fine-tuning with checkpoint resumption.
7. Introduction of configurable dataset split prefixes, including automatic selection of the corresponding checkpoint selection strategy based on the selected split.
8. Introduction of a large-scale unconditioned-fundus pretraining stage.
9. Modification of the diffusion classifier to support single-class pretraining.
10. Introduction of epoch-level training-loss and validation-loss calculation for single-class pretraining.
11. Skipping of unnecessary classification and majority-voting passes during single-class pretraining.
12. Introduction of validation-loss-based checkpoint selection for single-class pretraining.
13. Adaptation of the training configuration to hardware constraints, including batch-size and gradient-accumulation adjustments.
14. Correction of EMA updates in conjunction with gradient accumulation.
15. Introduction of Drusen-specific dataset preparation, augmentation, and dataset splitting.
16. Introduction of a clinical Drusen fine-tuning pipeline based on the pretrained model.
17. Creation of separate MOGON and Clinical PC execution environments and corresponding Git branches.
18. Introduction of MOGON-specific SLURM job execution and local-scratch dataset staging.
19. Introduction of a Docker-based execution environment using Python 3.11.11 for clinical fine-tuning.
20. Introduction of a separate discriminative baseline pipeline for comparison with the diffusion-based classifier.
21. Addition of a ResNet50-based baseline using ImageNet-pretrained weights.
22. Introduction of a k-fold cross-validation procedure for the Drusen fine-tuning stage, including fold-aware dataset splitting, run orchestration, unattended execution with notifications, and result aggregation.
23. Fix of environment-variable propagation in `run.sh` (`MODEL`, `FUNCTION`, `DATA`, `BACKBONE`, `VARIANT` were previously always reset to fixed defaults, breaking orchestrated runs that pre-set these values).
24. Addition of per-sample uncertainty quantification for the Drusen classifier, based on the Bernoulli variance of the majority-voting vote share, together with an uncertainty-based data-filtering analysis and plot.
25. Extension of the k-fold cross-validation procedure to the ResNet50 baseline classifier, reusing the diffusion classifier's fold splits and a generalized result-aggregation script.
26. Addition of a cross-validated training-from-scratch comparison for the Drusen classifier, and a unified `CROSS_VALIDATION` flag in `run.sh` to invoke any of the three cross-validation variants (pretrained finetuning, from scratch, baseline).
