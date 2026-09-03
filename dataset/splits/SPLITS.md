
# Overview of CSV splits in this folder

- **chexpert-train.csv / chexpert-valid.csv / chexpert-test.csv**: Train/validation/test (80/10/10) splits for experiments using the CheXpert dataset. Original datasplits from Favero et al.
- **isic-train.csv / isic-valid.csv / isic-test.csv**: Train/validation/test (80/10/10) splits for the ISIC skin-lesion dataset. Original datasplits from Favero et al.
- **pretrain-train.csv / pretrain-valid.csv / pretrain-test.csv**: (80/10/10) Non-Drusen-only splits used for private Phase-1 pretraining on clinic environment (used to train generative models on normal anatomy before finetuning) --> finally not used, because of hardware limitations. The private dataset contained 4952 non-drusen fundus images.
- **drusen-no-aug-train.csv / drusen-no-aug-valid.csv / drusen-no-aug-test.csv**: Train/validation/test (80/10/10) splits for the private Drusen dataset, without data augmentation and balanced classes (filled up to the number of drusen with random images from the non-drusen set). The private dataset contained 111 images showing optic disc drusen.
- **drusen-train.csv / drusen-valid.csv / drusen-test.csv**: Train/validation/test (80/10/10) splits for the private Drusen dataset, used for Phase-2 Finetuning and Training from Scratch experiment on clinical environment. The Train Set was expaned by 10 augmented variants of drusen and kept balanced between non-drusen and drusen class, leading to (0.8 * 111 * 10) drusen images in train, 111 * 0.1 drusen in test & valid split and filled up to same number of non-drusen images.
- **drusen-fold0-... - drusen-fold9-...**: 10-fold cross-validation splits for the private augmented Drusen dataset, for cross validating the Phase-2 Finetuning and Training from Scratch. Each fold produces `-train.csv`, `-valid.csv` and `-test.csv` (used for CV evaluation and reporting mean±std) with balanced classes and augmented variants only in train splits. The non-drusen pool used across all 10 folds is restricted to exactly the non-drusen images already present in drusen-train/valid/test.csv above, so the CV run and the hold-out split draw from the identical image pool (only the train/valid/test partitioning differs).

Scripts that produce these CSVs
- **create_splits_scripts/create_drusen_aug_split.py**
	- Builds balanced Drusen train/valid/test CSVs when augmented Drusen variants are available.
	- Splits by original-eye group to prevent augmentation leakage (all augmented variants of an eye stay together), and further by patient (see `patient_key()`), so a patient's images never end up split across train/valid/test.
	- Validation and test keep only untouched originals (`_aug00`); augmented variants are used for training only.
	- Enforces 50:50 class balance by sub-sampling non-drusen images (configurable).

- **create_splits_scripts/create_drusen_cv_splits.py**
	- Produces k-fold cross-validation splits for Drusen (grouped by original eye, and by patient — see `patient_key()`).
	- Each fold: `test` = fold i originals, `valid` = fold (i+1) originals, `train` = remaining folds (augmented variants allowed in train).
	- Balances non-drusen vs. drusen per split and writes `drusen-fold{i}-{train,valid,test}.csv` files, drawing the non-drusen pool for each fold fresh and independently from `--non-drusen-dir`.
	- Provides the Drusen-side grouping/partitioning logic (`patient_key()`, `original_id()`, `make_folds()`, ...) that `create_drusen_cv_from_holdout.py` below reuses; not used directly to produce the `drusen-fold{i}-*.csv` files currently shipped in this folder.
‚
- **create_splits_scripts/create_drusen_cv_from_holdout.py**
	- Produces the `drusen-fold0-...` – `drusen-fold9-...` CSVs actually shipped in this folder, so the 10-fold CV and the fixed `drusen-train/valid/test` hold-out split are directly comparable (same data pool, only the partitioning differs).
	- Imports and reuses `create_drusen_cv_splits.py`'s grouping/partitioning functions (`patient_key()`, `original_id()`, `is_original_variant()`, `make_folds()`) rather than reimplementing them.
	- Drusen (target=1): read fresh from `--drusen-dir` as in `create_drusen_cv_splits.py`, since this always yields the same 111 originals regardless of experiment.
	- Non-Drusen (target=0): instead of sampling `--non-drusen-dir` independently per fold, the candidate pool is restricted to exactly the union of non-drusen `image_name`s already used in `drusen-train.csv` / `drusen-valid.csv` / `drusen-test.csv` (passed in via `--holdout-train/valid/test`), then grouped by patient and partitioned into folds the same way.
	- Use this (instead of `create_drusen_cv_splits.py` directly) whenever the CV run needs to be compared one-to-one against the hold-out split.

- **create_splits_scripts/create_drusen_split.py**
	- Creates train/valid/test CSVs for Drusen WITHOUT augmentation.
	- Splits by patient (see `patient_key()` in `create_drusen_cv_splits.py`) and (optionally) enforces 50:50 balancing by sub-sampling.

- **create_splits_scripts/create_pretrain_split.py**
	- Builds non-drusen-only pretraining splits (`pretrain-train/valid/test`) for Phase 1 pretraining on clinical environment with private non-drusen data.
	- Excludes any non-drusen images that appear in the Phase-2 Drusen splits (train, valid AND test) to guarantee no image overlap between pretraining and finetuning/evaluation.
	- Splits the remaining pool by patient (see `patient_key()` in `create_drusen_cv_splits.py`).
	- Use this after running the Drusen split generation so exclusion lists match.
    - Finally in experiments don't used because of hardware constraints

Notes
- Filenames follow the pattern `<dataset>[-foldX]-{train,valid,test}.csv`.
- CSV format: two columns `image_name,target` where `image_name` is relative to the `--data-path` used by the script and `target` is `1` for positive (drusen) and `0` for non-drusen.
- When using augmented Drusen data, prefer the `_aug`-aware scripts to avoid leakage and ensure honest evaluation.


