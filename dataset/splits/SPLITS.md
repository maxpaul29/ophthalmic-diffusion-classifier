
# Overview of CSV splits in this folder

- **chexpert-train.csv / chexpert-valid.csv / chexpert-test.csv**: Train/validation/test (80/10/10) splits for experiments using the CheXpert dataset. Original datasplits from Favero et al.
- **isic-train.csv / isic-valid.csv / isic-test.csv**: Train/validation/test (80/10/10) splits for the ISIC skin-lesion dataset. Original datasplits from Favero et al.
- **pretrain-train.csv / pretrain-valid.csv / pretrain-test.csv**: (80/10/10) Healthy-only splits used for private Phase-1 pretraining on clinic environment (used to train generative models on normal anatomy before finetuning) --> finally not used, because of hardware limitations. The private dataset contained 4952 healty fundus images.
- **drusen-no-aug-train.csv / drusen-no-aug-valid.csv / drusen-no-aug-test.csv**: Train/validation/test (80/10/10) splits for the private Drusen dataset, without data augmentation and balanced classes (filled up to the number of drusen with random images from health set). The private dataset contained 111 images showing optic disc drusen.
- **drusen-train.csv / drusen-valid.csv / drusen-test.csv**: Train/validation/test (80/10/10) splits for the private Drusen dataset, used for Phase-2 Finetuning and Training from Scratch experiment on clinical environment. The Train Set was expaned by 10 augmented variants of drusen and kept balanced between healthy and drusen class, leading to (0.8 * 111 * 10) drusen images in train, 111 * 0.1 drusen in test & valid split and filled up to same number of healty images.
- **drusen-fold0-... - drusen-fold4-...**: 5-fold cross-validation splits for the private augmented Drusen dataset on the drusen-train/test/valid split mentioned above, for cross validating the Phase-2 Finetuning and Training from Scratch. Each fold produces `-train.csv`, `-valid.csv` and `-test.csv` (used for CV evaluation and reporting mean±std) with balanced classes and augemented variants only in train splits.

Scripts that produce these CSVs
- **create_splits_scripts/create_drusen_aug_split.py**
	- Builds balanced Drusen train/valid/test CSVs when augmented Drusen variants are available.
	- Splits by original-eye group to prevent augmentation leakage (all augmented variants of an eye stay together), and further by patient (see `patient_key()`), so a patient's images never end up split across train/valid/test.
	- Validation and test keep only untouched originals (`_aug00`); augmented variants are used for training only.
	- Enforces 50:50 class balance by sub-sampling healthy images (configurable).

- **create_splits_scripts/create_drusen_cv_splits.py**
	- Produces k-fold cross-validation splits for Drusen (grouped by original eye, and by patient — see `patient_key()`).
	- Each fold: `test` = fold i originals, `valid` = fold (i+1) originals, `train` = remaining folds (augmented variants allowed in train).
	- Balances healthy vs. drusen per split and writes `drusen-fold{i}-{train,valid,test}.csv` files.
	- Use this to report robust CV metrics rather than a single holdout.

- **create_splits_scripts/create_drusen_split.py**
	- Creates train/valid/test CSVs for Drusen WITHOUT augmentation.
	- Splits by patient (see `patient_key()` in `create_drusen_cv_splits.py`) and (optionally) enforces 50:50 balancing by sub-sampling.

- **create_splits_scripts/create_pretrain_split.py**
	- Builds healthy-only pretraining splits (`pretrain-train/valid/test`) for Phase 1 pretraining on clinical environment with private healthy data.
	- Excludes any healthy images that appear in the Phase-2 Drusen splits (train, valid AND test) to guarantee no image overlap between pretraining and finetuning/evaluation.
	- Splits the remaining pool by patient (see `patient_key()` in `create_drusen_cv_splits.py`).
	- Use this after running the Drusen split generation so exclusion lists match.
    - Finally in experiments don't used because of hardware constraints

Notes
- Filenames follow the pattern `<dataset>[-foldX]-{train,valid,test}.csv`.
- CSV format: two columns `image_name,target` where `image_name` is relative to the `--data-path` used by the script and `target` is `1` for positive (drusen) and `0` for healthy.
- When using augmented Drusen data, prefer the `_aug`-aware scripts to avoid leakage and ensure honest evaluation.


