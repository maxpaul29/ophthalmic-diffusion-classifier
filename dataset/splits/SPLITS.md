
# Overview of CSV splits in this folder

- **chexpert-train.csv / chexpert-valid.csv / chexpert-test.csv**: Train/validation/test (80/10/10) splits for experiments using the CheXpert dataset. Original datasplits from Favero et al.
- **isic-train.csv / isic-valid.csv / isic-test.csv**: Train/validation/test (80/10/10) splits for the ISIC skin-lesion dataset. Original datasplits from Favero et al.
- **fundus-train.csv / fundus-valid.csv / fundus-test.csv**: Train/validation/test (80/10/10) splits for the public Kaggle fundus dataset (`fundus-metadata.csv`), used for the common fundus-classification adaptation of the pipeline.
- **pretrain-mogon-train.csv / pretrain-mogon-valid.csv / pretrain-mogon-test.csv**: Train/validation/test splits (~409k/8.5k/8.5k images) for the large-scale, public, single-class Phase-1 pretraining pool run on MOGON (~400,000 fundus images total, combining several public sources; all rows written with `target=0` since Phase-1 pretraining is unconditioned — see `create_pretrain_mogon_split.py` below and CHANGELOG.md Section 1.7).

Script that produce these CSVs:

- **create_splits_scripts/create_pretrain_mogon_split.py**
	- Builds the Phase-1 `pretrain-mogon-{train,valid,test}` CSVs from a large public fundus image pool.
	- Unlike the private clinic Drusen data (fine-tuned on in Phase 2, on the `drusen` branch), this pool is a separate public dataset — no exclusion/leakage handling is needed, every image under `--input-dir` is eligible.
	- The pool may contain various pathologies, not just healthy eyes: irrelevant for Phase 1, since the model only trains on reconstruction/diffusion loss here, not on disease labels. All rows are written with `target=0` for compatibility with the single-class pretraining path (`split_prefix="pretrain-mogon"`).

Notes
- Filenames follow the pattern `<dataset>-{train,valid,test}.csv`.
- CSV format: two columns `image_name,target`, where `image_name` is relative to the `--data-path`/base path used by the script.
- This branch (`drusen-mogon`) only produces and consumes the splits above, used for Phase-1 pretraining. The private Drusen dataset splits (patient-grouped, augmented, cross-validated) are generated and documented on the `drusen` branch instead — see `dataset/splits/SPLITS.md` there.
