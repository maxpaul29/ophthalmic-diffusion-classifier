"""
Build train/valid/test CSV splits for the Optic Disc Drusen (ODD) classifier.

Combines two sources:
  - positive class (target=1): augmented Drusen crops   (from augment_drusen.py)
  - negative class (target=0): clinical non-drusen crops

Three safeguards make the resulting splits scientifically sound:

1. No augmentation leakage across splits.
   All augmented variants of one original Drusen image share a group id
   (the filename with the trailing `_augNN` removed). Splitting happens at the
   group level, so variants of the same eye never land in two different splits.

2. Honest evaluation.
   The validation and test splits keep ONLY the untouched originals
   (`_aug00`, written as variant 0 by augment_drusen.py). Augmented variants are
   used for training only — the model is never scored on synthetic images.

3. No patient leakage across splits.
   Grouping by individual original image alone is not enough: the same
   patient frequently contributes several original images (left/right eye,
   repeat visits, or even two crops of the same photo) that would otherwise
   be treated as independent samples and could be scattered across train and
   test — letting the model see, e.g., a patient's left eye during training
   and be evaluated on that same patient's right eye. `patient_key()` (see
   dataset/splits/create_splits_scripts/create_drusen_cv_splits.py, same
   logic) derives a patient identifier from the filename, and the
   train/valid/test split happens at the patient level, so every original
   image of one patient always ends up in the same split. Applied to both
   Drusen and non-drusen images.

Class balance (50:50) is enforced per split by sub-sampling the majority
(non-drusen) class, matching the paper's balanced-training setup.

Output CSVs use the same `image_name,target` format as create_fundus_split.py,
with paths relative to --data-path so dataset/fundus.py can load them directly.

Usage:
    python dataset/splits/create_splits_scripts/create_drusen_aug_split.py \
        --data-path      /data \
        --drusen-dir     /data/clinic/drusen_augmented \
        --non-drusen-dir /data/clinic/non_drusen \
        --output-dir     dataset/splits
"""

import argparse
import os
import random
import re
from pathlib import Path

import pandas as pd

from create_drusen_cv_splits import patient_key

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
AUG_SUFFIX = re.compile(r"_aug\d+$")


def list_images(directory):
    return [
        p for p in sorted(Path(directory).rglob("*"))
        if p.suffix.lower() in IMG_EXTENSIONS
    ]


def original_id(path):
    """Group id for an augmented file: filename with the `_augNN` suffix removed."""
    return AUG_SUFFIX.sub("", path.stem)


def is_original_variant(path):
    """True for the untouched original (variant `_aug00`) or a non-augmented file."""
    if AUG_SUFFIX.search(path.stem):
        return path.stem.endswith("_aug00")
    return True  # file does not follow the augmentation naming -> treat as original


def rel(path, data_path):
    # Always return POSIX-style relative paths (forward slashes), even on Windows.
    return os.path.relpath(str(path), data_path).replace('\\', '/')


def split_train_valid_test(items, valid_frac, test_frac, seed):
    """
    Shuffle `items` deterministically and cut it into train/valid/test.

    Replaces sklearn.model_selection.train_test_split (not installed) with a
    stdlib-only shuffle + slice, seeded for reproducibility.
    """
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n_valid = round(len(items) * valid_frac)
    n_test = round(len(items) * test_frac)

    test = items[:n_test]
    valid = items[n_test:n_test + n_valid]
    train = items[n_test + n_valid:]
    return train, valid, test


def main():
    parser = argparse.ArgumentParser(description="Create balanced Drusen train/valid/test splits.")
    parser.add_argument("--data-path", required=True, help="Base path; CSV image_name is relative to this.")
    parser.add_argument("--drusen-dir", required=True, help="Directory with augmented Drusen images (target=1).")
    parser.add_argument("--non-drusen-dir", required=True, help="Directory with non-drusen images (target=0).")
    parser.add_argument("--output-dir", default="dataset/splits", help="Where to write the CSVs.")
    parser.add_argument("--valid-frac", type=float, default=0.1, help="Fraction of originals for validation.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of originals for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-balance", action="store_true", help="Keep all non-drusen images (skip 50:50 balancing).")
    parser.add_argument("--n-drusen", type=int, default=None,
                        help="Use only this many original Drusen eyes (groups); all their augmented "
                             "variants are kept. Default: all.")
    parser.add_argument("--n-non-drusen", type=int, default=None,
                        help="Draw only this many non-drusen images into the pool before splitting. Default: all.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    drusen_files = list_images(args.drusen_dir)
    non_drusen_files = list_images(args.non_drusen_dir)
    if not drusen_files:
        raise FileNotFoundError(f"No Drusen images under {args.drusen_dir}")
    if not non_drusen_files:
        raise FileNotFoundError(f"No non-drusen images under {args.non_drusen_dir}")

    # ── Split Drusen at the group level to prevent augmentation leakage ─────────
    groups = {}
    for f in drusen_files:
        groups.setdefault(original_id(f), []).append(f)
    group_ids = sorted(groups.keys())

    # Optionally use only a subset of the original eyes (keeps whole groups intact).
    if args.n_drusen is not None and args.n_drusen < len(group_ids):
        group_ids = sorted(rng.sample(group_ids, args.n_drusen))

    # Optionally cap the non-drusen pool before splitting.
    if args.n_non_drusen is not None and args.n_non_drusen < len(non_drusen_files):
        non_drusen_files = rng.sample(non_drusen_files, args.n_non_drusen)

    # ── Group original ids further by patient to prevent patient leakage ───────
    # (safeguard 3): the train/valid/test split happens over patients, then
    # expanded back to their constituent original ids, so all original images
    # of one patient always end up in the same split.
    patient_to_gids = {}
    for gid in group_ids:
        patient_to_gids.setdefault(patient_key(gid), []).append(gid)
    patient_ids = sorted(patient_to_gids.keys())

    train_pids, valid_pids, test_pids = split_train_valid_test(
        patient_ids, args.valid_frac, args.test_frac, args.seed
    )
    train_ids = [gid for pid in train_pids for gid in patient_to_gids[pid]]
    valid_ids = [gid for pid in valid_pids for gid in patient_to_gids[pid]]
    test_ids = [gid for pid in test_pids for gid in patient_to_gids[pid]]

    def drusen_split(ids, originals_only):
        files = []
        for gid in ids:
            for f in groups[gid]:
                if originals_only and not is_original_variant(f):
                    continue
                files.append(f)
        return files

    drusen_train = drusen_split(train_ids, originals_only=False)  # all augmented variants
    drusen_valid = drusen_split(valid_ids, originals_only=True)   # originals only
    drusen_test = drusen_split(test_ids, originals_only=True)     # originals only

    # ── Split non-drusen images by patient, same as Drusen above ────────────────
    non_drusen_patient_to_files = {}
    for f in non_drusen_files:
        non_drusen_patient_to_files.setdefault(patient_key(original_id(f)), []).append(f)
    non_drusen_patient_ids = sorted(non_drusen_patient_to_files.keys())
    h_train_pids, h_valid_pids, h_test_pids = split_train_valid_test(
        non_drusen_patient_ids, args.valid_frac, args.test_frac, args.seed
    )
    h_train = [f for pid in h_train_pids for f in non_drusen_patient_to_files[pid]]
    h_valid = [f for pid in h_valid_pids for f in non_drusen_patient_to_files[pid]]
    h_test = [f for pid in h_test_pids for f in non_drusen_patient_to_files[pid]]

    # ── Balance each split 50:50 by sub-sampling non-drusen to the Drusen count ─
    def balance(non_drusen, n_drusen):
        if args.no_balance or len(non_drusen) <= n_drusen:
            return non_drusen
        return rng.sample(non_drusen, n_drusen)

    h_train = balance(h_train, len(drusen_train))
    h_valid = balance(h_valid, len(drusen_valid))
    h_test = balance(h_test, len(drusen_test))

    # ── Assemble and write CSVs ─────────────────────────────────────────────────
    def to_df(drusen, non_drusen):
        rows = [(rel(f, args.data_path), 1) for f in drusen]
        rows += [(rel(f, args.data_path), 0) for f in non_drusen]
        df = pd.DataFrame(rows, columns=["image_name", "target"])
        return df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "drusen-train": to_df(drusen_train, h_train),
        "drusen-valid": to_df(drusen_valid, h_valid),
        "drusen-test": to_df(drusen_test, h_test),
    }
    for name, df in splits.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        n_pos = int((df["target"] == 1).sum())
        n_neg = int((df["target"] == 0).sum())
        print(f"{name}: {len(df)} rows  (drusen={n_pos}, non_drusen={n_neg})  -> {path}")

    print(f"\nDrusen originals: {len(group_ids)} "
          f"(train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)})")


if __name__ == "__main__":
    main()
