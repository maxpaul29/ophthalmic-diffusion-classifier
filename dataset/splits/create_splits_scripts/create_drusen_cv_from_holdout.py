"""
Build k-fold CV splits that use EXACTLY the same pool of images as an existing
fixed hold-out split (drusen-train.csv / drusen-valid.csv / drusen-test.csv),
so that a CV-vs-hold-out comparison isolates only the effect of the
train/valid/test partitioning (and training-set size), without the additional
confound of independently re-sampled healthy images from the full ~4000-image
directory (see create_drusen_cv_splits.py, which draws a fresh random healthy
subsample per fold instead).

What differs from create_drusen_cv_splits.py:

- Drusen (target=1) images: unchanged — read from --drusen-dir exactly as
  before. This was already verified to yield the same 111 original images
  regardless of experiment, so no change is needed there.
- Healthy (target=0) images: instead of scanning --healthy-dir, the candidate
  pool is restricted to the union of healthy image_name entries already
  present in the three hold-out CSVs. Patient-level grouping (patient_key())
  and the fold-partitioning logic are otherwise identical.

Caveat (documented, not hidden): the hold-out split's own valid/test CSVs only
ever contained untouched originals (`_aug00`), never augmented variants — that
is safeguard 2 of create_drusen_cv_splits.py, applied when the hold-out split
was built. This script only reads healthy filenames (never augmented), so this
caveat does not actually apply to the healthy side; it is mentioned here only
for completeness since the analogous Drusen-side caveat does NOT apply either,
because Drusen images are still read fresh from --drusen-dir with full
augmentation available for whichever originals a new fold assigns to train.

Usage:
    python dataset/splits/create_splits_scripts/create_drusen_cv_from_holdout.py \
        --data-path      /data \
        --drusen-dir     /data/clinic/drusen_augmented \
        --holdout-train  dataset/splits/drusen-train.csv \
        --holdout-valid  dataset/splits/drusen-valid.csv \
        --holdout-test   dataset/splits/drusen-test.csv \
        --k-folds 5 \
        --output-dir     dataset/splits/cv_from_holdout
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd

from create_drusen_cv_splits import (
    original_id,
    patient_key,
    is_original_variant,
    rel,
    make_folds,
    list_images,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create k-fold Drusen CV splits restricted to the hold-out split's exact healthy image pool."
    )
    parser.add_argument("--data-path", required=True, help="Base path; CSV image_name is relative to this.")
    parser.add_argument("--drusen-dir", required=True, help="Directory with augmented Drusen images (target=1).")
    parser.add_argument("--holdout-train", required=True, help="Path to the hold-out split's drusen-train.csv.")
    parser.add_argument("--holdout-valid", required=True, help="Path to the hold-out split's drusen-valid.csv.")
    parser.add_argument("--holdout-test", required=True, help="Path to the hold-out split's drusen-test.csv.")
    parser.add_argument("--output-dir", default="dataset/splits/cv_from_holdout", help="Where to write the CSVs.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-balance", action="store_true", help="Keep all healthy images (skip 50:50 balancing).")
    args = parser.parse_args()

    if args.k_folds < 2:
        raise ValueError("--k-folds must be at least 2")

    # ── Drusen: unchanged, read fresh from --drusen-dir (already proven to be
    # the same 111-original pool regardless of experiment) ────────────────────
    drusen_files = list_images(args.drusen_dir)
    if not drusen_files:
        raise FileNotFoundError(f"No Drusen images under {args.drusen_dir}")

    groups = {}
    for f in drusen_files:
        groups.setdefault(original_id(f), []).append(f)
    group_ids = sorted(groups.keys())
    if len(group_ids) < args.k_folds:
        raise ValueError(f"Only {len(group_ids)} original Drusen eyes, cannot make {args.k_folds} folds.")

    patient_to_gids = {}
    for gid in group_ids:
        patient_to_gids.setdefault(patient_key(gid), []).append(gid)
    patient_ids = sorted(patient_to_gids.keys())
    if len(patient_ids) < args.k_folds:
        raise ValueError(f"Only {len(patient_ids)} distinct Drusen patients, cannot make {args.k_folds} folds.")

    patient_folds = make_folds(patient_ids, args.k_folds, args.seed)
    drusen_folds = [[gid for pid in fold for gid in patient_to_gids[pid]] for fold in patient_folds]

    # ── Healthy: restricted to exactly the hold-out split's own pool ───────────
    holdout_df = pd.concat([
        pd.read_csv(args.holdout_train),
        pd.read_csv(args.holdout_valid),
        pd.read_csv(args.holdout_test),
    ], ignore_index=True)
    holdout_healthy_names = sorted(set(holdout_df[holdout_df.target == 0].image_name))
    if not holdout_healthy_names:
        raise ValueError("No healthy (target=0) rows found across the three hold-out CSVs.")
    healthy_files = [Path(args.data_path) / name for name in holdout_healthy_names]
    print(f"Healthy pool restricted to hold-out split: {len(healthy_files)} images "
          f"(vs. the full --healthy-dir, which is NOT read by this script).")

    healthy_patient_to_files = {}
    for f in healthy_files:
        healthy_patient_to_files.setdefault(patient_key(original_id(f)), []).append(f)
    healthy_patient_ids = sorted(healthy_patient_to_files.keys())
    if len(healthy_patient_ids) < args.k_folds:
        raise ValueError(
            f"Only {len(healthy_patient_ids)} distinct healthy patients in the hold-out pool, "
            f"cannot make {args.k_folds} folds. Use a smaller --k-folds or the full "
            f"create_drusen_cv_splits.py instead."
        )
    healthy_patient_folds = make_folds(healthy_patient_ids, args.k_folds, args.seed)
    healthy_folds = [[f for pid in fold for f in healthy_patient_to_files[pid]] for fold in healthy_patient_folds]

    def collect(group_id_list, originals_only):
        files = []
        for gid in group_id_list:
            for f in groups[gid]:
                if originals_only and not is_original_variant(f):
                    continue
                files.append(f)
        return files

    def to_df(drusen, healthy):
        rows = [(rel(f, args.data_path), 1) for f in drusen]
        rows += [(rel(f, args.data_path), 0) for f in healthy]
        df = pd.DataFrame(rows, columns=["image_name", "target"])
        return df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    def balance(healthy, n_drusen, fold_seed):
        if args.no_balance or len(healthy) <= n_drusen:
            return healthy
        return random.Random(fold_seed).sample(healthy, n_drusen)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.k_folds):
        test_gids = drusen_folds[i]
        valid_gids = drusen_folds[(i + 1) % args.k_folds]
        train_gids = [gid for j, fold in enumerate(drusen_folds) if j not in (i, (i + 1) % args.k_folds) for gid in fold]

        d_test = collect(test_gids, originals_only=True)
        d_valid = collect(valid_gids, originals_only=True)
        d_train = collect(train_gids, originals_only=False)

        h_test = healthy_folds[i]
        h_valid = healthy_folds[(i + 1) % args.k_folds]
        h_train = [f for j, fold in enumerate(healthy_folds) if j not in (i, (i + 1) % args.k_folds) for f in fold]

        h_train = balance(h_train, len(d_train), args.seed + i * 3 + 0)
        h_valid = balance(h_valid, len(d_valid), args.seed + i * 3 + 1)
        h_test = balance(h_test, len(d_test), args.seed + i * 3 + 2)

        splits = {
            f"drusen-fold{i}-train": to_df(d_train, h_train),
            f"drusen-fold{i}-valid": to_df(d_valid, h_valid),
            f"drusen-fold{i}-test": to_df(d_test, h_test),
        }
        for name, df in splits.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            n_pos = int((df["target"] == 1).sum())
            n_neg = int((df["target"] == 0).sum())
            print(f"{name}: {len(df)} rows  (drusen={n_pos}, healthy={n_neg})  -> {path}")

        print(f"  fold {i}: test eyes={len(test_gids)}, valid eyes={len(valid_gids)}, train eyes={len(train_gids)}\n")

    print(f"Drusen originals: {len(group_ids)} across {args.k_folds} folds "
          f"(~{len(group_ids) // args.k_folds} eyes/fold)")
    print(f"Healthy originals used (restricted to hold-out pool): {len(healthy_files)}")


if __name__ == "__main__":
    main()
