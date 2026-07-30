"""
Build train/valid/test CSV splits for the Optic Disc Drusen (ODD) classifier
WITHOUT data augmentation.

Combines two sources:
  - positive class (target=1): Drusen crops   (raw clinical images, no augmentation)
  - negative class (target=0): clinical healthy crops

Every image is treated as an independent sample — there are no augmented
variants, so no group-level splitting or originals-only filtering is needed
(that logic lives in create_drusen_aug_split.py). Each source pool is split
independently into train/valid/test by fraction.

Class balance (50:50) is enforced per split by sub-sampling the majority
(healthy) class, matching the paper's balanced-training setup. Use --no-balance
to keep the natural class ratio, or --n-healthy / --n-drusen to cap how many
images are drawn from each folder before splitting.

Output CSVs use the same `image_name,target` format as create_fundus_split.py,
with paths relative to --data-path so dataset/fundus.py can load them directly.

Usage:
    python dataset/splits/create_splits_scripts/create_drusen_split.py \
        --data-path   /data \
        --drusen-dir  /data/clinic/drusen \
        --healthy-dir /data/clinic/healthy \
        --output-dir  dataset/splits

    # Drusen Split for Finetuning (Windows):
    python .\dataset\splits\create_splits_scripts\create_drusen_split.py --data-path "C:\Users\Max Paul\Documents\OpticDiscDrusen\AlleBilderQuality" --drusen-dir "C:\Users\Max Paul\Documents\OpticDiscDrusen\AlleBilderQuality\Drusen\usable" --healthy-dir "C:\Users\Max Paul\Documents\OpticDiscDrusen\AlleBilderQuality\keineDrusen\usable" --output-dir "dataset/splits"

    # With custom parameters:
    python dataset/splits/create_splits_scripts/create_drusen_split.py \
        --data-path /data \
        --drusen-dir /data/clinic/drusen \
        --healthy-dir /data/clinic/healthy \
        --output-dir dataset/splits \
        --valid-frac 0.15 \
        --test-frac 0.15 \
        --seed 42 \
        --n-drusen 500 \
        --n-healthy 1000

    # Without class balancing (keep natural ratio):
    python dataset/splits/create_splits_scripts/create_drusen_split.py \
        --data-path /data \
        --drusen-dir /data/clinic/drusen \
        --healthy-dir /data/clinic/healthy \
        --no-balance
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory):
    return [
        p for p in sorted(Path(directory).rglob("*"))
        if p.suffix.lower() in IMG_EXTENSIONS
    ]


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
    parser = argparse.ArgumentParser(description="Create balanced Drusen splits (no augmentation).")
    parser.add_argument("--data-path", required=True, help="Base path; CSV image_name is relative to this.")
    parser.add_argument("--drusen-dir", required=True, help="Directory with Drusen images (target=1).")
    parser.add_argument("--healthy-dir", required=True, help="Directory with healthy images (target=0).")
    parser.add_argument("--output-dir", default="dataset/splits", help="Where to write the CSVs.")
    parser.add_argument("--valid-frac", type=float, default=0.1, help="Fraction for validation.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-balance", action="store_true", help="Keep all healthy images (skip 50:50 balancing).")
    parser.add_argument("--n-drusen", type=int, default=None,
                        help="Draw only this many Drusen images before splitting. Default: all.")
    parser.add_argument("--n-healthy", type=int, default=None,
                        help="Draw only this many healthy images before splitting. Default: all.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    drusen_files = list_images(args.drusen_dir)
    healthy_files = list_images(args.healthy_dir)
    if not drusen_files:
        raise FileNotFoundError(f"No Drusen images under {args.drusen_dir}")
    if not healthy_files:
        raise FileNotFoundError(f"No healthy images under {args.healthy_dir}")

    # Optionally cap each pool before splitting.
    if args.n_drusen is not None and args.n_drusen < len(drusen_files):
        drusen_files = rng.sample(drusen_files, args.n_drusen)
    if args.n_healthy is not None and args.n_healthy < len(healthy_files):
        healthy_files = rng.sample(healthy_files, args.n_healthy)

    def split3(files):
        return split_train_valid_test(files, args.valid_frac, args.test_frac, args.seed)

    d_train, d_valid, d_test = split3(drusen_files)
    h_train, h_valid, h_test = split3(healthy_files)

    # ── Balance each split 50:50 by sub-sampling healthy to the Drusen count ────
    def balance(healthy, n_drusen):
        if args.no_balance or len(healthy) <= n_drusen:
            return healthy
        return rng.sample(healthy, n_drusen)

    h_train = balance(h_train, len(d_train))
    h_valid = balance(h_valid, len(d_valid))
    h_test = balance(h_test, len(d_test))

    # ── Assemble and write CSVs ─────────────────────────────────────────────────
    def to_df(drusen, healthy):
        rows = [(rel(f, args.data_path), 1) for f in drusen]
        rows += [(rel(f, args.data_path), 0) for f in healthy]
        df = pd.DataFrame(rows, columns=["image_name", "target"])
        return df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "drusen-no-aug-train": to_df(d_train, h_train),
        "drusen-no-aug-valid": to_df(d_valid, h_valid),
        "drusen-no-aug-test": to_df(d_test, h_test),
    }
    for name, df in splits.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        n_pos = int((df["target"] == 1).sum())
        n_neg = int((df["target"] == 0).sum())
        print(f"{name}: {len(df)} rows  (drusen={n_pos}, healthy={n_neg})  -> {path}")

    print(f"\nDrusen total: {len(drusen_files)}, healthy total: {len(healthy_files)}")


if __name__ == "__main__":
    main()
