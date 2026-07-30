"""
Build the Phase-1 pretraining CSV splits (healthy-only, class 0) for the
scratch → pretrain → finetune Drusen workflow.

Phase 1 trains the generative diffusion model from scratch on healthy optic-disc
crops only (all label=0), so it learns to reconstruct normal fundus anatomy.
Phase 2 then finetunes on a balanced Drusen-vs-healthy set.

To keep the two phases COMPLETELY disjoint — no healthy image is shared between
pretraining and finetuning, not even in the training portion — this script
excludes every healthy image that already appears in the Phase-2 Drusen split
CSVs (train, valid AND test). That full separation is trivial to argue in the
thesis: the pretraining set and the finetuning/evaluation set share no images at
all, so there is no possibility of leakage between phases.

Run this AFTER create_drusen_split.py (or create_drusen_aug_split.py), using the
same --data-path so relative image paths match for the exclusion.

Output: pretrain-train.csv, pretrain-valid.csv, pretrain-test.csv
(all healthy, target=0), in the same `image_name,target` format as the other
splits. Load them via dataset/fundus.py with split_prefix="pretrain".

Usage:
    python dataset/splits/create_splits_scripts/create_pretrain_split.py \
        --data-path   /data \
        --healthy-dir /data/clinic/healthy \
        --exclude-splits dataset/splits/drusen-train.csv \
                         dataset/splits/drusen-valid.csv \
                         dataset/splits/drusen-test.csv \
        --output-dir  dataset/splits
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
    parser = argparse.ArgumentParser(description="Create healthy-only pretraining splits (Phase 1).")
    parser.add_argument("--data-path", required=True,
                        help="Base path; must match the value used in create_drusen_split.py.")
    parser.add_argument("--healthy-dir", required=True, help="Directory with healthy images (target=0).")
    parser.add_argument("--exclude-splits", nargs="+", required=True,
                        help="Drusen split CSVs whose healthy images must be excluded from pretraining.")
    parser.add_argument("--output-dir", default="dataset/splits", help="Where to write the CSVs.")
    parser.add_argument("--valid-frac", type=float, default=0.05, help="Fraction for validation.")
    parser.add_argument("--test-frac", type=float, default=0.05, help="Fraction for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-healthy", type=int, default=None,
                        help="Use only this many healthy images (after exclusion). Default: all.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    healthy_files = list_images(args.healthy_dir)
    if not healthy_files:
        raise FileNotFoundError(f"No healthy images under {args.healthy_dir}")

    # Collect the image paths already used by the Phase-2 Drusen splits
    excluded = set()
    for csv_path in args.exclude_splits:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"exclude-split CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        excluded.update(df["image_name"].astype(str).tolist())

    # Keep only healthy images NOT used anywhere in Phase 2
    kept = [f for f in healthy_files if rel(f, args.data_path) not in excluded]
    n_removed = len(healthy_files) - len(kept)
    if not kept:
        raise RuntimeError("All healthy images were excluded — check --data-path matches the Drusen split.")

    if args.n_healthy is not None and args.n_healthy < len(kept):
        kept = rng.sample(kept, args.n_healthy)

    # Split into train/valid/test (all label=0)
    train, valid, test = split_train_valid_test(kept, args.valid_frac, args.test_frac, args.seed)

    def to_df(files):
        rows = [(rel(f, args.data_path), 0) for f in files]
        return pd.DataFrame(rows, columns=["image_name", "target"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "pretrain-train": to_df(train),
        "pretrain-valid": to_df(valid),
        "pretrain-test": to_df(test),
    }
    for name, df in splits.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} healthy images  -> {path}")

    print(f"\nHealthy total: {len(healthy_files)}, "
          f"excluded (used in Phase 2): {n_removed}, "
          f"used for pretraining: {len(kept)}")


if __name__ == "__main__":
    main()
