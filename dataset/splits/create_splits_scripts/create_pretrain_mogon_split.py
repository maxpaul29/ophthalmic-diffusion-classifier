"""
Build Phase-1 pretraining CSV splits from a large public fundus image pool
(e.g. ~400k images combining OIA-ODIR/REFUGE/EyePACS-style sources), intended
to run on Mogon rather than the clinic PC.

This pool is a public dataset, entirely separate from the private clinic
Drusen data used in Phase 2 (fine-tuning, on the `drusen` branch) — there is
no exclusion step needed here, every image in --input-dir is eligible.

The images may show various pathologies (glaucoma, diabetic retinopathy, etc.),
not just healthy eyes. That is fine for Phase 1: the model is trained purely on
the reconstruction/diffusion loss (how well can it regenerate this general
distribution of fundus photographs), not on the specific disease label — Phase 2
re-establishes the correct non-drusen-vs-Drusen class semantics on the small,
properly labelled clinic dataset afterwards. All rows are written with
target=0 for pipeline consistency with fundus.py/train.py's single-class
pretraining path (see split_prefix="pretrain-mogon").

Usage:
    python dataset/splits/create_splits_scripts/create_pretrain_mogon_split.py \
        --data-path  /path/to/mogon/dataset/root \
        --input-dir  /path/to/mogon/dataset/root/all_fundus_images \
        --output-dir dataset/splits
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
    return os.path.relpath(str(path), data_path)


def split_train_valid_test(items, valid_frac, test_frac, seed):
    """
    Shuffle `items` deterministically and cut it into train/valid/test.

    Stdlib-only shuffle + slice (no sklearn dependency), seeded for
    reproducibility — same approach as the other split scripts.
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
    parser = argparse.ArgumentParser(description="Create Phase-1 pretrain splits from a large public fundus pool (Mogon).")
    parser.add_argument("--data-path", required=True, help="Base path; CSV image_name is relative to this.")
    parser.add_argument("--input-dir", required=True, help="Directory (searched recursively) with the fundus images.")
    parser.add_argument("--output-dir", default="dataset/splits", help="Where to write the CSVs.")
    parser.add_argument("--valid-frac", type=float, default=0.02, help="Fraction for validation.")
    parser.add_argument("--test-frac", type=float, default=0.02, help="Fraction for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-images", type=int, default=None,
                        help="Use only this many images (random subset). Default: all.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    files = list_images(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No images found under {args.input_dir}")

    if args.n_images is not None and args.n_images < len(files):
        files = rng.sample(files, args.n_images)

    train, valid, test = split_train_valid_test(files, args.valid_frac, args.test_frac, args.seed)

    def to_df(items):
        rows = [(rel(f, args.data_path), 0) for f in items]
        return pd.DataFrame(rows, columns=["image_name", "target"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "pretrain-mogon-train": to_df(train),
        "pretrain-mogon-valid": to_df(valid),
        "pretrain-mogon-test": to_df(test),
    }
    for name, df in splits.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} images  -> {path}")

    print(f"\nTotal images found: {len(list_images(args.input_dir))}, used: {len(files)} "
          f"(train={len(train)}, valid={len(valid)}, test={len(test)})")


if __name__ == "__main__":
    main()
