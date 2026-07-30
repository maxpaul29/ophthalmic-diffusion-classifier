"""
Build k-fold cross-validation train/valid/test CSV splits for the Optic Disc
Drusen (ODD) classifier.

With only ~111 real Drusen cases, a single fixed train/valid/test split makes
the reported test result highly dependent on which handful of images happened
to land in the test set. k-fold cross-validation instead partitions the
original Drusen eyes into k folds and produces k independent splits, so every
real case is tested exactly once across the whole procedure — the final result
is the mean +/- std over the k folds' test metrics, not a single point value.

Three safeguards, applied per fold:

1. No augmentation leakage across splits.
   All augmented variants of one original Drusen image share a group id (the
   filename with the trailing `_augNN` removed). The k-way partition happens at
   the group level, so variants of the same eye never land in two different
   folds, let alone two different splits within a fold.

2. Honest evaluation.
   Each fold's validation and test splits keep ONLY the untouched originals
   (`_aug00`). Augmented variants are used for training only.

3. No patient leakage across splits.
   Grouping by individual original image alone is not enough: the same
   patient frequently contributes several original images (left/right eye,
   repeat visits, or even two crops of the same photo, e.g. `3906L1_1` /
   `3906L1_2`) that would otherwise be treated as independent samples and
   scattered across different folds — letting the model see, e.g., a
   patient's left eye during training and be evaluated on that same
   patient's right eye at test time. `patient_key()` derives a patient
   identifier from the filename (leading numeric ID, or the study/sequence
   number for `IM-...` scans) and the k-way partition happens at the patient
   level, so every original image of one patient always ends up in the same
   split. This is applied to both Drusen and healthy images. Filenames that
   don't match a known naming pattern are conservatively treated as their own
   single-image "patient" (never merged with anything else).

For fold i (i = 0..k-1):
  - test  = original groups in fold i (originals only)
  - valid = original groups in fold (i+1) % k (originals only)
  - train = all remaining folds' groups (all augmented variants)

Healthy images are balanced 1:1 per split the same way, independently
per fold (a fresh random subsample each time, seeded for reproducibility).

Output CSVs use the same `image_name,target` format as create_fundus_split.py,
named `drusen-fold{i}-{train,valid,test}.csv` so dataset/fundus.py can load
them directly via split_prefix=f"drusen-fold{i}" (no code changes needed there
— split_prefix is already a free-form string).

Usage:
    python dataset/splits/create_drusen_cv_splits.py \
        --data-path   /data \
        --drusen-dir  /data/clinic/drusen_augmented \
        --healthy-dir /data/clinic/healthy \
        --k-folds 5 \
        --output-dir  dataset/splits
"""

import argparse
import os
import random
import re
from pathlib import Path

import pandas as pd

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
AUG_SUFFIX = re.compile(r"_aug\d+$")

# Naming patterns observed in the clinical Drusen/healthy export, used to
# derive a patient identifier from an original (aug-suffix-stripped) filename.
# Order matters: more specific patterns are tried first.
_PATIENT_PATTERNS = [
    re.compile(r"^(\d+)_[LR]_\d{4}-\d{2}-\d{2}"),  # e.g. 51087925_L_2025-03-18
    re.compile(r"^(\d+)[LR]\d+"),                   # e.g. 12103L1, 3906L1_1, 10083R6
    re.compile(r"^IM-(\d+)-(\d+)-\d+"),              # e.g. IM-0001-10000-0002Diet
]


def list_images(directory):
    return [
        p for p in sorted(Path(directory).rglob("*"))
        if p.suffix.lower() in IMG_EXTENSIONS
    ]


def original_id(path):
    """Group id for an augmented file: filename with the `_augNN` suffix removed."""
    return AUG_SUFFIX.sub("", path.stem)


def patient_key(original_id_str):
    """
    Derive a patient identifier from an original (aug-stripped) filename, so
    all original images belonging to one patient can be grouped together
    before the k-way fold partition (see safeguard 3 above).

    Falls back to the input string itself (i.e. a single-image "patient") for
    any name that doesn't match a known pattern, so unrecognized names are
    never incorrectly merged with one another.
    """
    for pattern in _PATIENT_PATTERNS:
        match = pattern.match(original_id_str)
        if match:
            return "P:" + "-".join(match.groups())
    return "P:" + original_id_str


def is_original_variant(path):
    """True for the untouched original (variant `_aug00`) or a non-augmented file."""
    if AUG_SUFFIX.search(path.stem):
        return path.stem.endswith("_aug00")
    return True  # file does not follow the augmentation naming -> treat as original


def rel(path, data_path):
    # Always return POSIX-style relative paths (forward slashes), even on Windows.
    return os.path.relpath(str(path), data_path).replace('\\', '/')


def make_folds(items, k, seed):
    """Shuffle `items` deterministically and split into k roughly-equal folds."""
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    folds = [[] for _ in range(k)]
    for idx, item in enumerate(items):
        folds[idx % k].append(item)
    return folds


def main():
    parser = argparse.ArgumentParser(description="Create k-fold Drusen CV splits (grouped by original eye).")
    parser.add_argument("--data-path", required=True, help="Base path; CSV image_name is relative to this.")
    parser.add_argument("--drusen-dir", required=True, help="Directory with augmented Drusen images (target=1).")
    parser.add_argument("--healthy-dir", required=True, help="Directory with healthy images (target=0).")
    parser.add_argument("--output-dir", default="dataset/splits", help="Where to write the CSVs.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-balance", action="store_true", help="Keep all healthy images (skip 50:50 balancing).")
    parser.add_argument("--n-healthy", type=int, default=None,
                        help="Draw only this many healthy images into the pool before splitting. Default: all.")
    args = parser.parse_args()

    if args.k_folds < 2:
        raise ValueError("--k-folds must be at least 2")

    rng = random.Random(args.seed)

    drusen_files = list_images(args.drusen_dir)
    healthy_files = list_images(args.healthy_dir)
    if not drusen_files:
        raise FileNotFoundError(f"No Drusen images under {args.drusen_dir}")
    if not healthy_files:
        raise FileNotFoundError(f"No healthy images under {args.healthy_dir}")

    if args.n_healthy is not None and args.n_healthy < len(healthy_files):
        healthy_files = rng.sample(healthy_files, args.n_healthy)

    # ── Group Drusen files by original eye to prevent augmentation leakage ──────
    groups = {}
    for f in drusen_files:
        groups.setdefault(original_id(f), []).append(f)
    group_ids = sorted(groups.keys())
    if len(group_ids) < args.k_folds:
        raise ValueError(f"Only {len(group_ids)} original Drusen eyes, cannot make {args.k_folds} folds.")

    # ── Group original ids further by patient to prevent patient leakage ───────
    # (safeguard 3): the k-way partition is done over patients, then expanded
    # back to their constituent original ids, so all original images (and,
    # after collect(), all their augmented variants) of one patient always end
    # up in the same fold.
    patient_to_gids = {}
    for gid in group_ids:
        patient_to_gids.setdefault(patient_key(gid), []).append(gid)
    patient_ids = sorted(patient_to_gids.keys())
    if len(patient_ids) < args.k_folds:
        raise ValueError(f"Only {len(patient_ids)} distinct Drusen patients, cannot make {args.k_folds} folds.")

    patient_folds = make_folds(patient_ids, args.k_folds, args.seed)
    drusen_folds = [[gid for pid in fold for gid in patient_to_gids[pid]] for fold in patient_folds]

    healthy_patient_to_files = {}
    for f in healthy_files:
        healthy_patient_to_files.setdefault(patient_key(original_id(f)), []).append(f)
    healthy_patient_ids = sorted(healthy_patient_to_files.keys())
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


if __name__ == "__main__":
    main()
