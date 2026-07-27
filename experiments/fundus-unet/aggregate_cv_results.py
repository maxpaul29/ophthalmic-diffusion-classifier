"""
Aggregate k-fold cross-validation test results into mean +/- std per metric.

Reads inference_result.json from each fold's archived best_checkpoint
(written by experiments/fundus-unet/inference.py), and reports the mean and
standard deviation per metric across all folds — the number that should
actually be reported in the thesis instead of a single fold's point estimate.

Usage:
    python experiments/fundus-unet/aggregate_cv_results.py \
        --cv-root /checkpoints/final-models/drusen-unet/cv \
        --k-folds 5
"""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Aggregate k-fold CV inference results.")
    parser.add_argument("--cv-root", required=True,
                        help="Directory containing fold{i}/best_checkpoint/inference_result.json for each fold.")
    parser.add_argument("--k-folds", type=int, required=True, help="Number of folds.")
    args = parser.parse_args()

    cv_root = Path(args.cv_root)
    per_metric = {}
    per_fold = []

    for i in range(args.k_folds):
        result_path = cv_root / f"fold{i}" / "best_checkpoint" / "inference_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result for fold {i}: {result_path}")
        with open(result_path) as f:
            fold_result = json.load(f)
        per_fold.append(fold_result)
        for metric, value in fold_result.items():
            per_metric.setdefault(metric, []).append(value)

    print(f"Loaded {args.k_folds} fold results from {cv_root}\n")

    print("Per-fold results:")
    for i, fold_result in enumerate(per_fold):
        print(f"  fold {i}: {fold_result}")

    print("\nAggregated (mean +/- std over folds):")
    summary = {}
    for metric, values in per_metric.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[metric] = {"mean": round(mean, 4), "std": round(std, 4)}
        print(f"  {metric:>10}: {mean:.4f} +/- {std:.4f}")

    summary_path = cv_root / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"per_fold": per_fold, "summary": summary}, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
