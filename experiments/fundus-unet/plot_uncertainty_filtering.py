"""
Plot accuracy as a function of the fraction of most-uncertain test samples
removed, reproducing the uncertainty-filtering plot of Favero et al. (Fig.
showing "removed data" (0-100%) vs. accuracy on the remaining data).

Uncertainty per sample is the Bernoulli variance p*(1-p) of the winning-class
vote share over the N (eps, lambda) evaluations of classify() (see
diffusion/diffusion_classifier.py). Samples are removed in decreasing order of
uncertainty (most uncertain first); accuracy is recomputed on what remains
after each removal step.

Input: experiments/fundus-unet/uncertainty_predictions.json, produced by
running inference.py on the single 80/10/10 split (not cross-validation) with
UNCERTAINTY_ESTIMATION=true set for scripts/unet/fundus-unet.sh, e.g.:

    UNCERTAINTY_ESTIMATION=true FUNCTION=inference MODEL=unet bash scripts/run.sh

which writes uncertainty_predictions.json next to the checkpoint used
(config.checkpoint_folder, i.e. the finetune-epoch-300 checkpoint for the
plain single-split run). Copy/symlink that file to the path below before
running this script.

Run:
    python experiments/fundus-unet/plot_uncertainty_filtering.py
"""

import json
import os

import matplotlib.pyplot as plt

PREDICTIONS_PATH = "experiments/fundus-unet/uncertainty_predictions.json"
OUTPUT_DIR = "experiments/fundus-unet/plots"


def compute_filtering_curve(predictions):
    # Sort most uncertain first, so index i means "the i most uncertain
    # samples have been removed".
    ordered = sorted(predictions, key=lambda p: p["uncertainty"], reverse=True)
    n = len(ordered)

    removed_frac = []
    accuracy = []
    for i in range(n):  # i = number removed, from 0 up to n-1 remaining sample
        remaining = ordered[i:]
        if not remaining:
            break
        acc = sum(p["correct"] for p in remaining) / len(remaining)
        removed_frac.append(100.0 * i / n)
        accuracy.append(acc)

    return removed_frac, accuracy


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"Not found: {PREDICTIONS_PATH}")
        print("Run inference.py with UNCERTAINTY_ESTIMATION=true on the single "
              "80/10/10 split first, then copy its uncertainty_predictions.json here.")
        return

    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    removed_frac, accuracy = compute_filtering_curve(predictions)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(removed_frac, accuracy, marker="o", linewidth=2, markersize=4, color="#4C72B0")

    ax.set_xlabel("Removed data (%, most uncertain first)", fontsize=12)
    ax.set_ylabel("Accuracy on remaining data", fontsize=12)
    ax.set_title("Drusen Classification — Accuracy vs. Uncertainty-Based Filtering",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 102)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "uncertainty_filtering.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"n = {len(predictions)} test samples")


if __name__ == "__main__":
    main()
