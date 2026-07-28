"""
Plot mean accuracy (+/- std over folds) against the number of classification
steps (EVALUATION_PER_STAGE / Monte Carlo majority-voting sample count N),
using the aggregated CV results in experiments/fundus-unet/cv-results/.

Each cv_summary_n{N}.json is read directly (not hand-transcribed) so the
N -> accuracy mapping can't be mistyped.

Run:
    python experiments/fundus-unet/plot_cv_accuracy_vs_steps.py
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

RESULTS_DIR = "experiments/fundus-unet/cv-results"
OUTPUT_DIR = "experiments/fundus-unet/plots"


def load_results():
    points = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "cv_summary_n*.json")):
        match = re.search(r"cv_summary_n(\d+)\.json$", os.path.basename(path))
        if not match:
            continue
        n = int(match.group(1))
        with open(path) as f:
            data = json.load(f)
        acc = data["summary"]["accuracy"]
        points.append((n, acc["mean"], acc["std"]))
    points.sort(key=lambda p: p[0])
    return points


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    points = load_results()
    if not points:
        print(f"No cv_summary_n*.json files found in {RESULTS_DIR}.")
        return

    steps = [p[0] for p in points]
    means = [p[1] for p in points]
    stds = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(steps, means, yerr=stds, marker="o", linewidth=2, markersize=6,
                capsize=4, color="#4C72B0", ecolor="#4C72B0", elinewidth=1.2)

    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Classification steps (N)", fontsize=12)
    ax.set_ylabel("Accuracy (mean ± std over folds)", fontsize=12)
    ax.set_title("Drusen Cross-Validation — Accuracy vs. Classification Steps",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cv_accuracy_vs_steps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    for n, mean, std in points:
        print(f"  N={n}: accuracy={mean:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
