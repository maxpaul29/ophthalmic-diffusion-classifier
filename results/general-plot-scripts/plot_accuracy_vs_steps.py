"""
Plot accuracy against the number of classification steps (EVALUATION_PER_STAGE
/ Monte Carlo majority-voting sample count N) for the single finetuning run,
analogous to plot_cv_accuracy_vs_steps.py but without CV aggregation (a single
accuracy value per N, no mean/std over folds).

Reads results/fundus-unet/finetuned-model/single_run_results/inference_result_n{N}.json
directly (not hand-transcribed) so the N -> accuracy mapping can't be mistyped.

Run:
    python results/general-plot-scripts/plot_accuracy_vs_steps.py
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

RESULTS_DIR = "results/fundus-unet/finetuned-model/single_run_results"
OUTPUT_DIR = "experiments/fundus-unet/plots"


def load_results():
    points = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "inference_result_n*.json")):
        match = re.search(r"inference_result_n(\d+)\.json$", os.path.basename(path))
        if not match:
            continue
        n = int(match.group(1))
        with open(path) as f:
            data = json.load(f)
        data = data.get("results", data)  # some files nest the metrics under "results"
        points.append((n, data["accuracy"]))
    points.sort(key=lambda p: p[0])
    return points


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    points = load_results()
    if not points:
        print(f"No inference_result_n*.json files found in {RESULTS_DIR}.")
        return

    steps = [p[0] for p in points]
    accs = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, accs, marker="o", linewidth=2, markersize=6, color="#4C72B0")

    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Classification steps (N)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_vs_steps_single_run.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    for n, acc in points:
        print(f"  N={n}: accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
