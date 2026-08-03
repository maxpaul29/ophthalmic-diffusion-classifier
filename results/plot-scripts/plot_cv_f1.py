"""
Plot the per-fold F1-score trajectory across training epochs for the Drusen
(Phase-2 finetuning) cross-validation runs, one line per fold on a single graph.

Run:
    python results/plot-scripts/plot_cv_f1.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# List of (epoch, F1) pairs per fold, from the finetuning CV logs
# (log_finetuning_fold_0_cv.txt / log_finetuning_fold1_to_4_cv.log). Fold 0 is
# missing epochs 375/425/450/475 due to a log-capture gap (does not affect the
# reported final test metrics); those points are simply omitted here rather
# than interpolated, and can be added later if recovered.
F1_BY_FOLD = {
    "fold 0": [
    ],
    "fold 1": [
    ],
    "fold 2": [
    ],
    "fold 3": [
    ],
    "fold 4": [
    ],
}

OUTPUT_DIR = "experiments/fundus-unet/plots"
# ─────────────────────────────────────────────────────────────────────────────

FOLD_COLORS = {
    "fold 0": "#4C72B0",
    "fold 1": "#DD8452",
    "fold 2": "#55A868",
    "fold 3": "#C44E52",
    "fold 4": "#8172B2",
}
DEFAULT_COLOR = "#808080"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    active = {k: v for k, v in F1_BY_FOLD.items() if v}
    if not active:
        print("No F1 values entered yet. Fill in F1_BY_FOLD and re-run.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for fold_name, points in active.items():
        epochs = [e for e, _ in points]
        values = [v for _, v in points]
        color = FOLD_COLORS.get(fold_name, DEFAULT_COLOR)
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=4,
                color=color, label=fold_name.capitalize())

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_title("Drusen Cross-Validation — F1 Score per Fold", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cv_f1_per_fold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
