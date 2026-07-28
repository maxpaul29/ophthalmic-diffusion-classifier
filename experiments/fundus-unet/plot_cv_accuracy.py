"""
Plot the per-fold accuracy trajectory across training epochs for the Drusen
cross-validation runs, one line per fold on a single graph (analogous to the
per-fold accuracy plots in Favero et al.).

Folds 1-4 are read from scripts/logs/training_cv_51.log (complete, 400
epochs). Fold 0 is read from scripts/logs/training_fold_0.txt, a separate
rerun log started after the original fold 0 run/archive — also complete.

Run:
    python experiments/fundus-unet/plot_cv_accuracy.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# One value per evaluation epoch (every SAVE_IMAGE_EPOCHS=25, NUM_EPOCHS=400
# -> 17 points at epochs 0,25,...,375,399), per fold.
ACCURACY_BY_FOLD = {
    "fold 0": [0.5455, 0.5455, 0.5682, 0.5682, 0.5455, 0.6364, 0.6818, 0.7727, 0.6818,
               0.7045, 0.8182, 0.7500, 0.8636, 0.7727, 0.7955, 0.8182, 0.8864],
    "fold 1": [0.5682, 0.5682, 0.6591, 0.6818, 0.6364, 0.7273, 0.7045, 0.7500, 0.7500,
               0.8636, 0.7955, 0.7955, 0.8409, 0.7955, 0.8409, 0.7955, 0.8864],
    "fold 2": [0.3864, 0.4773, 0.5000, 0.5000, 0.5682, 0.6591, 0.6818, 0.6591, 0.7045,
               0.7955, 0.7727, 0.7955, 0.8864, 0.8636, 0.9091, 0.9091, 0.8864],
    "fold 3": [0.4545, 0.5227, 0.5455, 0.5909, 0.6591, 0.7273, 0.7045, 0.7273, 0.7045,
               0.7500, 0.7273, 0.7955, 0.7955, 0.8409, 0.8636, 0.8182, 0.8636],
    "fold 4": [0.5870, 0.5435, 0.6522, 0.6304, 0.7391, 0.6957, 0.6522, 0.7391, 0.7826,
               0.8043, 0.7826, 0.8913, 0.8696, 0.8478, 0.8478, 0.8696, 0.8261],
}

EVAL_EPOCHS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 399]

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

    active = {k: v for k, v in ACCURACY_BY_FOLD.items() if v}
    if not active:
        print("No accuracy values entered yet. Fill in ACCURACY_BY_FOLD and re-run.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for fold_name, values in active.items():
        epochs = EVAL_EPOCHS[:len(values)]
        color = FOLD_COLORS.get(fold_name, DEFAULT_COLOR)
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=4,
                color=color, label=fold_name.capitalize())

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Drusen Cross-Validation — Accuracy per Fold", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cv_accuracy_per_fold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
