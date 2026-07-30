"""
Plot the per-fold accuracy trajectory across training epochs for the Drusen
cross-validation runs, one line per fold on a single graph (analogous to the
per-fold accuracy plots in Favero et al.).

Run:
    python experiments/fundus-unet/plot_cv_accuracy.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# One value per evaluation epoch (every SAVE_IMAGE_EPOCHS=25, NUM_EPOCHS=400
ACCURACY_BY_FOLD = {
    "fold 0": [], # insert values
    "fold 1": [], # insert values
    "fold 2": [], # insert values
    "fold 3": [], # insert values
    "fold 4": [], # insert values
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
