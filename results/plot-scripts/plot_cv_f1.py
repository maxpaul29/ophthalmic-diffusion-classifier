"""
Plot the per-fold F1-score trajectory across training epochs for the Drusen
cross-validation runs, one line per fold on a single graph.

Run:
    python experiments/fundus-unet/plot_cv_f1.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# One value per evaluation epoch (every SAVE_IMAGE_EPOCHS=25, NUM_EPOCHS=400
F1_BY_FOLD = {
    "fold 0": [], # insert values
    "fold 1": [], # insert values
    "fold 2": [], # insert values
    "fold 3": [], # insert values
    "fold 4": [], # insert values
}

FIRST_EVAL_EPOCH = 0        # = SAVE_IMAGE_EPOCHS
EVAL_INTERVAL = 25          # = SAVE_IMAGE_EPOCHS

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


def epoch_axis(n_values):
    return [FIRST_EVAL_EPOCH + i * EVAL_INTERVAL for i in range(n_values)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    active = {k: v for k, v in F1_BY_FOLD.items() if v}
    if not active:
        print("No F1 values entered yet. Fill in F1_BY_FOLD and re-run.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for fold_name, values in active.items():
        epochs = epoch_axis(len(values))
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
