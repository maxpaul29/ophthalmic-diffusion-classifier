"""
Plot the per-fold F1-score trajectory across training epochs for the Drusen
cross-validation runs, one line per fold on a single graph.

Folds 1-4 are read from scripts/logs/training_cv_51.log (complete, 400
epochs). Fold 0 is read from scripts/logs/training_fold_0.txt, a separate
rerun log started after the original fold 0 run/archive — also complete.

Run:
    python experiments/fundus-unet/plot_cv_f1.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# One value per evaluation epoch (every SAVE_IMAGE_EPOCHS=25, NUM_EPOCHS=400
# -> 17 points at epochs 0,25,...,400), per fold. From training_cv_51.log.
F1_BY_FOLD = {
    "fold 0": [0.5833, 0.5455, 0.5957, 0.5366, 0.5238, 0.6522, 0.6667, 0.7826, 0.6818,
               0.7234, 0.8261, 0.7442, 0.8696, 0.7826, 0.8085, 0.8333, 0.8889],
    "fold 1": [0.5581, 0.5581, 0.6512, 0.6818, 0.6190, 0.6667, 0.7111, 0.7317, 0.7442,
               0.8696, 0.8085, 0.7907, 0.8511, 0.8163, 0.8627, 0.8085, 0.8936],
    "fold 2": [0.4000, 0.4889, 0.5417, 0.5600, 0.5366, 0.6341, 0.6667, 0.6512, 0.6977,
               0.7907, 0.7727, 0.7805, 0.8889, 0.8696, 0.9130, 0.9167, 0.8936],
    "fold 3": [0.4783, 0.4878, 0.5652, 0.6250, 0.6154, 0.7143, 0.6977, 0.7273, 0.7234,
               0.7843, 0.7500, 0.8085, 0.8163, 0.8511, 0.8696, 0.8462, 0.8750],
    "fold 4": [0.5581, 0.4878, 0.6667, 0.6222, 0.7273, 0.6500, 0.6522, 0.7391, 0.8000,
               0.8085, 0.7826, 0.8936, 0.8800, 0.8571, 0.8679, 0.8800, 0.8400],
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
    ax.set_ylim(0.0, 1.0)
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
