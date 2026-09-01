"""
Plot the per-fold accuracy trajectory across training epochs for the Drusen
(Phase-2 finetuning) cross-validation runs, one line per fold on a single
graph (analogous to the per-fold accuracy plots in Favero et al.).

Run:
    python results/plot-scripts/plot_cv_accuracy.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# List of (epoch, accuracy) pairs per fold, from the finetuning CV logs
# (log_finetuning_fold_0_cv.txt / log_finetuning_fold1_to_4_cv.log). Fold 0 is
# missing epochs 375/425/450/475 due to a log-capture gap (does not affect the
# reported final test metrics); those points are simply omitted here rather
# than interpolated, and can be added later if recovered.
ACCURACY_BY_FOLD = {
    "fold 0": [
        (0, 0.4091), (25, 0.4318), (50, 0.5227), (75, 0.5227), (100, 0.6591),
        (125, 0.6818), (150, 0.6818), (175, 0.6591), (200, 0.8182), (225, 0.6591),
        (250, 0.7727), (275, 0.7727), (300, 0.8182), (325, 0.7045), (350, 0.7955),
        (499, 0.8182),
    ],
    "fold 1": [
        (0, 0.5000), (25, 0.5000), (50, 0.5625), (75, 0.6250), (100, 0.5625),
        (125, 0.7292), (150, 0.7083), (175, 0.7292), (200, 0.7083), (225, 0.8125),
        (250, 0.8125), (275, 0.6875), (300, 0.7708), (325, 0.7917), (350, 0.7708),
        (375, 0.8125), (400, 0.7917), (425, 0.7917), (450, 0.8750), (475, 0.8958),
        (499, 0.8542),
    ],
    "fold 2": [
        (0, 0.5000), (25, 0.4773), (50, 0.5227), (75, 0.6136), (100, 0.6136),
        (125, 0.6136), (150, 0.6818), (175, 0.6818), (200, 0.7727), (225, 0.8409),
        (250, 0.8182), (275, 0.9545), (300, 0.8636), (325, 0.9091), (350, 0.8636),
        (375, 0.9091), (400, 0.8636), (425, 0.8864), (450, 0.8864), (475, 0.8864),
        (499, 0.8864),
    ],
    "fold 3": [
        (0, 0.5208), (25, 0.5625), (50, 0.5833), (75, 0.7083), (100, 0.6458),
        (125, 0.6250), (150, 0.7292), (175, 0.6875), (200, 0.8542), (225, 0.7292),
        (250, 0.8542), (275, 0.8333), (300, 0.7708), (325, 0.8333), (350, 0.8125),
        (375, 0.8750), (400, 0.7708), (425, 0.8125), (450, 0.8542), (475, 0.8750),
        (499, 0.7917),
    ],
    "fold 4": [
        (0, 0.5000), (25, 0.5526), (50, 0.5789), (75, 0.5526), (100, 0.6579),
        (125, 0.6579), (150, 0.7368), (175, 0.7895), (200, 0.7368), (225, 0.8158),
        (250, 0.7895), (275, 0.7895), (300, 0.8158), (325, 0.8421), (350, 0.7895),
        (375, 0.8158), (400, 0.8421), (425, 0.8158), (450, 0.8158), (475, 0.8684),
        (499, 0.8684),
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

    active = {k: v for k, v in ACCURACY_BY_FOLD.items() if v}
    if not active:
        print("No accuracy values entered yet. Fill in ACCURACY_BY_FOLD and re-run.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for fold_name, points in active.items():
        epochs = [e for e, _ in points]
        values = [v for _, v in points]
        color = FOLD_COLORS.get(fold_name, DEFAULT_COLOR)
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=4,
                color=color, label=fold_name.capitalize())

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Drusen Cross-Validation — Accuracy per Fold", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.02)
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
