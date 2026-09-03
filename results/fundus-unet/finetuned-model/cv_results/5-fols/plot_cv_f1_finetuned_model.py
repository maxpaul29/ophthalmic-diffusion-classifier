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
        (0, 0.4348), (25, 0.4444), (50, 0.5333), (75, 0.4878), (100, 0.6154),
        (125, 0.6316), (150, 0.6818), (175, 0.6512), (200, 0.8095), (225, 0.6341),
        (250, 0.7619), (275, 0.7826), (300, 0.8261), (325, 0.7347), (350, 0.8235),
        (499, 0.8400),
    ],
    "fold 1": [
        (0, 0.4545), (25, 0.4783), (50, 0.5714), (75, 0.6087), (100, 0.5532),
        (125, 0.7234), (150, 0.6957), (175, 0.7111), (200, 0.6957), (225, 0.8085),
        (250, 0.8000), (275, 0.7059), (300, 0.7843), (325, 0.8077), (350, 0.7843),
        (375, 0.8235), (400, 0.8000), (425, 0.8077), (450, 0.8846), (475, 0.9057),
        (499, 0.8727),
    ],
    "fold 2": [
        (0, 0.5000), (25, 0.5306), (50, 0.5333), (75, 0.6047), (100, 0.5854),
        (125, 0.6047), (150, 0.6957), (175, 0.6500), (200, 0.7917), (225, 0.8444),
        (250, 0.8182), (275, 0.9565), (300, 0.8696), (325, 0.9091), (350, 0.8636),
        (375, 0.9130), (400, 0.8696), (425, 0.8889), (450, 0.8837), (475, 0.8889),
        (499, 0.8837),
    ],
    "fold 3": [
        (0, 0.5106), (25, 0.5882), (50, 0.5652), (75, 0.6957), (100, 0.6222),
        (125, 0.5714), (150, 0.7347), (175, 0.6512), (200, 0.8571), (225, 0.7111),
        (250, 0.8511), (275, 0.8261), (300, 0.7556), (325, 0.8400), (350, 0.8085),
        (375, 0.8800), (400, 0.7925), (425, 0.8235), (450, 0.8627), (475, 0.8800),
        (499, 0.7917),
    ],
    "fold 4": [
        (0, 0.4865), (25, 0.5854), (50, 0.6000), (75, 0.5854), (100, 0.6486),
        (125, 0.6286), (150, 0.7500), (175, 0.7778), (200, 0.7500), (225, 0.8205),
        (250, 0.8000), (275, 0.7895), (300, 0.8205), (325, 0.8500), (350, 0.8000),
        (375, 0.8372), (400, 0.8500), (425, 0.8293), (450, 0.8372), (475, 0.8780),
        (499, 0.8780),
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
