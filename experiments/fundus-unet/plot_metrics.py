"""
Metric plotting script for fundus-unet training runs.

Fill in the METRICS dict below with your logged values and run:
    python experiments/fundus-unet/plot_metrics.py

One value per evaluation epoch (every SAVE_IMAGE_EPOCHS steps).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# Key   = metric name (used as legend label and y-axis label)
# Value = list of floats, one per evaluation checkpoint
#         (e.g. if SAVE_IMAGE_EPOCHS=50 and NUM_EPOCHS=200 → 4 values)

METRICS = {
    "accuracy": [
        0.3977, 0.4610, 0.6664, 0.7175, 0.7427, 0.7394, 0.7573, 0.7581, 0.7808, 0.7873, 0.7890, 0.7914, 0.7873, 0.8011, 0.7946, 0.8019 #, 0.7995, 0.8003, 0.7971, 0.7971, 0.7979
    ],
    "f1": [
        0.5333, 0.5337, 0.5910, 0.6159, 0.6846, 0.6880, 0.7054, 0.6972, 0.7228, 0.7259, 0.7297, 0.7353, 0.7316, 0.7418, 0.7400, 0.7469 #, 0.7430, 0.7427, 0.7374, 0.7407, 0.7393
    ],
    "precision": [
        0.3810, 0.4013, 0.5625, 0.6503, 0.6515, 0.6431, 0.6654, 0.6765, 0.7082, 0.7244, 0.7237, 0.7227, 0.7154, 0.7458, 0.7258, 0.7392 #, 0.7376, 0.7411, 0.7389, 0.7331, 0.7385
    ],
    "recall": [
        0.8889, 0.7966, 0.6226, 0.5849, 0.7212, 0.7421, 0.7505, 0.7191, 0.7379, 0.7275, 0.7358, 0.7484, 0.7484, 0.7379, 0.7547, 0.7547 #, 0.7484, 0.7442, 0.7358, 0.7484, 0.7400
    ],
    "auc": [
        # e.g. 0.784, 0.7920, 0.8077, 0.8161, 0.8258, 0.8233, 0.8435, 0.8401, 0.8371, 0.8429, 0.8313, 0.8380, 0.8456, 0.8435, 0.8420, 0.8378
    ],
}

# Epoch at which the first evaluation was logged and the interval between evals
FIRST_EVAL_EPOCH = 0        # = SAVE_IMAGE_EPOCHS in your shell config
EVAL_INTERVAL    = 50        # = SAVE_IMAGE_EPOCHS

OUTPUT_DIR = "experiments/fundus-unet/plots"
# ─────────────────────────────────────────────────────────────────────────────


COLORS = {
    "accuracy":  "#4C72B0",
    "f1":        "#DD8452",
    "precision": "#55A868",
    "recall":    "#C44E52",
    "auc":       "#8172B2",
}
DEFAULT_COLOR = "#808080"

YLIMITS = {
    "accuracy":  (0.0, 1.0),
    "f1":        (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall":    (0.0, 1.0),
    "auc":       (0.0, 1.0),
}


def epoch_axis(n_values):
    return [FIRST_EVAL_EPOCH + i * EVAL_INTERVAL for i in range(n_values)]


def plot_individual(metrics, output_dir):
    """One figure per metric."""
    for name, values in metrics.items():
        if not values:
            continue
        epochs = epoch_axis(len(values))
        fig, ax = plt.subplots(figsize=(7, 4))

        color = COLORS.get(name, DEFAULT_COLOR)
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=5,
                color=color, label=name)
        ax.fill_between(epochs, values, alpha=0.10, color=color)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(name.capitalize(), fontsize=12)
        ax.set_title(f"{name.capitalize()} over Training", fontsize=13, fontweight="bold")
        ax.set_ylim(*YLIMITS.get(name, (None, None)))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)

        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_combined(metrics, output_dir):
    """All metrics in one figure."""
    active = {k: v for k, v in metrics.items() if v}
    if not active:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, values in active.items():
        epochs = epoch_axis(len(values))
        color = COLORS.get(name, DEFAULT_COLOR)
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=5,
                color=color, label=name.capitalize())

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Metrics over Training", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    path = os.path.join(output_dir, "all_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_grid(metrics, output_dir):
    """Subplot grid: one panel per metric."""
    active = {k: v for k, v in metrics.items() if v}
    if not active:
        return

    n = len(active)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax_flat, (name, values) in zip(axes.flat, active.items()):
        epochs = epoch_axis(len(values))
        color = COLORS.get(name, DEFAULT_COLOR)
        ax_flat.plot(epochs, values, marker="o", linewidth=2, markersize=5, color=color)
        ax_flat.fill_between(epochs, values, alpha=0.10, color=color)
        ax_flat.set_title(name.capitalize(), fontsize=11, fontweight="bold")
        ax_flat.set_xlabel("Epoch", fontsize=10)
        ax_flat.set_ylabel("Score", fontsize=10)
        ax_flat.set_ylim(*YLIMITS.get(name, (None, None)))
        ax_flat.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_flat.grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplot panels
    for ax_flat in axes.flat[n:]:
        ax_flat.set_visible(False)

    fig.suptitle("Fundus UNet — Finetuning Metrics", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "metrics_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filled = {k: v for k, v in METRICS.items() if v}
    if not filled:
        print("No metric values entered yet. Fill in the METRICS dict and re-run.")
        return

    plot_individual(METRICS, OUTPUT_DIR)
    plot_combined(METRICS, OUTPUT_DIR)
    plot_grid(METRICS, OUTPUT_DIR)
    print(f"\nAll plots written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
