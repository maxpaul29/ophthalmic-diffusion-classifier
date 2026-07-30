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
    # training06.log (epochs 0-200, 9 points) + training07.log (resumed to
    # NUM_EPOCHS=400; its first eval at raw epoch 200 is ~1 epoch after
    # training06's last point and is dropped as a near-duplicate, leaving 8
    # points at epochs 225-400) — 17 points total, epochs 0,25,...,200,225,...,400.
    "accuracy": [
        0.4545, 0.3636, 0.5000, 0.5909, 0.5455, 0.6364, 0.6364, 0.8182, 0.8182,
        0.8182, 0.9091, 0.9091, 0.9545 #, 0.9091, 0.9091, 0.8636, 0.9545,
    ],
    "f1": [
        0.4545, 0.4167, 0.4762, 0.5714, 0.5000, 0.6000, 0.6364, 0.8000, 0.8182,
        0.8333, 0.9091, 0.9091, 0.9565 # , 0.9167, 0.9091, 0.8696, 0.9565,
    ],
    "precision": [
        0.4545, 0.3846, 0.5000, 0.6000, 0.5556, 0.6667, 0.6364, 0.8889, 0.8182,
        0.7692, 0.9091, 0.9091, 0.9167 # , 0.8462, 0.9091, 0.8333, 0.9167,
    ],
    "recall": [
        0.4545, 0.4545, 0.4545, 0.5455, 0.4545, 0.5455, 0.6364, 0.7273, 0.8182,
        0.9091, 0.9091, 0.9091, 1.0000 # , 1.0000, 0.9091, 0.9091, 1.0000,
    ],
    "auc": [
        0.2893, 0.5124, 0.6033, 0.6612, 0.5620, 0.5537, 0.8099, 0.7934, 0.8595,
        0.7686, 0.8595, 0.9008, 0.9339 # , 1.0000, 0.9504, 0.9091, 0.9421,
    ],
}

# Epoch at which the first evaluation was logged and the interval between evals
FIRST_EVAL_EPOCH = 0        # = SAVE_IMAGE_EPOCHS in your shell config
EVAL_INTERVAL    = 25        # = SAVE_IMAGE_EPOCHS

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
