"""
Metric plotting script for classification training/from-scratch runs.

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
    # From log_training_from_scratch_single_run.log: 29 eval points at epochs
    # 0,25,...,675,699 (NUM_EPOCHS=700; last point is 699, not 700, due
    # to 0-indexed epochs).
    "accuracy": [
        0.5000, 0.8333, 0.8333, 0.8333, 0.8333, 0.7778, 0.8333, 0.8333, 0.8889,
        0.8889, 0.8333, 0.8889, 0.8333, 0.8889, 0.7778, 0.8333, 0.8333, 0.7778,
        0.7778, 0.7778, 0.7778, 0.7778, 0.8333, 0.8333, 0.8889, 0.8889, 0.8333,
        0.8889, 0.8889,
    ],
    "f1": [
        0.4706, 0.8000, 0.8000, 0.8000, 0.8000, 0.7143, 0.8000, 0.8000, 0.8750,
        0.8750, 0.8000, 0.8750, 0.8000, 0.8750, 0.7143, 0.8000, 0.8000, 0.7143,
        0.7143, 0.7143, 0.7143, 0.7143, 0.8000, 0.8000, 0.8750, 0.8750, 0.8000,
        0.8750, 0.8750,
    ],
    "precision": [
        0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000,
    ],
    "recall": [
        0.4444, 0.6667, 0.6667, 0.6667, 0.6667, 0.5556, 0.6667, 0.6667, 0.7778,
        0.7778, 0.6667, 0.7778, 0.6667, 0.7778, 0.5556, 0.6667, 0.6667, 0.5556,
        0.5556, 0.5556, 0.5556, 0.5556, 0.6667, 0.6667, 0.7778, 0.7778, 0.6667,
        0.7778, 0.7778,
    ],
    "auc": [
        0.6420, 0.8519, 0.8272, 0.8148, 0.9012, 0.9012, 0.9753, 0.9630, 0.9259,
        0.9506, 0.8889, 0.9753, 0.9383, 0.9630, 0.9630, 0.9506, 1.0000, 0.9753,
        0.9877, 1.0000, 0.9877, 1.0000, 1.0000, 1.0000, 1.0000, 0.9877, 1.0000,
        0.9877, 1.0000,
    ],
}

# Epoch at which the first evaluation was logged and the interval between evals
FIRST_EVAL_EPOCH = 0        # = SAVE_IMAGE_EPOCHS in your shell config
EVAL_INTERVAL    = 25        # = SAVE_IMAGE_EPOCHS
# Note: the last eval point above is at raw epoch 699 (not 700), one epoch
# short of the regular 25-step grid due to 0-indexing; epoch_axis() below
# still assumes a perfectly even grid, so the plotted x-position for that
# last point is off by 1 (700 instead of 699) — negligible at this scale.

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


def adjusted_ylim(name, values):
    """Return y-axis limits with a small upper margin to avoid clipping top markers."""
    ymin, ymax = YLIMITS.get(name, (None, None))
    if ymin is None or ymax is None:
        return ymin, ymax
    margin = max(0.02, (ymax - ymin) * 0.02)
    return ymin, ymax + margin


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
        ax.set_ylim(*adjusted_ylim(name, values))
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
    ax.set_ylim(0.0, 1.02)
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
        ax_flat.set_ylim(*adjusted_ylim(name, values))
        ax_flat.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_flat.grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplot panels
    for ax_flat in axes.flat[n:]:
        ax_flat.set_visible(False)

    fig.suptitle("Fundus UNet — From-Scratch Training Metrics", fontsize=14, fontweight="bold", y=1.01)
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
