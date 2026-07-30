"""
Train/validation loss plotting script for the Phase-1 (single-class) pretraining
run, where classification metrics (F1/AUC/etc.) are undefined and the
reconstruction/diffusion loss is the only monitoring signal.

Fill in TRAIN_LOSS and VAL_LOSS below with the console values printed by
train_loop each epoch ("train_loss: ...") and each eval epoch ("Val loss: ..."),
then run:
    python experiments/fundus-unet/plot_loss.py

TRAIN_LOSS is logged every epoch; VAL_LOSS only every SAVE_IMAGE_EPOCHS epochs
(same cadence as the metric plots) — they are plotted on their own epoch axes.

"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# One value per EPOCH (train_loss is printed every epoch).
TRAIN_LOSS = [
    # insert values
]

# This run had metrics set (2-class finetuning), so train_loop never computes
# val_loss (that path only runs when metrics=None, i.e. Phase-1 pretraining) —
# no "Val loss: ..." lines exist in this log, so there is nothing to fill in here.
VAL_LOSS = []

# Epoch at which the first eval was logged and the interval between evals
# (must match SAVE_IMAGE_EPOCHS in fundus-unet.sh).
FIRST_EVAL_EPOCH = 0
EVAL_INTERVAL = 25

OUTPUT_DIR = "experiments/fundus-unet/plots"
# ─────────────────────────────────────────────────────────────────────────────


def eval_epoch_axis(n_values):
    return [FIRST_EVAL_EPOCH + i * EVAL_INTERVAL for i in range(n_values)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not TRAIN_LOSS and not VAL_LOSS:
        print("No loss values entered yet. Fill in TRAIN_LOSS/VAL_LOSS and re-run.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    if TRAIN_LOSS:
        epochs = list(range(len(TRAIN_LOSS)))
        ax.plot(epochs, TRAIN_LOSS, linewidth=2, color="#4C72B0", label="Train loss")

    if VAL_LOSS:
        epochs = eval_epoch_axis(len(VAL_LOSS))
        ax.plot(epochs, VAL_LOSS, marker="o", linewidth=2, markersize=5,
                color="#C44E52", label="Val loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Reconstruction loss", fontsize=12)
    ax.set_title("Phase 2 Drusen Finetuning — Train Loss", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "finetune_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    if VAL_LOSS:
        best_epoch = eval_epoch_axis(len(VAL_LOSS))[VAL_LOSS.index(min(VAL_LOSS))]
        print(f"Lowest val loss: {min(VAL_LOSS):.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
