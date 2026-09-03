"""
Train/validation loss plotting script for the Phase-1 (single-class) pretraining
run, where classification metrics (F1/AUC/etc.) are undefined and the
reconstruction/diffusion loss is the only monitoring signal.

Fill in TRAIN_LOSS and VAL_LOSS below with the console values printed by
train_loop each epoch ("train_loss: ...") and each eval epoch ("Val loss: ..."),
then run:
    python results/fundus-unet/pretraining/plots/plot_loss_pretrain_mogon.py

TRAIN_LOSS is logged every epoch; VAL_LOSS only every SAVE_IMAGE_EPOCHS epochs
(same cadence as the metric plots) — they are plotted on their own epoch axes.

"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── USER INPUT ────────────────────────────────────────────────────────────────
# One value per EPOCH (train_loss is printed every epoch).
TRAIN_LOSS = [
    0.002520, 0.000729, 0.000566, 0.000504, 0.000475, 0.000443, 0.000431, 0.000420, 0.000410, 0.000398, 0.000389, 0.000391, 0.000385, 0.000383, 0.000376, 0.000379, 0.000373, 0.000366, 0.000365, 0.000362, 0.000362, 0.000358, 0.000361, 0.000357, 0.000359, 0.000357, 0.000359, 0.000354, 0.000353, 0.000354
]

# One value per EVAL EPOCH (val_loss is printed only every SAVE_IMAGE_EPOCHS).
VAL_LOSS = [
    0.000944, 0.000543, 0.000569, 0.000432, 0.000415, 0.000402, 0.000469, 0.000353, 0.000371, 0.000406, 0.000337, 0.000336, 0.000341, 0.000396, 0.000315, 0.000289
]

# Epoch at which the first eval was logged and the interval between evals
# (must match SAVE_IMAGE_EPOCHS in fundus-unet.sh).
FIRST_EVAL_EPOCH = 0
EVAL_INTERVAL = 2

OUTPUT_DIR = "results/fundus-unet/pretraining/plots"
# ─────────────────────────────────────────────────────────────────────────────


def eval_epoch_axis(n_values):
    epochs = [FIRST_EVAL_EPOCH + i * EVAL_INTERVAL for i in range(n_values)]
    if TRAIN_LOSS and epochs:
        last_epoch = len(TRAIN_LOSS) - 1
        if epochs[-1] != last_epoch:
            epochs[-1] = last_epoch
    return epochs


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
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pretrain_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    if VAL_LOSS:
        best_epoch = eval_epoch_axis(len(VAL_LOSS))[VAL_LOSS.index(min(VAL_LOSS))]
        print(f"Lowest val loss: {min(VAL_LOSS):.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
