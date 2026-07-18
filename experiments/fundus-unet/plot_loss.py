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
# From scripts/logs/training06.log — Phase 2 (drusen) finetuning run, 200 epochs.
TRAIN_LOSS = [
    0.000382, 0.000348, 0.000305, 0.000306, 0.000279, 0.000295, 0.000302, 0.000356, 0.000318, 0.000266,
    0.000332, 0.000331, 0.000350, 0.000282, 0.000291, 0.000265, 0.000278, 0.000249, 0.000290, 0.000300,
    0.000311, 0.000314, 0.000278, 0.000267, 0.000247, 0.000259, 0.000312, 0.000296, 0.000291, 0.000276,
    0.000270, 0.000287, 0.000313, 0.000245, 0.000247, 0.000316, 0.000249, 0.000251, 0.000307, 0.000270,
    0.000284, 0.000338, 0.000300, 0.000290, 0.000289, 0.000342, 0.000276, 0.000300, 0.000269, 0.000233,
    0.000284, 0.000295, 0.000274, 0.000314, 0.000268, 0.000284, 0.000285, 0.000260, 0.000306, 0.000313,
    0.000293, 0.000324, 0.000310, 0.000252, 0.000268, 0.000277, 0.000301, 0.000266, 0.000271, 0.000260,
    0.000273, 0.000248, 0.000283, 0.000258, 0.000262, 0.000273, 0.000228, 0.000242, 0.000268, 0.000274,
    0.000257, 0.000309, 0.000289, 0.000260, 0.000383, 0.000205, 0.000270, 0.000425, 0.000250, 0.000293,
    0.000311, 0.000309, 0.000251, 0.000252, 0.000247, 0.000324, 0.000244, 0.000256, 0.000232, 0.000312,
    0.000265, 0.000203, 0.000294, 0.000248, 0.000279, 0.000252, 0.000264, 0.000339, 0.000306, 0.000230,
    0.000243, 0.000286, 0.000307, 0.000305, 0.000246, 0.000238, 0.000331, 0.000216, 0.000264, 0.000277,
    0.000277, 0.000311, 0.000243, 0.000248, 0.000262, 0.000234, 0.000249, 0.000262, 0.000310, 0.000276,
    0.000259, 0.000319, 0.000286, 0.000269, 0.000259, 0.000232, 0.000362, 0.000284, 0.000261, 0.000257,
    0.000283, 0.000308, 0.000263, 0.000279, 0.000287, 0.000272, 0.000348, 0.000316, 0.000323, 0.000281,
    0.000296, 0.000314, 0.000277, 0.000271, 0.000340, 0.000250, 0.000287, 0.000239, 0.000274, 0.000237,
    0.000249, 0.000234, 0.000258, 0.000252, 0.000304, 0.000251, 0.000298, 0.000269, 0.000254, 0.000236,
    0.000248, 0.000247, 0.000266, 0.000219, 0.000306, 0.000242, 0.000259, 0.000234, 0.000289, 0.000320,
    0.000260, 0.000248, 0.000376, 0.000292, 0.000304, 0.000254, 0.000277, 0.000286, 0.000253, 0.000292,
    0.000242, 0.000287, 0.000268, 0.000293, 0.000254, 0.000247, 0.000248, 0.000258, 0.000246, 0.000269,
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
