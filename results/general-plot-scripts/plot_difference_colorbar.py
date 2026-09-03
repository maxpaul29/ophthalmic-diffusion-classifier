"""
Generate a standalone colorbar legend for the "Difference" panel produced by
explain.py, to place next to already-generated counterfactual images in the
thesis (rather than regenerating those images).

Uses the same colormap and fixed scale as explain.py's fundus_plotter
(cmap='jet', vmin=0, vmax=DIFFERENCE_VMAX), so the legend matches images
generated with that scale.

Run:
    python experiments/fundus-unet/plot_difference_colorbar.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib as mpl

DIFFERENCE_VMAX = 0.3

OUTPUT_DIR = "experiments/fundus-unet/plots"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=DIFFERENCE_VMAX)

    # Vertical colorbar, sized to sit next to a single square image.
    fig, ax = plt.subplots(figsize=(1.2, 5))
    cbar = mpl.colorbar.ColorbarBase(ax, cmap="jet", norm=norm, orientation="vertical")
    cbar.set_label("Absolute intensity change\n(blue = none, red = strong)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "difference_colorbar_vertical.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Horizontal variant, for placing below a row of images instead.
    fig, ax = plt.subplots(figsize=(5, 1.0))
    cbar = mpl.colorbar.ColorbarBase(ax, cmap="jet", norm=norm, orientation="horizontal")
    cbar.set_label("Absolute intensity change (blue = none, red = strong)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "difference_colorbar_horizontal.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Unscaled variant (no tick numbers): for the older explain.py images that
    # were generated before DIFFERENCE_VMAX was fixed, i.e. each image was
    # auto-scaled by matplotlib to its OWN min/max difference. There the exact
    # numeric scale differs per image and can't be reconstructed after the
    # fact, so only the qualitative direction (blue = no/little change,
    # red = the strongest change within that specific image) can honestly be
    # shown - not an absolute, cross-patient-comparable scale.
    fig, ax = plt.subplots(figsize=(1.0, 5))
    cbar = mpl.colorbar.ColorbarBase(ax, cmap="jet", norm=mpl.colors.Normalize(vmin=0, vmax=1))
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.02, "no change", ha="center", va="top", transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.5, 1.02, "strong change", ha="center", va="bottom", transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(1.8, 0.5, "(relative to this image only —\nnot comparable across images)",
                 ha="left", va="center", transform=cbar.ax.transAxes, fontsize=7, style="italic")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "difference_colorbar_unscaled.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
