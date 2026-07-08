"""
Offline augmentation of the clinical Optic Disc Drusen (ODD) images.

The clinic provided only ~120 positive (Drusen) cases against ~5000 negatives.
Training a diffusion classifier on such an imbalance means the model barely ever
sees the positive class. This script expands the ~120 Drusen images to a target
count (default ~1000) by applying label-preserving augmentations and writing the
results to disk, so the expensive augmentation happens once — not every epoch.

The input images are tight crops centred on the optic nerve head (where the
Drusen sit). Such a crop has no anatomical "up" — unlike a full fundus image,
where the macula and vessel arcades fix an orientation — so it is rotation- and
flip-invariant, and rotation can be sampled over the full circle.

IMPORTANT — these images train a *generative* diffusion model, not a
discriminative classifier. The model must learn to *reconstruct* the augmented
images, and classification is driven by the reconstruction error (Eq. 3). This
makes two otherwise-standard augmentations counterproductive here, so they are
OFF by default and gated behind flags:

  --gaussian-noise : baking noise into the clean training target teaches the
                     model to produce noisy reconstructions, shrinking the
                     between-class error gap the classifier relies on.
  --random-erasing : (1) the model would learn to generate black boxes; (2) on a
                     disc-centred crop the Drusen sit in the centre, so erasing
                     there removes the pathology while keeping label=1
                     (label-inconsistent).

Enabled by default (safe for the generative setting):
  - rotation over [0, 360)      (orientation-invariant disc crop)
  - scaling in [0.9, 1.1]        (camera magnification / working distance)
  - colour jitter (mild)         (camera / illumination differences; hue kept
                                  small to preserve the yellowish Drusen colour)
  - elastic deformation (mild)   (plausible anatomical variation)

No plain cropping is applied — the optic nerve head must stay fully in frame.

Usage:
    python dataset/augment_drusen.py \
        --input-dir  /data/clinic/drusen \
        --output-dir /data/clinic/drusen_augmented \
        --target-count 1000
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(input_dir):
    paths = [
        p for p in sorted(Path(input_dir).rglob("*"))
        if p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found under {input_dir}")
    return paths


# Individual augmentation operations (operate on RGB uint8 HxWx3 arrays)

def aug_rotate_scale(img, rng):
    """Rotation over the full circle combined with a mild scale in one warp."""
    h, w = img.shape[:2]
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,  # black corners, like a fundus border
    )


def aug_color_jitter(img, rng):
    """Mild brightness/saturation/hue (HSV) + contrast (RGB) jitter."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-3, 3)) % 180        # hue: tiny, preserve Drusen colour
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)  # saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.85, 1.15), 0, 255)  # brightness (value)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    contrast = rng.uniform(0.85, 1.15)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = np.clip((out - mean) * contrast + mean, 0, 255)
    return out.astype(np.uint8)


def aug_elastic(img, rng, alpha=25.0, sigma=5.0):
    """Mild elastic deformation via smoothed random displacement fields."""
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def aug_gaussian_noise(img, rng, sigma_range=(3.0, 8.0)):
    """OFF by default — see module docstring. Additive Gaussian sensor noise."""
    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_random_erasing(img, rng, area_range=(0.02, 0.08), aspect_range=(0.3, 3.3)):
    """OFF by default — see module docstring. Occlude a random rectangle."""
    h, w = img.shape[:2]
    area = h * w * rng.uniform(*area_range)
    aspect = rng.uniform(*aspect_range)
    eh = int(round((area * aspect) ** 0.5))
    ew = int(round((area / aspect) ** 0.5))
    if eh >= h or ew >= w:
        return img
    top = rng.integers(0, h - eh)
    left = rng.integers(0, w - ew)
    out = img.copy()
    out[top:top + eh, left:left + ew] = rng.integers(0, 256, (eh, ew, img.shape[2]), dtype=np.uint8)
    return out


def augment(img, rng, use_noise, use_erasing):
    """Apply the enabled augmentation chain in a sensible order."""
    img = aug_rotate_scale(img, rng)                       # geometric first
    if rng.random() < 0.5:
        img = aug_elastic(img, rng)
    if rng.random() < 0.8:
        img = aug_color_jitter(img, rng)                   # then photometric
    if use_noise and rng.random() < 0.5:
        img = aug_gaussian_noise(img, rng)
    if use_erasing and rng.random() < 0.5:
        img = aug_random_erasing(img, rng)
    return img


def main():
    parser = argparse.ArgumentParser(description="Augment clinical Drusen images offline.")
    parser.add_argument("--input-dir", required=True, help="Directory with original Drusen images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write augmented images.")
    parser.add_argument("--target-count", type=int, default=1000, help="Approximate number of output images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--gaussian-noise", action="store_true",
                        help="Enable additive Gaussian noise (NOT recommended for the diffusion model).")
    parser.add_argument("--random-erasing", action="store_true",
                        help="Enable random erasing (NOT recommended for the diffusion model).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    sources = list_images(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # How many variants per source image to reach the target count.
    per_image = max(1, -(-args.target_count // len(sources)))  # ceil division

    print(f"Sources         : {len(sources)}")
    print(f"Variants/image  : {per_image}")
    print(f"Expected output : ~{len(sources) * per_image}")
    print(f"Gaussian noise  : {'on' if args.gaussian_noise else 'off'}")
    print(f"Random erasing  : {'on' if args.random_erasing else 'off'}")

    written = 0
    for src in sources:
        try:
            image = np.array(Image.open(src).convert("RGB"))
        except Exception as exc:
            print(f"  skipping unreadable {src}: {exc}")
            continue

        for i in range(per_image):
            if i == 0:
                out = image  # variant 0 is always the untouched original
            else:
                out = augment(image, rng, args.gaussian_noise, args.random_erasing)
            out_name = f"{src.stem}_aug{i:02d}.png"
            Image.fromarray(out).save(os.path.join(args.output_dir, out_name))
            written += 1

    print(f"\nDone. Wrote {written} augmented images to {args.output_dir}")


if __name__ == "__main__":
    main()
