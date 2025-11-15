"""
recolor_image.py
----------------
Module for recoloring segmented outline images into FG/BG color variants.
Supports pixel-wise and patch-wise recoloring.

Usage:
    from recolor_image import generate_variants

    paths = generate_variants(row, target_color="red", out_dir=Path("data/colored"), use_patches=True)
"""

from pathlib import Path
from PIL import Image
import numpy as np
import colorsys
import gc

SEED = 42
rng = np.random.default_rng(SEED)


def color_remap(pixel, target_color: str, blend: float = 0.75):
    """
    Recolor a single RGB pixel toward the target color.

    Parameters
    ----------
    pixel : tuple(int, int, int)
        Original (R, G, B) pixel values in 0–255.
    target_color : str
        Target color name ("red", "green", etc.).
    blend : float
        Blend strength between 0 (original) and 1 (fully recolored).

    Returns
    -------
    (r, g, b) : tuple(int)
        The recolored pixel.
    """
    target_color = target_color.lower()
    r0, g0, b0 = pixel
    h, s, v = colorsys.rgb_to_hsv(r0/255.0, g0/255.0, b0/255.0)

    # Protect outlines and very dark pixels
    if v < 0.12 or (v < 0.25 and s < 0.25):
        return pixel

    color_hue_map = {
        "red":    0.0,
        "brown":  0.05,
        "pink":   0.9,
        "orange": 0.07,
        "yellow": 0.15,
        "gold":   0.12,
        "green":  0.33,
        "blue":   0.55,
        "purple": 0.78,
        "black":  0.0,
        "grey":   0.0,
        "silver": 0.0,
        "white":  0.0,
    }

    neutral_targets = {"grey", "silver", "black"}

    # Neutral colors
    if target_color in neutral_targets:
        if target_color == "grey":
            new_v = min(max(v, 0.55), 0.75)
            new_s = 0.0
            per_color_blend = 0.85
        elif target_color == "silver":
            new_v = min(max(v, 0.65), 0.9)
            new_s = 0.0
            per_color_blend = 0.85
        elif target_color == "black":
            new_v = min(max(v * 0.2, 0.05), 0.19)
            new_s = 0.0
            per_color_blend = 1.0

        r1, g1, b1 = colorsys.hsv_to_rgb(0.0, new_s, new_v)
        r1, g1, b1 = int(r1 * 255), int(g1 * 255), int(b1 * 255)

        w = per_color_blend
        r = int((1 - w) * r0 + w * r1)
        g = int((1 - w) * g0 + w * g1)
        b = int((1 - w) * b0 + w * b1)
        return r, g, b

    # Colorful targets
    target_hue = color_hue_map.get(target_color, 0.0)
    new_h, new_s, new_v = target_hue, 0.75, min(max(v, 0.8), 0.95)

    # Per-color tweaks
    tweaks = {
        "red":    (0.85, 0.9),
        "brown":  (0.7,  0.55),
        "pink":   (0.5,  1.0),
        "orange": (0.85, 0.95),
        "yellow": (0.65, 1.0),
        "gold":   (0.6,  0.85),
        "green":  (0.75, 0.7),
        "blue":   (0.9,  0.7),
        "purple": (0.7,  0.85),
    }
    if target_color in tweaks:
        new_s, new_v = tweaks[target_color]

    r1, g1, b1 = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    r1, g1, b1 = int(r1 * 255), int(g1 * 255), int(b1 * 255)

    r = int((1 - blend) * r0 + blend * r1)
    g = int((1 - blend) * g0 + blend * g1)
    b = int((1 - blend) * b0 + blend * b1)

    return r, g, b


def recolor_subset(arr_rgb, idx_flat, k, target_color):
    """Randomly recolor k pixels from idx_flat."""
    if k <= 0 or len(idx_flat) == 0:
        return arr_rgb

    chosen = rng.choice(idx_flat, size=min(k, len(idx_flat)), replace=False)
    flat = arr_rgb.reshape(-1, 3)

    for i in chosen:
        r, g, b = flat[i]
        flat[i] = color_remap((int(r), int(g), int(b)), target_color)
    return arr_rgb


def recolor_subset_patches(arr_rgb, idx_flat, pct, target_color, H, W, patch_size=16):
    """
    Recolor image in patches rather than per pixel.
    Selects patches that overlap the mask region.
    """
    if len(idx_flat) == 0 or pct <= 0:
        return arr_rgb

    flat = arr_rgb.reshape(-1, 3)
    rows, cols = idx_flat // W, idx_flat % W

    patch_rows, patch_cols = rows // patch_size, cols // patch_size
    patch_ids_unique = np.unique(np.stack([patch_rows, patch_cols], axis=1), axis=0)
    n_patches = len(patch_ids_unique)
    if n_patches == 0:
        return arr_rgb

    target_pixels = int(round(pct / 100.0 * len(idx_flat)))
    avg_pixels_per_patch = max(1, len(idx_flat) // n_patches)
    k_patches = max(1, min(n_patches, target_pixels // avg_pixels_per_patch))

    chosen_idx = rng.choice(n_patches, size=k_patches, replace=False)
    chosen_patches = patch_ids_unique[chosen_idx]

    for (pr, pc) in chosen_patches:
        r_start, c_start = pr * patch_size, pc * patch_size
        r_end, c_end = min(r_start + patch_size, H), min(c_start + patch_size, W)

        in_patch = ((rows >= r_start) & (rows < r_end) & (cols >= c_start) & (cols < c_end))
        flat_indices = idx_flat[in_patch]

        for fi in flat_indices:
            r, g, b = flat[fi]
            flat[fi] = color_remap((int(r), int(g), int(b)), target_color)

    return arr_rgb


def generate_variants(row, target_color, out_dir: Path, use_patches=False, patch_size=16, step_size=10):
    """
    Generate FG/BG recolored variants for one image + one target color.

    Returns a list of saved variant paths (strings).
    """
    img = Image.open(row["image_path"]).convert("RGB")
    W, H = img.size

    m = Image.open(row["cv_mask_path"]).convert("L").resize((W, H), Image.BILINEAR)
    mask = (np.array(m, dtype=np.uint8) > 127)

    stem = Path(row["image_path"]).stem
    color_dir = out_dir / f"{stem}_{target_color}"
    color_dir.mkdir(parents=True, exist_ok=True)

    base = np.array(img, dtype=np.uint8)
    idx_all = np.arange(H * W)
    idx_fg = idx_all[mask.flatten()]
    idx_bg = idx_all[~mask.flatten()]

    fg_paths, bg_paths = [], []

    # Foreground recolor series
    for pct in range(0, 101, step_size):
        arr = base.copy()
        if use_patches:
            recolor_subset_patches(arr, idx_fg, pct, target_color, H, W, patch_size)
        else:
            k = int(round(pct / 100.0 * len(idx_fg)))
            recolor_subset(arr, idx_fg, k, target_color)
        out_path = color_dir / f"FG_{pct:03d}.png"
        Image.fromarray(arr).save(out_path)
        fg_paths.append(out_path)

    # Background recolor series
    for pct in range(10, 101, step_size):
        arr = base.copy()
        if use_patches:
            recolor_subset_patches(arr, idx_bg, pct, target_color, H, W, patch_size)
        else:
            k = int(round(pct / 100.0 * len(idx_bg)))
            recolor_subset(arr, idx_bg, k, target_color)
        out_path = color_dir / f"BG_{pct:03d}.png"
        Image.fromarray(arr).save(out_path)
        bg_paths.append(out_path)

    gc.collect()
    return [str(p) for p in (fg_paths + bg_paths)]