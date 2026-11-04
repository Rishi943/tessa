"""Soft night clouds lit from below by the horizon glow, for the opening's sky (the art reference's wisps).

  ../../.venv/Scripts/python make_clouds.py

Writes assets/sky-clouds.png, 1920 x 3000 like the sky layers: thin near the top, thickest in the lower sky, gone
just above the horizon so the glow behind the tree stays clear. Seeded noise; the same file every run.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
W, H = 1920, 3000
nr = np.random.default_rng(31)


def fbm(cells, octaves=5):
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        c = cells * 2 ** k
        g = nr.random((max(int(H / W * c / 2.2), 2) + 2, c + 2)).astype(np.float32)  # stretched wide: wisps, not puffs
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        out += amp * np.asarray(im, np.float32) / 255
        tot += amp
        amp *= 0.5
    return out / tot


yy = np.arange(H, dtype=np.float32)[:, None] / H
n = fbm(3)
cover = np.clip((n - 0.5) / 0.22, 0, 1) ** 1.6
band = np.exp(-((yy - 0.8) / 0.13) ** 2) + 0.35 * np.exp(-((yy - 0.6) / 0.08) ** 2)  # lower sky, and a thin upper band
band *= np.clip((0.965 - yy) / 0.05, 0, 1)  # clear just above the horizon
alpha = np.clip(cover * band * 0.42, 0, 0.42)
lit = np.clip((yy - 0.55) / 0.4, 0, 1)  # lit from below: bluer and brighter toward the horizon
rgb = np.stack([0.32 + 0.38 * lit, 0.48 + 0.4 * lit, 0.72 + 0.25 * lit], -1) * np.ones((H, W, 1), np.float32)
img = np.dstack([np.clip(rgb, 0, 1) * 255, alpha * 255]).astype(np.uint8)
Image.fromarray(img, "RGBA").filter(ImageFilter.GaussianBlur(6)).save(HERE / "assets" / "sky-clouds.png", optimize=True)
print("sky-clouds.png")
