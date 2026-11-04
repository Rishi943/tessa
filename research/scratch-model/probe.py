"""Train only: planet share in a few hand-picked slices."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent / "work"
names = json.loads((WORK / "features.json").read_text(encoding="utf-8"))
d = np.load(WORK / "train.npz")
X, y = d["X"], d["y"]
f = {n: X[:, i] for i, n in enumerate(names)}


def share(label, m):
    m = m & np.ones(len(y), bool)
    print(f"{label:60s} n {m.sum():4d}  planet share {y[m].mean() if m.sum() else float('nan'):.2f}")


for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)):
    share(f"rv_error in [{lo}, {hi})", (f["rv_error"] >= lo) & (f["rv_error"] < hi))
for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 1), (1, 2), (2, 1e9)):
    m = (f["rv_error"] >= lo) & (f["rv_error"] < hi)
    share(f"  rv_error in [{lo}, {hi}) and teff < 6200", m & (f["teff"] < 6200))
    share(f"  rv_error in [{lo}, {hi}) and teff >= 6200", m & (f["teff"] >= 6200))
for flo in (0, 0.05, 0.1, 0.2):
    for slo, shi in ((0, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)):
        share(f"odd_even_fraction >= {flo}, sigma in [{slo}, {shi})",
              (f["odd_even_fraction"] >= flo) & (f["odd_even_fraction"] < flo + (0.05 if flo < 0.1 else 0.1 if flo < 0.2 else 9))
              & (f["odd_even_sigma"] >= slo) & (f["odd_even_sigma"] < shi))
big = f["planet_radius"] > 10
for t in (5500, 6200, 6800, 7500):
    share(f"planet_radius > 10 and teff >= {t}", big & (f["teff"] >= t))
    share(f"planet_radius <= 10 and teff >= {t}", ~big & (f["teff"] >= t))
for r in (1.5, 2, 3):
    share(f"planet_radius > 10 and star_radius >= {r}", big & (f["star_radius"] >= r))
for lo, hi in ((10, 14), (14, 18), (18, 22), (22, 30), (30, 1e9)):
    share(f"planet_radius in [{lo}, {hi})", (f["planet_radius"] >= lo) & (f["planet_radius"] < hi))
share("code-rejected", d["code_reject"].astype(bool))
