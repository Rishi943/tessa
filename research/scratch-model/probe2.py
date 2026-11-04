import json
from pathlib import Path
import numpy as np
WORK = Path("work")
names = json.loads((WORK / "features.json").read_text(encoding="utf-8"))
d = np.load(WORK / "train.npz"); X, y = d["X"], d["y"]
f = {n: X[:, i] for i, n in enumerate(names)}
def row(label, m):
    print(f"{label:40s} n {m.sum():4d} share {y[m].mean() if m.sum() else float('nan'):.2f}")
T = [0, 4000, 5000, 5500, 6000, 6200, 6500, 6800, 7500, 1e9]
for name, m in (("rp<=4", f["planet_radius"] <= 4), ("4<rp<=10", (f["planet_radius"] > 4) & (f["planet_radius"] <= 10)), ("rp>10", f["planet_radius"] > 10)):
    for a, b in zip(T, T[1:]):
        row(f"{name} teff [{a},{b})", m & (f["teff"] >= a) & (f["teff"] < b))
R = [0, 0.6, 0.9, 1.2, 1.5, 2, 3, 1e9]
for name, m in (("rp<=10", f["planet_radius"] <= 10), ("rp>10", f["planet_radius"] > 10)):
    for a, b in zip(R, R[1:]):
        row(f"{name} rstar [{a},{b})", m & (f["star_radius"] >= a) & (f["star_radius"] < b))
