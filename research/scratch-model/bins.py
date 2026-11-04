"""Train split only: planet share by decile of each feature (and when missing)."""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent / "work"
names = json.loads((WORK / "features.json").read_text(encoding="utf-8"))
d = np.load(WORK / "train.npz")
X, y = d["X"], d["y"]
pick = sys.argv[1:] or names
for n in pick:
    col = X[:, names.index(n)]
    ok = np.isfinite(col)
    edges = np.unique(np.nanpercentile(col, np.linspace(0, 100, 11)))
    parts = []
    if (~ok).sum():
        parts.append(f"NA:{y[~ok].mean():.2f}({(~ok).sum()})")
    idx = np.clip(np.searchsorted(edges, col[ok], side="right") - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            parts.append(f"<={edges[b + 1]:.3g}:{y[ok][m].mean():.2f}({m.sum()})")
    print(f"{n}: " + "  ".join(parts))
