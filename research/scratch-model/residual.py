"""Train only: among signals above the 2%-planet-loss line, which features still split planets from FPs?
Usage: python residual.py MARGINS_NAME"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

WORK = Path(__file__).resolve().parent / "work"
names = json.loads((WORK / "features.json").read_text(encoding="utf-8"))
d = np.load(WORK / "train.npz")
X, y = d["X"], d["y"]
s = np.load(WORK / f"margins-{sys.argv[1]}.npz")["train"]
cut = np.quantile(s[y == 1], 0.02)
keep = s >= cut
print(f"above the line: {(y[keep] == 1).sum()} planets, {(y[keep] == 0).sum()} FPs (code-rejected among FPs: "
      f"{d['code_reject'][keep & (y == 0)].sum()})")
Xk, yk = X[keep], y[keep]
out = []
for j, n in enumerate(names):
    col = Xk[:, j]
    ok = np.isfinite(col)
    if ok.sum() < 30 or len(set(yk[ok])) < 2:
        continue
    auc = roc_auc_score(yk[ok], col[ok])
    miss = (1 - ok[yk == 1].mean(), 1 - ok[yk == 0].mean())
    out.append((abs(auc - 0.5), n, auc, miss, np.nanmedian(col[yk == 1]), np.nanmedian(col[yk == 0])))
for _, n, auc, miss, mp, mf in sorted(out, reverse=True)[:25]:
    print(f"{n:30s} AUC {auc:.3f}  miss P {miss[0]:.0%} F {miss[1]:.0%}  median planet {mp:.4g}  fp {mf:.4g}")
# the ten most planet-like FPs
fp = np.where((y == 0) & keep)[0]
top = fp[np.argsort(-s[fp])][:10]
show = ["v_shape_sigma", "centroid_offset_sigma", "depth_ppm", "planet_radius", "period", "star_radius", "teff",
        "odd_even_sigma", "secondary_snr", "contamination", "plausible_within_1px", "distance"]
print("\nmost planet-like FPs:")
for i in top:
    print(f"  {d['toi'][i]:>10s} m {s[i]:5.2f} code_reject {d['code_reject'][i]} " +
          " ".join(f"{k}={X[i, names.index(k)]:.3g}" for k in show))
