"""Train split only: per feature, how often it is missing in each class, its one-feature AUC, and class quantiles."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

WORK = Path(__file__).resolve().parent / "work"
names = json.loads((WORK / "features.json").read_text(encoding="utf-8"))
d = np.load(WORK / "train.npz")
X, y = d["X"], d["y"]
print(f"train {len(y)}: {y.sum()} planets, {len(y) - y.sum()} FPs; code-rejected {d['code_reject'].sum()} "
      f"(planets among them {y[d['code_reject']].sum()})")
rows = []
for j, n in enumerate(names):
    col = X[:, j]
    ok = np.isfinite(col)
    mp, mf = 1 - ok[y == 1].mean(), 1 - ok[y == 0].mean()
    auc = roc_auc_score(y[ok], col[ok]) if ok.sum() > 20 and len(set(y[ok])) == 2 else float("nan")
    qp = np.nanpercentile(col[y == 1], [10, 50, 90]) if ok[y == 1].any() else [np.nan] * 3
    qf = np.nanpercentile(col[y == 0], [10, 50, 90]) if ok[y == 0].any() else [np.nan] * 3
    rows.append((abs(auc - 0.5) if np.isfinite(auc) else 0, n, mp, mf, auc, qp, qf))
for _, n, mp, mf, auc, qp, qf in sorted(rows, reverse=True):
    f = lambda q: " ".join(f"{v:9.3g}" for v in q)  # noqa: E731
    print(f"{n:30s} miss P {mp:4.0%} F {mf:4.0%}  AUC {auc:.3f}  planet[{f(qp)}]  fp[{f(qf)}]")
