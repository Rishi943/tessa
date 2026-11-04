"""Paired bootstrap on the test split (resampling host stars): is model B's AUC / catch rate / planet loss at
the CV+ 2% cut really different from XGBoost's? No model changes; read-only on frozen margins."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from evaluate import WORK, conformal_cut, load, margins

tr, va, te = load("train"), load("val"), load("test")
yt = np.load(WORK / "sealed" / "test_labels.npz")["y"]


def calibrated(name):
    m = margins(name)
    cal = LogisticRegression(C=1e6).fit(m["val"].reshape(-1, 1), va["y"])
    p = lambda z: cal.predict_proba(z.reshape(-1, 1))[:, 1]  # noqa: E731
    cut = conformal_cut(np.concatenate([p(m["train"])[tr["y"] == 1], p(m["val"])[va["y"] == 1]]), 0.02)
    return p(m["test"]), cut


def stats(y, p, cut):
    return roc_auc_score(y, p), (p[y == 0] < cut).mean(), (p[y == 1] < cut).mean()


hosts = np.unique(te["tic"])
rows = {h: np.where(te["tic"] == h)[0] for h in hosts}
rng = np.random.default_rng(3407)
draws = [np.concatenate([rows[h] for h in rng.choice(hosts, len(hosts))]) for _ in range(2000)]
base_p, base_cut = calibrated("xgb")
for name in ("nn", "nn-hybrid", "hand", "xgb-hybrid"):
    p, cut = calibrated(name)
    for label, sub in (("all test", np.ones(len(yt), bool)), ("agent-bound", te["agent_bound"]), ("K/M hosts", te["km_host"])):
        d = []
        for idx in draws:
            idx = idx[sub[idx]]
            a, b = stats(yt[idx], p[idx], cut), stats(yt[idx], base_p[idx], base_cut)
            d.append(np.subtract(a, b))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        point = np.subtract(stats(yt[sub], p[sub], cut), stats(yt[sub], base_p[sub], base_cut))
        print(f"{name:10s} - xgb, {label:11s}: AUC {point[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]  "
              f"FPs caught {point[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]  planets lost {point[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}]")
