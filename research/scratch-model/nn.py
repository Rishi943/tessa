"""Arm 2: a neural network written from scratch in numpy (no ML framework), trained by gradient descent.

Inputs: each check mapped through its train-fold empirical CDF to [-1, 1] (heavy tails and odd scales stop
mattering), 0 when missing, plus a missing flag for every check that is ever missing.
Net: an MLP with SiLU, dropout and weight decay, trained with Adam on log loss. Each member early-stops on a
grouped 15% holdout of its training rows; an ensemble of members (different seeds and holdouts) is averaged
in logit space.

Dev protocol (same as XGBoost's): 5-fold GroupKFold by host star on train gives out-of-fold margins; the
model for val and test is the ensemble trained on all of train. Val and test only enter evaluate.py.

Run: python nn.py NAME [--config JSON]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from evaluate import fps_at_loss

WORK = Path(__file__).resolve().parent / "work"
DEFAULT = dict(hidden=[64, 32], dropout=0.2, in_dropout=0.1, l2=1e-3, lr=3e-3, batch=128, epochs=300, patience=40,
               members=8, noise=0.0)


class Prep:
    """Train-fold ECDF per column -> [-1, 1]; NaN -> 0 with a missing flag."""

    def __init__(self, X, hand=False):
        self.hand = hand
        self.grid = []
        for j in range(X.shape[1]):
            v = np.sort(X[:, j][np.isfinite(X[:, j])])
            q = np.unique(np.quantile(v, np.linspace(0, 1, 101))) if len(v) else np.array([0.0])
            self.grid.append(q)
        self.flags = np.where(np.isnan(X).any(axis=0))[0]

    def __call__(self, X):
        base = self.transform(X)
        if self.hand:  # the hand ledger's per-clue log-odds as extra inputs (hybrid arm)
            import hand
            base = np.concatenate([base, hand.contributions(X) / 2], axis=1)
        return base

    def transform(self, X):
        out = np.zeros(X.shape)
        for j, q in enumerate(self.grid):
            col = X[:, j]
            ok = np.isfinite(col)
            if len(q) > 1:
                out[ok, j] = np.interp(col[ok], q, np.linspace(-1, 1, len(q)))
        return np.concatenate([out, np.isnan(X[:, self.flags]).astype(float)], axis=1)


def silu(x):
    s = 1 / (1 + np.exp(-x))
    return x * s, s * (1 + x * (1 - s))


class MLP:
    def __init__(self, n_in, hidden, rng):
        sizes = [n_in] + hidden + [1]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X, rng=None, dropout=0.0, in_dropout=0.0):
        cache, h = [], X
        if rng is not None and in_dropout:
            h = h * (rng.random(h.shape) > in_dropout) / (1 - in_dropout)
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            a, da = silu(z)
            m = (rng.random(a.shape) > dropout) / (1 - dropout) if rng is not None and dropout else 1.0
            cache.append((h, da, m))
            h = a * m
        cache.append((h, None, None))
        return (h @ self.W[-1] + self.b[-1])[:, 0], cache

    def backward(self, dz, cache):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = dz[:, None]
        for i in range(len(self.W) - 1, -1, -1):
            h, _, _ = cache[i]
            gW[i], gb[i] = h.T @ g, g.sum(axis=0)
            if i:
                _, da, m = cache[i - 1]
                g = (g @ self.W[i].T) * m * da
        return gW, gb


def train_member(Xt, yt, Xh, yh, cfg, seed):
    rng = np.random.default_rng(seed)
    net = MLP(Xt.shape[1], cfg["hidden"], rng)
    params = net.W + net.b
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    best, best_state, since, t = np.inf, None, 0, 0
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(yt))
        for s in range(0, len(order), cfg["batch"]):
            idx = order[s:s + cfg["batch"]]
            xb = Xt[idx] + (rng.normal(0, cfg["noise"], Xt[idx].shape) if cfg["noise"] else 0)
            z, cache = net.forward(xb, rng, cfg["dropout"], cfg["in_dropout"])
            dz = (1 / (1 + np.exp(-z)) - yt[idx]) / len(idx)
            gW, gb = net.backward(dz, cache)
            grads = [g + cfg["l2"] * w for g, w in zip(gW, net.W)] + gb
            t += 1
            for k, (p, g) in enumerate(zip(params, grads)):
                m1[k] = 0.9 * m1[k] + 0.1 * g
                m2[k] = 0.999 * m2[k] + 0.001 * g * g
                p -= cfg["lr"] * (m1[k] / (1 - 0.9 ** t)) / (np.sqrt(m2[k] / (1 - 0.999 ** t)) + 1e-8)
        zh, _ = net.forward(Xh)
        loss = np.mean(np.logaddexp(0, zh) - yh * zh)
        if loss < best - 1e-5:
            best, since = loss, 0
            best_state = ([w.copy() for w in net.W], [b.copy() for b in net.b])
        else:
            since += 1
            if since > cfg["patience"]:
                break
    net.W, net.b = best_state
    return net


def ensemble(X, y, groups, cfg, seed=0):
    """Fit Prep and cfg['members'] nets on (X, y); return a margin function."""
    prep = Prep(X, cfg.get("hand", False))
    P = prep(X)
    nets = []
    for k in range(cfg["members"]):
        split = GroupShuffleSplit(1, test_size=0.15, random_state=seed * 100 + k)
        tr, ho = next(split.split(P, y, groups))
        nets.append(train_member(P[tr], y[tr], P[ho], y[ho], cfg, seed * 100 + k))
    return lambda Xn: np.mean([n.forward(prep(Xn))[0] for n in nets], axis=0)


def oof(X, y, groups, cfg):
    out = np.zeros(len(y))
    for f, (fit, held) in enumerate(GroupKFold(5).split(X, y, groups)):
        out[held] = ensemble(X[fit], y[fit], groups[fit], cfg, seed=f + 1 + 10 * cfg.get("seed", 0))(X[held])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--cv-only", action="store_true")
    a = ap.parse_args()
    cfg = dict(DEFAULT, **json.loads(a.config))
    tr = np.load(WORK / "train.npz")
    X, y, g = tr["X"], tr["y"].astype(float), tr["tic"]
    o = oof(X, y, g, cfg)
    print(f"{a.name}: OOF AUC {roc_auc_score(y, o):.4f}, FPs caught at 2% planet loss {fps_at_loss(y, o):.3f}  {json.dumps(cfg)}")
    if a.cv_only:
        return
    full = ensemble(X, y, g, cfg, seed=0)
    np.savez(WORK / f"margins-{a.name}.npz", train=o, val=full(np.load(WORK / "val.npz")["X"]),
             test=full(np.load(WORK / "test.npz")["X"]))
    (WORK / f"config-{a.name}.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
