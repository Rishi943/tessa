"""Train-only CV sweep for the numpy net: each config over 3 seeds, mean and spread of OOF AUC and catch rate."""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate import fps_at_loss
from nn import DEFAULT, WORK, oof

CONFIGS = {
    "default": {},
    "wide": dict(hidden=[128, 64]),
    "narrow": dict(hidden=[32, 16]),
    "one-layer": dict(hidden=[64]),
    "three-layer": dict(hidden=[64, 64, 32]),
    "more-dropout": dict(dropout=0.35, in_dropout=0.2),
    "less-dropout": dict(dropout=0.1, in_dropout=0.0),
    "strong-l2": dict(l2=1e-2),
    "noise": dict(noise=0.1),
    "hybrid": dict(hand=True),
    "hybrid-dropout": dict(hand=True, dropout=0.35, in_dropout=0.2),
    "max-dropout": dict(dropout=0.5, in_dropout=0.3),
    "dropout-wide": dict(hidden=[128, 64], dropout=0.35, in_dropout=0.2),
}
pick = sys.argv[1:] or list(CONFIGS)
tr = np.load(WORK / "train.npz")
X, y, g = tr["X"], tr["y"].astype(float), tr["tic"]
for name in pick:
    aucs, fps = [], []
    for seed in range(3):
        cfg = dict(DEFAULT, **CONFIGS[name], seed=seed)
        o = oof(X, y, g, cfg)
        aucs.append(roc_auc_score(y, o))
        fps.append(fps_at_loss(y, o))
    print(f"{name:14s} AUC {np.mean(aucs):.4f} +- {np.std(aucs):.4f}   caught {np.mean(fps):.3f} +- {np.std(fps):.3f}", flush=True)
