"""Control: production XGBoost (same params, same grouped-CV tree count, same folds) given the hand ledger's
per-clue terms as extra inputs. Tells whether the hybrid's gain comes from the net or from the hand features.

Run: python xgb_control.py
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import hand
from evaluate import fps_at_loss

WORK = Path(__file__).resolve().parent / "work"
ROOT = Path(__file__).resolve().parents[2]
params = json.loads((ROOT / "data" / "finetune" / "system1" / "results-clues.json").read_text(encoding="utf-8"))["params"]


def inputs(X):
    return np.concatenate([X, hand.contributions(X)], axis=1)


tr = np.load(WORK / "train.npz")
X, y, g = inputs(tr["X"]), tr["y"], tr["tic"]
folds = list(GroupKFold(5).split(X, y, g))
cv = xgb.cv(params, xgb.DMatrix(X, y), num_boost_round=3000, early_stopping_rounds=50, folds=folds, seed=3407)
rounds = len(cv)
oof = np.zeros(len(y))
for fit, held in folds:
    m = xgb.train(params, xgb.DMatrix(X[fit], y[fit]), num_boost_round=rounds)
    oof[held] = m.predict(xgb.DMatrix(X[held]), output_margin=True)
full = xgb.train(params, xgb.DMatrix(X, y), num_boost_round=rounds)
margin = lambda s: full.predict(xgb.DMatrix(inputs(np.load(WORK / f"{s}.npz")["X"])), output_margin=True)  # noqa: E731
np.savez(WORK / "margins-xgb-hybrid.npz", train=oof, val=margin("val"), test=margin("test"))
print(f"xgb + hand terms: {rounds} trees, OOF AUC {roc_auc_score(y, oof):.4f}, FPs caught at 2% planet loss {fps_at_loss(y, oof):.3f}")
