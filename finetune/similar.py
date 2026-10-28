"""Similar past cases: for each TOI, the 5 training TOIs whose checks look most alike, and what follow-up
found for each. They go into Glimmer's dossier prompt ("4 of the 5 most similar candidates were binaries").

Distance is over the System 1 features, each turned into its rank among training TOIs (so units and
outliers don't matter) and weighted by how much System 1 relies on it (XGBoost gain). Missing values
sit at the median rank. Neighbours come from train only, so nothing leaks into val or test; a train
TOI never gets itself or a TOI on its own star.

Run: python -m finetune.similar [--name base]   writes data/finetune/similar.json, prints a vote check
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from finetune.system1 import FOLDER, OUT, load, matrix

K = 5


def ranks(train_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each column's value as its rank among the training values, 0..1; NaN -> 0.5."""
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        ref = np.sort(train_x[:, j][np.isfinite(train_x[:, j])])
        col = x[:, j]
        out[:, j] = np.where(np.isfinite(col), np.searchsorted(ref, col) / max(len(ref), 1), 0.5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="base", help="System 1 version whose features and gains to use")
    args = ap.parse_args()
    results = json.loads((OUT / f"results-{args.name}.json").read_text(encoding="utf-8"))
    names = results["features"]
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(OUT / f"xgb-{args.name}.json")
    full_gain = booster.get_score(importance_type="gain")
    weights = np.array([full_gain.get(n, 0.0) for n in names])
    weights = weights / weights.sum()

    rows = load(multis="other_tois_on_star" in names)
    train = [r for r in rows if r["label"]["split"] == "train" and r["label"]["target"] in ("gate2", "reject", "flag")]
    tx = matrix(train, names)
    R_train = ranks(tx, tx)
    R_all = ranks(tx, matrix(rows, names))
    tics = np.array([r["tic"] for r in train])

    out, votes = {}, []
    for r, v in zip(rows, R_all):
        d = np.sqrt(((R_train - v) ** 2 * weights).sum(axis=1))
        d[tics == r["tic"]] = np.inf  # never itself or its own star
        idx = np.argsort(d)[:K]
        out[r["toi"]] = [{"target": train[i]["label"]["target"], "reason": train[i]["label"].get("reason"),
                          "distance": round(float(d[i]), 4)} for i in idx]
        if r["label"]["split"] == "test" and r["label"]["target"] in ("gate2", "reject"):
            labelled = [n for n in out[r["toi"]] if n["target"] in ("gate2", "reject")]
            share = np.mean([n["target"] == "gate2" for n in labelled]) if labelled else 0.5
            votes.append((r["label"]["target"] == "gate2", share))
    (FOLDER / "similar.json").write_text(json.dumps(out), encoding="utf-8")
    y, s = zip(*votes)
    print(f"{len(out)} TOIs; test check: share of planets among the {K} neighbours has AUC {roc_auc_score(y, s):.3f} "
          f"on {len(votes)} test TOIs (System 1 alone: {results['test']['auc']:.3f})")
    top = sorted(zip(names, weights), key=lambda kv: -kv[1])[:6]
    print("distance weights, top:", ", ".join(f"{n} {w:.2f}" for n, w in top))


if __name__ == "__main__":
    main()
