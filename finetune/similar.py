"""Similar past cases: for each TOI, the 5 training TOIs whose checks look most alike, and what follow-up
found for each. They go into Glimmer's dossier prompt ("4 of the 5 most similar candidates were binaries").

Distance is over the System 1 features, each turned into its rank among training TOIs (so units and
outliers don't matter) and weighted by how much System 1 relies on it (XGBoost gain). Missing values
sit at the median rank. Neighbours come from train only, so nothing leaks into val or test; a train
TOI never gets itself or a TOI on its own star. The ranking and search live in pipeline/system1.py, which
finds the same cases for survey signals from the index written here.

Run: python -m finetune.similar [--name clues]   writes data/finetune/similar.json and the index
     (system1/similar-NAME.npz + .json), prints a vote check
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from finetune.system1 import FOLDER, OUT, load, matrix
from pipeline.system1 import SIMILAR_K, nearest, ranks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="clues", help="System 1 version whose features and gains to use")
    args = ap.parse_args()
    results = json.loads((OUT / f"results-{args.name}.json").read_text(encoding="utf-8"))
    names = results["features"]
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(OUT / f"xgb-{args.name}.json")
    full_gain = booster.get_score(importance_type="gain")
    weights = np.array([full_gain.get(n, 0.0) for n in names])
    weights = weights / weights.sum()

    rows = load(multis="other_tois_on_star" in names, groups=tuple(results.get("clues", ())))
    train = [r for r in rows if r["label"]["split"] == "train" and r["label"]["target"] in ("gate2", "reject", "flag")]
    tx = matrix(train, names)
    R_train = ranks(tx, tx)
    R_all = ranks(tx, matrix(rows, names))
    tics = np.array([r["tic"] for r in train])
    meta = [{"toi": r["toi"], "target": r["label"]["target"], "reason": r["label"].get("reason")} for r in train]
    np.savez_compressed(OUT / f"similar-{args.name}.npz", x=tx, weights=weights, tics=tics)
    (OUT / f"similar-{args.name}.json").write_text(json.dumps(meta), encoding="utf-8")

    out, votes = {}, []
    for r, v in zip(rows, R_all):
        idx, d = nearest(R_train, weights, v, tics == r["tic"])  # never itself or its own star
        out[r["toi"]] = [{"target": meta[i]["target"], "reason": meta[i]["reason"], "distance": round(float(d[i]), 4)}
                         for i in idx]
        if r["label"]["split"] == "test" and r["label"]["target"] in ("gate2", "reject"):
            labelled = [n for n in out[r["toi"]] if n["target"] in ("gate2", "reject")]
            share = np.mean([n["target"] == "gate2" for n in labelled]) if labelled else 0.5
            votes.append((r["label"]["target"] == "gate2", share))
    (FOLDER / "similar.json").write_text(json.dumps(out), encoding="utf-8")
    y, s = zip(*votes)
    print(f"{len(out)} TOIs; test check: share of planets among the {SIMILAR_K} neighbours has AUC {roc_auc_score(y, s):.3f} "
          f"on {len(votes)} test TOIs (System 1 alone: {results['test']['auc']:.3f})")
    top = sorted(zip(names, weights), key=lambda kv: -kv[1])[:6]
    print("distance weights, top:", ", ".join(f"{n} {w:.2f}" for n, w in top))


if __name__ == "__main__":
    main()
