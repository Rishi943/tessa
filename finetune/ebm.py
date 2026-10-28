"""EBM (Explainable Boosting Machine) as a System 1 challenger: one learned curve per check, plus a few
pairwise interactions, so every score is a sum of readable parts. Same features, splits, calibration and
conformal cut as XGBoost (finetune/system1.py), so finetune/compare.py can put them side by side.

Run: python -m finetune.ebm [--clues all] [--name ebm]
"""
import argparse
import json

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from finetune.system1 import CLUE_GROUPS, OUT, binary, load, matrix, scorecard, y


def fit(X, Y, names):
    return ExplainableBoostingClassifier(feature_names=names, interactions=10, random_state=3407, n_jobs=4).fit(X, Y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clues", default="all")
    ap.add_argument("--name", default="ebm")
    args = ap.parse_args()
    groups = CLUE_GROUPS if args.clues == "all" else tuple(g for g in args.clues.split(",") if g)
    rows = load(groups=groups)
    split = {s: [r for r in rows if r["label"]["split"] == s] for s in ("train", "val", "test")}
    names = list(rows[0]["x"])
    train = binary(split["train"])
    X, Y = matrix(train, names), y(train)
    model = fit(X, Y, names)

    oof = np.zeros(len(train))  # out-of-fold, as for XGBoost: no TOI scored by a model that saw its star
    for part, held in GroupKFold(5).split(X, Y, [r["tic"] for r in train]):
        oof[held] = fit(X[part], Y[part], names).decision_function(X[held])
    print(f"train {len(train)}, out-of-fold AUC {roc_auc_score(Y, oof):.4f}")

    raw = lambda rs: model.decision_function(matrix(rs, names))  # noqa: E731
    cal, results = scorecard(raw, split)
    scored = split["val"] + split["test"]
    margins = np.concatenate([oof, raw(scored)])
    preds = dict(zip((r["toi"] for r in train + scored), cal.predict_proba(margins.reshape(-1, 1))[:, 1].round(4).tolist()))
    importance = sorted(zip(model.term_names_, model.term_importances()), key=lambda t: -t[1])[:12]
    print("top terms:", ", ".join(f"{n} {v:.2f}" for n, v in importance))
    results.update({"features": names, "name": args.name, "clues": list(groups),
                    "terms": {n: round(float(v), 4) for n, v in importance}})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results-{args.name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (OUT / f"preds-{args.name}.json").write_text(json.dumps(preds), encoding="utf-8")


if __name__ == "__main__":
    main()
