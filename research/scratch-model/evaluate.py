"""The scorecard every scratch model goes through, the same steps as finetune/system1.py's.

A model gives margins (any score where higher = more planet-like) for train (out-of-fold if it is
trained; in-sample if it is hand-written), val and test. Then:
  Platt scaling fitted on val margins -> P(planet)
  conformal reject cuts: on val planets alone, and CV+ (train planets + val planets, as production)
  AUC, Brier, and planets lost / false positives caught at each cut.

Dev mode reports train (OOF) and val. --final opens the sealed test labels; run it once per frozen model.

Run: python research/scratch-model/evaluate.py MODEL_NAME [--final]
     (reads work/margins-MODEL_NAME.npz with keys train, val, test; 'xgb' is the production baseline)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
ROOT = HERE.parents[1]
ALPHAS = (0.01, 0.02, 0.05)


def load(split: str) -> dict:
    return dict(np.load(WORK / f"{split}.npz"))


def conformal_cut(planet_scores, alpha):
    s = np.sort(planet_scores)
    k = int(np.floor((len(s) + 1) * alpha))
    return float(s[k - 1]) if k >= 1 else 0.0


def certified(n, alpha, confidence=0.95):
    k = int(np.floor((n + 1) * alpha))
    return float(beta.ppf(confidence, k, n + 1 - k)) if k >= 1 else 0.0


def xgb_margins() -> dict:
    """Production System 1's calibrated P (train out-of-fold, val, test) as logits."""
    preds = json.loads((ROOT / "data" / "finetune" / "system1" / "preds-clues.json").read_text(encoding="utf-8"))
    out = {}
    for s in ("train", "val", "test"):
        p = np.clip(np.array([preds[t] for t in load(s)["toi"]]), 1e-6, 1 - 1e-6)
        out[s] = np.log(p / (1 - p))
    return out


def margins(name: str) -> dict:
    if name == "xgb":
        return xgb_margins()
    return dict(np.load(WORK / f"margins-{name}.npz"))


def block(y, p, code, cuts) -> dict:
    out = {"n": int(len(y)), "planets": int(y.sum()), "auc": round(roc_auc_score(y, p), 4),
           "brier": round(brier_score_loss(y, p), 4)}
    for label, c in cuts.items():
        drop = p < c
        out[f"{label}: planets lost"] = round(float(drop[y == 1].mean()), 3)
        out[f"{label}: fps caught"] = round(float(drop[y == 0].mean()), 3)
        both = code | drop
        out[f"{label}+rules: planets lost"] = round(float(both[y == 1].mean()), 3)
        out[f"{label}+rules: fps caught"] = round(float(both[y == 0].mean()), 3)
    return out


def fps_at_loss(y, s, loss=0.02):
    """Share of false positives below the score that loses `loss` of the planets (threshold-free, for dev)."""
    c = np.quantile(s[y == 1], loss)
    return float((s[y == 0] < c).mean())


def scorecard(name: str, final: bool = False) -> dict:
    m = margins(name)
    tr, va = load("train"), load("val")
    cal = LogisticRegression(C=1e6).fit(m["val"].reshape(-1, 1), va["y"])
    prob = lambda z: cal.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]  # noqa: E731
    p_tr, p_va = prob(m["train"]), prob(m["val"])
    cuts = {}
    for a in ALPHAS:
        cuts[f"val {a:.0%}"] = conformal_cut(p_va[va["y"] == 1], a)
        cuts[f"CV+ {a:.0%}"] = conformal_cut(np.concatenate([p_tr[tr["y"] == 1], p_va[va["y"] == 1]]), a)
    res = {"model": name, "cuts": {k: round(v, 4) for k, v in cuts.items()},
           "train (OOF) auc": round(roc_auc_score(tr["y"], m["train"]), 4),
           "train (OOF) fps caught at 2% planet loss": round(fps_at_loss(tr["y"], m["train"]), 3),
           "val": block(va["y"], p_va, va["code_reject"], cuts)}
    if final:
        te = load("test")
        sealed = np.load(WORK / "sealed" / "test_labels.npz")
        assert (sealed["toi"] == te["toi"]).all()
        yt, pt = sealed["y"], prob(m["test"])
        res["test"] = block(yt, pt, te["code_reject"], cuts)
        res["test, agent-bound"] = block(yt[te["agent_bound"]], pt[te["agent_bound"]], te["code_reject"][te["agent_bound"]], cuts)
        res["test, K/M hosts"] = block(yt[te["km_host"]], pt[te["km_host"]], te["code_reject"][te["km_host"]], cuts)
        log = WORK / "final_runs.log"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model": name}) + "\n")
    (WORK / f"results-{name}{'-final' if final else ''}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def show(res: dict) -> None:
    print(f"== {res['model']}: train OOF AUC {res['train (OOF) auc']}, "
          f"FPs caught at 2% planet loss (train OOF) {res['train (OOF) fps caught at 2% planet loss']}")
    for part in ("val", "test", "test, agent-bound", "test, K/M hosts"):
        if part in res:
            b = res[part]
            print(f"  {part:18s} n {b['n']:3d}  AUC {b['auc']:.4f}  Brier {b['brier']:.4f}  "
                  f"CV+2%: lost {b['CV+ 2%: planets lost']:.3f} caught {b['CV+ 2%: fps caught']:.3f}  "
                  f"val2%: lost {b['val 2%: planets lost']:.3f} caught {b['val 2%: fps caught']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    for n in a.names:
        show(scorecard(n, a.final))
    sys.exit(0)
