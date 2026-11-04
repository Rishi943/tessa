"""Score the survey's 2,698 dips with the frozen net and compare with production XGBoost. Read-only on the pipeline.

1. Rebuild the frozen net (deterministic: same data, config, seeds) and check it reproduces margins-nn.npz.
2. Rebuild each dip's 65 inputs exactly as pipeline.system1.score does, and check XGBoost on them reproduces the
   P(planet) stored in the vet file (so both models see the same inputs).
3. Net P(planet): Platt on val, as evaluate.py; net rank: its place among the calibration planets (train
   out-of-fold + val), as production's planet_rank.

Run from anywhere: python research/scratch-model/survey.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import xgboost as xgb  # noqa: E402

from nn import ensemble  # noqa: E402
from pipeline import system1 as s1  # noqa: E402
from pipeline.paths import DATA, batches  # noqa: E402

WORK = HERE / "work"
NAMES = json.loads((WORK / "features.json").read_text(encoding="utf-8"))
CANDIDATES = {9066956: "likely", 71503196: "likely", 116046773: "likely", 166321785: "likely",
              321290256: "nearby open", 50099421: "nearby open", 61220826: "nearby open", 79490453: "nearby open",
              458191673: "known (pre-hackathon)", 426450210: "flagged (FPP 1.0)", 376928980: "flagged (no FPP)"}


def row(c: dict) -> np.ndarray:
    clue = c.get("clues") or {}
    x = s1.features(c)
    x.update(s1.clue_features({"checks": c}, clue, clue.get("gaia") or {}, clue.get("eb") or {}, s1.CLUE_GROUPS))
    return np.array([np.nan if x.get(n) is None else float(x[n]) for n in NAMES])


def main() -> None:
    cfg = json.loads((WORK / "config-nn.json").read_text(encoding="utf-8"))
    tr, va = np.load(WORK / "train.npz"), np.load(WORK / "val.npz")
    net = ensemble(tr["X"], tr["y"].astype(float), tr["tic"], cfg, seed=0)
    frozen = np.load(WORK / "margins-nn.npz")
    drift = float(np.max(np.abs(net(va["X"]) - frozen["val"])))
    print(f"rebuilt net vs frozen val margins: max difference {drift:.2e}")
    assert drift < 1e-6, "rebuilt net differs from the frozen one"

    cal = LogisticRegression(C=1e6).fit(frozen["val"].reshape(-1, 1), va["y"])
    prob = lambda z: cal.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]  # noqa: E731
    planets = np.sort(np.concatenate([prob(frozen["train"])[tr["y"] == 1], prob(frozen["val"])[va["y"] == 1]]))

    status = {}
    for b in batches():
        f = DATA / "vet" / f"{b}_v10" / "status.csv"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                status |= {int(r["tic"]): r["status"] for r in csv.DictReader(fh)}
    m = s1.model()
    rows, X = [], []
    for b in batches():
        for p in sorted((DATA / "vet" / f"{b}_v10").glob("[0-9]*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            x = row(c)
            X.append(x)
            margin = float(m["booster"].predict(xgb.DMatrix(x[None], feature_names=m["names"]), output_margin=True)[0])
            p_x = 1 / (1 + math.exp(-(m["a"] * margin + m["b"])))
            stored = (c.get("system1") or {}).get("p_planet")
            rows.append({"tic": int(c["tic"]), "batch": b, "code": c.get("code_verdict"), "status": status.get(int(c["tic"]), ""),
                         "xgb_p_stored": stored, "xgb_p": round(p_x, 4),
                         "xgb_rank": round(float(np.searchsorted(m["planets"], p_x) / len(m["planets"])), 3)})
    X = np.array(X)
    assert m["names"] == NAMES
    have = [r for r in rows if r["xgb_p_stored"] is not None]
    err = np.abs(np.array([r["xgb_p_stored"] - r["xgb_p"] for r in have]))
    print(f"inputs rebuilt for {len(rows)} dips ({len(rows) - len(have)} have no stored System 1 score); XGBoost on the "
          f"rebuilt inputs vs stored P: max difference {err.max():.4f}, {(err > 0.001).sum()} of {len(have)} differ by > 0.001")
    nostore = [r for r in rows if r["xgb_p_stored"] is None]
    print("  no stored score, by code verdict:", {v: sum(r["code"] == v for r in nostore) for v in {r["code"] for r in nostore}})
    pn = prob(net(X))
    for r, p in zip(rows, pn):
        r["nn_p"] = round(float(p), 4)
        r["nn_rank"] = round(float(np.searchsorted(planets, p) / len(planets)), 3)
    with open(WORK / "survey_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    xr, nr = np.array([r["xgb_rank"] for r in rows]), np.array([r["nn_rank"] for r in rows])
    print(f"\nrank agreement (Spearman) across all dips: {spearmanr(xr, nr).statistic:.3f}")
    for lo, hi, label in ((0, 0.05, "bottom 5% (the floor)"), (0.05, 0.32, "0.05-0.32"), (0.32, 1.01, ">= 0.32 (v9 pass rank)")):
        a, b_ = (xr >= lo) & (xr < hi), (nr >= lo) & (nr < hi)
        print(f"  {label:24s} XGBoost {a.sum():5d}  net {b_.sum():5d}  both {(a & b_).sum():5d}")
    print("\nby gate-2 status (median rank, XGBoost / net):")
    for s in sorted({r["status"] for r in rows}):
        sub = [r for r in rows if r["status"] == s]
        print(f"  {s or '(not at gate 2)':18s} n {len(sub):5d}  {np.median([r['xgb_rank'] for r in sub]):.3f} / "
              f"{np.median([r['nn_rank'] for r in sub]):.3f}")
    print("\nsent to the agent (code verdict 'agent'), floor check:")
    ag = [r for r in rows if r["code"] == "agent"]
    print(f"  {len(ag)} dips; below the 5% floor: XGBoost {sum(r['xgb_rank'] < 0.05 for r in ag)}, net {sum(r['nn_rank'] < 0.05 for r in ag)}")
    print("\nnamed stars (best-ranked dip per star):")
    for tic, label in CANDIDATES.items():
        sub = [r for r in rows if r["tic"] == tic]
        if not sub:
            print(f"  TIC {tic:>10} ({label}): not in the vet files")
            continue
        r = max(sub, key=lambda r: r["xgb_rank"])
        print(f"  TIC {tic:>10} {label:22s} XGBoost P {r['xgb_p']:.3f} rank {r['xgb_rank']:.3f} | net P {r['nn_p']:.3f} rank {r['nn_rank']:.3f}"
              f"  [{r['status']}]")


if __name__ == "__main__":
    main()
