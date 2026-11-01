"""Can Glimmer v10 and System 1 together beat ExoMiner++? Chosen on dev, then frozen, then test once.

Reads the rows of finetune/exo_bench.py, which runs no model. Dev is val + the train signals not in Glimmer's training rows; the
rows themselves are left out, as they were filtered to signals Spark Open already got right. Two candidates:
  floor  Tessa's pass with the rank floor raised: Glimmer says gate2 and System 1's planet rank is at least the floor
  blend  logistic regression on the verdict (pass, stopped short, rejected) and System 1's log-odds, cut at a threshold
Each is set on dev to recover at least as many planets as ExoMiner++ at 0.5 with the fewest false positives, the blend's numbers
from 5-fold cross-validation. `dev` writes the choice to data/finetune/blend_frozen.json. `test` reads that file, refits nothing
but the blend on all of dev, and prints test against ExoMiner++ at its 0.5 threshold.

Run: python -m finetune.blend dev
     python -m finetune.blend test
"""
import argparse
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from finetune.exo_bench import auc_ci, ci
from finetune.valgate import val_signals
from pipeline import system1
from pipeline.paths import DATA

FROZEN = DATA / "finetune" / "blend_frozen.json"
FLOORS = (0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.32, 0.4, 0.5)
SEED = 20261101


def load() -> list[dict]:
    rows = json.loads((DATA / "finetune" / "exo_bench.json").read_text(encoding="utf-8"))
    planets = system1.model()["planets"]
    for r in rows:
        r["rank"] = float(np.searchsorted(planets, r["s1"]) / len(planets))
    return rows


def x(rows: list[dict]) -> np.ndarray:
    logit = [np.log(max(r["s1"], 1e-6) / max(1 - r["s1"], 1e-6)) for r in rows]
    return np.array([[r["tier"] == 2, r["tier"] == 1, v] for r, v in zip(rows, logit)], float)


def y(rows: list[dict]) -> np.ndarray:
    return np.array([r["planet"] for r in rows], int)


def rates(passed: np.ndarray, truth: np.ndarray) -> tuple[int, int]:
    return int(passed[truth == 1].sum()), int(passed[truth == 0].sum())


def lowest_leak(score: np.ndarray, truth: np.ndarray, want: int) -> float:
    """The highest cut that still recovers `want` planets, so the fewest false positives at that recovery."""
    return float(np.sort(score[truth == 1])[::-1][want - 1])


def dev() -> None:
    rows = [r for r in load() if r["split"] == "val" or (r["split"] == "train" and not r["trained"])]
    t, exo = y(rows), np.array([r["exo"] for r in rows])
    gp = np.array([r["tier"] == 2 or r["tier"] == 1 for r in rows])
    rank = np.array([r["rank"] for r in rows])
    want, leak = rates(exo, t)
    print(f"dev: {len(rows)} TOIs, {t.sum()} planets, {(1 - t).sum()} FPs; ExoMiner++ {want} planets, {leak} FPs "
          f"({leak / (1 - t).sum():.1%}), AUC {roc_auc_score(t, [r['exo_score'] for r in rows]):.3f}")
    print(f"System 1 AUC {roc_auc_score(t, [r['s1'] for r in rows]):.3f}")
    for f in FLOORS:
        a, b = rates(np.array([r["tessa"] for r in rows]) & (rank >= f), t)
        print(f"  floor {f:<5} planets {a:>3}  FPs {b:>3}")
    oof = np.zeros(len(rows))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x(rows), t):
        oof[te] = LogisticRegression(max_iter=2000).fit(x(rows)[tr], t[tr]).predict_proba(x(rows)[te])[:, 1]
    cut = lowest_leak(oof, t, want)
    a, b = rates(oof >= cut, t)
    print(f"blend (cross-validated): AUC {roc_auc_score(t, oof):.3f}; at ExoMiner++'s {want} planets, cut {cut:.3f}: {a} planets, {b} FPs")
    s1cut = lowest_leak(np.array([r["s1"] for r in rows]), t, want)
    a, b = rates(np.array([r["s1"] for r in rows]) >= s1cut, t)
    print(f"System 1 alone at the same planets: cut {s1cut:.3f}: {a} planets, {b} FPs")
    lr = LogisticRegression(max_iter=2000).fit(x(rows), t)
    print("blend weights: pass, stopped short, System 1 log-odds =", np.round(lr.coef_[0], 2), "intercept", round(float(lr.intercept_[0]), 2))
    FROZEN.write_text(json.dumps({"cut": cut, "s1_cut": s1cut, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                                  "dev": len(rows)}), encoding="utf-8")


def test() -> None:
    fz = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows = [r for r in load() if r["split"] == "test"]
    t = y(rows)
    score = x(rows) @ np.array(fz["coef"]) + fz["intercept"]
    p = 1 / (1 + np.exp(-score))
    em = np.array([r["exo_score"] for r in rows])
    s1 = np.array([r["s1"] for r in rows])
    rng = np.random.default_rng(SEED)
    print(f"test: {len(rows)} TOIs, {t.sum()} planets, {(1 - t).sum()} FPs")
    for name, passed in (("blend", p >= fz["cut"]), ("System 1 alone", s1 >= fz["s1_cut"]), ("ExoMiner++ at 0.5", em >= 0.5)):
        a, b = rates(passed, t)
        print(f"  {name:18} planets {a:>3}/{t.sum()} ({a / t.sum():4.0%})  FPs {b:>3}/{(1 - t).sum()} ({b / (1 - t).sum():5.1%})")
    print(f"  AUC blend {roc_auc_score(t, p):.3f}, System 1 {roc_auc_score(t, s1):.3f}, ExoMiner++ {roc_auc_score(t, em):.3f}")
    lo, hi = auc_ci(t, p, em, rng)
    passed, e = (p >= fz["cut"]).astype(float), (em >= 0.5).astype(float)
    rp, rf = ci(passed[t == 1], e[t == 1], rng), ci(passed[t == 0], e[t == 0], rng)
    print(f"  blend minus ExoMiner++: AUC {roc_auc_score(t, p) - roc_auc_score(t, em):+.3f} [{lo:+.3f}, {hi:+.3f}], "
          f"planets {passed[t == 1].mean() - e[t == 1].mean():+.1%} [{rp[0]:+.1%}, {rp[1]:+.1%}], "
          f"FPs {passed[t == 0].mean() - e[t == 0].mean():+.1%} [{rf[0]:+.1%}, {rf[1]:+.1%}]")


def scorer() -> None:
    """System 1 decides, code rejects stand: the cut comes from dev, the fewest FPs at ExoMiner++'s planets; then test, once."""
    code = {s["toi"]: not s["reaches"] for sp in ("train", "val", "test") for s in val_signals(sp)}
    rows = load()
    devs = [r for r in rows if r["split"] == "val" or (r["split"] == "train" and not r["trained"])]
    t = y(devs)
    s1 = np.array([r["s1"] for r in devs])
    ok = ~np.array([code[r["toi"]] for r in devs])
    want = rates(np.array([r["exo"] for r in devs]), t)[0]
    cut = max(c for c in np.unique(s1) if rates((s1 >= c) & ok, t)[0] >= want)
    print(f"cut from dev: P(planet) >= {cut:.4f}: {rates((s1 >= cut) & ok, t)} planets, FPs; ExoMiner++ {rates(np.array([r['exo'] for r in devs]), t)}")
    rng = np.random.default_rng(SEED)
    with open(DATA / "finetune" / "exominer" / "matched_FFI.csv", encoding="utf-8") as f:
        ffi = {r["toi"]: float(r["score"]) for r in csv.DictReader(f)}
    test_rows = [r for r in rows if r["split"] == "test"]
    for name, rs, exo in (("test, ExoMiner++ both runs", test_rows, {r["toi"]: r["exo_score"] for r in test_rows}),
                          ("test, ExoMiner++ full-frame run", [r for r in test_rows if r["toi"] in ffi], ffi)):
        t = y(rs)
        p = np.array([r["s1"] >= cut and not code[r["toi"]] for r in rs], float)
        e = np.array([exo[r["toi"]] >= 0.5 for r in rs], float)
        a, b = rates(p > 0, t)
        c, d = rates(e > 0, t)
        rp, rf = ci(p[t == 1], e[t == 1], rng), ci(p[t == 0], e[t == 0], rng)
        print(f"{name}: {len(rs)} TOIs, {t.sum()} planets, {(1 - t).sum()} FPs")
        print(f"  scorer decides  planets {a}/{t.sum()} ({a / t.sum():.0%})  FPs {b}/{(1 - t).sum()} ({b / (1 - t).sum():.1%})")
        print(f"  ExoMiner++      planets {c}/{t.sum()} ({c / t.sum():.0%})  FPs {d}/{(1 - t).sum()} ({d / (1 - t).sum():.1%})")
        print(f"  difference      planets {(a - c) / t.sum():+.1%} [{rp[0]:+.1%}, {rp[1]:+.1%}]  FPs {(b - d) / (1 - t).sum():+.1%} [{rf[0]:+.1%}, {rf[1]:+.1%}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["dev", "test", "scorer"])
    {"dev": dev, "test": test, "scorer": scorer}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
