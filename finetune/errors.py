"""Where the vetting goes wrong and what could still help, on train out-of-fold + val (test untouched).

  1. Rejections by reason: what each reject rule costs alone (signals with no other reason to reject).
  2. The rule table against System 1's rank alone: planets passed at the same number of FPs passed.
  3. Do the table's rows add anything to the rank? Grouped 5-fold logistic stack, planets vs FPs.
  4. Learning curve: System 1 on a share of its training stars, for whether more labels would help.

Run: python -m finetune.errors [--curve]
"""
import argparse
import json
import re
from collections import Counter

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from pipeline.gate2 import table_rows
from pipeline.paths import DATA

from . import system1 as s1

FOLDER = DATA / "finetune"
KIND = {"gate2": "planet", "reject": "FP", "flag": "APC"}
ROWS = ("sectors", "size", "odd_even", "centroid", "neighbours", "gaps", "secondary", "shape", "rotation", "phase_curve",
        "duration", "refit", "known")


def records() -> list[dict]:
    out = []
    for path in (FOLDER / "vet_s1").glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["label"]["split"] in ("train", "val"):
            d["kind"] = KIND[d["label"]["target"]]
            d["rank"] = (d["checks"].get("system1") or {}).get("planet_rank") or 0
            d["rows"] = {x.split(":")[0] for x in table_rows(d["checks"], d.get("evidence"))}  # the table alone, not v9's rank policy
            out.append(d)
    return out


def why(reason: str) -> str:
    if reason.startswith("System 1"):
        return "System 1 cut"
    if reason.startswith("the dip comes from Gaia"):
        return "centroid on a Gaia star"
    return re.sub(r"[\d.,]+", "#", reason)[:34]


def rejections(recs: list[dict]) -> None:
    print("1. rejections: any reason / the only reason")
    every, alone = Counter(), Counter()
    for d in recs:
        if d["checks"]["code_verdict"] == "reject":
            reasons = {why(r) for r in d["checks"]["reasons"]}
            for r in reasons:
                every[(r, d["kind"])] += 1
                if len(reasons) == 1:
                    alone[(r, d["kind"])] += 1
    for r in sorted({r for r, _ in every}, key=lambda r: -every[(r, "planet")] - every[(r, "FP")])[:12]:
        print(f"  {r:<36} any: planet {every[(r, 'planet')]:>3} FP {every[(r, 'FP')]:>3} APC {every[(r, 'APC')]:>3} | "
              f"only: planet {alone[(r, 'planet')]:>3} FP {alone[(r, 'FP')]:>3} APC {alone[(r, 'APC')]:>3}")


def table_vs_rank(recs: list[dict]) -> None:
    print("\n2. passed: the table, or rank alone at the same FPs passed (signals past the code rejections)")
    for split in (("train", "val"), ("train",), ("val",)):
        rs = [d for d in recs if d["label"]["split"] in split and d["checks"]["code_verdict"] != "reject"]
        count = lambda keep, k: sum(keep(d) and d["kind"] == k for d in rs)  # noqa: E731
        table = lambda d: not d["rows"]  # noqa: E731
        print(f"  {'+'.join(split)}: table: planet {count(table, 'planet')} FP {count(table, 'FP')} APC {count(table, 'APC')} "
              f"of planet {sum(d['kind'] == 'planet' for d in rs)} FP {sum(d['kind'] == 'FP' for d in rs)}")
        fps = sorted((d["rank"] for d in rs if d["kind"] == "FP"), reverse=True)
        cut = fps[min(count(table, "FP"), len(fps) - 1)]
        by_rank = lambda d, c=cut: d["rank"] > c  # noqa: E731
        print(f"    rank > {cut:.3f}: planet {count(by_rank, 'planet')} FP {count(by_rank, 'FP')} APC {count(by_rank, 'APC')}")
        for c in (0.3, 0.4, 0.5, 0.6):
            both = lambda d, c=c: not d["rows"] or d["rank"] >= c  # noqa: E731
            print(f"    table, or rank >= {c}: planet {count(both, 'planet')} FP {count(both, 'FP')} APC {count(both, 'APC')}")
        for row in ("centroid", "gaps", "secondary", "neighbours"):
            drop = lambda d, row=row: not (d["rows"] - {row})  # noqa: E731
            print(f"    table without {row:<10}: planet {count(drop, 'planet')} FP {count(drop, 'FP')} APC {count(drop, 'APC')}")


def policy(recs: list[dict], cut: float = 0.32) -> None:
    """A candidate policy against today's: rank alone passes, and the rejections that are only a centroid or a
    phase curve become flags unless the rank passes them."""
    print(f"\n2b. today vs rank >= {cut} passes, centroid/phase-curve-only rejections no longer reject")
    def now(d):
        return "reject" if d["checks"]["code_verdict"] == "reject" else "flag" if d["rows"] else "pass"

    def new(d):
        hard = d["checks"]["code_verdict"] == "reject" and {why(r) for r in d["checks"]["reasons"]} - {
            "centroid on a Gaia star", "the star varies by # ppm at this p"}
        return "reject" if hard else "pass" if d["rank"] >= cut else "flag"

    for split in (("train", "val"), ("train",), ("val",)):
        rs = [d for d in recs if d["label"]["split"] in split]
        for name, f in (("today", now), ("new", new)):
            c = Counter((d["kind"], f(d)) for d in rs)
            print(f"  {'+'.join(split):<10}{name:<6}" + "  ".join(
                f"{k}: pass {c[(k, 'pass')]} flag {c[(k, 'flag')]} reject {c[(k, 'reject')]}" for k in ("planet", "FP", "APC")))
    only = [d for d in recs if d["checks"]["code_verdict"] == "reject" and not (
        {why(r) for r in d["checks"]["reasons"]} - {"centroid on a Gaia star", "the star varies by # ppm at this p"})]
    print("  rule-only rejects, rank quantiles (10,50,90):", {k: np.round(np.percentile(
        [d["rank"] for d in only if d["kind"] == k], [10, 50, 90]), 2).tolist() for k in ("planet", "FP")})


def stack(recs: list[dict]) -> None:
    rs = [d for d in recs if d["checks"]["code_verdict"] != "reject" and d["kind"] != "APC"]
    y = np.array([d["kind"] == "planet" for d in rs])
    rank = np.array([d["rank"] for d in rs]).clip(1e-3, 1 - 1e-3)
    logit = np.log(rank / (1 - rank))
    rows = np.array([[r in d["rows"] for r in ROWS] for d in rs], float)
    hosts = [d["tic"] for d in rs]
    both, alone = np.zeros(len(rs)), np.zeros(len(rs))
    for fit, held in GroupKFold(5).split(rows, y, hosts):
        X = np.column_stack([logit, rows])
        both[held] = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit], y[fit]).decision_function(X[held])
        alone[held] = logit[held]
    print(f"\n3. past the code rejections, planets vs FPs ({y.sum()} / {(~y).sum()}): AUC rank alone "
          f"{roc_auc_score(y, alone):.4f}, rank + table rows {roc_auc_score(y, both):.4f}")
    m = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack([logit, rows]), y)
    print("   log-odds per row given the rank:", {r: round(float(c), 2) for r, c in zip(ROWS, m.coef_[0][1:]) if rows[:, ROWS.index(r)].sum()})


def curve() -> None:
    """Each share of the training stars, scored on one held-out fold of stars + val, repeated over folds and seeds."""
    rows = s1.load(False, s1.CLUE_GROUPS)
    train = s1.binary([r for r in rows if r["label"]["split"] == "train"])
    val = s1.binary([r for r in rows if r["label"]["split"] == "val"])
    names = list(rows[0]["x"])
    X, Y, hosts = s1.matrix(train, names), s1.y(train), np.array([r["tic"] for r in train])
    Xv, Yv = s1.matrix(val, names), s1.y(val)
    params = json.loads((s1.OUT / "params.json").read_text(encoding="utf-8"))
    print(f"\n4. learning curve: {len(train)} train TOIs, {len(names)} inputs")
    for share in (0.25, 0.5, 0.75, 1.0):
        cv, on_val, caught = [], [], []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            for fit, held in GroupKFold(5).split(X, Y, hosts):
                stars = np.unique(hosts[fit])
                keep = np.isin(hosts[fit], rng.choice(stars, int(share * len(stars)), replace=False))
                part = fit[keep]
                m = xgb.train(params, xgb.DMatrix(X[part], Y[part], feature_names=names), num_boost_round=300)
                p = m.predict(xgb.DMatrix(X[held], feature_names=names))
                pv = m.predict(xgb.DMatrix(Xv, feature_names=names))
                cv.append(roc_auc_score(Y[held], p))
                on_val.append(roc_auc_score(Yv, pv))
                cut = np.quantile(p[Y[held] == 1], 0.02)
                caught.append((p[Y[held] == 0] < cut).mean())
        print(f"  {share:>4.0%} of stars: held-out AUC {np.mean(cv):.4f}, val AUC {np.mean(on_val):.4f}, "
              f"FPs caught at 2% of planets lost {np.mean(caught):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--curve", action="store_true", help="also the learning curve (some minutes of CPU)")
    args = ap.parse_args()
    recs = records()
    rejections(recs)
    table_vs_rank(recs)
    policy(recs)
    stack(recs)
    if args.curve:
        curve()


if __name__ == "__main__":
    main()
